import numpy as np
import h5py
import os
from datetime import datetime

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils import data

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from resnet1D_torch import ResNet50

import resnet_radioml, resnet1D_radioml, resnet_amc

import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--batch-size", type=int, default=512, help="Input batch size on each device (default: 512)")
parser.add_argument("--total-epochs", type=int, default=120, help="Total epochs to train the model (default: 120)")
parser.add_argument("--save-every", type=int, default=10, help="How often to save a snapshot (default: 10)")
parser.add_argument("--output-dir", type=str, default="./resource/ckpt/", help="Directory to save a snapshot")


args = parser.parse_args()


def load_split_dataset(dataset_directory, snr):
    for filename in os.listdir(dataset_directory):
        if f'_{snr}' not in filename:
            continue
        file = h5py.File(dataset_directory + filename, 'r')
        data = np.array(file['data'])
        label = np.array(file['label'])
        dataset = torch.utils.data.TensorDataset(torch.permute(torch.from_numpy(data), (0, 2, 1)), torch.from_numpy(label).argmax(1))
        train_ds, val_ds = torch.utils.data.random_split(dataset, [0.8, 0.2])
        return train_ds, val_ds

def load_dataset_snr(dataset_path, snr):
    file = h5py.File(dataset_path, 'r')
    data = np.array(file['X'])
    label = np.array(file['Y'])
    snrs = np.array(file['Z'])
    dataset = torch.utils.data.TensorDataset(torch.permute(torch.from_numpy(data), (0, 2, 1)), torch.from_numpy(label).argmax(1))
    dataset = torch.utils.data.Subset(dataset, (snrs == snr).nonzero()[0])
    train_ds, val_ds = torch.utils.data.random_split(dataset, [0.8, 0.2])
    return train_ds, val_ds

def load_dataset(dataset_path):
    file = h5py.File(dataset_path, 'r')
    data = np.array(file['X'])
    label = np.array(file['Y'])
    snrs = np.array(file['Z'])
    dataset = torch.utils.data.TensorDataset(torch.permute(torch.from_numpy(data), (0, 2, 1)), torch.from_numpy(label).argmax(1))
    datasets_snr = {}
    train_ds_snr = {}
    val_ds_snr = {}
    for snr in np.unique(snrs):
        datasets_snr[snr] = torch.utils.data.Subset(dataset, (snrs == snr).nonzero()[0])
        train_ds_snr[snr], val_ds_snr[snr] = torch.utils.data.random_split(datasets_snr[snr], [0.8, 0.2])        
    train_ds, val_ds = torch.utils.data.ConcatDataset(train_ds_snr.values()), torch.utils.data.ConcatDataset(val_ds_snr.values())
    return train_ds, val_ds

class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        val_data: DataLoader,
        loss_fn,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: torch.optim.lr_scheduler,
        save_every: int,
        output_dir: str,
        es_patience: int
    ) -> None:
        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.global_rank = int(os.environ["RANK"])
        self.model = model.to(self.local_rank)
        self.train_data = train_data
        self.val_data = val_data
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.save_every = save_every
        self.epochs_run = 0
        self.output_dir = output_dir
        self.es_patience = es_patience

        self.best_val_loss = float('inf')
        self.last_val_loss = float('inf')
        self.es_cnt = 0
        if os.path.exists(self.output_dir + "snapshot.pt"):
            print("Loading snapshot")
            self._load_snapshot(self.output_dir + "snapshot.pt")

        self.model = DDP(self.model, device_ids=[self.local_rank])

    def _load_snapshot(self):
        loc = f"cuda:{self.local_rank}"
        snapshot = torch.load(self.output_dir + "snapshot.pt", map_location=loc)
        self.model.load_state_dict(snapshot["MODEL_STATE"])
        self.epochs_run = snapshot["EPOCHS_RUN"]
        print(f"Resuming training from snapshot at Epoch {self.epochs_run}")

    def _run_batch(self, source, targets):
        self.optimizer.zero_grad()
        output = self.model(source)
        loss = self.loss_fn(output, targets)
        loss.backward()
        self.optimizer.step()
        return loss

    def _run_epoch(self, epoch):
        size = len(self.train_data.dataset)
        print(f"[GPU{self.global_rank}] Epoch {epoch} | Batchsize: {len(next(iter(self.train_data))[0])} | Steps: {len(self.train_data)}")
        self.train_data.sampler.set_epoch(epoch)
        running_loss = 0.0
        last_loss = 0.0
        self.model.train()
        for batch, (X, y) in enumerate(self.train_data):
            X, y = X.to(self.local_rank), y.to(self.local_rank)

            loss = self._run_batch(X, y)

            running_loss += loss.item()
            if batch % 100 == 99:
                last_loss, loss, current = running_loss / 100.0, loss.item(), (batch + 1) * len(X)
                print(f"avg loss: {last_loss:>7f} loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
                running_loss = 0.0
        return last_loss

    def _validate(self):
        size = len(self.val_data.dataset)
        num_batches = len(self.val_data)
        self.model.eval()
        val_loss, val_acc = 0, 0
        with torch.no_grad():
            for X, y in self.val_data:
                X, y = X.to(self.local_rank), y.to(self.local_rank)
                pred = self.model(X)
                val_loss += self.loss_fn(pred, y).item()
                val_acc += (pred.argmax(1) == y).type(torch.float).sum().item()
        val_loss /= num_batches
        val_acc /= size
        print(f"Validation Error: \n Accuracy: {(100*val_acc):>0.1f}%, Avg loss: {val_loss:>8f} \n")
        return val_loss, val_acc

    def _save_snapshot(self, epoch):
        snapshot = {
            "MODEL_STATE": self.model.module.state_dict(),
            "EPOCHS_RUN": epoch,
        }
        torch.save(snapshot, self.output_dir + "snapshot.pt")
        print(f"Epoch {epoch} | Training snapshot saved at {self.output_dir}snapshot.pt")

    def _save_best(self):
        torch.save(self.model.module, f"{self.output_dir}best_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt")

    def train(self, max_epochs: int):
        for epoch in range(self.epochs_run, max_epochs):
            self._run_epoch(epoch)
            if self.local_rank == 0 and epoch % self.save_every == 0:
                self._save_snapshot(epoch)

    def train_and_validate(self, max_epochs: int):
        for epoch in range(self.epochs_run, max_epochs):
            print(f"Epoch {epoch+1}\n-------------------------------")
            self._run_epoch(epoch)
            val_loss, val_acc = self._validate()
            self.lr_scheduler.step(val_loss)
            if self.local_rank == 0 and epoch % self.save_every == 0:
                self._save_snapshot(epoch)
            if self.local_rank == 0 and val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self._save_best()
            if self.last_val_loss < val_loss:
                self.es_cnt += 1
            else:
                self.es_cnt = 0
            if self.es_cnt > self.es_patience:
                print(f"Early Stopping: Validation loss doesn't decrease after {self.es_patience} epochs")
                break
            self.last_val_loss = val_loss

def load_checkpoint(ckpt_directory):
    latest_checkpoint = datetime.min
    for filename in os.listdir(ckpt_directory):
        if 'model_' in filename:
            time = filename.replace('model_', '')
            time = datetime.strptime(time, '%Y%m%d_%H%M%S')
            if time > latest_checkpoint:
                latest_checkpoint = time
    model = None
    if latest_checkpoint != datetime.min:
        model = torch.load('{}model_{}'.format(ckpt_directory, latest_checkpoint.strftime('%Y%m%d_%H%M%S')))
    return model

def main():
    ddp_setup()

    dataset_directory = "/media/mohammad/Data/dataset/radioml2018/2018.01/"
    # dataset_directory = "/home/admin/dataset/radioml2018/2018.01/"
    dataset_filename = "GOLD_XYZ_OSC.0001_1024.hdf5"
    split_dataset_directory = "/media/mohammad/Data/radioml/split_dataset/"
    # split_dataset_directory = "/home/admin/dataset/radioml2018/split_dataset/"

    # train_ds, val_ds = load_split_dataset(split_dataset_directory, 20)
    # train_ds, val_ds = load_dataset_snr(dataset_directory + dataset_filename, 20)
    train_ds, val_ds = load_dataset(dataset_directory + dataset_filename)

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, pin_memory=True, shuffle=False, sampler=DistributedSampler(train_ds))
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=args.batch_size, pin_memory=True, shuffle=False, sampler=DistributedSampler(val_ds))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = None
    # model = load_checkpoint('./resource/ckpt/')
    if model == None:
        model = resnet_amc.ResNet_AMC()

    loss_fn = torch.nn.CrossEntropyLoss() 
    optimizer = torch.optim.Adam(model.parameters())
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5, min_lr=0.0000001)

    trainer = Trainer(model, train_loader, val_loader, loss_fn, optimizer, lr_scheduler, save_every=args.save_every, output_dir=args.output_dir, es_patience=10)

    trainer.train_and_validate(args.total_epochs)
    destroy_process_group()
    print("Done!")

def ddp_setup():
    init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

if __name__ == '__main__':
    os.system("mkdir -p ./resource/ckpt/")
    main()
