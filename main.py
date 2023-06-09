import numpy as np
import h5py
import os
from datetime import datetime

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils import data

from resnet1D_torch import ResNet50

import resnet_radioml, resnet_1d, resnet_amc

# import args

import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--snrs", nargs="*", type=int, default=None, help="List of SNRs to train on (default: None)")
parser.add_argument("--batch-size", type=int, default=1024, help="Input batch size on each device (default: 1024)")
parser.add_argument("--total-epochs", type=int, default=100, help="Total epochs to train the model (default: 100)")
parser.add_argument("--output-dir", type=str, default="./resource/ckpt/", help="Directory to save the best model checkpoint")

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

def load_dataset(dataset_path, snrs=None):
    file = h5py.File(dataset_path, 'r')
    data = np.array(file['X'])
    label = np.array(file['Y'])
    all_snrs = np.array(file['Z'])
    dataset = torch.utils.data.TensorDataset(torch.permute(torch.from_numpy(data), (0, 2, 1)), torch.from_numpy(label).argmax(1))
    datasets_snr = {}
    train_ds_snr = {}
    val_ds_snr = {}
    if snrs == None:
        snrs = np.unique(all_snrs)
    generator1 = torch.Generator().manual_seed(1994)
    for snr in snrs:
        datasets_snr[snr] = torch.utils.data.Subset(dataset, (all_snrs == snr).nonzero()[0])
        train_ds_snr[snr], val_ds_snr[snr] = torch.utils.data.random_split(datasets_snr[snr], [0.8, 0.2], generator=generator1)        
    # train_ds, val_ds = torch.utils.data.ConcatDataset(train_ds_snr.values()), torch.utils.data.ConcatDataset(val_ds_snr.values())
    return train_ds_snr, val_ds_snr

def train_one_epoch(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    running_loss = 0.0
    last_loss = 0.0
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if batch % 100 == 99:
            last_loss, loss, current = running_loss / 100.0, loss.item(), (batch + 1) * len(X)
            print(f"avg loss: {last_loss:>7f} loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            running_loss = 0.0
    return last_loss

def test(dataloader, model, loss_fn, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    return correct, test_loss

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
    dataset_directory = "/media/mohammad/Data/dataset/radioml2018/2018.01/"
    # dataset_directory = "/home/admin/dataset/radioml2018/2018.01/"
    dataset_filename = "GOLD_XYZ_OSC.0001_1024.hdf5"
    split_dataset_directory = "/media/mohammad/Data/radioml/split_dataset/"
    # split_dataset_directory = "/home/admin/dataset/radioml2018/split_dataset/"

    train_ds, val_ds = load_dataset(dataset_directory + dataset_filename, args.snrs)

    train_loader = torch.utils.data.DataLoader(torch.utils.data.ConcatDataset(train_ds.values()), batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = torch.utils.data.DataLoader(torch.utils.data.ConcatDataset(val_ds.values()), batch_size=args.batch_size, shuffle=False, num_workers=4)
    val_loader_snr = {}
    for snr in val_ds:
        val_loader_snr[snr] = torch.utils.data.DataLoader(val_ds[snr], batch_size=args.batch_size, shuffle=False, num_workers=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = None
    # model = load_checkpoint('./resource/ckpt/')
    if model == None:
        model = resnet_1d.resnet18()
    model = nn.DataParallel(model)
    model.to(device)

    loss_fn = torch.nn.CrossEntropyLoss() 
    optimizer = torch.optim.Adam(model.parameters())
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5, min_lr=0.0000001)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    es_patience = 10
    best_val_loss = float('inf')
    last_val_loss = float('inf')
    earlystop_cnt = 0
    accuracy_snr = {}
    val_loss_snr = {}
    for t in range(args.total_epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_one_epoch(train_loader, model, loss_fn, optimizer, device)
        accuracy, val_loss = test(val_loader, model, loss_fn, device)
        print(f"Validation Error:\n\tAccuracy: {(100*accuracy):>0.1f}%, Avg loss: {val_loss:>8f} \n")
        scheduler.step(val_loss)
        # Track best performance, and save the model's state
        if val_loss < best_val_loss:
            for snr in val_loader_snr:
                accuracy_snr[snr], val_loss_snr[snr] = test(val_loader_snr[snr], model, loss_fn, device)
                print(f"Validation Error: SNR: {snr}\n\tAccuracy: {(100*accuracy_snr[snr]):>0.1f}%, Avg loss: {val_loss_snr[snr]:>8f} \n")
            best_val_loss = val_loss
            os.system("mkdir -p " + args.output_dir)
            model_path = f'{args.output_dir}model_{args.snrs}.pt'
            model_state_path = f'{args.output_dir}model_{args.snrs}_state.pt'
            torch.save(model.module, model_path)
            torch.save(model.module.state_dict(), model_state_path)
        if last_val_loss < val_loss:
            earlystop_cnt += 1
        else:
            earlystop_cnt = 0
        if earlystop_cnt > es_patience:
            print(f"Early Stopping: Validation loss doesn't decrease after {es_patience} epochs")
            break
        last_val_loss = val_loss
    print("Done!")


if __name__ == '__main__':
    main()
