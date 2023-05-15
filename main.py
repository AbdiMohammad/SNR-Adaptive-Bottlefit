import numpy as np
import h5py
import os
from datetime import datetime

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils import data

from resnet1D_torch import ResNet50

import resnet_radioml


def load_dataset(dataset_directory, snr):
    for filename in os.listdir(dataset_directory):
        if f'_{snr}' not in filename:
            continue
        file = h5py.File(dataset_directory + filename, 'r')
        data = np.array(file['data'])
        label = np.array(file['label'])
        dataset = torch.utils.data.TensorDataset(torch.permute(torch.from_numpy(data), (0, 2, 1)).unsqueeze(1), torch.from_numpy(label).argmax(1))
        train_ds, val_ds = torch.utils.data.random_split(dataset, [0.8, 0.2])
        return train_ds, val_ds

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
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
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
    dataset_directory = '/media/mohammad/Data/dataset/radioml2018/2018.01/'
    dataset_filename = 'GOLD_XYZ_OSC.0001_1024.hdf5'
    split_dataset_directory = '/media/mohammad/Data/radioml/dataset_splited/'
    split_dataset_directory = '/home/admin/dataset/radioml2018/split_dataset/'

    train_ds, val_ds = load_dataset(split_dataset_directory, 20)
    batch_size = 512
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = None
    # model = load_checkpoint('./resource/ckpt/')
    if model == None:
        model = resnet_radioml.resnet18()
    print(model)
    model.to(device)

    loss_fn = torch.nn.CrossEntropyLoss() 
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0005, momentum=0.9)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    epochs = 100
    best_val_loss = float('inf')
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_one_epoch(train_loader, model, loss_fn, optimizer, device)
        accuracy, val_loss = test(val_loader, model, loss_fn, device)
        # Track best performance, and save the model's state
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_path = './resource/ckpt/model_{}'.format(timestamp)
            torch.save(model, model_path)
    print("Done!")


if __name__ == '__main__':
    main()
