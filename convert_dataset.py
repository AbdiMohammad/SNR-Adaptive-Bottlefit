import h5py
import numpy as np
import torch

dataset_directory = "/media/mohammad/Data/dataset/radioml2018/2018.01/"
dataset_directory = "/home/admin/dataset/radioml2018/2018.01/"
dataset_filename = "GOLD_XYZ_OSC.0001_1024.hdf5"

file = h5py.File(dataset_directory + dataset_filename, 'r')
data = np.array(file['X'])
label = np.array(file['Y'])
snrs = np.array(file['Z'])

torch.save(torch.permute(torch.from_numpy(data), (0, 2, 1)), f"{dataset_directory}data.pt")
torch.save(torch.from_numpy(label).argmax(1), f"{dataset_directory}label.pt")
torch.save(torch.from_numpy(snrs), f"{dataset_directory}snrs.pt")