import os
import numpy as np
import torch
from torch.utils.data import Dataset


class TrainDataset(Dataset):
    def __init__(self, in_path,out_path,ptId):
        super().__init__()
        self.in_path = in_path
        self.out_path = out_path

        fpath = 'classtxt/class'+str(ptId)+'.txt'
        with open(fpath, 'r') as f:
            lines = f.readlines()
        lines = [line.strip() for line in lines]
        if len(lines)>50:
            lines = lines[:50]
        self.tensor_paths = lines

        self.label_paths = lines


    def __len__(self):
        return len(self.tensor_paths)

    def __getitem__(self, idx):
        tensor_path = self.tensor_paths[idx]
        label_path = self.label_paths[idx]

        tensor = torch.load(os.path.join(self.in_path, tensor_path))

        label = torch.load(os.path.join(self.out_path, label_path))

        return tensor, label

class TrainsampleDataset(Dataset):
    def __init__(self, in_path,out_path,sample):
        super().__init__()
        self.in_path = in_path
        self.out_path = out_path
        self.uvi = np.random.choice(260000, sample, replace=False)
        print(self.uvi[0])
        self.tensor_paths = os.listdir(self.in_path)

        self.label_paths = os.listdir(self.out_path)


    def __len__(self):
        return len(self.uvi)

    def __getitem__(self, idx):
        tensor_path = self.tensor_paths[self.uvi[idx]]
        label_path = self.label_paths[self.uvi[idx]]
        tensor = torch.load(os.path.join(self.in_path, tensor_path))
        label = torch.load(os.path.join(self.out_path, label_path))

        return tensor, label

class TrainvoxelDataset(Dataset):
    def __init__(self, in_path,out_path,vox_path):
        super().__init__()
        self.in_path = in_path
        self.out_path = out_path
        self.vox_path = vox_path
        self.tensor_paths = os.listdir(self.in_path)
        self.vox_paths = os.listdir(self.vox_path)
        self.label_paths = os.listdir(self.out_path)


    def __len__(self):
        return len(self.tensor_paths)

    def __getitem__(self, idx):
        path = self.tensor_paths[idx]

        tensor = torch.load(os.path.join(self.in_path, path))
        vox = torch.load(os.path.join(self.vox_path, path))
        label = torch.load(os.path.join(self.out_path, path))

        return tensor, label,vox


class TrainmeanDataset(Dataset):
    def __init__(self, in_path,out_path):
        super().__init__()
        self.in_path = in_path
        self.out_path = out_path

        self.tensor_paths = os.listdir(self.in_path)

        self.label_paths = os.listdir(self.out_path)


    def __len__(self):
        return len(self.tensor_paths)

    def __getitem__(self, idx):
        tensor_path = self.tensor_paths[idx]
        label_path = self.label_paths[idx]
        tensor = torch.load(os.path.join(self.in_path, tensor_path))
        label = torch.load(os.path.join(self.out_path, label_path))

        return tensor, label


class TestDataset(Dataset):
    def __init__(self, in_path):
        super().__init__()
        self.in_path = in_path
        self.tensor_paths = os.listdir(self.in_path)



    def __len__(self):
        return len(self.tensor_paths)

    def __getitem__(self, idx):
        tensor_path = self.tensor_paths[idx]

        tensor = torch.load(os.path.join(self.in_path, tensor_path))


        return tensor