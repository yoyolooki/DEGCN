import torch
import pandas as pd
from torch.utils.data import Dataset


class Omics_Data_VAE(Dataset):
    def __init__(self, filepath1, filepath2, filepath3):
        omics1 = pd.read_csv(filepath1, sep=',', header=0, index_col=None)
        omics1 = omics1.iloc[:, 1:]  # Column 0 is sample names

        omics2 = pd.read_csv(filepath2, sep=',', header=0, index_col=None)
        omics2 = omics2.iloc[:, 1:]

        omics3 = pd.read_csv(filepath3, sep=',', header=0, index_col=None)
        omics3 = omics3.iloc[:, 1:]

        self.length = omics1.shape[0]

        self.omics1_data = torch.tensor(omics1.values, dtype=torch.float32)
        self.omics2_data = torch.tensor(omics2.values, dtype=torch.float32)
        self.omics3_data = torch.tensor(omics3.values, dtype=torch.float32)

    def __getitem__(self, index):
        return self.omics1_data[index], self.omics2_data[index], self.omics3_data[index]

    def __len__(self):
        return self.length


class Omics_Data_GCN(Dataset):
    def __init__(self, path_laten_data):
        laten_data = pd.read_csv(path_laten_data, sep=',', header=0, index_col=None)
        laten_data = laten_data.iloc[:, 1:]  # 第0列是样本名

        self.length = laten_data.shape[0]
        self.laten_data = torch.tensor(laten_data.values, dtype=torch.float32)

    def __getitem__(self, index):
        return self.laten_data[index]

    def __len__(self):
        return self.length
