# This file contains data utility functions for preparing datasets.
import torch


class PrepareDataset(torch.utils.data.Dataset):
    """A simple dataset class to prepare data for training.
    inputs:
        data_path: Path to the dataset file.
    """
    def __init__(self, data_path):
        # Load and preprocess the dataset from the given path
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = f.readlines()
        # Clean the data
        self.data = [line.strip() for line in self.data]

    def __len__(self):
        return len(self.data)

    