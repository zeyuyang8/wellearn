import numpy as np
import torch
from torch.utils.data import Dataset

class TabularDataset(Dataset):
    """
    Container for a tabular data set
    """

    def __init__(self, x_data, y_data, 
                 transform=torch.from_numpy, 
                 target_transform=torch.from_numpy):
        """
        Input: 2D numpy array x_data, y_data, optional function transform, target_transform
        Output: Object storing a tabular data set
        """
        
        self._inputs = x_data
        self._outputs = y_data
        self._transform = transform
        self._target_transform = target_transform

    def __getitem__(self, idx):
        """ Loads and returns a sample from the dataset at the given index idx """
        
        input = self._inputs[idx, :]
        output = np.asarray(self._outputs[idx])
        if self._transform:
            input = self._transform(input)
        if self._target_transform:
            output = self._target_transform(output)
        return input, output

    def __len__(self):
        """ Returns the number of samples in our dataset """

        return len(self._outputs)