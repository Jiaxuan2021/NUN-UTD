import scipy.io as sio
import numpy as np
import torch
from torch.utils.data import Dataset


class RiverDataset(Dataset):
    def __init__(self, data, data_name, is_pseudo):
        num_bands = data.shape[-1]
        if is_pseudo:  # add water mask, only water
            try:
                water_mask = np.load(fr'water_mask/NDWI_{data_name}.npy')  
                data = data[np.where(water_mask == 0)]
                self.data = np.reshape(data, (-1, num_bands)) 
            except FileNotFoundError:
                print("There is no water mask file.")
                self.data = np.reshape(data, (-1, num_bands))
        else:
            self.data = np.reshape(data, (-1, num_bands))

    def __getitem__(self, index):
        data = self.data[index]
        data = torch.tensor(data)
        return data
    
    def __len__(self):
        return self.data.shape[0]


    