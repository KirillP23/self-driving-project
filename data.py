from pathlib import Path

from PIL import Image
import numpy as np

from torch.utils.data import Dataset
from matplotlib import pyplot as plt

class Udacity(Dataset):
    def __init__(self, data_path, df_data, transform):
        self.df_data = df_data
        self.data_path = data_path
        self.transform = transform

    def __len__(self):
        return len(self.df_data)

    def __getitem__(self, idx):
        centre_image = np.array(Image.open((str(Path(self.data_path, self.df_data.iloc[idx][0])))))
        if self.transform:
            centre_image = self.transform(centre_image)
        centre_steering = float(self.df_data.iloc[idx][3])
        if idx % 2 == 1:
            centre_steering = -centre_steering
        return centre_image, centre_steering
