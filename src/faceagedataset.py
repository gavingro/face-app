import os
from PIL import Image

import torch
import numpy as np
import pandas as pd

from .hyperparameters import DEVICE


# Make Custom Dataset for our Faces and Age
class FaceAgeDataset(torch.utils.data.Dataset):
    """
    Face - Age UTKFace dataset.

    Learning from
    https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
    """

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.ages_df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.ages_df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.ages_df.iloc[idx, 0])
        # image = io.imread(img_name)
        image = Image.open(img_name)
        ages = self.ages_df.iloc[idx, 1]
        ages = np.array(ages)
        ages = torch.tensor(ages, dtype=torch.float)

        if self.transform:
            image = self.transform(image)
        return image.to(DEVICE), ages.to(DEVICE)
