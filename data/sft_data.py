import torch
from PIL import Image
import random
import pandas as pd
from pathlib import Path

class RetinalFundusDatasetSFT(torch.utils.data.Dataset):
    def __init__(
        self,
        df,
        transform=None,
        seed=42,
        img_path_key='path',
        diagnosis_col_key='Unhealthy',
    ):
        
        self.df = df
        self.transform = transform
        self.img_path_key = img_path_key
        self.diagnosis_col_key = diagnosis_col_key

        random.seed(seed)

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.df[self.img_path_key].iloc[idx]
        im = Image.open(img_path).convert("RGB")
        if self.transform:
            im = self.transform(im)

        return im, self.df[self.diagnosis_col_key].iloc[idx]