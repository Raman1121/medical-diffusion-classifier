import torch
from PIL import Image
import random
import pandas as pd
from pathlib import Path

class RetinalFundusDatasetCLIP(torch.utils.data.Dataset):
    def __init__(
        self,
        df,
        tokenizer=None,
        transform=None,
        seed=42,
        img_path_key='path',
        caption_col_key='Text',
    ):
        
        pass

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        pass