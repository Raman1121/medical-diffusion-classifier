import torch
from PIL import Image
import random
import pandas as pd
from pathlib import Path
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

class RetinalFundusDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        df,
        tokenizer=None,
        transform=None,
        seed=42,
        img_path_key='path',
        caption_col_key='Text',
    ):
        
        self.df = df
        self.transform = transform
        self.tokenizer = tokenizer
        self.img_path_key = img_path_key
        self.caption_col_key = caption_col_key

        random.seed(seed)

        if self.tokenizer is not None:
            self.tokens = self.tokenizer(
                self.df[self.caption_col_key].tolist(),
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
            )
            self.uncond_tokens = self.tokenizer(
                "",
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
            )

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.df[self.img_path_key].iloc[idx]
        im = Image.open(img_path).convert("RGB")
        if self.transform:
            im = self.transform(im)

        sample = {
            "pixel_values": im,
            "text": self.df[self.caption_col_key].iloc[idx],
        }

        if self.tokenizer is not None:
            input_ids, attention_mask = torch.LongTensor(
                    self.tokens.input_ids[idx]
                ), torch.LongTensor(self.tokens.attention_mask[idx])
            sample["input_ids"] = input_ids
            sample["attention_mask"] = attention_mask

        return sample
    
class MimicCXRDataset(torch.utils.data.Dataset):
    """Mimic CXR dataset."""

    def __init__(
        self,
        df,
        tokenizer=None,
        transform=None,
        seed=42,
        classifier_guidance_dropout=0.1,
        img_path_key='path',
        caption_col_key='text',
    ):
        
        self.transform = transform
        self.tokenizer = tokenizer
        self.classifier_guidance_dropout = classifier_guidance_dropout

        random.seed(seed)

        self.df = df
        self.img_path_key = img_path_key
        self.caption_col_key = caption_col_key

        assert all(
            [
                isinstance(text, str)
                for text in self.df["text"].to_list()
            ]
        ), "All text must be strings"

        if self.tokenizer is not None:
            self.tokens = self.tokenizer(
                self.df[self.caption_col_key].to_list(),
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
            )
            self.uncond_tokens = self.tokenizer(
                "",
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
            )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.df[self.img_path_key].iloc[idx]
        try:
            im = Image.open(img_path).convert("RGB")
        except:
            print("ERROR IN LOADING THE IMAGE {}".format(img_path))
        if self.transform:
            im = self.transform(im)
        
        sample = {
            "pixel_values": im,
            "text": self.df[self.caption_col_key].iloc[idx],
        }

        if self.tokenizer is not None:
            if random.randint(0, 100) / 100 < self.classifier_guidance_dropout:
                input_ids, attention_mask = torch.LongTensor(
                    self.uncond_tokens.input_ids
                ), torch.LongTensor(self.uncond_tokens.attention_mask)
            else:
                input_ids, attention_mask = torch.LongTensor(
                    self.tokens.input_ids[idx]
                ), torch.LongTensor(self.tokens.attention_mask[idx])
            sample["input_ids"] = input_ids
            sample["attention_mask"] = attention_mask

        return sample
    
class CheXpertDataset(torch.utils.data.Dataset):
    """Chexpert dataset."""

    def __init__(
        self,
        df,
        tokenizer=None,
        transform=None,
        seed=42,
        classifier_guidance_dropout=0.1,
        img_path_key='Path',
        caption_col_key='Simple_prompt',
    ):
        
        self.transform = transform
        self.tokenizer = tokenizer
        self.classifier_guidance_dropout = classifier_guidance_dropout

        random.seed(seed)

        self.df = df
        self.img_path_key = img_path_key
        self.caption_col_key = caption_col_key

        assert all(
            [
                isinstance(text, str)
                for text in self.df[self.caption_col_key].to_list()
            ]
        ), "All text must be strings"

        if self.tokenizer is not None:
            self.tokens = self.tokenizer(
                self.df[self.caption_col_key].to_list(),
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
            )
            self.uncond_tokens = self.tokenizer(
                "",
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
            )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.df[self.img_path_key].iloc[idx]
        try:
            im = Image.open(img_path).convert("RGB")
        except:
            print("ERROR IN LOADING THE IMAGE {}".format(img_path))
        if self.transform:
            im = self.transform(im)
        
        sample = {
            "pixel_values": im,
            "text": self.df[self.caption_col_key].iloc[idx],
        }

        if self.tokenizer is not None:
            if random.randint(0, 100) / 100 < self.classifier_guidance_dropout:
                input_ids, attention_mask = torch.LongTensor(
                    self.uncond_tokens.input_ids
                ), torch.LongTensor(self.uncond_tokens.attention_mask)
            else:
                input_ids, attention_mask = torch.LongTensor(
                    self.tokens.input_ids[idx]
                ), torch.LongTensor(self.tokens.attention_mask[idx])
            sample["input_ids"] = input_ids
            sample["attention_mask"] = attention_mask

        return sample