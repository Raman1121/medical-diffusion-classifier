import torch
from transformers import AutoTokenizer, CLIPTokenizer
from datasets import load_dataset, Dataset, Features, Value, Array3D, Sequence
from diffusers import StableDiffusionPipeline, AutoencoderKL
import pandas as pd
import numpy as np
import os
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser(description='Preprocess data for training')
parser.add_argument('--csv_file', type=str, help='Path to the CSV file containing the image paths and text captions')
parser.add_argument('--pretrained_model_name_or_path', type=str, default="CompVis/stable-diffusion-v1-4", help='Name or path of the pretrained model')
parser.add_argument('--resolution', type=int, default=512, help='Resolution of the images')
parser.add_argument('--center_crop', action='store_true', help='Whether to center crop the images')
parser.add_argument('--random_flip', action='store_true', help='Whether to randomly flip the images')
# parser.add_argument('--output_dir', type=str, help='Path to save the processed dataset')
parser.add_argument('--dataset_type', type=str, default='iid', help='Type of dataset (iid or ood)')
args = parser.parse_args()

PATH_MAPPING = {
    'iid': '/raid/s2198939/Fundus_Images/In-Distribution-Splits',
    'ood': '/raid/s2198939/Fundus_Images/OOD-Splits'
}

# Load the CSV file
csv_file = args.csv_file
data_df = pd.read_csv(csv_file)
pretrained_model_name_or_path = args.pretrained_model_name_or_path
resolution = args.resolution
center_crop = args.center_crop
random_flip = args.random_flip

output_dir = os.path.join(PATH_MAPPING[args.dataset_type], 'Saved_datasets', csv_file.split('/')[-1].split('.')[0])
# output_dir = '/raid/s2198939/Fundus_Images/In-Distribution-Splits/Saved_datasets/' + csv_file.split('/')[-1].split('.')[0]
# output_dir = output_dir + "_" + csv_file.split('/')[-1].split('.')[0]
os.makedirs(output_dir, exist_ok=True)

# Load the Stable Diffusion Pipeline for the VAE model
vae = AutoencoderKL.from_pretrained(
            pretrained_model_name_or_path, subfolder="vae"
        )
vae.eval()  # Set VAE to evaluation mode

# Load the  tokenizer
tokenizer = CLIPTokenizer.from_pretrained(
        pretrained_model_name_or_path, subfolder="tokenizer", revision=None
    )

image_transform = transforms.Compose(
        [
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(resolution) if center_crop else transforms.RandomCrop(resolution),
            transforms.RandomHorizontalFlip() if random_flip else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            # transforms.Normalize([0.5], [0.5]),
        ]
    )

# Function to encode images
def encode_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = image_transform(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        latents = vae.encode(image).latent_dist.sample() * vae.config.scaling_factor
    return latents.squeeze(0).cpu().numpy()

# Function to tokenize 
def tokenize_text():
    tokens = tokenizer(text, return_tensors='np', padding='max_length', truncation=True, max_length=77)
    return tokens.input_ids.squeeze(0)  

encoded_images = []
tokenized_texts = []

# Wrap the following in tqdm to get a progress bar
for idx, row in tqdm(data_df.iterrows(), total=len(data_df)):
    # Encode the image

    if(idx % 100 == 0):
        print(f"Processing image {idx} of {len(data_df)}")
    image_path = row['path']
    latents = encode_image(image_path)
    encoded_images.append(latents)

    # Tokenize the text
    text = row['Text']
    tokens = tokenize_text()
    tokenized_texts.append(tokens)

# Define the dataset features
features = Features({
    'image_latents': Array3D(dtype='float32', shape=(4, 64, 64)),  # Shape of VAE latent space
    'text_tokens': Sequence(Value(dtype='int64'))  # Token IDs of text captions
})

# Create the dataset
dataset = Dataset.from_dict({
    'image_latents': encoded_images,
    'text_tokens': tokenized_texts
}, features=features)

dataset.save_to_disk(output_dir)

print(f"Dataset saved to {output_dir}")