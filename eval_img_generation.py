import argparse
import os
import pandas as pd
import torch
import torchvision.transforms as transforms
import torchvision.utils as vutils
from tqdm import tqdm
from PIL import Image
from diffusers import StableDiffusionPipeline
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure
from torchmetrics.image.psnr import PeakSignalNoiseRatio
from torchvision.models import inception_v3
import glob

# Function to generate images
def generate_images(model, df, prompts, output_dir, device):
    output_dir = os.path.join(output_dir, "images")
    os.makedirs(output_dir, exist_ok=True)
    images = []
    
    for i, prompt in tqdm(enumerate(prompts), total=len(prompts), desc="Generating Images"):
        image = model(prompt).images[0]  # Generate image
        image_path = os.path.join(output_dir, f"image_{i}.png")
        image.save(image_path)
        images.append(image_path)

    return images

# Function to load images as tensors
# def load_images(image_paths, device):
#     transform = transforms.Compose([
#         transforms.Resize((299, 299)),  # Resize for InceptionV3
#         transforms.ToTensor()
#     ])
#     # images = torch.stack([transform(vutils.read_image(img_path).convert("RGB")) for img_path in image_paths])
#     images = torch.stack([transform(Image.open(img_path).convert("RGB")) for img_path in image_paths])

#     return images.to(device)

def load_images(image_paths, device):
    transform = transforms.Compose([
        transforms.Resize((299, 299)),  # Resize for InceptionV3
        transforms.ToTensor()
    ])

    images = []
    for img_path in image_paths:
        img = Image.open(img_path).convert("RGB")  # Open image with PIL
        img_tensor = transform(img)  # Convert to tensor (float32)
        img_tensor = (img_tensor * 255).byte()  # Convert to uint8 (0-255)
        images.append(img_tensor)

    images = torch.stack(images).to(device)  # Stack and move to device
    return images, image_paths

# Function to compute metrics
def compute_metrics(generated_images, real_images, device):
    fid = FrechetInceptionDistance().to(device)
    kid = KernelInceptionDistance().to(device)
    ssim = StructuralSimilarityIndexMeasure().to(device)
    psnr = PeakSignalNoiseRatio().to(device)

    # Ensure we have at least 2 images in both real and generated sets
    if real_images.shape[0] < 2 or generated_images.shape[0] < 2:
        raise ValueError("FID and KID require at least two real and two generated images.")

    # Compute FID and KID by comparing real vs. generated images
    fid.update(real_images, real=True)
    fid.update(generated_images, real=False)

    # kid.update(real_images, real=True)
    # kid.update(generated_images, real=False)

    # Compute FID & KID scores
    fid_score = fid.compute().item()
    # kid_score = kid.compute().item()

    # Compute SSIM & PSNR (self-comparison among generated images)
    # ssim_score = ssim(generated_images, generated_images).item()
    psnr_score = psnr(generated_images, generated_images).item()

    return {
        "FID": fid_score,
        #  "KID": kid_score,
        # "SSIM": ssim_score,
        "PSNR": psnr_score
    }

def main():
    parser = argparse.ArgumentParser(description="Evaluate Stable Diffusion Image Generation")
    parser.add_argument("--model_name", type=str, default="CompVis/stable-diffusion-v1-4", help="Pretrained model name or path")
    parser.add_argument("--test_csv", type=str, required=True, help="CSV file containing text prompts")
    parser.add_argument("--prompt_key", type=str, required=False, default='Text', help="Key indicating the prompt column in the CSV")
    parser.add_argument("--output_dir", type=str, default="RESULTS/T2I_generation_results", help="Directory to save generated images")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    pipeline = StableDiffusionPipeline.from_pretrained(args.model_name).to(device)

    # Read prompts from CSV
    prompts_df = pd.read_csv(args.test_csv)
    prompts = prompts_df[args.prompt_key].tolist()

    # Generate images
    image_paths, _ = generate_images(pipeline, prompts_df, prompts, args.output_dir, device)
    real_img_paths = prompts_df["path"].tolist()

    # Load images as tensors
    generated_images, _ = load_images(image_paths, device)
    real_images, _ = load_images(real_img_paths, device)

    # Compute and print metrics
    metrics = compute_metrics(generated_images, real_images, device)
    print(metrics)

    # Save results to CSV
    results_df = pd.DataFrame({
        "FID": [metrics["FID"]],
        # "KID": [metrics["KID"]],
        # "SSIM": [metrics["SSIM"]],
        "PSNR": [metrics["PSNR"]]
    })

    results_df.to_csv(os.path.join(args.output_dir, "FID_results_"+args.test_csv.split("/")[-1]), index=False)
    print("Results saved to CSV at ", os.path.join(args.output_dir, "FID_results_"+args.test_csv.split("/")[-1]))
    
if __name__ == "__main__":
    main()
