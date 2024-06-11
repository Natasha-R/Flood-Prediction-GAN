import models
import utils

import pandas as pd
import numpy as np
import argparse
import os
import tifffile as tf
import matplotlib.pyplot as plt

import torch
from torchvision.transforms import Resize

device = "cuda" if torch.cuda.is_available() else "cpu"

def main(model,
        image,
        saved_model_path,
        data_path,
        dem,
        resize,
        crop,
        crop_index,
        save_generator_output,
        save_attention_mask):
    
    saved_model = torch.load(saved_model_path)
    epoch = saved_model["starting_epoch"] - 1
    add_identity_loss = False
    not_input_topography = saved_model["not_input_topography"]
    if not_input_topography:
        input_channels = 3
    else: # if input_topography
        input_channels = 9
        
    if model.lower()=="pix2pix":
        generator = models.Pix2PixGenerator(input_channels=input_channels).to(device)
        generator.load_state_dict(saved_model["generator"])
        generator.eval()
    elif model.lower() == "cyclegan" or model.lower()=="attentiongan":
        if model.lower() == "cyclegan":
            generator_architecture = models.CycleGANGenerator
            generator = generator_architecture(input_channels=input_channels).to(device)
        else: # if model.lower()=="attentiongan":
            generator_architecture = models.AttentionGANGenerator
            generator = generator_architecture(input_channels=input_channels, save_attention_mask=True).to(device)
        add_identity_loss = saved_model["add_identity_loss"]
        generator.load_state_dict(saved_model["pre_to_post_generator"])
        generator.eval()
    else:
        raise NotImplementedError("Model must be one of: Pix2Pix, CycleGAN or AttentionGAN")   
    
    dataset_split = pd.read_csv("dataset_split.csv")
    dem_string = dataset_split[dataset_split["image"]==image][f"{dem}_DEM"].head(1).item()
    input_path = f"{data_path}/data/dataset_input/{image}_{dem_string}.tif"
    input_image = torch.from_numpy(tf.imread(input_path).transpose(2, 0, 1))
    if not_input_topography:
        input_image = input_image[:3, :, :]
    if resize:
        input_image = Resize(resize, antialias=True)(input_image)
    if crop:
        channels, rows, cols = input_image.shape
        num_divisions = int(np.sqrt(crop))
        rows_size = rows // num_divisions
        cols_size = cols // num_divisions
        row_index = crop_index // num_divisions
        col_index = crop_index % num_divisions
        start_row = row_index * rows_size
        start_col = col_index * cols_size
        input_image = input_image[:, start_row:start_row + rows_size, start_col:start_col + cols_size]
    input_image = torch.unsqueeze(input_image, dim=0).to(device)
    generator_output = np.clip(generator(input_image).squeeze().cpu().detach().numpy().transpose(1, 2, 0), 0, 1)

    if save_generator_output:
        generator_output_path = utils.create_path("image", model, data_path, "generatorOutput", image, dem, not_input_topography, resize, crop, epoch, add_identity_loss)
        print(f"\nSaving generator output of image '{image}' to {generator_output_path}")
        plt.imsave(generator_output_path, generator_output, vmin=0, vmax=1)

    if save_attention_mask and model.lower()=="attentiongan":
        attention_mask = np.clip(generator.last_attention_mask.squeeze().cpu().detach().numpy(), 0, 1)
        attention_mask_path = utils.create_path("image", model, data_path, "attentionMask", image, dem, not_input_topography, resize, crop, epoch, add_identity_loss)
        print(f"\nSaving attention mask of image '{image}' to {attention_mask_path}")
        plt.imsave(attention_mask_path, attention_mask, cmap="gray", vmin=0, vmax=1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate and save output from a trained Pix2Pix, CycleGAN or AttentionGAN model.")
    parser.add_argument("--model", required=True, help="Model can be one of: Pix2Pix, CycleGAN or AttentionGAN")
    parser.add_argument("--image", required=True, help="The name of the input image")
    parser.add_argument("--saved_model_path", required=True, help="Path to the trained model")
    parser.add_argument("--data_path", required=True, help="The path to the location of the data folder")
    parser.add_argument("--dem", required=True, help="Whether the 'best' or 'same' DEM should be used in the input to the model")
    parser.add_argument("--resize", type=int, default=256, help="Resize the images to the given size. The resize is applied before the crop")
    parser.add_argument("--crop", type=int, default=None, help="Crop each image into the given number of images. The resize is applied before the crop")
    parser.add_argument("--crop_index", type=int, default=0, help="Select the quadrant of the cropped image to input. 0 is top left, 1 is top right, etc")
    parser.add_argument("--save_generator_output", default=False, action="store_true", help="Save the generator output")
    parser.add_argument("--save_attention_mask", default=False, action="store_true", help="Save the generation attention mask (for AttentionGAN only)")
    args = parser.parse_args()
    if not os.path.isfile(args.saved_model_path):
        raise FileNotFoundError("Saved model not found. Check the path to the saved model.")
    main(**vars(args))    