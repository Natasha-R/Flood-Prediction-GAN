import numpy as np

import torch
from torchvision.transforms import Resize, Normalize, InterpolationMode
device = "cuda" if torch.cuda.is_available() else "cpu"

def tensor_to_numpy(image):
    """
    Convert an image stored as a tensor to an image stored as a numpy array.
    """
    image = image.squeeze().cpu().detach().numpy()
    if len(image.shape)==3:
        image = image.transpose(1, 2, 0)[:, :, :3]
        image = (image + 1) * 0.5
        image = np.clip(image, 0, 1)
    return image

def apply_transformations(image_name,
                          input_image,
                          ground_truth,
                          not_input_topography,
                          resize,
                          crop,
                          to_loader=False,
                          crop_index=0):
    """
    Transform the input and output images, by resizing, cropping and normalisation.
    """
    if not_input_topography:
        input_image = input_image[:3, :, :]
    if resize:
        input_image = Resize(resize, antialias=True, interpolation=InterpolationMode.BICUBIC)(input_image)
        ground_truth = Resize(resize, antialias=True, interpolation=InterpolationMode.BICUBIC)(ground_truth)
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
        ground_truth = ground_truth[:, start_row:start_row + rows_size, start_col:start_col + cols_size]
        image_name = f"{image_name}_{crop_index}"
    if not_input_topography:
        input_image = Normalize(mean=(0.5, 0.5, 0.5), 
                        std=(0.5, 0.5, 0.5))(input_image)
    else:
        input_image = Normalize(mean=(0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5), 
                                std=(0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5))(input_image)
    ground_truth = Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))(ground_truth)

    if to_loader==False:
        input_image = torch.unsqueeze(input_image, dim=0).to(device)
        ground_truth = torch.unsqueeze(ground_truth, dim=0).to(device)

    return input_image, ground_truth, image_name