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
                          output_image,
                          topography,
                          resize,
                          crop,
                          to_loader=False,
                          crop_index=0):
    """
    Transform the input and output images, by resizing, cropping and normalisation.
    """
    if topography=="dem":
        input_image = input_image[:4, :, :]
    elif topography=="flow":
        input_image = torch.cat((input_image[:3, :, :], input_image[4, :, :].unsqueeze(dim=0)), 0)
    elif topography=="river":
        input_image = torch.cat((input_image[:3, :, :], input_image[5, :, :].unsqueeze(dim=0)), 0)
    elif topography=="map":
        input_image = torch.cat((input_image[:3, :, :], input_image[6: :, :]), 0)
    elif topography==None:
        input_image = input_image[:3, :, :]

    if resize:
        input_image = Resize(resize, antialias=True, interpolation=InterpolationMode.BICUBIC)(input_image)
        output_image = Resize(resize, antialias=True, interpolation=InterpolationMode.BICUBIC)(output_image)

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
        output_image = output_image[:, start_row:start_row + rows_size, start_col:start_col + cols_size]
        image_name = f"{image_name}_{crop_index}"

    topography_channels = {"all": 9, "map": 6, "dem": 4, "flow": 4, "river": 4, None: 3}
    input_image = Normalize(mean=(0.5,)*topography_channels[topography], 
                            std=(0.5,)*topography_channels[topography])(input_image)
    output_image = Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))(output_image)

    if to_loader==False:
        input_image = torch.unsqueeze(input_image, dim=0).to(device)
        output_image = torch.unsqueeze(output_image, dim=0).to(device)

    return input_image, output_image, image_name

def extract_input_topography(input_image, topography):
    """
    When comparing models with different topography inputs, extract the correct topography from the input image.
    """
    topography_inputs = {"all": input_image,
                         "dem" : input_image[:, :4, :, :],
                         "flow" : torch.cat((input_image[:, :3, :, :], input_image[:, 4, :, :].unsqueeze(dim=0)), 1),
                         "river" : torch.cat((input_image[:, :3, :, :], input_image[:, 5, :, :].unsqueeze(dim=0)), 1),
                         "map" : torch.cat((input_image[:, :3, :, :], input_image[:, 6: :, :]), 1),
                         "none" : input_image[:, :3, :, :]}
    return topography_inputs[topography]