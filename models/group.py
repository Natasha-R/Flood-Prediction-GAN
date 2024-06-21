from models import model
from models import utils

import os
import pandas as pd
import tifffile as tf
from datetime import datetime
import matplotlib.pyplot as plt

import torch

class ModelsGroup():
    """
    A class containing the four Pix2Pix, CycleGAN, AttentionGAN and PairedAttention models,
    for the purpose of making comparisons between them.
    """
    def __init__(self,
                 pix2pix_path,
                 cyclegan_path,
                 attentiongan_path,
                 pairedattention_path,
                 dataset_subset,
                 dataset_dem,
                 data_path,
                 resize,
                 crop,
                 crop_index,
                 not_input_topography):

        self.pix2pix_path = pix2pix_path
        self.cyclegan_path = cyclegan_path
        self.attentiongan_path = attentiongan_path
        self.pairedattention_path = pairedattention_path
        self.dataset_subset = dataset_subset
        self.dataset_dem = dataset_dem
        self.data_path = data_path
        self.resize = resize
        self.crop = crop
        self.crop_index = crop_index
        self.not_input_topography = not_input_topography

        self.generators = dict()
        for model_name, model_path in zip(["Pix2Pix", "CycleGAN", "AttentionGAN", "PairedAttention"],
                                            [self.pix2pix_path, self.cyclegan_path, self.attentiongan_path, self.pairedattention_path]):
            if not os.path.isfile(model_path):
                raise FileNotFoundError(f"Saved {model_name} model not found. Check the path to the {model_name} model.")
            full_model = model.Model(model=model_name.lower(),
                                     dataset_subset="all",
                                     dataset_dem=self.dataset_dem,
                                     data_path=self.data_path,
                                     resize=self.resize,
                                     crop=self.crop,
                                     load_pretrained_model=True,
                                     pretrained_model_path=model_path,
                                     training_model=False,
                                     not_input_topography=self.not_input_topography)
            self.generators[model_name] = full_model.pre_to_post_generator if full_model.model_is_cycle else full_model.generator
    
    def create_path(self):
        """
        Defines an informative path string to save images to.
        """
        current_time = str(datetime.now())[:-7].replace(' ', '-').replace(':', '-')
        path = (f"{self.data_path}/images/"
                f"models_comparison_topography{not self.not_input_topography}_"
                f"{self.dataset_subset}Data_{self.dataset_dem}DEM_"
                f"resize{self.resize}_crop{self.crop}_"
                f"date{current_time}.png")
        return path

    def compare_output_images(self, image_names):
        """
        Compare the outputs of each of the models on the given input images.
        """
        dataset_split = pd.read_csv("dataset_split.csv")
        fig, axes = plt.subplots(nrows=len(image_names), ncols=6, figsize=(30, len(image_names) * 5))
        for ax in axes.ravel():
            ax.set_axis_off()

        for i, image_name in enumerate(image_names):

            dem_string = dataset_split[dataset_split["image"]==image_name][f"{self.dataset_dem}_DEM"].head(1).item()
            input_path = f"{self.data_path}/dataset_input/{image_name}_{dem_string}.tif"
            input_image = torch.from_numpy(tf.imread(input_path).transpose(2, 0, 1))
            ground_truth = torch.from_numpy(tf.imread(f"{self.data_path}/dataset_output/{image_name}.tif").transpose(2, 0, 1))

            input_image, ground_truth, image_name = utils.apply_transformations(image_name=image_name,
                                                                                input_image=input_image, 
                                                                                output_image=ground_truth, 
                                                                                not_input_topography=self.not_input_topography, 
                                                                                resize=self.resize, 
                                                                                crop=self.crop, 
                                                                                crop_index=self.crop_index)
            
            pix2pix_output = utils.tensor_to_numpy(self.generators["Pix2Pix"](input_image))
            cyclegan_output = utils.tensor_to_numpy(self.generators["CycleGAN"](input_image))
            attentiongan_output = utils.tensor_to_numpy(self.generators["AttentionGAN"](input_image))
            pairedattention_output = utils.tensor_to_numpy(self.generators["PairedAttention"](input_image))
            input_image = utils.tensor_to_numpy(input_image)
            ground_truth = utils.tensor_to_numpy(ground_truth)

            axes[i, 0].imshow(input_image, vmin=0, vmax=1)
            axes[i, 0].set_title(f"Input ({image_name})")

            axes[i, 1].imshow(ground_truth, vmin=0, vmax=1)
            axes[i, 1].set_title("Ground truth")

            axes[i, 2].imshow(pix2pix_output, vmin=0, vmax=1)
            axes[i, 2].set_title("Pix2Pix")

            axes[i, 3].imshow(pairedattention_output, vmin=0, vmax=1)
            axes[i, 3].set_title("PairedAttention")

            axes[i, 4].imshow(cyclegan_output, vmin=0, vmax=1)
            axes[i, 4].set_title("CycleGAN")

            axes[i, 5].imshow(attentiongan_output, vmin=0, vmax=1)
            axes[i, 5].set_title("AttentionGAN")

        fig.tight_layout()
        images_path = self.create_path()
        print(f"Saving comparison of models images to {images_path}")
        fig.savefig(images_path, bbox_inches="tight")
        plt.close()