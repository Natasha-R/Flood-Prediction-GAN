from models import model
from models import utils
from models import data
from models import segmentation_model

import os

import time
import numpy as np
import pandas as pd
import tifffile as tf
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt
from collections import defaultdict

import torch
from torchmetrics.regression import MeanSquaredError
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.classification import BinaryAccuracy, BinaryF1Score, BinaryPrecision, BinaryRecall
from torchmetrics.image import PeakSignalNoiseRatio, MultiScaleStructuralSimilarityIndexMeasure, StructuralSimilarityIndexMeasure

device = "cuda" if torch.cuda.is_available() else "cpu"

class ModelsGroup():
    """
    A class containing the four Pix2Pix, CycleGAN, AttentionGAN and PairedAttention models,
    for the purpose of making comparisons between them.
    """
    def __init__(self,
                 paths,
                 compare,
                 dataset_subset,
                 dataset_dem,
                 data_path,
                 resize,
                 crop,
                 crop_index,
                 topography):

        self.paths = paths
        self.compare = compare
        self.generators = {}
        self.dataset_subset = dataset_subset
        self.dataset_dem = dataset_dem
        self.data_path = data_path
        self.resize = resize
        self.crop = crop
        self.crop_index = crop_index
        self.topography = topography

        # initialise the models
        for model_name in self.paths:
            if not os.path.isfile(self.paths[model_name]):
                raise FileNotFoundError(f"Saved {model_name} model not found. Check the path to the {model_name} model.")
            if self.compare == "topography":
                model_topography = model_name.lower()
                if model_topography == "none": model_topography = None 
            else:
                model_topography = self.topography
            full_model = model.Model(model=self.paths[model_name].split("/")[-1].split("_")[0].lower(),
                                    dataset_subset=self.dataset_subset,
                                    dataset_dem=self.dataset_dem,
                                    data_path=self.data_path,
                                    resize=self.resize,
                                    crop=self.crop,
                                    load_pretrained_model=True,
                                    pretrained_model_path=self.paths[model_name],
                                    training_model=False,
                                    topography=model_topography,
                                    verbose=True)
            self.generators[model_name] = full_model.pre_to_post_generator if full_model.model_is_cycle else full_model.generator

        # initialise data
        overall_topography = "all" if self.compare == "topography" else self.topography
        self.train_loader, self.val_loader, self.test_loader = data.create_flood_dataset(self.dataset_subset, 
                                                                                   self.dataset_dem, 
                                                                                   self.data_path, 
                                                                                   overall_topography, 
                                                                                   self.resize, 
                                                                                   self.crop)
        
    def extract_input_topography(self, input_image):
        """
        When comparing models with different topography inputs, extract the correct topography from the input image.
        """
        topography_inputs = dict()
        topography_inputs["All"] = input_image
        topography_inputs["DEM"] = input_image[:, :4, :, :]
        topography_inputs["Flow accumulation"] = torch.cat((input_image[:, :3, :, :], input_image[:, 4, :, :].unsqueeze(dim=0)), 1)
        topography_inputs["Distance to rivers"] = torch.cat((input_image[:, :3, :, :], input_image[:, 5, :, :].unsqueeze(dim=0)), 1)
        topography_inputs["Map"] = torch.cat((input_image[:, :3, :, :], input_image[:, 6: :, :]), 1)
        topography_inputs["None"] = input_image[:, :3, :, :]
        return topography_inputs

    def create_path(self, save_type, info=""):
        """
        Defines an informative path string to save images and csv files to.
        """
        file_type = ".png" if save_type=="image" else ".csv"
        current_time = str(datetime.now())[:-7].replace(' ', '-').replace(':', '-')
        topography = "different" if self.compare == "topography" else self.topography
        path = (f"{self.data_path}/{save_type}s/"
                f"{self.compare}_comparison_{info}_{topography}Topography_"
                f"{self.dataset_subset}Data_{self.dataset_dem}DEM_"
                f"resize{self.resize}_crop{self.crop}_"
                f"date{current_time}{file_type}")
        path = path.replace("__", "_")
        return path
    
    def tensor_to_mask(self, tensor):
        return (torch.sigmoid(tensor.detach().clone()) > 0.5).float()

    def compare_metrics(self, use_test_data, seg_model_path):
        """
        Calculate automated metrics to compare the models.
        """
        metrics = {"PSNR": PeakSignalNoiseRatio(data_range=(0, 1)).to(device),
                    "SSIM": StructuralSimilarityIndexMeasure(data_range=(0, 1)).to(device),
                    "MS-SSIM": MultiScaleStructuralSimilarityIndexMeasure(data_range=(0, 1)).to(device),
                    "LPIPS": LearnedPerceptualImagePatchSimilarity().to(device),
                    "MSE":MeanSquaredError().to(device),
                    "Accuracy":BinaryAccuracy().to(device),
                    "F1_Flood":BinaryF1Score().to(device),
                    "Precision_Flood":BinaryPrecision().to(device),
                    "Recall_Flood":BinaryRecall().to(device),
                    "F1_No_Flood":BinaryF1Score().to(device),
                    "Precision_No_Flood":BinaryPrecision().to(device),
                    "Recall_No_Flood":BinaryRecall().to(device)}
        metrics_results = {metric: defaultdict(list) for metric in list(metrics.keys()) + ["Inference"]}
        all_true_flood_masks = defaultdict(lambda: "None")
        all_output_flood_masks = defaultdict(lambda: "None")
        all_true_flood_masks_grouped = defaultdict(lambda: "None")
        all_output_flood_masks_grouped = defaultdict(lambda: "None")
        image_names = []
        seg_model = segmentation_model.SegmentationModel(data_path=self.data_path,
                                                         pretrained_model_path=seg_model_path,
                                                         train=False).model

        print("\nCalculating metrics...")
        loader = self.test_loader if use_test_data else self.val_loader
        for input_stack, ground_truth, image_name in tqdm(loader, desc="Images", leave=False):

            input_stack = input_stack.to(device)
            ground_truth = ground_truth.to(device)
            image_names.append(image_name[0])
            disaster_name = image_name[0].split("_")[0]

            if self.compare == "topography": topography_inputs = self.extract_input_topography(input_stack)

            for generator_name in self.generators:

                input_stack_copy = input_stack.detach().clone()
                ground_truth_copy = ground_truth.detach().clone()
                start_time = time.time()
                torch.manual_seed(47)
                if self.compare == "topography": input_stack_copy = topography_inputs[generator_name].detach().clone()
                generator_output = self.generators[generator_name](input_stack_copy)
                inference_time = time.time()-start_time
                ground_truth_copy = torch.clamp((ground_truth_copy + 1) * 0.5, min=0, max=1)
                generator_output = torch.clamp((generator_output + 1) * 0.5, min=0, max=1)
                output_mask = self.tensor_to_mask(seg_model(generator_output.detach().clone()))
                true_mask = self.tensor_to_mask(seg_model(ground_truth_copy.detach().clone()))
                flat_true_mask = true_mask.detach().clone().squeeze().flatten().squeeze()
                flat_output_mask = output_mask.detach().clone().squeeze().flatten().squeeze()

                for metrics_name in ["PSNR", "SSIM", "MS-SSIM", "LPIPS"]:
                    metrics_results[metrics_name][generator_name].append(metrics[metrics_name](generator_output.detach().clone(), ground_truth_copy.detach().clone()).item())
                    metrics[metrics_name].reset()
                metrics_results["Inference"][generator_name].append(inference_time)

                all_true_flood_masks[generator_name] = torch.cat((all_true_flood_masks[generator_name], flat_true_mask), dim=0) if not all_true_flood_masks[generator_name]=="None" else flat_true_mask
                all_output_flood_masks[generator_name] = torch.cat((all_output_flood_masks[generator_name], flat_output_mask), dim=0) if not all_output_flood_masks[generator_name]=="None" else flat_output_mask

                all_true_flood_masks_grouped[f"{generator_name}_{disaster_name}"] = torch.cat((all_true_flood_masks_grouped[f"{generator_name}_{disaster_name}"], flat_true_mask), dim=0) if not all_true_flood_masks_grouped[f"{generator_name}_{disaster_name}"]=="None" else flat_true_mask
                all_output_flood_masks_grouped[f"{generator_name}_{disaster_name}"] = torch.cat((all_output_flood_masks_grouped[f"{generator_name}_{disaster_name}"], flat_output_mask), dim=0) if not all_output_flood_masks_grouped[f"{generator_name}_{disaster_name}"]=="None" else flat_output_mask

        disaster_names = list(set(image_name.split("_")[0] for image_name in image_names))
        grouped_results = {value:[] for value in ["Metric_Model"] + disaster_names}

        for generator_name in self.generators:
            for metrics_name in ["MSE", "Accuracy", "F1_Flood", "Precision_Flood", "Recall_Flood", "F1_No_Flood", "Precision_No_Flood", "Recall_No_Flood"]:
                if "No" in metrics_name:
                    all_output_masks_inverted = torch.abs(all_output_flood_masks[generator_name]-1)
                    all_true_masks_inverted = torch.abs(all_true_flood_masks[generator_name]-1)
                    metrics_results[metrics_name][generator_name].append(metrics[metrics_name](all_output_masks_inverted, all_true_masks_inverted).item())
                else:
                    metrics_results[metrics_name][generator_name].append(metrics[metrics_name](all_output_flood_masks[generator_name], all_true_flood_masks[generator_name]).item())
                grouped_results["Metric_Model"].append(f"{metrics_name}_{generator_name}")
                for disaster_name in disaster_names:
                    if "No" in metrics_name:
                        all_output_grouped_inverted = torch.abs(all_output_flood_masks_grouped[f"{generator_name}_{disaster_name}"]-1)
                        all_true_grouped_inverted = torch.abs(all_true_flood_masks_grouped[f"{generator_name}_{disaster_name}"]-1)
                        grouped_results[disaster_name].append(metrics[metrics_name](all_output_grouped_inverted, all_true_grouped_inverted).item())
                    else:
                        grouped_results[disaster_name].append(metrics[metrics_name](all_output_flood_masks_grouped[f"{generator_name}_{disaster_name}"], all_true_flood_masks_grouped[f"{generator_name}_{disaster_name}"]).item())
        
        for generator_name in self.generators: 
            metrics_results["Inference"][generator_name] = metrics_results["Inference"][generator_name][5:]
            break
        
        average_metrics = pd.concat([pd.Series([np.mean(metrics_results[metrics_name][generator_name])
                                                for generator_name in self.generators], 
                                                index=self.generators.keys(), 
                                                name=metrics_name)
                                     for metrics_name in metrics_results], axis=1)
        print(average_metrics)
        average_metrics.index.name = "Model"
        average_metrics.to_csv(self.create_path("metric"))
        
        grouped_by_disaster = list()
        for metrics_name in ["PSNR", "SSIM", "MS-SSIM", "LPIPS"]:
            single_metric_df = pd.DataFrame([metrics_results[metrics_name][generator_name] for generator_name in self.generators]).transpose()
            single_metric_df.columns = [f"{metrics_name}_{generator_name}" for generator_name in self.generators.keys()]
            single_metric_df["disaster"] = [image_name.split("_")[0] for image_name in image_names]
            grouped_by_disaster.append(single_metric_df.groupby("disaster").mean().transpose())
        grouped_metrics = pd.concat(grouped_by_disaster, axis=0).reset_index()
        grouped_metrics.rename(columns={"index":"Metric_Model"}, inplace=True)
        grouped_metrics = pd.concat([grouped_metrics, pd.DataFrame(grouped_results)])
        grouped_metrics = grouped_metrics.sort_values(["Metric_Model"])
        grouped_metrics.to_csv(self.create_path("metric", info="grouped"), index=False)
        
    def compare_output_images(self, image_names):
        """
        Compare the outputs of each of the models on the given input images.
        """
        dataset_split = pd.read_csv("metadata/dataset_split.csv")
        fig, axes = plt.subplots(nrows=len(image_names), 
                                 ncols=len(self.generators)+2, 
                                 figsize=((len(self.generators)+2) * 5, (len(image_names) * 5) + (0.5 * len(image_names))))
        for ax in axes.ravel():
            ax.set_axis_off()
        if len(image_names)==1:
            axes = np.expand_dims(axes, axis=0)

        for i, image_name in enumerate(image_names):
            if image_name[-2] == "_": 
                final_crop_index = int(image_name[-1])
                image_name = image_name[:-2]
            else:
                final_crop_index = self.crop_index
            dem_string = dataset_split[dataset_split["image"]==image_name][f"{self.dataset_dem}_DEM"].head(1).item()
            input_path = f"{self.data_path}/dataset_input/{image_name}_{dem_string}.tif"
            
            input_image = torch.from_numpy(tf.imread(input_path).transpose(2, 0, 1))
            ground_truth = torch.from_numpy(tf.imread(f"{self.data_path}/dataset_output/{image_name}.tif").transpose(2, 0, 1))

            topography = "all" if self.compare == "topography" else self.topography
            input_image, ground_truth, image_name = utils.apply_transformations(image_name=image_name,
                                                                                input_image=input_image, 
                                                                                output_image=ground_truth, 
                                                                                topography=topography,
                                                                                resize=self.resize, 
                                                                                crop=self.crop, 
                                                                                crop_index=final_crop_index)
            topography_inputs = self.extract_input_topography(input_image)
            
            outputs = dict()
            for generator_name in self.generators:
                final_input = topography_inputs[generator_name] if self.compare=="topography" else input_image
                torch.manual_seed(47)
                outputs[generator_name] = utils.tensor_to_numpy(self.generators[generator_name](final_input.detach().clone()))
         
            input_image = utils.tensor_to_numpy(input_image)
            ground_truth = utils.tensor_to_numpy(ground_truth)

            axes[i, 0].imshow(input_image, vmin=0, vmax=1)
            axes[i, 0].set_title(f"Input ({image_name})")
            axes[i, 1].imshow(ground_truth, vmin=0, vmax=1)
            axes[i, 1].set_title("Ground truth")
            for j, generator_name in enumerate(self.generators, start=2):
                axes[i, j].imshow(outputs[generator_name], vmin=0, vmax=1)
                axes[i, j].set_title(generator_name)

        fig.tight_layout()
        info = image_names[0] if len(image_names) == 1 else ""
        images_path = self.create_path(save_type="image", info=info)
        print(f"\nSaving comparison of {self.compare} images to {images_path}")
        fig.savefig(images_path, bbox_inches="tight")
        plt.close()