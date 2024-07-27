from models import utils

import pandas as pd
import numpy as np
import tifffile as tf

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Normalize

def create_flood_dataset(dataset_subset,
                   dataset_dem,
                   path, 
                   topography,
                   resize=None, 
                   crop=None,
                   batch_size=1, 
                   num_workers=0):
    """
    Returns train, validation and test set dataloaders for the specified dataset.
    dataset_subset : "usa", "india", "hurricane-harvey", "hurricane-florence", "midwest-flooding", "nepal-flooding", "testing", "all"
    dataset_dem : "best, "same"
    """
    dataset_train = FloodDataset(dataset_subset, dataset_dem, "train", path, topography, resize, crop)
    dataset_val = FloodDataset(dataset_subset, dataset_dem, "validation", path, topography, resize, crop)
    dataset_test = FloodDataset(dataset_subset, dataset_dem, "test", path, topography, resize, crop)
    
    train_loader = DataLoader(dataset=dataset_train,
                              batch_size=batch_size,
                              num_workers=num_workers,
                              shuffle=True,
                              pin_memory=True)
    val_loader = DataLoader(dataset=dataset_val,
                              batch_size=batch_size,
                              num_workers=num_workers,
                              shuffle=True,
                              pin_memory=True) 
    test_loader = DataLoader(dataset=dataset_test,
                              batch_size=batch_size,
                              num_workers=num_workers,
                              shuffle=True,
                              pin_memory=True)
    
    return train_loader, val_loader, test_loader
    
class FloodDataset(Dataset):
    """
    Dataset class for the flood dataset.
    """
    def __init__(self, dataset_subset, dataset_dem, split, path, topography, resize, crop):
        self.data_files = determine_flood_dataset(dataset_subset, dataset_dem, crop)[split]
        self.resize = resize
        self.path = path
        self.crop = crop
        self.topography = topography

    def __getitem__(self, index):
        image = self.data_files[index]
        image_path = image[0]
        version = image[1]
        image_name = image_path[:-8]
        crop_index = image[2] if self.crop else 0
        if version == "flipped":
            input_image = torch.from_numpy(np.fliplr(tf.imread(f"{self.path}/dataset_input/{image_path}")).transpose(2, 0, 1).copy())
            output_image = torch.from_numpy(np.fliplr(tf.imread(f"{self.path}/dataset_output/{image_name + '.tif'}")).transpose(2, 0, 1).copy())
        else:
            input_image = torch.from_numpy(tf.imread(f"{self.path}/dataset_input/{image_path}") .transpose(2, 0, 1))
            output_image = torch.from_numpy(tf.imread(f"{self.path}/dataset_output/{image_name + '.tif'}").transpose(2, 0, 1))

        input_image, output_image, image_name = utils.apply_transformations(image_name=image_name,
                                                                            input_image=input_image, 
                                                                            output_image=output_image, 
                                                                            topography=self.topography, 
                                                                            resize=self.resize, 
                                                                            crop=self.crop, 
                                                                            to_loader=True,
                                                                            crop_index=crop_index)
        return input_image, output_image, image_name

    def __len__(self):
        return len(self.data_files)
    
def determine_flood_dataset(subset, dem, crop=None):
    """
    Determines the image files contained within each dataset subset.
    """
    dataset_split = pd.read_csv("metadata/dataset_split.csv")
    locations = ["usa", "india"]
    disasters = ["hurricane-harvey", "hurricane-florence", "midwest-flooding", "nepal-flooding"]
    
    if subset.lower() in locations:
        dataset = dataset_split[dataset_split["country"]==subset.lower()].copy()
    elif subset.lower() in disasters:
        dataset = dataset_split[dataset_split["disaster"]==subset.lower()].copy()
    elif subset.lower()=="harveyflorence":
        dataset = dataset_split[dataset_split["country"]=="usa"].copy()
        flipped_test = dataset[((dataset["disaster"]=="hurricane-harvey") | (dataset["disaster"]=="hurricane-florence")) & (dataset["split"]=="test")].copy()
        flipped_test["version"] = "flipped"
        dataset = pd.concat([dataset, flipped_test], axis=0)
        dataset.loc[(dataset["disaster"]=="hurricane-harvey") | (dataset["disaster"]=="hurricane-florence"), "split"] = "train"
        dataset.loc[dataset["disaster"]=="midwest-flooding", "split"] = "validation"
        all_val = dataset[dataset["disaster"]=="midwest-flooding"].copy()
        all_val["split"] = "test"
        dataset = pd.concat([dataset, all_val], axis=0).reset_index(drop=True)
        dataset = dataset.drop(dataset[((dataset["split"] == "test") | (dataset["split"] == "validation")) & (dataset["version"] == "flipped")].index)
    elif subset.lower()=="harveyonflorence":
        dataset = dataset_split[(dataset_split["disaster"]=="hurricane-harvey") | (dataset_split["disaster"]=="hurricane-florence")].copy()
        flipped_test = dataset[(dataset["disaster"]=="hurricane-harvey") & (dataset["split"]=="test")].copy()
        flipped_test["version"] = "flipped"
        dataset = pd.concat([dataset, flipped_test], axis=0)
        dataset.loc[dataset["disaster"]=="hurricane-harvey", "split"] = "train"
        dataset.loc[dataset["disaster"]=="hurricane-florence", "split"] = "validation"
        all_val = dataset[dataset["disaster"]=="hurricane-florence"].copy()
        all_val["split"] = "test"
        dataset = pd.concat([dataset, all_val], axis=0).reset_index(drop=True)
        dataset = dataset.drop(dataset[((dataset["split"] == "test") | (dataset["split"] == "validation")) & (dataset["version"] == "flipped")].index)
    elif subset.lower()=="testing":
        dataset = dataset_split[dataset_split["disaster"]=="hurricane-harvey"].copy()
        dataset = dataset[dataset["version"]=="original"]
        dataset = dataset.sample(n=50, random_state=47)
    elif subset.lower()=="all":
        dataset = dataset_split.copy()
    else:
        raise NotImplementedError("Unrecognised dataset subset name")
    if dem not in ["best", "same"]:
        raise NotImplementedError("Unrecognised DEM name - provide 'best' or 'same'")
        
    dataset["file_name"] = dataset["image"] + "_" + dataset[f"{dem}_DEM"] + ".tif"
    dataset = dataset.sample(frac=1, random_state=47)
    
    if crop:
        crops = [dataset.copy() for i in range(crop)] 
        for i in range(crop):
            crops[i]["crop"] = i
        dataset = pd.concat(crops)
        splits = [list(zip(dataset[dataset["split"] == split_name]["file_name"], 
                           dataset[dataset["split"] == split_name]["version"],
                           dataset[dataset["split"] == split_name]["crop"]))
                  for split_name in ["train", "validation", "test"]]
    
    else:
        splits = [list(zip(dataset[dataset["split"] == split_name]["file_name"], 
                           dataset[dataset["split"] == split_name]["version"]))
                  for split_name in ["train", "validation", "test"]]
    
    return {"train": splits[0], "validation": splits[1], "test": splits[2]}

def create_masks_dataset(dataset_subset, path, train_on_all, batch_size=1, num_workers=0):
    """
    Returns train, validation and test set dataloaders for the specified dataset.
    When training on "all", only a train dataloader is created.
    The dataset_subset may be "usa" or "india".
    """
    train_data, val_data, test_data = determine_masks_dataset(dataset_subset, train_on_all)
    train_loader, val_loader, test_loader = None, None, None

    dataset_train = MaskDataset(train_data, path)
    train_loader = DataLoader(dataset=dataset_train,
                             batch_size=batch_size,
                             num_workers=num_workers,
                             shuffle=True,
                             pin_memory=True)
    if not train_on_all:
        dataset_val = MaskDataset(val_data, path)
        val_loader = DataLoader(dataset=dataset_val,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            shuffle=True,
                            pin_memory=True) 
        dataset_test = MaskDataset(test_data, path)
        test_loader = DataLoader(dataset=dataset_test,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            shuffle=True,
                            pin_memory=True) 
   
    return train_loader, val_loader, test_loader

class MaskDataset(Dataset):
    """
    Dataset class for the flood mask (labels) dataset.
    """
    def __init__(self, data, path):
        self.data_files = data
        self.path = path

    def __getitem__(self, index):
        image = image = self.data_files[index]
        image_path = image[0]
        version = image[1]
        if version == "flipped":
            input_image = torch.from_numpy(np.fliplr(tf.imread(f"{self.path}/masks_input/{image_path}")).transpose(2, 0, 1).copy())
            output_image = torch.from_numpy(np.fliplr(tf.imread(f"{self.path}/masks_output/{image_path}")).copy()).unsqueeze(dim=0)
        else:
            input_image = torch.from_numpy(tf.imread(f"{self.path}/masks_input/{image_path}").transpose(2, 0, 1))
            output_image = torch.from_numpy(tf.imread(f"{self.path}/masks_output/{image_path}")).unsqueeze(dim=0)

        return input_image, output_image, image_path

    def __len__(self):
        return len(self.data_files)
    
def determine_masks_dataset(subset, train_on_all):
    """
    Determines the image files contained within each dataset subset.
    """
    masks_split = pd.read_csv("metadata/masks_metadata.csv")
    if subset.lower() not in ["usa", "india"]:
        raise NotImplementedError("Unrecognised dataset subset name")
    dataset = masks_split[masks_split["country"]==subset.lower()]

    if train_on_all:
        return list(zip(dataset["image"], dataset["version"])), None, None
    else:
        splits = [list(zip(dataset[dataset["split"] == split_name]["image"], 
                        dataset[dataset["split"] == split_name]["version"]))
                for split_name in ["train", "validation", "test"]]
        return splits[0], splits[1], splits[2]