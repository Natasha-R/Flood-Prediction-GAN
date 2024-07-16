import pandas as pd
import numpy as np
import json
import requests
import os
import shutil
import time
from tqdm import tqdm

import tifffile as tf
import matplotlib.pyplot as plt
from PIL import Image 

from osgeo import gdal
gdal.UseExceptions()

### Metadata ####################################

def create_metadata(path):
    # create a metadata csv describing the selected images
    images_path = f"{path}/xBD/pngs_selected"
    labels_path = f"{path}/xBD/labels_all/"
    tiffs_path = f"{path}/xBD/tiffs_all/"

    selected_images = list({image.split("_")[0] + "_" + image.split("_")[1] for image in os.listdir(images_path)})
    disaster = [image.split("_")[0] for image in selected_images]

    post_date = []
    pre_date = []
    for image in selected_images:
        with open(labels_path + image + "_post_disaster.json") as file: 
            post_date.append(json.load(file)["metadata"]["capture_date"])
        with open(labels_path + image + "_pre_disaster.json") as file: 
            pre_date.append(json.load(file)["metadata"]["capture_date"])

    all_tiffs = [image + "_pre_disaster.tif" for image in selected_images]
    x_min = []
    y_min = []
    x_max = []
    y_max = []
    for tiff in all_tiffs:
        image = gdal.Open(tiffs_path + tiff)
        width = image.RasterXSize
        height = image.RasterYSize
        geo = image.GetGeoTransform()
        x_min.append(geo[0])
        y_min.append(geo[3] + width*geo[4] + height*geo[5])
        x_max.append(geo[0] + width*geo[1] + height*geo[2])
        y_max.append(geo[3]) 

    metadata = pd.DataFrame({"image": selected_images, 
                             "disaster": disaster, 
                             "pre_date": pd.to_datetime(pre_date), 
                             "post_date": pd.to_datetime(post_date),
                             "date_difference": pd.to_datetime(post_date) - pd.to_datetime(pre_date),
                             "x_min": x_min,
                             "y_min": y_min,
                             "x_max": x_max,
                             "y_max": y_max,
                            })
    metadata.insert(5, "days_difference", metadata["date_difference"].dt.days)
    metadata["x_max_extended"] = metadata["x_max"] + 0.002
    metadata["polygon"] = ("POLYGON ((" + metadata.x_min.astype(str) + " " + metadata.y_min.astype(str) + ", " + metadata.x_min.astype(str) + " " + metadata.y_max.astype(str) + ", " + metadata.x_max.astype(str) + " " + metadata.y_max.astype(str) + ", " + metadata.x_max.astype(str) + " " + metadata.y_min.astype(str)).tolist()
    metadata = metadata.sort_values("image").reset_index(drop=True)
    metadata.to_csv("metadata.csv", index=False)
    
    midwest_left = metadata[metadata["disaster"]=="midwest-flooding"].tail(132).groupby("disaster")[["x_min", "y_min", "x_max_extended", "y_max"]].agg(['min','max'])
    midwest_right = metadata[metadata["disaster"]=="midwest-flooding"].head(15).groupby("disaster")[["x_min", "y_min", "x_max_extended", "y_max"]].agg(['min','max'])
    midwest_left.index = ["midwest_left"]
    midwest_right.index = ["midwest_right"]
    extents_metadata = pd.concat([metadata[metadata["disaster"]!="midwest-flooding"].groupby("disaster")[["x_min", "y_min", "x_max_extended", "y_max"]].agg(['min','max']), midwest_left, midwest_right])
    extents_metadata = extents_metadata.reset_index(names="disaster")
    extents_metadata = extents_metadata[[("x_min", "min"), ("x_max_extended", "max"), ("y_min", "min"), ("y_max", "max")]]
    extents_metadata = extents_metadata.reset_index(names="disaster")
    extents_metadata.columns = extents_metadata.columns.get_level_values(0)
    extents_metadata["string"] = extents_metadata.x_min.astype("str") + ", " + extents_metadata.x_max_extended.astype("str") + ", " + extents_metadata.y_min.astype("str") + ", " + extents_metadata.y_max.astype("str") 
    extents_metadata.to_csv("extents_metadata.csv", index=False)
            
def create_dataset_split_metadata(metadata_path, path):
    # determine the training/validation/test split and record the relevant metadata
    metadata = pd.read_csv(metadata_path)
    
    training = metadata.groupby("disaster", group_keys=False).apply(lambda group: group.sample(frac=0.8, random_state=47))
    training = training[["image", "disaster"]]
    training["split"] = "train"
    training["version"] = "original"

    training_flipped = training[["image", "disaster"]].copy()
    training_flipped["version"] = "flipped"
    training_flipped["split"] = "train"

    validation_and_test = metadata[~metadata.index.isin(training.index)]
    validation_and_test = validation_and_test[["image", "disaster"]]

    validation = validation_and_test.groupby("disaster", group_keys=False).apply(lambda group: group.sample(frac=0.5, random_state=47))
    validation["split"] = "validation"
    validation["version"] = "original"

    validation_flipped = validation.copy()
    validation_flipped["version"] = "flipped"
    validation_flipped["split"] = "validation"

    test = validation_and_test[~validation_and_test.index.isin(validation.index)].copy()
    test["split"] = "test"
    test["version"] = "original"

    dataset_split = pd.concat([training, training_flipped, validation, validation_flipped, test]).reset_index(drop=True)

    dataset_split["country"] = "india"
    dataset_split["country"].loc[(dataset_split.disaster=="hurricane-florence") | (dataset_split.disaster=="hurricane-harvey") | (dataset_split.disaster=="midwest-flooding")] = "usa"

    dataset_split["best_DEM"] = "10m"
    dataset_split["best_DEM"].loc[dataset_split.disaster=="hurricane-harvey"] = "01m"
    dataset_split["best_DEM"].loc[dataset_split.disaster=="nepal-flooding"] = "30m"
    for file_name in os.listdir(f"{path}/dataset_input"):
        if ("midwest-flooding" in file_name) and ("01m" in file_name):
            image_name = "_".join(file_name.split("_")[:2])
            dataset_split.loc[dataset_split.image==image_name, "best_DEM"] = "01m"                                     
    dataset_split["same_DEM"] = "10m"
    dataset_split["same_DEM"].loc[dataset_split.disaster=="nepal-flooding"] = "30m"

    dataset_split = dataset_split[["image", "best_DEM", "same_DEM", "version", "split", "disaster", "country"]]

    dataset_split.to_csv("dataset_split.csv", index=False)

def create_masks_metadata(masks_path, country):
    # create the metadata for training the flood segmentation model on the masks data
    disasters = ["hurricane-harvey", "hurricane-florence", "midwest-flooding"] if country.lower()=="usa" else ["nepal-flooding"]
    images = [image_name for image_name in os.listdir(masks_path) if any(disaster in image_name for disaster in disasters)]

    masks_metadata = pd.DataFrame({"image": images})
    train_masks_metadata = masks_metadata.sample(frac=0.8, random_state=47)
    val_test_masks_metadata = masks_metadata[~masks_metadata.index.isin(train_masks_metadata.index)]

    train_masks_metadata["split"] = "train"
    train_masks_metadata["version"] = "original"
    flipped_masks_train = train_masks_metadata.copy()
    flipped_masks_train["version"] = "flipped"

    val_masks_metadata = val_test_masks_metadata.sample(frac=0.5, random_state=47)
    test_masks_metadata = val_test_masks_metadata[~val_test_masks_metadata.index.isin(val_masks_metadata.index)]
    val_masks_metadata["split"] = "validation"
    val_masks_metadata["version"] = "original"
    test_masks_metadata["split"] = "test"
    test_masks_metadata["version"] = "original"

    val_test_masks_metadata["version"] = "flipped"
    val_test_masks_metadata["split"] = None

    masks_metadata = pd.concat([train_masks_metadata, flipped_masks_train, val_masks_metadata, test_masks_metadata, val_test_masks_metadata])
    masks_metadata["country"] = country

    masks_metadata.to_csv("metadata/masks_metadata.csv", mode="a", header=not os.path.exists("metadata/masks_metadata.csv"), index=False)

### Digital Elevation Model ####################################

def download_DEM(metadata_path, api_key, path, api_name="usgsdem", resolution="10m"):
    # download the DEM images from the OpenTopography API
    metadata = pd.read_csv(metadata_path)
    api = f"https://portal.opentopography.org/API/{api_name}"
    if api_name == "usgsdem":
        dataset = "datasetName"
        dataset_name = f"USGS{resolution}"
    else: # if api=="globaldem":
        dataset = "demtype"
        dataset_name = "COP30"
        
    for index, image in metadata.iterrows():
        response = requests.get(api, params={dataset: dataset_name, 
                                         "south":image["y_min"], 
                                         "north":image["y_max"],
                                         "west":image["x_min"], 
                                         "east":image["x_max_extended"],
                                         "outputFormat":"GTiff",
                                         "API_Key":api_key,
                                         })
        if response.status_code == 200:
            with open(f"{path}/DEM/DEM_images/{image.image}_{resolution}_DEM.tif", "wb") as file:
                file.write(response.content)
        time.sleep(1)
        
def project_DEM(path):
    # project the DEM to the correct coordinate system
    for image in os.listdir(f"{path}/DEM/DEM_images/"):
        if "nepal-flooding" not in image and "1m" not in image:
            with open(f"project_DEM.bat", "a+") as file:
                file.write(f"\ngdalwarp -overwrite -s_srs EPSG:4269 -t_srs EPSG:4326 -r near -of GTiff {path}/DEM/DEM_images/{image} {path}/DEM/DEM_projected/{image[:-4]}_proj.tif")
        elif "1m" in image:
            image_array = gdal.Open(f"{path}/DEM/DEM_images/{image}")
            source_proj = image_array.GetProjection()[-8:-3]
            with open(f"project_DEM.bat", "a+") as file:
                file.write(f"\ngdalwarp -overwrite -s_srs EPSG:{source_proj} -t_srs EPSG:4326 -r near -of GTiff {path}/DEM/DEM_images/{image} {path}/DEM/DEM_projected/{image[:-4]}_proj.tif")
        else:
            shutil.copyfile(f"{path}/DEM/DEM_images/{image}", f"{path}/DEM/DEM_projected/{image[:-4]}_proj.tif")

# project_DEM.bat
                
def render_DEM(path):
    # render the DEM with a consistent representation 
    for image in os.listdir(f"{path}/DEM/DEM_projected/"):
        image_array = tf.imread(f"{path}/DEM/DEM_projected/" + image)
        if "1m" in image:
            image_array[image_array < 0] = np.min(image_array[image_array > 0])
        image_array = (image_array - np.min(image_array))/100
        tf.imsave(f"{path}/DEM/DEM_render/" + image[:-9] + "_render.tif", image_array, photometric="minisblack")
    
### Open Street Map ####################################

def create_pbf(metadata_path, path):
    # use osmium to create the image specific .pbfs from the larger countrywide .pbf files
    metadata = pd.read_csv(metadata_path)
    for index, image in metadata.iterrows():
        with open("create_pbf.bat", "a+") as file:
            file.write(f"\nosmium extract -b {image.x_min},{image.y_min},{image.x_max_extended},{image.y_max} {path}/OSM/country_pbf/{image.disaster}.osm.pbf -o {path}/OSM/image_pbf/{image.image}.osm.pbf -s smart -S types=any")
                            
# create_pbf.bat
            
def create_osm(metadata_path, path):
    # use Maperitive to convert the image .pbfs to TIF images with a custom visual
    metadata = pd.read_csv(metadata_path)
    for index, image in metadata.iterrows():
        with open(f"create_osm.mscript", "a+") as file:
            file.write(f'\nclear-map')
            file.write(f'\nload-source "{path}/OSM/image_pbf/{image.image}.osm.pbf"')
            file.write(f'\nuse-ruleset alias="OSMNoText"')
            file.write(f'\napply-ruleset')
            file.write(f'\nset-geo-bounds {image.x_min},{image.y_min},{image.x_max_extended},{image.y_max}')
            file.write(f'\nset-print-bounds-geo')
            file.write(f'\nexport-bitmap file={path}/OSM/osm_img/{image.image}_osm.tif height=1024')
                            
# Maperitive create_osm.mscript
                            
def georeference_osm(metadata_path, path):
    # georeference the OSM TIFs using gdal_translate and gdal_warp
    metadata = pd.read_csv(metadata_path)
    for index, image in metadata.iterrows():
        img = Image.open(f"{path}/OSM/osm_img/{image.image}_osm.tif") 
        width = img.width 
        height = img.height 
        with open("georeference_osm.bat", "a+") as file:
            file.write(f"\ngdal_translate -of GTiff -gcp 0 0 {image.x_min} {image.y_max} -gcp {width} 0 {image.x_max_extended} {image.y_max} -gcp 0 {height} {image.x_min} {image.y_min} -gcp {width} {height} {image.x_max_extended} {image.y_min} {path}/OSM/osm_img/{image.image}_osm.tif {path}/OSM/osm_render/{image.image}_osm_gt.tif")
            file.write(f"\ngdalwarp -r near -order 1 -co COMPRESS=NONE -t_srs EPSG:4326 -dstalpha {path}/OSM/osm_render/{image.image}_osm_gt.tif {path}/OSM/osm_render/{image.image}_osm_render.tif")
            file.write(f"\ndel {path}/OSM/osm_render/{image.image}_osm_gt.tif")

# georeference_osm.bat
                            
### River Distance ####################################

def create_river_distance(metadata_path, path):
    # create images of the distance to the nearest river
    metadata = pd.read_csv(metadata_path)
    for index, image in metadata.iterrows():
        img = Image.open(f"{path}/OSM/osm_render/{image.image}_osm_render.tif") 
        width = img.width 
        height = img.height 
        with open("create_river_distance.bat", "a+") as file:
            file.write(f"\ngdal_rasterize -l river_distance_projected -a color_code -ts {width} {height} -a_nodata 0.0 -te {image.x_min} {image.y_min} {image.x_max_extended} {image.y_max} -ot Float32 -of GTiff {path}/river_distance/qgis/river_distance_projected.gpkg {path}/river_distance/river_distance_images/{image.image}_river_distance.tif")
 
# create_river_distance.bat                          
                            
def render_river_distance(metadata_path, path):
    # render the river distance as images
    metadata = pd.read_csv(metadata_path)
    for index, image in metadata.iterrows():
        image_array = tf.imread(f"{path}/river_distance/river_distance_images/{image.image}_river_distance.tif")
        image_array = image_array/255
        plt.imsave(f"{path}/river_distance/river_distance_render/{image.image}_rd_render.tiff", image_array, cmap="gray", vmin=0, vmax=1)
        
### Flow Accumulation ####################################
                            
def create_flow_accumulation(metadata_path, path):
    # extract flow accumulation maps for each image
    metadata = pd.read_csv(metadata_path)
    for index, image in metadata.iterrows():
        if image.disaster == "hurricane-florence":
            flow_map = "florence"
        elif image.disaster == "hurricane-harvey":
            flow_map = "harvey"
        elif image.disaster == "midwest-flooding":
            if image.x_min > -94:
                flow_map = "midwest_right"
            else:
                flow_map = "midwest_left"
        else: # if image.disaster == nepal-flooding
            flow_map = "india"
        with open(f"create_flow_accumulation.bat", "a+") as file:
            file.write(f"\ngdal_translate -projwin {image.x_min} {image.y_max} {image.x_max_extended} {image.y_min} -of GTiff {path}/flow_accumulation/fa_full_maps/{flow_map}_flow_accumulation.tif {path}/flow_accumulation/fa_images/{image.image}_flow_acc.tif")

# create_flow_accumulation.bat                  
                            
def render_flow_accumulation(path):
    # render the flow accumulation maps with a consistent representation
    for image in os.listdir(f"{path}/flow_accumulation/fa_images/"):
        image_array = tf.imread(f"{path}/flow_accumulation/fa_images/" + image)
        image_array = image_array/5.5
        tf.imsave(f"{path}/flow_accumulation/fa_render/" + image[:-12] + "fa_render.tif", image_array, photometric="minisblack")
        
### Image Stacks ####################################

def create_stacked_image_folders(metadata_path, path):
    # create folders for each image stack
    metadata = pd.read_csv(metadata_path)
    for index, image in metadata.iterrows():
        stack_path = f"{path}/image_stacks/{image.image}/" 
        if not os.path.exists(stack_path):
            os.makedirs(stack_path)

        shutil.copyfile(f"{path}/xBD/pngs_selected/{image.image}_pre_disaster.png", f"{stack_path}/1_pre_image.png")

        shutil.copyfile(f"{path}/river_distance/river_distance_render/{image.image}_rd_render.tiff", f"{stack_path}/2_river_dist.tif")

        shutil.copyfile(f"{path}/OSM/osm_render/{image.image}_osm_render.tif", f"{stack_path}/3_osm.tif")

        def DEM_path(resolution):
            return f"{path}/DEM/DEM_render/{image.image}_{resolution}_DEM_render.tif"
        if os.path.exists(DEM_path("10m")):
            shutil.copyfile(DEM_path("10m"), f"{stack_path}/4_10m_DEM.tif")
        if os.path.exists(DEM_path("1m")):
            shutil.copyfile(DEM_path("1m"), f"{stack_path}/45_1m_DEM.tif")
        if os.path.exists(DEM_path("30m")):
            shutil.copyfile(DEM_path("30m"), f"{stack_path}/4_30m_DEM.tif")

        shutil.copyfile(f"{path}/flow_accumulation/fa_render/{image.image}_fa_render.tif", f"{stack_path}/5_flow_acc.tif")
        
def apply_masks(path):
    # apply masks to some the satellite images to hide clouds, etc
    for image_folder in os.listdir(f"{path}/image_stacks/"):

        folder_path = f"{path}/image_stacks/{image_folder}"
        all_images = os.listdir(folder_path)

        if "mask.tif" in all_images:

            pre_satellite = tf.imread(f"{folder_path}/pre_satellite.tif")
            post_satellite = tf.imread(f"{folder_path}/post_satellite.tif")

            mask = (tf.imread(f"{folder_path}/mask.tif")/255).astype(np.int16)
            mask = np.repeat(mask[..., np.newaxis], 3, axis=-1)

            modified_post_satellite = np.multiply(post_satellite, mask)
            modified_pre_satellite = np.multiply(pre_satellite, mask)

            tf.imsave(f"{folder_path}/pre_satellite.tif", modified_pre_satellite)
            tf.imsave(f"{folder_path}/post_satellite.tif", modified_post_satellite)
            
def create_input_stack(path):
    # create a single tif image representing the input stack of images
    for image_folder in tqdm(os.listdir(f"{path}/image_stacks/")):
    
        folder_path = f"{path}/image_stacks/{image_folder}"
        all_images = os.listdir(folder_path)

        pre_satellite = (tf.imread(f"{folder_path}/pre_satellite.tif")/255).astype(np.float32)

        osm = tf.imread(f"{folder_path}/osm.tif")

        river_dist = tf.imread(f"{folder_path}/river_dist.tif")
        river_dist = np.mean(river_dist, 2)
        river_dist = np.expand_dims(river_dist, axis=-1)

        flow_acc = tf.imread(f"{folder_path}/flow_acc.tif")
        flow_acc = np.mean(flow_acc, 2)
        flow_acc = np.expand_dims(flow_acc, axis=-1)

        if "1m_DEM.tif" in all_images:
            DEM_1 = tf.imread(f"{folder_path}/1m_DEM.tif")
            DEM_1 = np.mean(DEM_1, 2)
            DEM_1 = np.expand_dims(DEM_1, axis=-1)

            full_image_1 = np.concatenate((pre_satellite, 
                                           DEM_1, 
                                           flow_acc,
                                           river_dist,
                                           osm,), 
                                           axis=-1)

            tf.imsave(f"{path}/dataset_input/{image_folder}_01m.tif", 
                      full_image_1, 
                      planarconfig="contig")

        if "10m_DEM.tif" in all_images:
            DEM_10 = tf.imread(f"{folder_path}/10m_DEM.tif")
            DEM_10 = np.mean(DEM_10, 2)
            DEM_10 = np.expand_dims(DEM_10, axis=-1)

            full_image_10 = np.concatenate((pre_satellite, 
                                           DEM_10, 
                                           flow_acc,
                                           river_dist,
                                           osm,), 
                                           axis=-1)

            tf.imsave(f"{path}/dataset_input/{image_folder}_10m.tif", 
                      full_image_10, 
                      planarconfig="contig")

        if "30m_DEM.tif" in all_images:
            DEM_30 = tf.imread(f"{folder_path}/30m_DEM.tif")
            DEM_30 = np.mean(DEM_30, 2)
            DEM_30 = np.expand_dims(DEM_30, axis=-1)

            full_image_30 = np.concatenate((pre_satellite, 
                                           DEM_30, 
                                           flow_acc,
                                           river_dist,
                                           osm,), 
                                           axis=-1)

            tf.imsave(f"{path}/dataset_input/{image_folder}_30m.tif", 
                      full_image_30, 
                      planarconfig="contig")
            
def create_output(path):
    # create the output images
    for image_folder in tqdm(os.listdir(f"{path}/image_stacks/")):
        folder_path = f"{path}/image_stacks/{image_folder}"
        post_satellite = (tf.imread(f"{folder_path}/post_satellite.tif")/255).astype(np.float32)
        tf.imsave(f"{path}/dataset_output/{image_folder}.tif", 
              post_satellite, 
              planarconfig="contig")
    