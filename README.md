# Generating Aerial Flood Prediction Imagery with Generative Adversarial Networks

This repository contains the code for training generative adversarial networks (GANs) to generate aerial flood prediction imagery. The GANs are input with a pre-flooding satellite image in addition to informative factors such as a digital elevation model, flow accumulation, distance to rivers, and OSM map. The GANs output a photorealistic post-flooding image prediction. 

This repository also contains code for training a flood segmentation model, which when input a post-flooding satellite image, outputs a binary mask indicating the locations of floodwaters. The segmentation model can hence be used to evaluate the predictions of the GANs, by comparing the flood masks of a pair of predicted and ground truth images.

<p align="center">
<img src="https://github.com/user-attachments/assets/e01d782d-c269-47e7-9a0a-5c1dc3d77eb7" width="750">
<img src="https://github.com/user-attachments/assets/59834fb7-1577-46fc-a506-774519f00b1b" width="600">
</p>

## Sample output

Sample generated images from different model architectures:
<img src="https://github.com/user-attachments/assets/92468c4f-fb1c-46f1-b0aa-885dda3ed837">

Sample generated images from different combinations of input factors:
<img src="https://github.com/user-attachments/assets/bbb39027-dfdf-4f95-952f-a90149a6f041">

## Dataset

The dataset and associated metadata are available on Zenodo (https://zenodo.org/doi/10.5281/zenodo.13366121) under the Creative Commons Attribution Non-Commercial Share-Alike 4.0 International licence. 

## Training

To train a GAN model for flood image generation:   
``python train.py --model=PairedAttention --dataset_subset=usa --dataset_dem=same --data_path=path/to/data --num_epochs=200 --topography=all --resize=512 --crop=4 --save_model_interval=50 --save_images_interval=25 --verbose``

To train a segmentation model:  
``python segment.py --train --dataset_subset=usa --data_path=path/to/data --num_epochs=100 --save_model_interval=25 --save_images_interval=10 --verbose``

## Evaluating

A model can be evaluated by calculating metrics, plotting the losses over the epochs, plotting a random sample of generated images, or generating a specific named image:  
``python evaluate.py --model=PairedAttention --dataset_subset=usa --dataset_dem=same --use_test_data --data_path=path/to/data --resize=512 --crop=4 --topography=all --pretrained_model_path=path/to/model --plot_losses --plot_sample_images --num_images=10 --calculate_metrics --segmentation_model_path=path/to/segmentation --plot_single_image=output --image_name=hurricane-harvey_00000257 --crop_index=3``

Multiple models can be compared by their calculated metrics or generated images:  
``python compare.py --compare=models --dataset_subset=usa --dataset_dem=same --use_test_data --data_path=path/to/data --resize=512 --crop=4 --topography=all --segmentation_model_path=path/to/segmentation --pix2pix_path=pix2pix/path --cyclegan_path=cyclegan/path --attentiongan_path=attentiongan/path --pairedattention_path=pairedattention/path --calculate_metrics --image_names hurricane-harvey_00000257_3 hurricane-harvey_00000268_1``
