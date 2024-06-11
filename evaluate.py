import models
import data
import utils

import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

import torch
from torch import nn

device = "cuda" if torch.cuda.is_available() else "cpu"

def generate_images(dataset_subset,
                    dataset_dem,
                    data_path,
                    model,
                    resize,
                    crop,
                    num_images,
                    saved_model_path=None,
                    add_identity_loss=False,
                    not_input_topography=False,
                    trained_model=None,
                    epoch=None):
    
    # loading a saved model from path
    if saved_model_path:
        saved_model = torch.load(saved_model_path)
        epoch = saved_model["starting_epoch"] - 1
        not_input_topography = saved_model["not_input_topography"]
        if not_input_topography:
            input_channels = 3
        else: # if input_topography
            input_channels = 9
        if model.lower()=="pix2pix":
            generator = models.Pix2PixGenerator(input_channels=input_channels).to(device)
            generator.load_state_dict(saved_model["generator"])
            generators = [generator.eval()]
        elif model.lower() == "cyclegan" or model.lower()=="attentiongan":
            add_identity_loss = saved_model["add_identity_loss"]
            if model.lower() == "cyclegan":
                generator_architecture = models.CycleGANGenerator
            else: # if model.lower()=="attentiongan":
                generator_architecture = models.AttentionGANGenerator
            pre_to_post_generator = generator_architecture(input_channels=input_channels).to(device)
            post_to_pre_generator = generator_architecture(input_channels=input_channels).to(device)
            pre_to_post_generator.load_state_dict(saved_model["pre_to_post_generator"])
            post_to_pre_generator.load_state_dict(saved_model["post_to_pre_generator"])
            generators = [pre_to_post_generator.eval(), post_to_pre_generator.eval()]
        else:
            raise NotImplementedError("Model must be one of: Pix2Pix, CycleGAN or AttentionGAN")
        
    # loading a trained model
    elif trained_model:
        if model.lower()=="pix2pix":
            generators = [trained_model.eval()]
        elif model.lower() == "cyclegan" or model.lower() == "attentiongan":
            pre_to_post_generator = trained_model[0].eval()
            post_to_pre_generator = trained_model[1].eval()
            generators = [pre_to_post_generator, post_to_pre_generator]
    else:
        NotImplementedError("Either the path to a saved model, or a trained model, must be provided")
        
    train_loader, val_loader, _ = data.create_dataset(dataset_subset, dataset_dem, data_path, not_input_topography, resize, crop)
    if num_images > min(len(train_loader), len(val_loader)):
        raise ValueError(f"Enter num_images as {min(len(train_loader), len(val_loader))} or fewer")

    index_to_generator={0:"pre_to_post",
                        1:"post_to_pre"}
    for gen_index, generator in enumerate(generators):
        with torch.no_grad():
            for split, loader in zip(["train", "val"], [train_loader, val_loader]):
                
                fig, axes = plt.subplots(nrows=num_images, ncols=3, figsize=(15, num_images * 5))
                for ax in axes.ravel():
                    ax.set_axis_off()
                torch.manual_seed(47)
                
                for i, (input_stack, output_image) in enumerate(loader):
                    if gen_index==1: ## if generator is post-to-pre then flip the inputs
                        store_output = output_image.clone()
                        if not_input_topography:
                            output_image = input_stack.to(device)
                            input_stack = store_output.to(device)
                        else:
                            topography = input_stack[:, 3:, :, :].detach().clone()
                            output_image = input_stack[:, :3, :, :].to(device)
                            input_stack = torch.cat((store_output, topography), dim=1).to(device)
                    else:
                        input_stack = input_stack.to(device)
                        output_image = output_image.to(device)
                        
                    axes[i, 0].imshow(input_stack.squeeze().cpu().detach().numpy().transpose(1, 2, 0)[:, :, :3], vmin=0, vmax=1)
                    axes[i, 1].imshow(np.clip(generator(input_stack).squeeze().cpu().detach().numpy().transpose(1, 2, 0)[:, :, :3], 0, 1), vmin=0, vmax=1)
                    axes[i, 2].imshow(output_image.squeeze().cpu().detach().numpy().transpose(1, 2, 0)[:, :, :3], vmin=0, vmax=1)
                    axes[i, 0].set_title("Input")
                    axes[i, 1].set_title("Generator Output")
                    axes[i, 2].set_title("Ground Truth Output")
                    if i >= num_images-1:
                        break
                fig.tight_layout() 
                if "cyclegan" in model.lower():
                    model = f"{index_to_generator[gen_index]}_cyclegan"
                elif "attentiongan" in model.lower():
                    model = f"{index_to_generator[gen_index]}_attentiongan"
                images_path = utils.create_path("image", model.lower(), data_path, split, dataset_subset, dataset_dem, not_input_topography, resize, crop, epoch, add_identity_loss)
                print(f"Saving {split} images to {images_path}")
                fig.savefig(images_path)
                plt.close();
            
def print_losses(dataset_subset,
                 dataset_dem,
                 data_path,
                 model,
                 resize,
                 crop,
                 saved_model_path):
    
    saved_model = torch.load(saved_model_path)
    not_input_topography = saved_model["not_input_topography"]
    epoch = saved_model["starting_epoch"] - 1 
    train_loader, val_loader, _ = data.create_dataset(dataset_subset, dataset_dem, data_path, not_input_topography, resize, crop)
    if not_input_topography:
        input_channels = 3
    else: # if input_topography
        input_channels = 9
        
    if model.lower()=="pix2pix":
        generator = models.Pix2PixGenerator(input_channels=input_channels).to(device)
        discriminator = models.Pix2PixDiscriminator(input_channels=input_channels).to(device)
        discriminator.load_state_dict(saved_model["discriminator"])
        generator.load_state_dict(saved_model["generator"])
        discriminator.eval()
        generator.eval()
        loss_func = nn.BCEWithLogitsLoss()
        l1_loss = nn.L1Loss()  
        
        for loader_name, loader in zip(["train", "validation"], [train_loader, val_loader]):
            losses = utils.initialise_loss_storage("Pix2Pix", overall=False)
            all_losses = utils.initialise_loss_storage("Pix2Pix", overall=True)
            with torch.no_grad():
                torch.manual_seed(47)
                for input_stack, output_image in loader:
                    input_stack = input_stack.to(device)
                    output_image = output_image.to(device)
                    synthetic_output = generator(input_stack)
                    concat_real = torch.cat((input_stack, output_image), 1)
                    concat_synthetic = torch.cat((input_stack, synthetic_output), 1)
                    prediction_discriminator_real = discriminator(concat_real)
                    prediction_discriminator_synthetic = discriminator(concat_synthetic)
                    discriminator_prediction_shape = prediction_discriminator_synthetic.shape

                    loss_discriminator_synthetic = loss_func(prediction_discriminator_synthetic,
                                                             torch.full(discriminator_prediction_shape, 0., dtype=torch.float32, device=device))
                    loss_discriminator_real = loss_func(prediction_discriminator_real,
                                                        torch.full(discriminator_prediction_shape, 1., dtype=torch.float32, device=device))
                    loss_generator_synthetic = loss_func(prediction_discriminator_synthetic, 
                                                         torch.full(discriminator_prediction_shape, 1., dtype=torch.float32, device=device))
                    l1_loss_generator_synthetic = l1_loss(synthetic_output, output_image) * 100

                    losses["losses_discriminator_real"].append(loss_discriminator_real.detach().cpu().item())
                    losses["losses_discriminator_synthetic"].append(loss_discriminator_synthetic.detach().cpu().item())
                    losses["losses_generator_synthetic"].append(loss_generator_synthetic.detach().cpu().item())
                    losses["l1_losses_generator_synthetic"].append(l1_loss_generator_synthetic.detach().cpu().item())

                for key in all_losses.keys():
                    all_losses[key].append(np.mean(losses[key[4:]]))
                print(f"\nLosses on the {loader_name} dataset:")
                utils.print_losses("pix2pix", epoch, all_losses)
                
    elif model.lower() == "cyclegan" or model.lower()=="attentiongan":
        if model.lower() == "cyclegan":
            generator_architecture = models.CycleGANGenerator
            discriminator_architecture = models.CycleGANDiscriminator
        else: # if model.lower()=="attentiongan":
            generator_architecture = models.AttentionGANGenerator
            discriminator_architecture = models.AttentionGANDiscriminator
        pre_to_post_generator = generator_architecture(input_channels=input_channels).to(device)
        post_to_pre_generator = generator_architecture(input_channels=input_channels).to(device)
        pre_discriminator = discriminator_architecture(input_channels=input_channels).to(device)
        post_discriminator = discriminator_architecture(input_channels=input_channels).to(device)
        pre_to_post_generator.load_state_dict(saved_model["pre_to_post_generator"])
        post_to_pre_generator.load_state_dict(saved_model["post_to_pre_generator"])
        pre_discriminator.load_state_dict(saved_model["pre_discriminator"])
        post_discriminator.load_state_dict(saved_model["post_discriminator"])
        add_identity_loss = saved_model["add_identity_loss"]
        pre_to_post_generator.eval()
        post_to_pre_generator.eval()
        pre_discriminator.eval()
        post_discriminator.eval()
        loss_func = nn.MSELoss()
        cycle_loss = nn.L1Loss()
        identity_loss = nn.L1Loss()
        
        for loader_name, loader in zip(["train", "validation"], [train_loader, val_loader]):
            losses = utils.initialise_loss_storage(model, overall=False, add_identity_loss=add_identity_loss)
            all_losses = utils.initialise_loss_storage(model, overall=True, add_identity_loss=add_identity_loss)
            with torch.no_grad():
                torch.manual_seed(47)
                for input_stack, output_image in loader:
                    real_pre_image = input_stack.to(device)
                    real_post_image = output_image.to(device)
                    if not not_input_topography:
                        topography = input_stack[:, 3:, :, :].detach().clone()
                        real_post_image = torch.cat((real_post_image, topography.clone().to(device)), dim=1)
                    synthetic_post_image = pre_to_post_generator(real_pre_image)
                    synthetic_pre_image = post_to_pre_generator(real_post_image)
                    if not not_input_topography:
                        synthetic_post_image = torch.cat((synthetic_post_image, topography.clone().to(device)), dim=1)
                        synthetic_pre_image = torch.cat((synthetic_pre_image, topography.clone().to(device)), dim=1)
                    recreated_post_image = pre_to_post_generator(synthetic_pre_image)  
                    recreated_pre_image = post_to_pre_generator(synthetic_post_image)
                    if add_identity_loss:
                        identity_loss_post = identity_loss(pre_to_post_generator(real_post_image), real_post_image[:, :3, :, :]) * 5
                        identity_loss_pre = identity_loss(post_to_pre_generator(real_pre_image), real_pre_image[:, :3, :, :]) * 5
                    
                    discriminator_prediction_shape = post_discriminator(synthetic_post_image).shape
                    post_generator_loss = loss_func(post_discriminator(synthetic_post_image), 
                                                    torch.full(discriminator_prediction_shape, 1., dtype=torch.float32, device=device))
                    pre_generator_loss = loss_func(pre_discriminator(synthetic_pre_image), 
                                                   torch.full(discriminator_prediction_shape, 1., dtype=torch.float32, device=device))
                    pre_to_post_cycle_loss = cycle_loss(recreated_pre_image, real_pre_image[:, :3, :, :]) * 10
                    post_to_pre_cycle_loss = cycle_loss(recreated_post_image, real_post_image[:, :3, :, :]) * 10                                       
                    loss_discriminator_real_pre = loss_func(pre_discriminator(real_pre_image),
                                                           torch.full(discriminator_prediction_shape, 1., dtype=torch.float32, device=device))
                    loss_discriminator_synthetic_pre = loss_func(pre_discriminator(synthetic_pre_image),
                                                                torch.full(discriminator_prediction_shape, 0., dtype=torch.float32, device=device))
                    loss_discriminator_real_post = loss_func(post_discriminator(real_post_image),
                                                            torch.full(discriminator_prediction_shape, 1., dtype=torch.float32, device=device))
                    loss_discriminator_synthetic_post = loss_func(post_discriminator(synthetic_post_image),
                                                                 torch.full(discriminator_prediction_shape, 0., dtype=torch.float32, device=device))
                    
                    losses["losses_generator_post"].append(post_generator_loss.detach().cpu().item())
                    losses["losses_generator_pre"].append(pre_generator_loss.detach().cpu().item())
                    losses["losses_pre_to_post_cycle"].append(pre_to_post_cycle_loss.detach().cpu().item())
                    losses["losses_post_to_pre_cycle"].append(post_to_pre_cycle_loss.detach().cpu().item())
                    losses["losses_discriminator_pre_real"].append(loss_discriminator_real_pre.detach().cpu().item())
                    losses["losses_discriminator_post_real"].append(loss_discriminator_real_post.detach().cpu().item())
                    losses["losses_discriminator_pre_synthetic"].append(loss_discriminator_synthetic_pre.detach().cpu().item())
                    losses["losses_discriminator_post_synthetic"].append(loss_discriminator_synthetic_post.detach().cpu().item())
                    if add_identity_loss:
                        losses["losses_identity_post"].append(identity_loss_post.detach().cpu().item())
                        losses["losses_identity_pre"].append(identity_loss_pre.detach().cpu().item())
                    
                for key in all_losses.keys():
                    all_losses[key].append(np.mean(losses[key[4:]]))
                print(f"\nLosses on the {loader_name} dataset:")
                utils.print_losses(model, epoch, all_losses, add_identity_loss)
                
    else:
        raise NotImplementedError("Model must be one of: Pix2Pix, CycleGAN or AttentionGAN")
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained Pix2Pix, CycleGAN or AttentionGAN model on the flood images dataset.")
    parser.add_argument("--model", required=True, help="Model can be one of: Pix2Pix, CycleGAN or AttentionGAN")
    parser.add_argument("--dataset_subset", required=True, help="Specify the dataset subset, e.g. USA, India, Hurricane-Harvery.")
    parser.add_argument("--dataset_dem", required=True, help="Specify whether the DEM used should be 'best' available or all the 'same'.")
    parser.add_argument("--data_path", required=True, help="The path to the location of the data folder")
    parser.add_argument("--saved_model_path", required=True, help="Path to the trained model")
    parser.add_argument("--num_images", type=int, default=5, help="The number of images the generator should create")
    parser.add_argument("--resize", type=int, default=256, help="Resize the images to the given size. The resize is applied before the crop")
    parser.add_argument("--crop", type=int, default=None, help="Crop each image into the given number of images. The resize is applied before the crop")
    parser.add_argument("--print_losses", action="store_true", default=False, help="Print the model's losses on the dataset")
    parser.add_argument("--save_images", action="store_true", default=False, help="Save generated images")
    args = parser.parse_args()
    
    if not os.path.isfile(args.saved_model_path):
        raise FileNotFoundError("Saved model not found. Check the path to the saved model.")
            
    if args.save_images:
        generate_images(args.dataset_subset,
                        args.dataset_dem,
                        args.data_path,
                        args.model,
                        args.resize,
                        args.crop,
                        args.num_images,
                        args.saved_model_path)
    if args.print_losses:
        print_losses(args.dataset_subset,
                     args.dataset_dem,
                     args.data_path,
                     args.model,
                     args.resize,
                     args.crop,
                     args.saved_model_path)