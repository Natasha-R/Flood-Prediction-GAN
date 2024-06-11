import models
import data
import evaluate
import utils

import numpy as np
import argparse
import os
import itertools
import random

import torch
from torch import nn
from torch.optim import lr_scheduler

device = "cuda" if torch.cuda.is_available() else "cpu"
compile_model = False
#if torch.__version__[0] == "2" and os.name != "nt":
#    compile_model = True

def train_conditional(model,
                      dataset_subset, 
                      dataset_dem, 
                      data_path,
                      num_epochs, 
                      not_input_topography,
                      resize,
                      crop,
                      save_model_interval,
                      save_images_interval,
                      verbose,
                      continue_training, 
                      saved_model_path,
                      add_identity_loss):

    ### set-up #######
    
    if verbose:
        print(f"\nSetting up the {model} model...") 

    if continue_training:
        saved_model = torch.load(saved_model_path)
        model = saved_model["model"]
        num_epochs = saved_model["num_epochs"]
        all_losses = saved_model["all_losses"]
        not_input_topography = saved_model["not_input_topography"]
        
    if not_input_topography:
        input_channels = 3
    else: # if input_topography
        input_channels = 9
        
    torch.manual_seed(47)
    discriminator = models.Pix2PixDiscriminator(input_channels=input_channels).apply(utils.initialise_weights).to(device)
    generator = models.Pix2PixGenerator(input_channels=input_channels).apply(utils.initialise_weights).to(device)
    if compile_model:
        discriminator = torch.compile(discriminator)
        generator = torch.compile(generator)
                  
    loss_func = nn.BCEWithLogitsLoss()
    l1_loss = nn.L1Loss()  
    
    optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_generator = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
                        
    lambda_rule = utils.define_lambda_rule(num_epochs)
    scheduler_discriminator = lr_scheduler.LambdaLR(optimizer_discriminator, lr_lambda=lambda_rule)
    scheduler_generator = lr_scheduler.LambdaLR(optimizer_generator, lr_lambda=lambda_rule) 
                  
    starting_epoch = 1
    all_losses = utils.initialise_loss_storage(model, overall=True)

    if continue_training:
        
        starting_epoch = saved_model["starting_epoch"]
        
        discriminator.load_state_dict(saved_model["discriminator"])
        generator.load_state_dict(saved_model["generator"])
        
        optimizer_discriminator.load_state_dict(saved_model["optimizer_discriminator"])
        optimizer_generator.load_state_dict(saved_model["optimizer_generator"])
        
        scheduler_discriminator.load_state_dict(saved_model["scheduler_discriminator"])
        scheduler_generator.load_state_dict(saved_model["scheduler_generator"])

    train_loader, val_loader, _ = data.create_dataset(dataset_subset, dataset_dem, data_path, not_input_topography, resize, crop)
    
    if verbose:
        utils.print_setup(model, continue_training, num_epochs, starting_epoch, not_input_topography, dataset_subset, dataset_dem, resize, crop, save_model_interval, save_images_interval, add_identity_loss)
        
    ### training #######
    
    for epoch in range(starting_epoch, num_epochs + 1):
        
        losses = utils.initialise_loss_storage(model, overall=False)

        discriminator.train()
        generator.train()

        torch.manual_seed(epoch)

        for input_stack, output_image in train_loader:

            input_stack = input_stack.to(device)
            output_image = output_image.to(device)
            synthetic_output = generator(input_stack)
            concat_real = torch.cat((input_stack, output_image), 1)
            concat_synthetic = torch.cat((input_stack, synthetic_output), 1)

            # train the discriminator
            for parameter in discriminator.parameters(): 
                parameter.requires_grad = True
            optimizer_discriminator.zero_grad()
            
            prediction_discriminator_synthetic = discriminator(concat_synthetic.detach()) 
            discriminator_prediction_shape = prediction_discriminator_synthetic.shape
            loss_discriminator_synthetic = loss_func(prediction_discriminator_synthetic,
                                                     torch.full(discriminator_prediction_shape, 0., dtype=torch.float32, device=device))
            prediction_discriminator_real = discriminator(concat_real)
            loss_discriminator_real = loss_func(prediction_discriminator_real,
                                               torch.full(discriminator_prediction_shape, 1., dtype=torch.float32, device=device))
            loss_discriminator = (loss_discriminator_synthetic + loss_discriminator_real) * 0.5
            loss_discriminator.backward()
            optimizer_discriminator.step()

            # train the generator
            for parameter in discriminator.parameters(): 
                parameter.requires_grad = False
            optimizer_generator.zero_grad()

            prediction_discriminator_synthetic = discriminator(concat_synthetic)
            loss_generator_synthetic = loss_func(prediction_discriminator_synthetic, 
                                                torch.full(discriminator_prediction_shape, 1., dtype=torch.float32, device=device))
            l1_loss_generator_synthetic = l1_loss(synthetic_output, output_image) * 100
            loss_generator = loss_generator_synthetic + l1_loss_generator_synthetic
            loss_generator.backward()
            optimizer_generator.step()
            
            losses["losses_discriminator_real"].append(loss_discriminator_real.detach().cpu().item())
            losses["losses_discriminator_synthetic"].append(loss_discriminator_synthetic.detach().cpu().item())
            losses["losses_generator_synthetic"].append(loss_generator_synthetic.detach().cpu().item())
            losses["l1_losses_generator_synthetic"].append(l1_loss_generator_synthetic.detach().cpu().item())

        scheduler_discriminator.step()
        scheduler_generator.step()

        ### save results #######
        for key in all_losses.keys():
            all_losses[key].append(np.mean(losses[key[4:]]))
        
        if verbose:
            utils.print_losses(model, epoch, all_losses)

        if save_model_interval != 0 and ((epoch % save_model_interval) == 0):
            saved_model = {
                    "model": model,
                    "starting_epoch": epoch + 1,
                    "num_epochs": num_epochs,
                    "not_input_topography": not_input_topography,
                    "discriminator": discriminator.state_dict(),
                    "generator": generator.state_dict(),
                    "optimizer_discriminator": optimizer_discriminator.state_dict(),
                    "optimizer_generator": optimizer_generator.state_dict(),
                    "scheduler_discriminator": scheduler_discriminator.state_dict(),
                    "scheduler_generator": scheduler_generator.state_dict(),
                    "all_losses": all_losses,
                    }
            model_path = utils.create_path("model", model, data_path, "model", dataset_subset, dataset_dem, not_input_topography, resize, crop, epoch, add_identity_loss)
            print(f"Saving {model} model to {model_path}")
            torch.save(saved_model, model_path)

        if save_images_interval != 0 and ((epoch % save_images_interval) == 0):
            evaluate.generate_images(dataset_subset, dataset_dem, data_path, model, resize, crop, 5, add_identity_loss=add_identity_loss, not_input_topography=not_input_topography, trained_model=generator, epoch=epoch)
            
            
def train_cycle(model,
                dataset_subset, 
                dataset_dem, 
                data_path,
                num_epochs, 
                not_input_topography,
                resize,
                crop,
                save_model_interval,
                save_images_interval,
                verbose,
                continue_training, 
                saved_model_path,
                add_identity_loss):

    ### set-up #######

    if verbose:
        print(f"\nSetting up the {model} model...") 

    if continue_training:
        saved_model = torch.load(saved_model_path)
        model = saved_model["model"]
        num_epochs = saved_model["num_epochs"]
        all_losses = saved_model["all_losses"]
        not_input_topography = saved_model["not_input_topography"]
        add_identity_loss = saved_model["add_identity_loss"]
        
    if not_input_topography:
        input_channels = 3
    else: # if input_topography
        input_channels = 9
        
    torch.manual_seed(47)
    if model.lower() == "cyclegan":
        generator_architecture = models.CycleGANGenerator
        discriminator_architecture = models.CycleGANDiscriminator
    else: # if model.lower()=="attentiongan":
        generator_architecture = models.AttentionGANGenerator
        discriminator_architecture = models.AttentionGANDiscriminator
        
    pre_to_post_generator = generator_architecture(input_channels=input_channels).apply(utils.initialise_weights).to(device)
    post_to_pre_generator = generator_architecture(input_channels=input_channels).apply(utils.initialise_weights).to(device)
    pre_discriminator = discriminator_architecture(input_channels=input_channels).apply(utils.initialise_weights).to(device)
    post_discriminator = discriminator_architecture(input_channels=input_channels).apply(utils.initialise_weights).to(device)

    loss_func = nn.MSELoss()
    cycle_loss = nn.L1Loss()
    identity_loss = nn.L1Loss()

    optimizer_generators = torch.optim.Adam(itertools.chain(pre_to_post_generator.parameters(), post_to_pre_generator.parameters()), 
                                            lr=0.0002, betas=(0.5, 0.999))
    optimizer_discriminators = torch.optim.Adam(itertools.chain(post_discriminator.parameters(), pre_discriminator.parameters()), 
                                                lr=0.0002, betas=(0.5, 0.999))
    lambda_rule = utils.define_lambda_rule(num_epochs)
    scheduler_generators = lr_scheduler.LambdaLR(optimizer_generators, lr_lambda=lambda_rule) 
    scheduler_discriminators = lr_scheduler.LambdaLR(optimizer_discriminators, lr_lambda=lambda_rule)

    starting_epoch = 1
    all_losses = utils.initialise_loss_storage(model, overall=True, add_identity_loss=add_identity_loss)

    if continue_training:
        
        starting_epoch = saved_model["starting_epoch"]
        
        pre_to_post_generator.load_state_dict(saved_model["pre_to_post_generator"])
        post_to_pre_generator.load_state_dict(saved_model["post_to_pre_generator"])
        pre_discriminator.load_state_dict(saved_model["pre_discriminator"])
        post_discriminator.load_state_dict(saved_model["post_discriminator"])

        optimizer_generators.load_state_dict(saved_model["optimizer_generators"])
        optimizer_discriminators.load_state_dict(saved_model["optimizer_discriminators"])

        scheduler_generators.load_state_dict(saved_model["scheduler_generators"])
        scheduler_discriminators.load_state_dict(saved_model["scheduler_discriminators"])

    train_loader, val_loader, _ = data.create_dataset(dataset_subset, dataset_dem, data_path, not_input_topography, resize, crop)
    pre_images_buffer = []
    post_images_buffer = []

    if verbose:
        utils.print_setup(model, continue_training, num_epochs, starting_epoch, not_input_topography, dataset_subset, dataset_dem, resize, crop, save_model_interval, save_images_interval, add_identity_loss)

    ### training #######

    for epoch in range(starting_epoch, num_epochs + 1):

        losses = utils.initialise_loss_storage(model, overall=False, add_identity_loss=add_identity_loss)

        pre_to_post_generator.train()
        post_to_pre_generator.train()
        pre_discriminator.train()
        post_discriminator.train()

        torch.manual_seed(epoch)

        for input_stack, output_image in train_loader:
            
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
            # train the generators
            for parameter in pre_discriminator.parameters(): 
                parameter.requires_grad = False
            for parameter in post_discriminator.parameters(): 
                parameter.requires_grad = False
            optimizer_generators.zero_grad()
            
            identity_loss_post = 0
            identity_loss_pre = 0
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
            generator_loss = post_generator_loss + pre_generator_loss + pre_to_post_cycle_loss + post_to_pre_cycle_loss + identity_loss_post + identity_loss_pre
            generator_loss.backward()
            optimizer_generators.step()

            # train the discriminators
            for parameter in pre_discriminator.parameters(): 
                parameter.requires_grad = True
            for parameter in post_discriminator.parameters(): 
                parameter.requires_grad = True
            optimizer_discriminators.zero_grad()

            synthetic_pre_image = utils.get_buffer_image(synthetic_pre_image, pre_images_buffer)
            synthetic_post_image = utils.get_buffer_image(synthetic_post_image, post_images_buffer)

            loss_discriminator_real_pre = loss_func(pre_discriminator(real_pre_image),
                                                   torch.full(discriminator_prediction_shape, 1., dtype=torch.float32, device=device))
            loss_discriminator_synthetic_pre = loss_func(pre_discriminator(synthetic_pre_image),
                                                        torch.full(discriminator_prediction_shape, 0., dtype=torch.float32, device=device))
            loss_discriminator_pre = (loss_discriminator_real_pre + loss_discriminator_synthetic_pre) * 0.5
            loss_discriminator_pre.backward()

            loss_discriminator_real_post = loss_func(post_discriminator(real_post_image),
                                                    torch.full(discriminator_prediction_shape, 1., dtype=torch.float32, device=device))
            loss_discriminator_synthetic_post = loss_func(post_discriminator(synthetic_post_image),
                                                         torch.full(discriminator_prediction_shape, 0., dtype=torch.float32, device=device))
            loss_discriminator_post = (loss_discriminator_real_post + loss_discriminator_synthetic_post) * 0.5
            loss_discriminator_post.backward()
            optimizer_discriminators.step()

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

        scheduler_generators.step()
        scheduler_discriminators.step()

        ### save results #######

        for key in all_losses.keys():
            all_losses[key].append(np.mean(losses[key[4:]]))

        if verbose:
            utils.print_losses(model, epoch, all_losses, add_identity_loss)

        if save_model_interval != 0 and ((epoch % save_model_interval) == 0):
            saved_model = {
                    "model": model,
                    "starting_epoch": epoch + 1,
                    "num_epochs": num_epochs,
                    "pre_to_post_generator": pre_to_post_generator.state_dict(),
                    "post_to_pre_generator": post_to_pre_generator.state_dict(),
                    "pre_discriminator": pre_discriminator.state_dict(),
                    "post_discriminator": post_discriminator.state_dict(),
                    "optimizer_generators": optimizer_generators.state_dict(),
                    "optimizer_discriminators": optimizer_discriminators.state_dict(),
                    "scheduler_generators": scheduler_generators.state_dict(),
                    "scheduler_discriminators": scheduler_discriminators.state_dict(),
                    "all_losses": all_losses,
                    "not_input_topography": not_input_topography,
                    "add_identity_loss": add_identity_loss,
                    }
            model_path = utils.create_path("model", model, data_path, "model", dataset_subset, dataset_dem, not_input_topography, resize, crop, epoch, add_identity_loss)
            print(f"Saving model to {model_path}")
            torch.save(saved_model, model_path)

        if save_images_interval != 0 and ((epoch % save_images_interval) == 0):
            evaluate.generate_images(dataset_subset, dataset_dem, data_path, model, resize, crop, 5, add_identity_loss=add_identity_loss, not_input_topography=not_input_topography, trained_model=[pre_to_post_generator, post_to_pre_generator], epoch=epoch)            
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the Pix2Pix, CycleGAN or AttentionGAN model on the flood images dataset")
    parser.add_argument("--model", required=True, help="Model can be one of: Pix2Pix, CycleGAN or AttentionGAN")
    parser.add_argument("--dataset_subset", required=True, help="Specify the dataset subset, e.g. USA, India, Hurricane-Harvery")
    parser.add_argument("--dataset_dem", required=True, help="Specify whether the DEM used should be 'best' available or all the 'same'")
    parser.add_argument("--data_path", required=True, help="The path to the location of the data folder")
    parser.add_argument("--num_epochs", type=int, default=10, help="Number of epochs to train for")
    parser.add_argument("--not_input_topography", default=False, action="store_true", help="The additional topographical factors (DEM/flow accumulation/distance to rivers/map) should NOT be input to the model")
    parser.add_argument("--resize", type=int, default=256, help="Resize the images to the given size. The resize is applied before the crop")
    parser.add_argument("--crop", type=int, default=None, help="Crop each image into the given number of images. The resize is applied before the crop")
    parser.add_argument("--save_model_interval", type=int, default=100, help="Save the model every given number of epochs. Set to 0 if you don't want to save the model")
    parser.add_argument("--save_images_interval", type=int, default=100, help="Save some sample generator outputs every given number of epochs Set to 0 if you don't want to save images")
    parser.add_argument("--verbose", default=False, action="store_true", help="Print out the losses on every epoch")
    parser.add_argument("--continue_training", default=False, action="store_true", help="Whether training should be resumed from a pre-trained model")
    parser.add_argument("--saved_model_path", default=None, help="If continue_training==True, then this path should point to the pre-trained model")
    parser.add_argument("--add_identity_loss", action="store_true", default=False, help="Add identity loss to the CycleGAN or AttentionGAN's loss function")
    
    args = parser.parse_args()
    
    if args.continue_training:
        if not args.saved_model_path:
            raise ValueError("Provide a saved model.")
        if not os.path.isfile(args.saved_model_path):
            raise FileNotFoundError("Saved model not found. Check the path to the saved model.")
    
    if args.model.lower() == "pix2pix":
        train_conditional(**vars(args))
    elif args.model.lower() == "cyclegan" or args.model.lower() == "attentiongan":
        train_cycle(**vars(args))
    else:
        raise NotImplementedError("Model must be one of: Pix2Pix, CycleGAN or AttentionGAN")
