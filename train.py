import models
import data

import numpy as np
from datetime import datetime
import argparse
import os
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.optim import lr_scheduler

device = "cuda" if torch.cuda.is_available() else "cpu"
compile_model = False
#if torch.__version__[0] == "2" and os.name != "nt":
#    compile_model = True

def train_pix2pix(dataset_subset, 
                  dataset_dem, 
                  data_path,
                  num_epochs, 
                  resize,
                  crop,
                  save_model_interval,
                  save_images_interval,
                  verbose,
                  continue_training, 
                  saved_model_path):

    ### set-up #######
    
    if verbose:
        print(f"\nSetting up the Pix2Pix model...") 

    torch.manual_seed(47)
    discriminator = models.Pix2PixDiscriminator().apply(initialise_weights).to(device)
    generator = models.Pix2PixGenerator().apply(initialise_weights).to(device)
    if compile_model:
        discriminator = torch.compile(discriminator)
        generator = torch.compile(generator)
                  
    optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_generator = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
                  
    loss_func = nn.BCEWithLogitsLoss()
    l1_loss = nn.L1Loss()  
                  
    def lambda_rule(epoch):
        learning_rate = 1.0 - max(0, epoch + 1 - (num_epochs / 2)) / float((num_epochs/2) + 1)
        return learning_rate
    scheduler_discriminator = lr_scheduler.LambdaLR(optimizer_discriminator, lr_lambda=lambda_rule)
    scheduler_generator = lr_scheduler.LambdaLR(optimizer_generator, lr_lambda=lambda_rule) 
                  
    starting_epoch = 1
    losses = {"all_losses_discriminator_real":[],
              "all_losses_discriminator_synthetic":[],
              "all_losses_generator_synthetic":[],
              "all_l1_losses_generator_synthetic":[]}

    if continue_training:
        if not saved_model_path:
            raise ValueError("Provide a saved model.")
        if not os.path.isfile(saved_model_path):
            raise FileNotFoundError("Saved model not found. Check the path to the saved model.")
        saved_model = torch.load(saved_model_path)
            
        starting_epoch = saved_model["starting_epoch"]
        num_epochs = saved_model["num_epochs"]
        losses = saved_model["losses"]
        resize = saved_model["resize"]
        
        discriminator.load_state_dict(saved_model["discriminator"])
        generator.load_state_dict(saved_model["generator"])
        
        optimizer_discriminator.load_state_dict(saved_model["optimizer_discriminator"])
        optimizer_generator.load_state_dict(saved_model["optimizer_generator"])
        
        scheduler_discriminator.load_state_dict(saved_model["scheduler_discriminator"])
        scheduler_generator.load_state_dict(saved_model["scheduler_generator"])

    train_loader, val_loader, _ = data.create_dataset(dataset_subset, dataset_dem, path=data_path, resize=resize, crop=crop)
    
    if verbose:
        print(f"\n{'Continuing' if continue_training is True else 'Beginning'} training:")
        print(f"{num_epochs} epochs")
        print(f"Starting from epoch {starting_epoch}")
        print(f"Dataset: '{dataset_subset}' '{dataset_dem} DEM'")
        print(f"Data resized to {resize} pixels with {crop} crops")
        print(f"Models saved every {save_model_interval} epochs")
        print(f"Sample generator output images saved every {save_images_interval} epochs.\n")
        
    ### training #######
    
    for epoch in range(starting_epoch, num_epochs + 1):
        
        losses_discriminator_real = []
        losses_discriminator_synthetic = []
        losses_generator_synthetic = []
        l1_losses_generator_synthetic = []

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
            label_0 = torch.full(prediction_discriminator_synthetic.shape, 0., dtype=torch.float32, device=device)
            loss_discriminator_synthetic = loss_func(prediction_discriminator_synthetic, label_0)
            prediction_discriminator_real = discriminator(concat_real)
            label_1 = torch.full(prediction_discriminator_real.shape, 1., dtype=torch.float32, device=device)
            loss_discriminator_real = loss_func(prediction_discriminator_real, label_1)
            loss_discriminator = (loss_discriminator_synthetic + loss_discriminator_real) * 0.5
            loss_discriminator.backward()
            optimizer_discriminator.step()

            # train the generator
            for parameter in discriminator.parameters(): 
                parameter.requires_grad = False
            optimizer_generator.zero_grad()

            prediction_discriminator_synthetic = discriminator(concat_synthetic)
            label_1 = torch.full(prediction_discriminator_synthetic.shape, 1., dtype=torch.float32, device=device)
            loss_generator_synthetic = loss_func(prediction_discriminator_synthetic, label_1)
            l1_loss_generator_synthetic = l1_loss(synthetic_output, output_image) * 100
            loss_generator = loss_generator_synthetic + l1_loss_generator_synthetic
            loss_generator.backward()
            optimizer_generator.step()
            losses_discriminator_real.append(loss_discriminator_real.detach().cpu().item())
            losses_discriminator_synthetic.append(loss_discriminator_synthetic.detach().cpu().item())
            losses_generator_synthetic.append(loss_generator_synthetic.detach().cpu().item())
            l1_losses_generator_synthetic.append(l1_loss_generator_synthetic.detach().cpu().item())

        scheduler_discriminator.step()
        scheduler_generator.step()

        ### save results #######
        losses["all_losses_discriminator_real"].append(np.mean(losses_discriminator_real))
        losses["all_losses_discriminator_synthetic"].append(np.mean(losses_discriminator_synthetic))
        losses["all_losses_generator_synthetic"].append(np.mean(losses_generator_synthetic))
        losses["all_l1_losses_generator_synthetic"].append(np.mean(l1_losses_generator_synthetic))
        
        if verbose:
            print(f"{epoch=} | discriminator_real_loss={losses['all_losses_discriminator_real'][-1]:.2f} | discriminator_synthetic_loss={losses['all_losses_discriminator_synthetic'][-1]:.2f} | generator_synthetic_loss={losses['all_losses_generator_synthetic'][-1]:.2f} | l1_generator_loss={losses['all_l1_losses_generator_synthetic'][-1]:.2f}")

        if save_model_interval != 0 and ((epoch % save_model_interval) == 0):
            saved_model = {
                    "starting_epoch": epoch + 1,
                    "num_epochs": num_epochs,
                    "resize": resize,
                    "discriminator": discriminator.state_dict(),
                    "generator": generator.state_dict(),
                    "optimizer_discriminator": optimizer_discriminator.state_dict(),
                    "optimizer_generator": optimizer_generator.state_dict(),
                    "scheduler_discriminator": scheduler_discriminator.state_dict(),
                    "scheduler_generator": scheduler_generator.state_dict(),
                    "losses": losses,
                    }
            model_path = f"{data_path}/data/models/pix2pix_{dataset_subset}_{dataset_dem}DEM_resize{resize}_crop{crop}_epoch{epoch}_{str(datetime.now())[:-7].replace(' ', '-').replace(':', '-')}.pth.tar"
            print(f"Saving model to {model_path}")
            torch.save(saved_model, model_path)

        if save_images_interval != 0 and ((epoch % save_images_interval) == 0):
            
            num_images = 5
            generator.eval()
            with torch.no_grad():
                for split, loader in zip(["train", "val"], [train_loader, val_loader]):
                    fig, axes = plt.subplots(nrows=num_images, ncols=3, figsize=(15, num_images * 5))
                    for ax in axes.ravel():
                        ax.set_axis_off()
                    torch.manual_seed(47)
                    for i, (input_stack, output_image) in enumerate(loader):
                        input_stack = input_stack.to(device)
                        output_image = output_image.to(device)
                        axes[i, 0].imshow(input_stack.squeeze().detach().cpu().numpy().transpose(1, 2, 0)[:, :, :3], vmin=0, vmax=1)
                        axes[i, 1].imshow(np.clip(generator(input_stack).squeeze().detach().cpu().numpy().transpose(1, 2, 0)[:, :, :3], 0, 1), vmin=0, vmax=1)
                        axes[i, 2].imshow(output_image.squeeze().detach().cpu().numpy().transpose(1, 2, 0), vmin=0, vmax=1)
                        axes[i, 0].set_title("Input")
                        axes[i, 1].set_title("Generator Output")
                        axes[i, 2].set_title("Ground Truth Output")
                        if i >= num_images-1:
                            break
                    fig.tight_layout()
                    images_path = f"{data_path}/data/images/pix2pix_{split}_{dataset_subset}_{dataset_dem}DEM_resize{resize}_crop{crop}_epoch{epoch}_{str(datetime.now())[:-7].replace(' ', '-').replace(':', '-')}.png"
                    print(f"Saving generated images to {images_path}")
                    fig.savefig(images_path)
                    plt.close();

def initialise_weights(m):
    classname = m.__class__.__name__
    if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm2d') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the Pix2Pix, CycleGAN or AttentionGAN model on the flood images dataset")
    parser.add_argument("--model", required=True, help="Model can be one of: Pix2Pix, CycleGAN or AttentionGAN")
    parser.add_argument("--dataset_subset", required=True, help="Specify the dataset subset, e.g. USA, India, Hurricane-Harvery")
    parser.add_argument("--dataset_dem", required=True, help="Specify whether the DEM used should be 'best' available or all the 'same'")
    parser.add_argument("--data_path", required=True, help="The path to the location of the data folder")
    parser.add_argument("--num_epochs", type=int, required=True, help="Number of epochs to train for")
    parser.add_argument("--resize", type=int, default=256, help="Resize the images to the given size. The resize is applied before the crop")
    parser.add_argument("--crop", type=int, default=None, help="Crop each image into the given number of images. The resize is applied before the crop")
    parser.add_argument("--save_model_interval", type=int, default=100, help="Save the model every given number of epochs. Set to 0 if you don't want to save the model")
    parser.add_argument("--save_images_interval", type=int, default=100, help="Save some sample generator outputs every given number of epochs Set to 0 if you don't want to save images")
    parser.add_argument("--continue_training", default=False, action="store_true", help="Whether training should be resumed from a pre-trained model")
    parser.add_argument("--saved_model_path", default=None, help="If continue_training==True, then this path should point to the pre-trained model")
    parser.add_argument("--verbose", default=False, action="store_true", help="Print out the losses on every epoch")
    args = parser.parse_args()
    
    if args.model.lower() == "pix2pix":
        train_pix2pix(args.dataset_subset, 
                      args.dataset_dem, 
                      args.data_path,
                      args.num_epochs, 
                      args.resize,
                      args.crop,
                      args.save_model_interval,
                      args.save_images_interval,
                      args.verbose,
                      args.continue_training, 
                      args.saved_model_path)
    elif args.model.lower() == "cyclegan":
        None
    elif args.model.lower() == "attentiongan":
        None
    else:
        raise NotImplementedError("Model must be one of: Pix2Pix, CycleGAN or AttentionGAN")
