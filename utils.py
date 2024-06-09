import torch
from torch import nn
from datetime import datetime
import random
device = "cuda" if torch.cuda.is_available() else "cpu"

def initialise_weights(m):
    classname = m.__class__.__name__
    if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm2d') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)
        
def get_buffer_image(image, images_buffer):
    image = image.detach()
    if len(images_buffer) < 50: # if the buffer is not yet full, always append the new image AND return it
        images_buffer.append(image.cpu())
        return image
    else:
        p = random.uniform(0, 1)
        if p > 0.5: # 50% chance the new image is stored and a random old image is returned
            index = random.randint(0, 50 - 1)
            old_image = images_buffer[index].clone()
            images_buffer[index] = image.cpu()
            return old_image.to(device)
        else: # 50% chance the new image is not stored and the new image is returned
            return image
        
def define_lambda_rule(num_epochs):
    def lambda_rule(epoch):
        learning_rate = 1.0 - max(0, epoch + 1 - (num_epochs / 2)) / float((num_epochs/2) + 1)
        return learning_rate
    return lambda_rule
        
def print_setup(model, continue_training, num_epochs, starting_epoch, not_input_topography, dataset_subset, dataset_dem, resize, crop, save_model_interval, save_images_interval, add_identity_loss):
        print(f"\n{'Continuing' if continue_training is True else 'Beginning'} training {model}:")
        print(f"{num_epochs} epochs")
        print(f"Starting from epoch {starting_epoch}")
        print(f"Additional topographical factors will {'NOT ' if not_input_topography else ''}be input to the model")
        if (model.lower()=="cyclegan" or model.lower()=="attentiongan") and add_identity_loss:
            print(f"Using identity mapping loss")
        print(f"Dataset: '{dataset_subset}' '{dataset_dem} DEM'")
        print(f"Data resized to {resize} pixels with {crop} crops")
        print(f"Model saved every {save_model_interval} epochs")
        print(f"Sample generator output images saved every {save_images_interval} epochs\n")
        
def create_path(save, model, data_path, split, dataset_subset, dataset_dem, not_input_topography, resize, crop, epoch, add_identity_loss):
    if save=="image":
        file_type = ".png"
    elif save=="model":
        file_type = ".pth.tar"
    else:
        raise NotImplementedError("Path must relate to either images or a model")
    path = f"{data_path}/data/{save}s/{model}_topography{not not_input_topography}_{split}_{dataset_subset}_{dataset_dem}DEM_resize{resize}_crop{crop}_identityLoss{add_identity_loss}_epoch{epoch}_{str(datetime.now())[:-7].replace(' ', '-').replace(':', '-')}{file_type}"
    return path

def initialise_loss_storage(model, overall, add_identity_loss=False):
    pre_string = ""
    if overall:
        pre_string = "all_"
        
    if model.lower()=="pix2pix":
        return {f"{pre_string}losses_discriminator_real":[],
                f"{pre_string}losses_discriminator_synthetic":[],
                f"{pre_string}losses_generator_synthetic":[],
                f"{pre_string}l1_losses_generator_synthetic":[]}
    
    elif model.lower()=="cyclegan" or model.lower()=="attentiongan":
        loss_dict = {f"{pre_string}losses_generator_post":[],
                     f"{pre_string}losses_generator_pre":[],
                     f"{pre_string}losses_pre_to_post_cycle":[],
                     f"{pre_string}losses_post_to_pre_cycle":[],
                     f"{pre_string}losses_discriminator_pre_real":[],
                     f"{pre_string}losses_discriminator_post_real":[],
                     f"{pre_string}losses_discriminator_pre_synthetic":[],
                     f"{pre_string}losses_discriminator_post_synthetic":[]}
        if add_identity_loss:
            loss_dict[f"{pre_string}losses_identity_post"]=[]
            loss_dict[f"{pre_string}losses_identity_pre"]=[]
        return loss_dict
        
    else:
        raise NotImplementedError("Model must be one of: Pix2Pix, CycleGAN or AttentionGAN")
        
def print_losses(model, epoch, all_losses, add_identity_loss=False):
    
    if model.lower()=="pix2pix":
        print(f"{epoch=} | Discriminator real loss = {all_losses['all_losses_discriminator_real'][-1]:.2f} | Discriminator synthetic loss = {all_losses['all_losses_discriminator_synthetic'][-1]:.2f} | Generator synthetic loss = {all_losses['all_losses_generator_synthetic'][-1]:.2f} | L1 Generator loss = {all_losses['all_l1_losses_generator_synthetic'][-1]:.2f}")
    
    elif model.lower()=="cyclegan" or model.lower()=="attentiongan":
        print(f"{epoch=} | Generator post image loss = {all_losses['all_losses_generator_post'][-1]:.2f} | Generator pre image loss = {all_losses['all_losses_generator_pre'][-1]:.2f} | Pre to post cycle loss = {all_losses['all_losses_pre_to_post_cycle'][-1]:.2f} | Post to pre cycle loss = {all_losses['all_losses_post_to_pre_cycle'][-1]:.2f} | Discriminator pre real image loss = {all_losses['all_losses_discriminator_pre_real'][-1]:.2f} | Discriminator post real image loss = {all_losses['all_losses_discriminator_post_real'][-1]:.2f} | Discriminator pre synthetic image loss = {all_losses['all_losses_discriminator_pre_synthetic'][-1]:.2f} | Discriminator post synthetic image loss = {all_losses['all_losses_discriminator_post_synthetic'][-1]:.2f}", end="" if add_identity_loss else "\n")
        if add_identity_loss:
            print(f" | Identity pre image loss = {all_losses['all_losses_identity_pre'][-1]:.2f} | Identity post image loss = {all_losses['all_losses_identity_post'][-1]:.2f}")
        
    else:
        raise NotImplementedError("Model must be one of: Pix2Pix, CycleGAN or AttentionGAN")
    