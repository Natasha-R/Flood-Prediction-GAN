import models
import data
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn

device = "cuda" if torch.cuda.is_available() else "cpu"

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
    parser.add_argument("--print_losses", action="store_true", help="Print the model's losses on the dataset")
    parser.add_argument("--save_images", action="store_true", help="Save generated images")
    args = parser.parse_args()
    
if args.save_images:
    generate_images(args.dataset_subset,
                    args.dataset_dem,
                    args.data_path,
                    args.model,
                    args.resize,
                    args.crop,
                    args.saved_model_path,
                    args.num_images)
if args.print_losses:
    print_losses(ags.dataset_subset,
                 args.dataset_dem,
                 args.data_path,
                 args.model,
                 args.resize,
                 args.crop,
                 args.saved_model_path)
    
def generate_images(dataset_subset,
                    dataset_dem,
                    data_path,
                    model,
                    resize,
                    crop,
                    saved_model_path,
                    num_images):
    
    # set-up
    train_loader, val_loader, _ = data.create_dataset(dataset_subset, dataset_dem, data_path, resize=resize, crop=crop)
    saved_model = torch.load(saved_model_path)
    epoch = saved_model["starting_epoch"]
    
    if model.lower()=="pix2pix":
        generator = models.Pix2PixGenerator().to(device)
        generator.load_state_dict(saved_model["generator"])
    elif model.lower() == "cyclegan":
        None
    elif model.lower() == "attentiongan":
        None
    else:
        raise NotImplementedError("Model must be one of: Pix2Pix, CycleGAN or AttentionGAN")
        
    if num_images > min(len(train_loader), len(val_loader)):
        raise ValueError(f"Enter num_images as {min(len(train_loader), len(val_loader))} or fewer")
          
    # save images
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
                axes[i, 0].imshow(input_stack.squeeze().cpu().detach().numpy().transpose(1, 2, 0)[:, :, :3], vmin=0, vmax=1)
                axes[i, 1].imshow(np.clip(generator(input_stack).squeeze().cpu().detach().numpy().transpose(1, 2, 0)[:, :, :3], 0, 1), vmin=0, vmax=1)
                axes[i, 2].imshow(output_image.squeeze().cpu().detach().numpy().transpose(1, 2, 0), vmin=0, vmax=1)
                axes[i, 0].set_title("Input")
                axes[i, 1].set_title("Generator Output")
                axes[i, 2].set_title("Ground Truth Output")
                if i >= num_images-1:
                    break
            fig.tight_layout()
            fig.savefig(f"{data_path}/data/images/{model}_{split}_{dataset_subset}_{dataset_dem}DEM_resize{resize}_crop{crop}_epoch{epoch}_{str(datetime.now())[:-7].replace(' ', '-').replace(':', '-')}.png")
            plt.close();
            
def print_losses(dataset_subset,
                 dataset_dem,
                 model,
                 resize,
                 crop,
                 saved_model_path):

    if model.lower()=="pix2pix":
        # set-up
        train_loader, val_loader, _ = data.create_dataset(dataset_subset, dataset_dem, resize=resize, crop=crop)
        saved_model = torch.load(saved_model_path)
        epoch = saved_model["starting_epoch"]
        generator = models.Pix2PixGenerator().to(device)
        generator.load_state_dict(saved_model["generator"])
        discriminator = models.Pix2PixDiscriminator().to(device)
        discriminator.load_state_dict(saved_model["discriminator"])
        train_losses_discriminator_synthetic = []
        train_losses_generator_synthetic = []
        train_l1_losses_generator_synthetic = []
        train_losses_discriminator_real = []
        val_losses_discriminator_synthetic = []
        val_losses_generator_synthetic = []
        val_l1_losses_generator_synthetic = []
        val_losses_discriminator_real = []

        # print losses
        for input_stack, output_image in train_loader:

            input_stack = input_stack.to(device)
            output_image = output_image.to(device)

            synthetic_output = generator(input_stack)
            concat_synthetic = torch.cat((input_stack, synthetic_output), 1)
            concat_real = torch.cat((input_stack, output_image), 1)
            prediction_discriminator_real = discriminator(concat_real)
            prediction_discriminator_synthetic = discriminator(concat_synthetic)

            loss_func = nn.BCEWithLogitsLoss()
            l1_loss = nn.L1Loss()  

            l1_loss_generator_synthetic = l1_loss(synthetic_output, output_image) * 100
            label_1 = torch.full(prediction_discriminator_synthetic.shape, 1., dtype=torch.float32, device=device)
            loss_generator_synthetic = loss_func(prediction_discriminator_synthetic, label_1)
            label_0 = torch.full(prediction_discriminator_synthetic.shape, 0., dtype=torch.float32, device=device)
            loss_discriminator_synthetic = loss_func(prediction_discriminator_synthetic, label_0)
            label_1 = torch.full(prediction_discriminator_real.shape, 1., dtype=torch.float32, device=device)
            loss_discriminator_real = loss_func(prediction_discriminator_real, label_1)

            train_losses_discriminator_real.append(loss_discriminator_real.detach().cpu().item())
            train_losses_discriminator_synthetic.append(loss_discriminator_synthetic.detach().cpu().item())
            train_losses_generator_synthetic.append(loss_generator_synthetic.detach().cpu().item())
            train_l1_losses_generator_synthetic.append(l1_loss_generator_synthetic.detach().cpu().item())

        for input_stack, output_image in val_loader:

            input_stack = input_stack.to(device)
            output_image = output_image.to(device)

            synthetic_output = generator(input_stack)
            concat_synthetic = torch.cat((input_stack, synthetic_output), 1)
            concat_real = torch.cat((input_stack, output_image), 1)
            prediction_discriminator_real = discriminator(concat_real)
            prediction_discriminator_synthetic = discriminator(concat_synthetic)

            loss_func = nn.BCEWithLogitsLoss()
            l1_loss = nn.L1Loss()  

            l1_loss_generator_synthetic = l1_loss(synthetic_output, output_image) * 100
            label_1 = torch.full(prediction_discriminator_synthetic.shape, 1., dtype=torch.float32, device=device)
            loss_generator_synthetic = loss_func(prediction_discriminator_synthetic, label_1)
            label_0 = torch.full(prediction_discriminator_synthetic.shape, 0., dtype=torch.float32, device=device)
            loss_discriminator_synthetic = loss_func(prediction_discriminator_synthetic, label_0)
            label_1 = torch.full(prediction_discriminator_real.shape, 1., dtype=torch.float32, device=device)
            loss_discriminator_real = loss_func(prediction_discriminator_real, label_1)

            val_losses_discriminator_real.append(loss_discriminator_real.detach().cpu().item())
            val_losses_discriminator_synthetic.append(loss_discriminator_synthetic.detach().cpu().item())
            val_losses_generator_synthetic.append(loss_generator_synthetic.detach().cpu().item())
            val_l1_losses_generator_synthetic.append(l1_loss_generator_synthetic.detach().cpu().item())

        print(f"Average discriminator loss on synthetic training data: {np.mean(train_losses_discriminator_synthetic):.2f}\n",
              f"Average discriminator loss on synthetic validation data: {np.mean(val_losses_discriminator_synthetic):.2f}\n",
              f"Average discriminator loss on real training data: {np.mean(train_losses_discriminator_real):.2f}\n",
              f"Average discriminator loss on real validation data: {np.mean(val_losses_discriminator_real):.2f}\n",
              f"Average generator loss on synthetic training data: {np.mean(train_losses_generator_synthetic):.2f}\n",
              f"Average generator loss on synthetic validation data: {np.mean(val_losses_generator_synthetic):.2f}\n",
              f"Average generator L1 loss on synthetic training data: {np.mean(train_l1_losses_generator_synthetic):.2f}\n",
              f"Average generator L1 loss on synthetic validation data: {np.mean(val_l1_losses_generator_synthetic):.2f}\n",)