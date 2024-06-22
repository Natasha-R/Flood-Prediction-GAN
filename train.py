from models import model

import os
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the Pix2Pix, CycleGAN, AttentionGAN or PairedAttention model on the flood images dataset")
    parser.add_argument("--model", required=True, help="Model can be one of: Pix2Pix, CycleGAN, AttentionGAN or PairedAttention")
    parser.add_argument("--dataset_subset", required=True, help="Specify the dataset subset, e.g. USA, India, Hurricane-Harvey")
    parser.add_argument("--dataset_dem", required=True, help="Specify whether the DEM used should be 'best' available or all the 'same'")
    parser.add_argument("--data_path", required=True, help="The path to the location of the data folder. Example: 'C:/data'")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of epochs to train for")
    parser.add_argument("--topography", default=None, help="Which topographical factors should be input to the model. 'all', 'dem', 'map', 'flow', or 'river'")
    parser.add_argument("--resize", type=int, default=None, help="Resize the images to the given size. The resize is applied before the crop")
    parser.add_argument("--crop", type=int, default=None, help="Crop each image into the given number of images. The resize is applied before the crop")
    parser.add_argument("--save_model_interval", type=int, default=0, help="Save the model every given number of epochs. Set to 0 if you don't want to save the model")
    parser.add_argument("--save_images_interval", type=int, default=0, help="Save some sample generator outputs every given number of epochs Set to 0 if you don't want to save images")
    parser.add_argument("--verbose", default=False, action="store_true", help="Print out the losses on every epoch")
    parser.add_argument("--load_pretrained_model", default=False, action="store_true", help="Whether training should be resumed from a pre-trained model")
    parser.add_argument("--pretrained_model_path", default=None, help="If load_pretrained_model==True, then this path should point to the model")
    parser.add_argument("--add_identity_loss", action="store_true", default=False, help="Add identity loss to the CycleGAN or AttentionGAN's loss function")
    parser.add_argument("--seed", type=int, default=47, help="The random seed to initialise the models")

    args = parser.parse_args()
    args.model = args.model.lower()

    if args.load_pretrained_model:
        if not args.pretrained_model_path:
            raise ValueError("Provide a saved model.")
        if not os.path.isfile(args.pretrained_model_path):
            raise FileNotFoundError("Saved model not found. Check the path to the model.")
    
    args.training_model = True
    train_model = model.Model(**vars(args))
    if train_model.model_is_cycle:
        train_model.train_cycle()
    else:
        train_model.train_paired()