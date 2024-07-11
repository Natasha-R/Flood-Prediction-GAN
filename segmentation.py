from models import segmentation_model

import os
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the Pix2Pix, CycleGAN, AttentionGAN or PairedAttention model on the flood images dataset")
    parser.add_argument("--train", action="store_true", default=False, help="Train the model, else evaluate a pre-trained model")
    parser.add_argument("--dataset_subset", required=True, help="Specify the dataset subset, either 'USA' or 'India'")
    parser.add_argument("--train_on_all", action="store_true", default=False, help="Whether the model should train on the full dataset (e.g. for deployment, not evaluation)")
    parser.add_argument("--data_path", required=True, help="The path to the location of the data folder. Example: 'C:/data'")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of epochs to train for")
    parser.add_argument("--learning_rate", type=float, default=0.0001, help="The learning rate")
    parser.add_argument("--batch_size", type=int, default=4, help="The batch size")
    parser.add_argument("--save_model_interval", type=int, default=0, help="Save the model every given number of epochs. Set to 0 if you don't want to save the model")
    parser.add_argument("--save_images_interval", type=int, default=0, help="Save some sample generator outputs every given number of epochs Set to 0 if you don't want to save images")
    parser.add_argument("--verbose", default=False, action="store_true", help="Print out the losses on every epoch")
    parser.add_argument("--pretrained_model_path", default=None, help="When evaluating (train==False), this path must point to a pre-trained model")
    parser.add_argument("--seed", type=int, default=47, help="The random seed to initialise the models")

    args = parser.parse_args()

    if not args.train:
        if not args.pretrained_model_path:
            raise ValueError("Provide a saved model.")
        if not os.path.isfile(args.pretrained_model_path):
            raise FileNotFoundError("Saved model not found. Check the path to the model.")
    
    model = segmentation_model.SegmentationModel(**vars(args))

    if args.train:
        model.train_model()
    else:
        model.plot_loss()
        model.plot_sample_images(10)
        model.calculate_metrics()
