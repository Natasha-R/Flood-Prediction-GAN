from models import segmentation_model

import os
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train (or evaluate) the flood segmentation model")
    parser.add_argument("--train", action="store_true", default=False, help="Train the model, else evaluate a pre-trained model")
    parser.add_argument("--dataset_subset", required=True, help="Specify the dataset subset, either 'USA' or 'India'")
    parser.add_argument("--train_on_all", action="store_true", default=False, help="Whether the model should train on the full dataset (e.g. for deployment, not evaluation)")
    parser.add_argument("--data_path", required=True, help="The path to the location of the data folder. Example: 'C:/data'")
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of epochs to train for")
    parser.add_argument("--save_model_interval", type=int, default=0, help="Save the model every given number of epochs. Set to 0 if you don't want to save the model")
    parser.add_argument("--save_images_interval", type=int, default=0, help="Save some sample generator outputs every given number of epochs Set to 0 if you don't want to save images")
    parser.add_argument("--verbose", default=False, action="store_true", help="Print out the losses on every epoch")
    parser.add_argument("--pretrained_model_path", default=None, help="When evaluating (train==False), this path must point to a pre-trained model")
    parser.add_argument("--plot_mask_image", default=None, help="Plot and save the mask generated from the image at the given path")
    parser.add_argument("--seed", type=int, default=47, help="The random seed to initialise the models")
    parser.add_argument("--use_test_data", action="store_true", default=False, help="Use the test dataset instead of the validation dataset.")

    args = parser.parse_args()

    if not args.train:
        if not args.pretrained_model_path:
            raise ValueError("Provide a saved model.")
        if not os.path.isfile(args.pretrained_model_path):
            raise FileNotFoundError("Saved model not found. Check the path to the model.")
    
    model = segmentation_model.SegmentationModel(**vars(args))

    if args.train:
        model.train_model()
    elif args.plot_mask_image:
        model.plot_mask_image(args.plot_mask_image)
    else:
        model.plot_loss()
        model.plot_sample_images(10, args.use_test_data)
        model.calculate_metrics(args.use_test_data)

