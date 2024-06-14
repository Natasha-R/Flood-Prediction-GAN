import model

import os
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained Pix2Pix, CycleGAN, AttentionGAN or PairedAttention model on the flood images dataset.")
    parser.add_argument("--model", required=True, help="Model can be one of: Pix2Pix, CycleGAN, AttentionGAN or PairedAttention")
    parser.add_argument("--dataset_subset", default="all", help="Specify the dataset subset, e.g. USA, India, Hurricane-Harvey")
    parser.add_argument("--dataset_dem", required=True, help="Specify whether the DEM used should be 'best' available or all the 'same'")
    parser.add_argument("--data_path", required=True, help="The path to the location of the data folder. Example: 'C:/data'")
    parser.add_argument("--resize", type=int, default=256, help="Resize the images to the given size. The resize is applied before the crop")
    parser.add_argument("--crop", type=int, default=None, help="Crop each image into the given number of images. The resize is applied before the crop")
    parser.add_argument("--crop_index", type=int, default=0, help="When saving an image with the crop transformation, the crop_index indicates which quadrant to save")
    parser.add_argument("--pretrained_model_path", required=True, help="Path to a pretrained model")
    parser.add_argument("--plot_losses", action="store_true", default=False, help="")
    parser.add_argument("--plot_sample_images", action="store_true", default=False, help="Plot 'num_images' generated images from the training and validation dataset")
    parser.add_argument("--num_images", type=int, default=5, help="When plotting sample images, num_images indicates how many generated images to plot")
    parser.add_argument("--seed", type=int, default=47, help="The random seed to generate sample images")
    parser.add_argument("--image_name", default=None, help="The name of the image to plot")
    parser.add_argument("--plot_single_image", default=None, help="Plot a single image of the given type, must be one of 'input' 'ground truth' 'output' or 'attention mask'")
    parser.add_argument("--plot_image_set", action="store_true", default=False, help="Plot a set of input, ground truth, output and attention mask (if appropriate)")
    args = parser.parse_args()
    args.model = args.model.lower()

    if not os.path.isfile(args.pretrained_model_path):
        raise FileNotFoundError("Saved model not found. Check the path to the model.")

    evaluate_model = model.Model(model=args.model,
                                 dataset_subset=args.dataset_subset,
                                 dataset_dem=args.dataset_dem,
                                 data_path=args.data_path,
                                 resize=args.resize,
                                 crop=args.crop,
                                 load_pretrained_model=True,
                                 pretrained_model_path=args.pretrained_model_path,
                                 training_model=False,
                                 seed=args.seed)

    if args.plot_losses:
        evaluate_model.plot_losses()

    if args.plot_sample_images:
        evaluate_model.plot_sample_images(args.num_images)
    
    if args.plot_single_image or args.plot_image_set:
        if not args.image_name:
            raise FileNotFoundError("Please specify an image to plot")
        evaluate_model.plot_image(image_name=args.image_name,
                                  plot_single_image=args.plot_single_image,
                                  plot_image_set=args.plot_image_set,
                                  crop_index=args.crop_index)