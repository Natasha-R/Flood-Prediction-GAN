from models import group
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare trained Pix2Pix, CycleGAN, AttentionGAN and PairedAttention models on the flood images dataset")
    parser.add_argument("--dataset_subset", required=True, help="The dataset subset that the models were trained on")
    parser.add_argument("--dataset_dem", required=True, help="Specify whether the DEM used should be 'best' available or all the 'same'")
    parser.add_argument("--data_path", required=True, help="The path to the location of the data folder. Example: 'C:/data'")
    parser.add_argument("--pix2pix_path", required=True, help="Path to the pretrained Pix2Pix model")
    parser.add_argument("--cyclegan_path", required=True, help="Path to the pretrained CycleGAN model")
    parser.add_argument("--attentiongan_path", required=True, help="Path to the pretrained AttentionGAN model")
    parser.add_argument("--pairedattention_path", required=True, help="Path to the pretrained PairedAttention model")
    parser.add_argument("--image_names", default=None, nargs="+", help="The names of the images to plot.")
    parser.add_argument("--resize", type=int, default=None, help="Resize the images to the given size. The resize is applied before the crop")
    parser.add_argument("--crop", type=int, default=None, help="Crop each image into the given number of images. The resize is applied before the crop")
    parser.add_argument("--crop_index", type=int, default=0, help="When saving an image with the crop transformation, the crop_index indicates which quadrant to save")
    parser.add_argument("--not_input_topography", default=False, action="store_true", help="The additional topographical factors will NOT be input to the model")
    
    args = parser.parse_args()

    all_models = group.ModelsGroup(pix2pix_path=args.pix2pix_path,
                               cyclegan_path=args.cyclegan_path,
                               attentiongan_path=args.attentiongan_path,
                               pairedattention_path=args.pairedattention_path,
                               dataset_subset=args.dataset_subset,
                               dataset_dem=args.dataset_dem,
                               data_path=args.data_path,
                               resize=args.resize,
                               crop=args.crop,
                               crop_index=args.crop_index,
                               not_input_topography=args.not_input_topography)
    
    if args.image_names:
        all_models.compare_output_images(args.image_names)
