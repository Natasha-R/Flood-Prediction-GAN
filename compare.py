from models import group
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare trained Pix2Pix, CycleGAN, AttentionGAN and PairedAttention models on the flood images dataset")
    parser.add_argument("--dataset_subset", required=True, help="The dataset subset that the models were trained on")
    parser.add_argument("--dataset_dem", required=True, help="Specify whether the DEM used should be 'best' available or all the 'same'")
    parser.add_argument("--use_test_data", action="store_true", default=False, help="Use the test dataset instead of the validation dataset.")
    parser.add_argument("--data_path", required=True, help="The path to the location of the data folder. Example: 'C:/data'")
    parser.add_argument("--resize", type=int, default=None, help="Resize the images to the given size. The resize is applied before the crop")
    parser.add_argument("--crop", type=int, default=None, help="Crop each image into the given number of images. The resize is applied before the crop")
    parser.add_argument("--crop_index", type=int, default=0, help="When saving an image with the crop transformation, the crop_index indicates which quadrant to save")
    parser.add_argument("--topography", default=None, help="Which topographical factors should be input to the model. 'all', 'dem', 'map', 'flow', or 'river'")

    parser.add_argument("--segmentation_model_path", default=None, help="Path to a pre-trained flood segmentation model")

    parser.add_argument("--pix2pix_path", default=None, help="Path to the pretrained Pix2Pix model")
    parser.add_argument("--cyclegan_path", default=None, help="Path to the pretrained CycleGAN model")
    parser.add_argument("--attentiongan_path", default=None, help="Path to the pretrained AttentionGAN model")
    parser.add_argument("--pairedattention_path", default=None, help="Path to the pretrained PairedAttention model")

    parser.add_argument("--all_topography_path", default=None, help="Path to a model trained on all topography")
    parser.add_argument("--none_topography_path", default=None, help="Path to a model trained on no topography")
    parser.add_argument("--dem_topography_path", default=None, help="Path to a model trained on only DEM topography")
    parser.add_argument("--river_topography_path", default=None, help="Path to a model trained on on only river distance topography")
    parser.add_argument("--flow_topography_path", default=None, help="Path to a model trained on on only flow accumulation topography")
    parser.add_argument("--map_topography_path", default=None, help="Path to a model trained on on only map topography")

    parser.add_argument("--model_1_path", default=None, help="Path to pre-trained model 1")
    parser.add_argument("--model_2_path", default=None, help="Path to pre-trained model 2")

    parser.add_argument("--compare", required=True, help="Compare the performance of either 'models' 'topography' or 'two'")
    parser.add_argument("--image_names", default=None, nargs="+", help=("The names of the images to compare on the models." 
                                                                        "Optionally add '_index' to the end of image names to specify the crop index"))
    parser.add_argument("--calculate_metrics", action="store_true", default=False, help=("Calculate automated metrics to compare the models"))

    args = parser.parse_args()

    if args.compare == "models":
        if not (args.pix2pix_path and args.cyclegan_path and args.attentiongan_path and args.pairedattention_path):
            raise ValueError("Paths to Pix2Pix, CycleGAN, AttentionGAN and PairedAttention models must be provided.")
        paths = {"PairedAttention":args.pairedattention_path,
                 "Pix2Pix":args.pix2pix_path,
                 "AttentionGAN":args.attentiongan_path,
                 "CycleGAN":args.cyclegan_path}
        
    elif args.compare == "topography":
        if not (args.all_topography_path and args.none_topography_path and args.dem_topography_path and args.river_topography_path and args.flow_topography_path and args.map_topography_path):
            raise ValueError("Paths to all, none, DEM, river distance, flow accumulation and map topography models must be provided.")
        paths = {"All": args.all_topography_path,
                 "DEM": args.dem_topography_path,
                 "Flow accumulation": args.flow_topography_path,
                 "Distance to rivers": args.river_topography_path,
                 "Map": args.map_topography_path,
                 "None": args.none_topography_path,}
                 
    elif args.compare == "two":
        paths = {"Model 1": args.model_1_path,
                 "Model 2": args.model_2_path}

    else:
        raise NotImplementedError("Comparisons must be made between 'models' 'topography' or 'two'")
    
    all_models = group.ModelsGroup(paths=paths,
                                   compare=args.compare,
                                   dataset_subset=args.dataset_subset,
                                   dataset_dem=args.dataset_dem,
                                   data_path=args.data_path,
                                   resize=args.resize,
                                   crop=args.crop,
                                   crop_index=args.crop_index,
                                   topography=args.topography)
    
    if args.calculate_metrics:
        if not args.segmentation_model_path:
            raise ValueError("To calculate metrics, a pre-trained flood segmentation model must be provided.")
        all_models.compare_metrics(args.use_test_data, args.segmentation_model_path)

    if args.image_names:
        all_models.compare_output_images(args.image_names)

    
