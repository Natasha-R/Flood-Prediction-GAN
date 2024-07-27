from models import data
from models import utils
from models import segmentation_model
from models import model_architectures

import time
import random
import itertools
import numpy as np
import pandas as pd
from tqdm import tqdm
import tifffile as tf
from datetime import datetime
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.optim import lr_scheduler
from torchmetrics.regression import MeanSquaredError
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.classification import BinaryAccuracy, BinaryF1Score, BinaryPrecision, BinaryRecall
from torchmetrics.image import PeakSignalNoiseRatio, MultiScaleStructuralSimilarityIndexMeasure, StructuralSimilarityIndexMeasure

device = "cuda" if torch.cuda.is_available() else "cpu"

class Model():
    """
    A class encapsulating the attributes and functions for training and evaluating a model.
    """
    def __init__(self, 
                model="Pix2Pix",
                dataset_subset="all", 
                dataset_dem="best", 
                data_path=None,
                num_epochs=1, 
                topography="all",
                resize=256,
                crop=None,
                save_model_interval=0,
                save_images_interval=0,
                verbose=False,
                load_pretrained_model=False, 
                pretrained_model_path=None,
                add_identity_loss=False,
                training_model=True,
                seed=47):

        if verbose:
            print(f"\nSetting up the {self.prettify_model_name(model)} model...") 

        # load information about the model
        if load_pretrained_model:
            saved_model = torch.load(pretrained_model_path)
            self.model = saved_model["model"]
            self.num_epochs = saved_model["num_epochs"]
            self.topography = saved_model["topography"]
            self.add_identity_loss = saved_model["add_identity_loss"]
        else:
            self.model = model
            self.num_epochs = num_epochs
            self.topography = topography
            self.add_identity_loss = add_identity_loss
        self.verbose = verbose
        self.save_model_interval = save_model_interval
        self.save_images_interval = save_images_interval
        self.load_pretrained_model = load_pretrained_model
        self.data_path = data_path
        self.dataset_subset = dataset_subset
        self.dataset_dem = dataset_dem
        self.resize = resize
        self.crop = crop
        self.training_model = training_model
        self.seed = seed
        self.model_is_cycle = self.model_is_cycle()
        self.model_is_attention = self.model_is_attention()

        # define the model architectures
        topography_channels = {"all": 9, "map": 6, "dem": 4, "flow": 4, "river": 4, None: 3}
        input_channels = topography_channels[self.topography]
        torch.manual_seed(self.seed)
        if self.model=="pix2pix":
            generator_architecture = model_architectures.Pix2PixGenerator
            discriminator_architecture = model_architectures.Pix2PixDiscriminator
        elif self.model=="pairedattention":
            generator_architecture = model_architectures.PairedAttentionGenerator
            discriminator_architecture = model_architectures.PairedAttentionDiscriminator
        elif self.model=="attentiongan":
            generator_architecture = model_architectures.AttentionGANGenerator
            discriminator_architecture = model_architectures.AttentionGANDiscriminator
        elif self.model=="cyclegan":
            generator_architecture = model_architectures.CycleGANGenerator
            discriminator_architecture = model_architectures.CycleGANDiscriminator
        else:
            raise NotImplementedError("Model must be one of: Pix2Pix, CycleGAN, AttentionGAN or PairedAttention") 
        if self.model_is_cycle:
            self.pre_to_post_generator = generator_architecture(input_channels=input_channels).apply(self.initialise_weights).to(device)
            self.post_to_pre_generator = generator_architecture(input_channels=input_channels).apply(self.initialise_weights).to(device)
            if self.training_model:
                self.pre_discriminator = discriminator_architecture(input_channels=input_channels).apply(self.initialise_weights).to(device)
                self.post_discriminator = discriminator_architecture(input_channels=input_channels).apply(self.initialise_weights).to(device)
        else:
            self.generator = generator_architecture(input_channels=input_channels).apply(self.initialise_weights).to(device)
            if self.training_model:
                self.discriminator = discriminator_architecture(input_channels=input_channels).apply(self.initialise_weights).to(device)

        # if training, define the loss functions, optimisers and schedulers
        if self.training_model:
            if self.model_is_cycle:
                self.loss_func = nn.MSELoss()
                self.cycle_loss = nn.L1Loss()
                self.identity_loss = nn.L1Loss()
                self.optimizer_generator = torch.optim.Adam(itertools.chain(self.pre_to_post_generator.parameters(), 
                                                                            self.post_to_pre_generator.parameters()), 
                                                            lr=0.0002, betas=(0.5, 0.999))
                self.optimizer_discriminator = torch.optim.Adam(itertools.chain(self.post_discriminator.parameters(), 
                                                                                self.pre_discriminator.parameters()), 
                                                                lr=0.0002, betas=(0.5, 0.999))
            else:     
                self.loss_func = nn.MSELoss()
                self.l1_loss = nn.L1Loss()
                self.optimizer_discriminator = torch.optim.Adam(self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
                self.optimizer_generator = torch.optim.Adam(self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
            self.scheduler_generator = lr_scheduler.LambdaLR(self.optimizer_generator, lr_lambda=self.lambda_rule) 
            self.scheduler_discriminator = lr_scheduler.LambdaLR(self.optimizer_discriminator, lr_lambda=self.lambda_rule)

        # load pre-saved state dicts for models, optimisers and schedulers
        if self.load_pretrained_model:
            self.starting_epoch = saved_model["starting_epoch"]
            self.all_losses = saved_model["all_losses"]
            if self.training_model:
                self.optimizer_discriminator.load_state_dict(saved_model["optimizer_discriminator"])
                self.optimizer_generator.load_state_dict(saved_model["optimizer_generator"])
                self.scheduler_discriminator.load_state_dict(saved_model["scheduler_discriminator"])
                self.scheduler_generator.load_state_dict(saved_model["scheduler_generator"])
            if self.model_is_cycle:
                self.pre_to_post_generator.load_state_dict(saved_model["pre_to_post_generator"])
                self.post_to_pre_generator.load_state_dict(saved_model["post_to_pre_generator"])
                if self.training_model:
                    self.pre_discriminator.load_state_dict(saved_model["pre_discriminator"])
                    self.post_discriminator.load_state_dict(saved_model["post_discriminator"])
            else:
                self.generator.load_state_dict(saved_model["generator"])
                if self.training_model:
                    self.discriminator.load_state_dict(saved_model["discriminator"])
        else:
            self.starting_epoch = 1
            self.all_losses = self.initialise_loss_storage(overall=True)
        self.current_epoch = self.starting_epoch
        
        # load the data
        self.train_loader, self.val_loader, self.test_loader = data.create_flood_dataset(self.dataset_subset, 
                                                                                   self.dataset_dem, 
                                                                                   self.data_path, 
                                                                                   self.topography, 
                                                                                   self.resize, 
                                                                                   self.crop)
        
        # print training set-up
        if self.verbose and self.training_model:
            self.print_training_setup()
            
    def initialise_weights(self, m):
        """
        Initialise the weights of a neural network.
        """
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0.0)

    def lambda_rule(self, epoch):
        """
        The learning rate stays the same for the first half of epochs,
        then decreases linearly for the second half of epochs.
        """
        learning_rate = 1.0 - max(0, epoch + 1 - (self.num_epochs / 2)) / float((self.num_epochs/2) + 1)
        return learning_rate
    
    def initialise_loss_storage(self, overall):
        """
        Initialises a dictionary to store the losses during training.
        """
        pre_string = "all_" if overall else ""
        if self.model_is_cycle:
            loss_dict = {f"{pre_string}losses_generator_post":[],
                    f"{pre_string}losses_generator_pre":[],
                    f"{pre_string}losses_pre_to_post_cycle":[],
                    f"{pre_string}losses_post_to_pre_cycle":[],
                    f"{pre_string}losses_discriminator_pre_real":[],
                    f"{pre_string}losses_discriminator_post_real":[],
                    f"{pre_string}losses_discriminator_pre_synthetic":[],
                    f"{pre_string}losses_discriminator_post_synthetic":[]}
            if self.add_identity_loss:
                loss_dict[f"{pre_string}losses_identity_post"]=[]
                loss_dict[f"{pre_string}losses_identity_pre"]=[]
            return loss_dict
        else:
            return {f"{pre_string}losses_discriminator_real":[],
                    f"{pre_string}losses_discriminator_synthetic":[],
                    f"{pre_string}losses_generator_synthetic":[],
                    f"{pre_string}l1_losses_generator_synthetic":[]}
    
    def model_is_cycle(self):
        """
        Returns whether the model uses the 'cycle' training strategy.
        """
        model_to_category = {"pix2pix":False,
                            "pairedattention":False,
                            "cyclegan":True,
                            "attentiongan":True}
        if self.model not in model_to_category.keys():
            raise NotImplementedError("Model must be one of: Pix2Pix, CycleGAN, AttentionGAN or PairedAttention")
        return model_to_category[self.model]

    def model_is_attention(self):
        """
        Returns whether the model uses an architecture with 'attention'.
        """
        model_to_category = {"pix2pix":False,
                            "pairedattention":True,
                            "cyclegan":False,
                            "attentiongan":True}
        if self.model not in model_to_category.keys():
            raise NotImplementedError("Model must be one of: Pix2Pix, CycleGAN, AttentionGAN or PairedAttention")
        return model_to_category[self.model]
    
    def prettify_model_name(self, model_name=None):
        """
        Prettifies the model name capitalisation, for pretty printing.
        """
        model_to_pretty = {"pix2pix":"Pix2Pix",
                           "cyclegan":"CycleGAN",
                           "attentiongan":"AttentionGAN",
                           "pairedattention":"PairedAttention"}
        return model_to_pretty[model_name.lower()] if model_name else model_to_pretty[self.model]
    
    def create_path(self, save_type, info=""):
        """
        Defines an informative path string to save images or models to, 
        containing information about the model and dataset.
        """
        file_types = {"image":".png", "figure":".png", "model":".pth.tar", "metric":".csv"}
        file_type = file_types[save_type]
        model_name = self.prettify_model_name()
        current_time = str(datetime.now())[:-7].replace(' ', '-').replace(':', '-')
        add_identity_loss = f"identity{self.add_identity_loss}" if self.model_is_cycle else ""
        path = (f"{self.data_path}/{save_type}s/"
        f"{model_name}_{info}_epoch{self.current_epoch if self.training_model else self.current_epoch-1}_"
        f"{self.topography}Topography_{add_identity_loss}_"
        f"{self.dataset_subset}Data_{self.dataset_dem}DEM_"
        f"resize{self.resize}_crop{self.crop}_"
        f"date{current_time}{file_type}")
        path = path.replace("__", "_")
        return path

    def print_training_setup(self):
        """
        Prints information about the training set-up.
        """
        print(f"\n{'Continuing' if self.load_pretrained_model is True else 'Beginning'} training {self.prettify_model_name()}:")
        print(f"{self.num_epochs} epochs")
        print(f"Starting from epoch {self.starting_epoch}")
        print(f"{self.topography.title() if self.topography else 'No' } topographical factors will be input to the model")
        if self.model_is_cycle and self.add_identity_loss:
            print(f"Using identity mapping loss")
        print(f"Dataset: {len(self.train_loader)} images from '{self.dataset_subset}' with '{self.dataset_dem}' DEM")
        print(f"Data resized to {self.resize} pixels with {self.crop} crops, scaled to [-1, 1]")
        print(f"Model saved every {self.save_model_interval} epochs")
        print(f"Sample generator output images saved every {self.save_images_interval} epochs\n")

    def get_buffer_image(self, image, images_buffer):
        """
        Return an image from the buffer.
        There is a 50% chance the new image is stored and a random old image is returned,
        and a 50% chance the new image is not stored and the new image is returned.
        If the buffer is not yet full, the new image is always stored and returned.
        """
        image = image.detach()
        if len(images_buffer) < 50: # if the buffer is not yet full, always store the new image AND return it
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

    def print_losses(self):
        """
        Prints the model losses from the previous epoch.
        """
        if self.model_is_cycle:
            print((f"| "
                    f"Generator post image loss = {self.all_losses['all_losses_generator_post'][-1]:.2f} | "
                    f"Generator pre image loss = {self.all_losses['all_losses_generator_pre'][-1]:.2f} | "
                    f"Pre to post cycle loss = {self.all_losses['all_losses_pre_to_post_cycle'][-1]:.2f} | "
                    f"Post to pre cycle loss = {self.all_losses['all_losses_post_to_pre_cycle'][-1]:.2f} | "
                    f"Discriminator pre real image loss = {self.all_losses['all_losses_discriminator_pre_real'][-1]:.2f} | "
                    f"Discriminator post real image loss = {self.all_losses['all_losses_discriminator_post_real'][-1]:.2f} | "
                    f"Discriminator pre synthetic image loss = {self.all_losses['all_losses_discriminator_pre_synthetic'][-1]:.2f} | "
                    f"Discriminator post synthetic image loss = {self.all_losses['all_losses_discriminator_post_synthetic'][-1]:.2f}"), 
                    end="" if self.add_identity_loss else "\n")
            if self.add_identity_loss:
                print((f" | Identity pre image loss = {self.all_losses['all_losses_identity_pre'][-1]:.2f} | "
                    f"Identity post image loss = {self.all_losses['all_losses_identity_post'][-1]:.2f}"))
        else:
            print((f"| "
                f"Discriminator real loss = {self.all_losses['all_losses_discriminator_real'][-1]:.2f} | "
                f"Discriminator synthetic loss = {self.all_losses['all_losses_discriminator_synthetic'][-1]:.2f} | "
                f"Generator synthetic loss = {self.all_losses['all_losses_generator_synthetic'][-1]:.2f} | "
                f"L1 generator loss = {self.all_losses['all_l1_losses_generator_synthetic'][-1]:.2f}"))

    def save_results(self, epoch, losses, epoch_start_time):
        """
        Save the current losses, model, and sample outputs, at given epochs.
        """
        self.current_epoch = epoch

        for key in self.all_losses.keys():
            self.all_losses[key].append(np.mean(losses[key[4:]]))

        if self.verbose:
            epoch_end_time = time.time()
            print(f"Epoch {epoch} ({epoch_end_time - epoch_start_time :.2f} seconds) ", end="")
            self.print_losses()

        if self.save_model_interval != 0 and ((epoch % self.save_model_interval) == 0):
            saved_model = {
                    "model": self.model,
                    "starting_epoch": epoch + 1,
                    "num_epochs": self.num_epochs,
                    "topography": self.topography,
                    "optimizer_generator": self.optimizer_generator.state_dict(),
                    "optimizer_discriminator": self.optimizer_discriminator.state_dict(),
                    "scheduler_generator": self.scheduler_generator.state_dict(),
                    "scheduler_discriminator": self.scheduler_discriminator.state_dict(),
                    "all_losses": self.all_losses,
                    "add_identity_loss": self.add_identity_loss}
            if self.model_is_cycle:
                saved_model["pre_to_post_generator"] = self.pre_to_post_generator.state_dict()
                saved_model["post_to_pre_generator"] = self.post_to_pre_generator.state_dict()
                saved_model["pre_discriminator"] = self.pre_discriminator.state_dict()
                saved_model["post_discriminator"] = self.post_discriminator.state_dict()
            else:
                saved_model["discriminator"] = self.discriminator.state_dict()
                saved_model["generator"] = self.generator.state_dict()
                    
            model_path = self.create_path(save_type="model")
            print(f"Saving {self.prettify_model_name()} model to {model_path}")
            torch.save(saved_model, model_path)

        if self.save_images_interval != 0 and ((epoch % self.save_images_interval) == 0):
            self.plot_sample_images(num_images=5, use_test_data=False)

    def calculate_metrics(self, use_test_data=False, seg_model_path=None):
        """
        Calculate metrics for the model.
        """
        metrics = {"PSNR": PeakSignalNoiseRatio(data_range=(0, 1)).to(device),
                    "SSIM": StructuralSimilarityIndexMeasure(data_range=(0, 1)).to(device),
                    "MS-SSIM": MultiScaleStructuralSimilarityIndexMeasure(data_range=(0, 1)).to(device),
                    "LPIPS": LearnedPerceptualImagePatchSimilarity().to(device),
                    "MSE":MeanSquaredError().to(device),
                    "Accuracy":BinaryAccuracy().to(device),
                    "F1_Flood":BinaryF1Score().to(device),
                    "Precision_Flood":BinaryPrecision().to(device),
                    "Recall_Flood":BinaryRecall().to(device),
                    "F1_No_Flood":BinaryF1Score().to(device),
                    "Precision_No_Flood":BinaryPrecision().to(device),
                    "Recall_No_Flood":BinaryRecall().to(device)}
        metrics_results = {metric: list() for metric in list(metrics.keys()) + ["Inference"]}
        seg_model = segmentation_model.SegmentationModel(data_path=self.data_path,
                                                         pretrained_model_path=seg_model_path,
                                                         train=False).model

        print("\nCalculating metrics...")
        loader = self.test_loader if use_test_data else self.val_loader
        all_true_flood_mask = None
        all_output_flood_mask = None
        for input_stack, ground_truth, _ in tqdm(loader, desc="Images", leave=False):

            input_stack = utils.extract_input_topography(input_stack, self.topography).detach().clone().to(device)
            ground_truth = ground_truth.to(device)
            start_time = time.time()
            torch.manual_seed(47)
            generator = self.pre_to_post_generator if self.model_is_cycle else self.generator
            generator_output = generator(input_stack)
            inference_time = time.time()-start_time
            ground_truth = torch.clamp((ground_truth + 1) * 0.5, min=0, max=1)
            generator_output = torch.clamp((generator_output + 1) * 0.5, min=0, max=1)
            output_mask = (torch.sigmoid(seg_model(generator_output.detach().clone())) > 0.5).float()
            true_mask = (torch.sigmoid(seg_model(ground_truth.detach().clone())) > 0.5).float()
            flat_true_mask = true_mask.detach().clone().squeeze().flatten().squeeze()
            flat_output_mask = output_mask.detach().clone().squeeze().flatten().squeeze()

            for metrics_name in ["PSNR", "SSIM", "MS-SSIM", "LPIPS"]:
                metrics_results[metrics_name].append(metrics[metrics_name](generator_output.detach().clone(), ground_truth.detach().clone()).item())
                metrics[metrics_name].reset()
            metrics_results["Inference"].append(inference_time)

            all_true_flood_mask = torch.cat((all_true_flood_mask, flat_true_mask), dim=0).to(device) if not all_true_flood_mask==None else flat_true_mask
            all_output_flood_mask = torch.cat((all_output_flood_mask, flat_output_mask), dim=0).to(device) if not all_output_flood_mask==None else flat_output_mask

        for metrics_name in ["MSE", "Accuracy", "F1_Flood", "Precision_Flood", "Recall_Flood"]:
            metrics_results[metrics_name].append(metrics[metrics_name](all_output_flood_mask, all_true_flood_mask).item())

        all_true_no_flood_mask = torch.abs(all_true_flood_mask-1)
        all_output_no_flood_mask = torch.abs(all_output_flood_mask-1)
        for metrics_name in ["F1_No_Flood", "Precision_No_Flood", "Recall_No_Flood"]:
            metrics_results[metrics_name].append(metrics[metrics_name](all_output_no_flood_mask, all_true_no_flood_mask).item())

        metrics_df = pd.DataFrame([(metrics_name, np.mean(metrics_results[metrics_name])) for metrics_name in metrics_results]).set_index(0).transpose()
        print(metrics_df)
        metrics_df.to_csv(self.create_path("metric"))

    def plot_losses(self):
        """
        Plot the training losses over the epochs.
        """
        if self.model_is_cycle:
            plot_parameters = {"all_losses_generator_post": {"colour":"#7BA4A9", "label":"Generator (post)", "linestyle":(0, (3, 1)), "plot":0},
                            "all_losses_generator_pre": {"colour":"#7BA4A9", "label":"Generator (pre)", "linestyle":"solid", "plot":0},
                            "all_losses_pre_to_post_cycle": {"colour":"#7BA4A9", "label":"Pre to post cycle loss", "linestyle":"solid", "plot":1},
                            "all_losses_post_to_pre_cycle": {"colour":"#9F799B", "label":"Post to pre cycle loss", "linestyle":"solid", "plot":1},
                            "all_losses_discriminator_pre_real": {"colour":"#5F2959", "label":"Discriminator (pre, real)", "linestyle":"solid", "plot":0},
                            "all_losses_discriminator_post_real": {"colour":"#5F2959", "label":"Discriminator (post, real)", "linestyle":(0, (3, 1)), "plot":0},
                            "all_losses_discriminator_pre_synthetic": {"colour":"#9F799B", "label":"Discriminator (pre, synthetic)", "linestyle":"solid", "plot":0},
                            "all_losses_discriminator_post_synthetic": {"colour":"#9F799B", "label":"Discriminator (post, synthetic)", "linestyle":(0, (3, 1)), "plot":0},}
            if self.add_identity_loss:
                plot_parameters["all_losses_identity_post"] = {"colour":"black", "label":"Identity (post)", "linestyle":(0, (3, 1)), "plot":2}
                plot_parameters["all_losses_identity_pre"] = {"colour":"black", "label":"Identity (pre)", "linestyle":"solid", "plot":2}
        else:
            plot_parameters = {"all_losses_discriminator_real": {"colour":"#5F2959", "label":"Discriminator (real)", "linestyle":"solid", "plot":0},
                            "all_losses_discriminator_synthetic": {"colour":"#9F799B", "label":"Discriminator (synthetic)", "linestyle":"solid", "plot":0},
                            "all_losses_generator_synthetic": {"colour": "#7BA4A9", "label":"Generator (synthetic)", "linestyle":"solid", "plot":0},
                            "all_l1_losses_generator_synthetic": {"colour":"black", "label":"L1 loss", "linestyle":"solid", "plot":1}}
            
        num_plots = 3 if self.add_identity_loss else 2
        fig, axes = plt.subplots(nrows=num_plots, ncols=1, figsize=(10, num_plots * 7))
        for ax in axes.ravel():
            ax.tick_params(axis="both", which="major", labelsize=14)
            ax.set_xlabel("Epoch", fontsize=14)
            ax.set_ylabel("Loss", fontsize=14)
            ax.grid(alpha=0.4)
        for loss in self.all_losses.keys():
            parameters = plot_parameters[loss]
            axes[parameters["plot"]].plot(range(1, self.starting_epoch),
                                          self.all_losses[loss],
                                          c=parameters["colour"], 
                                          linestyle=parameters["linestyle"],
                                          label=parameters["label"],
                                          linewidth=2)
        axes[0].set_title(f"{self.prettify_model_name()} Discriminator and Generator Losses", fontsize=15)
        axes[1].set_title(f"{self.prettify_model_name()} {'Cycle Losses' if self.model_is_cycle else 'L1 Losses'}", fontsize=15)
        axes[0].legend(fontsize=14)
        if self.model_is_cycle:
            axes[1].legend(fontsize=14)
        if self.add_identity_loss:
            axes[2].set_title(f"{self.prettify_model_name()} Identity Losses", fontsize=15)
            axes[2].legend(fontsize=14)

        fig.tight_layout();
        losses_path = self.create_path(save_type="figure", info="losses")
        print(f"\nSaving losses figure to {losses_path}")
        fig.savefig(losses_path, bbox_inches="tight")
    
    def plot_image(self, image_name, plot_single_image, plot_image_set, crop_index=0):
        """
        Plot the image specified by 'image_name'. 
        Can plot the associated 'input', 'ground truth', generator 'output', or 'attention mask',
        and/or plot a set of images containing all of them.
        """
        # import the input image
        dataset_split = pd.read_csv("metadata/dataset_split.csv")
        dem_string = dataset_split[dataset_split["image"]==image_name][f"{self.dataset_dem}_DEM"].head(1).item()
        input_path = f"{self.data_path}/dataset_input/{image_name}_{dem_string}.tif"
        input_image = torch.from_numpy(tf.imread(input_path).transpose(2, 0, 1))
        ground_truth = torch.from_numpy(tf.imread(f"{self.data_path}/dataset_output/{image_name}.tif").transpose(2, 0, 1))

        # apply transformations and generate the output image
        input_image, ground_truth, image_name = utils.apply_transformations(image_name=image_name,
                                                                            input_image=input_image, 
                                                                            output_image=ground_truth, 
                                                                            topography=self.topography, 
                                                                            resize=self.resize, 
                                                                            crop=self.crop, 
                                                                            crop_index=crop_index)
        generator = self.pre_to_post_generator if self.model_is_cycle else self.generator
        torch.manual_seed(47)
        generator_output = utils.tensor_to_numpy(generator(input_image))

        # plot the image
        if plot_single_image:
            if plot_single_image=="input":
                input_image_path = f"{self.data_path}/images/{image_name}_input.png"
                print(f"\nSaving input image of image '{image_name}' to {input_image_path}")
                plt.imsave(input_image_path, utils.tensor_to_numpy(input_image), vmin=0, vmax=1)
            elif plot_single_image=="ground truth":
                ground_truth_path = f"{self.data_path}/images/{image_name}_groundTruth.png"
                print(f"\nSaving ground truth of image '{image_name}' to {ground_truth_path}")
                plt.imsave(ground_truth_path, utils.tensor_to_numpy(ground_truth), vmin=0, vmax=1)
            elif plot_single_image=="output":
                generator_output_path = self.create_path(save_type="image", info=image_name)
                print(f"\nSaving generator output of image '{image_name}' to {generator_output_path}")
                plt.imsave(generator_output_path, generator_output, vmin=0, vmax=1)
            elif plot_single_image=="attention mask" and self.model_is_attention:
                attention_mask = utils.tensor_to_numpy(generator.last_attention_mask)
                attention_mask_path = self.create_path(save_type="image", info=f"{image_name}_attentionMask")
                print(f"\nSaving attention mask of image '{image_name}' to {attention_mask_path}")
                plt.imsave(attention_mask_path, attention_mask, vmin=0, vmax=1, cmap="gray_r")
            else:
                raise NotImplementedError("Type of image must be one of 'input', 'ground truth', 'output', or 'attention mask'")
            
        if plot_image_set:
            num_cols = 4 if self.model_is_attention else 3
            fig, axes = plt.subplots(nrows=1, ncols=num_cols, figsize=(num_cols * 5, 5))
            for ax in axes.ravel():
                ax.set_axis_off()
            axes[0].imshow(utils.tensor_to_numpy(input_image), vmin=0, vmax=1)
            axes[1].imshow(generator_output, vmin=0, vmax=1)
            axes[num_cols-1].imshow(utils.tensor_to_numpy(ground_truth), vmin=0, vmax=1)
            axes[0].set_title(f"Input ({image_name})")
            axes[1].set_title("Generator Output")
            axes[num_cols-1].set_title("Ground Truth Output")
            if self.model_is_attention:
                axes[2].imshow(utils.tensor_to_numpy(generator.last_attention_mask), cmap="gray_r", vmin=0, vmax=1)
                axes[2].set_title("Attention Mask")
            fig.tight_layout()
            images_path = self.create_path(save_type="image", info=image_name)
            print(f"Saving {image_name} image set to {images_path}")
            fig.savefig(images_path, bbox_inches="tight")
            plt.close()

    def plot_sample_images(self, num_images, use_test_data):
        """
        Plot 'num_images' random sample generator output images from both the training and validation datasets.
        The randomness of the images can be controlled via the seed.
        """
        if self.model_is_cycle:
            generators = [("pre-to-post", self.pre_to_post_generator),
                          ("post-to-pre", self.post_to_pre_generator)]
        else:
            generators = [("pre-to-post", self.generator)]
        splits = ["training", "validation"]
        loaders = [self.train_loader, self.val_loader]
        if use_test_data:
            splits += ["test"]
            loaders += [self.test_loader]
            
        for generator_label, generator in generators:
            for split, dataloader in zip(splits, loaders):
                num_cols = 4 if self.model_is_attention else 3
                fig, axes = plt.subplots(nrows=num_images, ncols=num_cols, figsize=(num_cols * 5, num_images * 5))
                for ax in axes.ravel():
                    ax.set_axis_off()
                torch.manual_seed(self.seed)
                for i, (input_stack, output_image, image_name) in enumerate(dataloader):
                    if generator_label=="post-to-pre": # flip the input and output
                        store_output = output_image.clone()
                        if self.topography:
                            condition = input_stack[:, 3:, :, :].detach().clone()
                            output_image = input_stack[:, :3, :, :].clone().to(device)
                            input_stack = torch.cat((store_output, condition), dim=1).to(device)
                        else:
                            output_image = input_stack.clone().to(device)
                            input_stack = store_output.to(device)
                    else:
                        input_stack = input_stack.to(device)
                        output_image = output_image.to(device)
                    axes[i, 0].imshow(utils.tensor_to_numpy(input_stack), vmin=0, vmax=1)
                    torch.manual_seed(47)
                    axes[i, 1].imshow(utils.tensor_to_numpy(generator(input_stack)), vmin=0, vmax=1)
                    axes[i, num_cols-1].imshow(utils.tensor_to_numpy(output_image), vmin=0, vmax=1)
                    axes[i, 0].set_title(f"Input ({image_name[0]})")
                    axes[i, 1].set_title("Generator Output")
                    axes[i, num_cols-1].set_title("Ground Truth Output")
                    if self.model_is_attention:
                        axes[i, 2].imshow(utils.tensor_to_numpy(generator.last_attention_mask), cmap="gray_r")
                        axes[i, 2].set_title("Attention Mask")
                    if i >= num_images-1:
                        break
                fig.tight_layout()
                
                images_path = self.create_path(save_type="image", info=f"{split}{'_' + generator_label if len(generators)>1 else ''}")
                print(f"Saving {split} {generator_label + ' ' if len(generators)>1 else ''}sample images to {images_path}")
                fig.savefig(images_path, bbox_inches="tight")

                plt.close();

    def train_paired(self):
        """
        Train a model using the paired images approach.
        """
        for epoch in range(self.starting_epoch, self.num_epochs + 1):
            epoch_start_time = time.time()
            losses = self.initialise_loss_storage(overall=False)

            self.discriminator.train()
            self.generator.train()

            torch.manual_seed(epoch)

            for input_stack, output_image, _ in tqdm(self.train_loader, desc="Iterations", leave=False, disable=not self.verbose):

                input_stack = input_stack.to(device)
                output_image = output_image.to(device)
                synthetic_output = self.generator(input_stack)
                concat_real = torch.cat((input_stack, output_image), 1)
                concat_synthetic = torch.cat((input_stack, synthetic_output), 1)

                # train the discriminator
                for parameter in self.discriminator.parameters(): 
                    parameter.requires_grad = True
                self.optimizer_discriminator.zero_grad()
                
                prediction_discriminator_synthetic = self.discriminator(concat_synthetic.detach()) 
                discriminator_prediction_shape = prediction_discriminator_synthetic.shape
                loss_discriminator_synthetic = self.loss_func(prediction_discriminator_synthetic,
                                                        torch.full(discriminator_prediction_shape, 0., dtype=torch.float32, device=device))
                prediction_discriminator_real = self.discriminator(concat_real)
                loss_discriminator_real = self.loss_func(prediction_discriminator_real,
                                                torch.full(discriminator_prediction_shape, 1., dtype=torch.float32, device=device))
                loss_discriminator = (loss_discriminator_synthetic + loss_discriminator_real) * 0.5
                loss_discriminator.backward()
                self.optimizer_discriminator.step()

                # train the generator
                for parameter in self.discriminator.parameters(): 
                    parameter.requires_grad = False
                self.optimizer_generator.zero_grad()

                prediction_discriminator_synthetic = self.discriminator(concat_synthetic)
                loss_generator_synthetic = self.loss_func(prediction_discriminator_synthetic, 
                                                    torch.full(discriminator_prediction_shape, 1., dtype=torch.float32, device=device))
                l1_loss_generator_synthetic = self.l1_loss(synthetic_output, output_image) * 100
                loss_generator = loss_generator_synthetic + l1_loss_generator_synthetic
                loss_generator.backward()
                self.optimizer_generator.step()
                
                losses["losses_discriminator_real"].append(loss_discriminator_real.detach().cpu().item())
                losses["losses_discriminator_synthetic"].append(loss_discriminator_synthetic.detach().cpu().item())
                losses["losses_generator_synthetic"].append(loss_generator_synthetic.detach().cpu().item())
                losses["l1_losses_generator_synthetic"].append(l1_loss_generator_synthetic.detach().cpu().item())

            self.scheduler_discriminator.step()
            self.scheduler_generator.step()

            self.save_results(epoch=epoch, 
                             losses=losses,
                             epoch_start_time=epoch_start_time)    

    def train_cycle(self):
        """
        Train a model using the cycle approach - works with unpaired images.
        """
        pre_images_buffer = []
        post_images_buffer = []

        for epoch in range(self.starting_epoch, self.num_epochs + 1):
            epoch_start_time = time.time()
            losses = self.initialise_loss_storage(overall=False)

            self.pre_to_post_generator.train()
            self.post_to_pre_generator.train()
            self.pre_discriminator.train()
            self.post_discriminator.train()

            torch.manual_seed(epoch)

            for input_stack, output_image, _ in tqdm(self.train_loader, desc="Iterations", leave=False, disable=not self.verbose):
                
                real_pre_image = input_stack.to(device)
                real_post_image = output_image.to(device)
                if self.topography:
                    conditions = input_stack[:, 3:, :, :].detach().clone()
                    real_post_image = torch.cat((real_post_image, conditions.detach().clone().to(device)), dim=1)
                synthetic_post_image = self.pre_to_post_generator(real_pre_image)
                synthetic_pre_image = self.post_to_pre_generator(real_post_image)
                if self.topography:
                    synthetic_post_image = torch.cat((synthetic_post_image, conditions.detach().clone().to(device)), dim=1)
                    synthetic_pre_image = torch.cat((synthetic_pre_image, conditions.detach().clone().to(device)), dim=1)
                recreated_post_image = self.pre_to_post_generator(synthetic_pre_image)  
                recreated_pre_image = self.post_to_pre_generator(synthetic_post_image)      

                # train the generators
                for parameter in self.pre_discriminator.parameters(): 
                    parameter.requires_grad = False
                for parameter in self.post_discriminator.parameters(): 
                    parameter.requires_grad = False
                self.optimizer_generator.zero_grad()
                
                identity_loss_post = 0
                identity_loss_pre = 0
                if self.add_identity_loss:
                    identity_loss_post = self.identity_loss(self.pre_to_post_generator(real_post_image), real_post_image[:, :3, :, :]) * 5
                    identity_loss_pre = self.identity_loss(self.post_to_pre_generator(real_pre_image), real_pre_image[:, :3, :, :]) * 5
                discriminator_prediction_shape = self.post_discriminator(synthetic_post_image).shape
                post_generator_loss = self.loss_func(self.post_discriminator(synthetic_post_image), 
                                                    torch.full(discriminator_prediction_shape, 1., dtype=torch.float32, device=device))
                pre_generator_loss = self.loss_func(self.pre_discriminator(synthetic_pre_image), 
                                                    torch.full(discriminator_prediction_shape, 1., dtype=torch.float32, device=device))
                pre_to_post_cycle_loss = self.cycle_loss(recreated_pre_image, real_pre_image[:, :3, :, :]) * 10
                post_to_pre_cycle_loss = self.cycle_loss(recreated_post_image, real_post_image[:, :3, :, :]) * 10                                       
                generator_loss = post_generator_loss + pre_generator_loss + pre_to_post_cycle_loss + post_to_pre_cycle_loss + identity_loss_post + identity_loss_pre
                generator_loss.backward()
                self.optimizer_generator.step()

                # train the discriminators
                for parameter in self.pre_discriminator.parameters(): 
                    parameter.requires_grad = True
                for parameter in self.post_discriminator.parameters(): 
                    parameter.requires_grad = True
                self.optimizer_discriminator.zero_grad()

                synthetic_pre_image = self.get_buffer_image(synthetic_pre_image, pre_images_buffer)
                synthetic_post_image = self.get_buffer_image(synthetic_post_image, post_images_buffer)

                loss_discriminator_real_pre = self.loss_func(self.pre_discriminator(real_pre_image),
                                                            torch.full(discriminator_prediction_shape, 1., dtype=torch.float32, device=device))
                loss_discriminator_synthetic_pre = self.loss_func(self.pre_discriminator(synthetic_pre_image.detach()),
                                                                 torch.full(discriminator_prediction_shape, 0., dtype=torch.float32, device=device))
                loss_discriminator_pre = (loss_discriminator_real_pre + loss_discriminator_synthetic_pre) * 0.5
                loss_discriminator_pre.backward()

                loss_discriminator_real_post = self.loss_func(self.post_discriminator(real_post_image),
                                                        torch.full(discriminator_prediction_shape, 1., dtype=torch.float32, device=device))
                loss_discriminator_synthetic_post = self.loss_func(self.post_discriminator(synthetic_post_image.detach()),
                                                            torch.full(discriminator_prediction_shape, 0., dtype=torch.float32, device=device))
                loss_discriminator_post = (loss_discriminator_real_post + loss_discriminator_synthetic_post) * 0.5
                loss_discriminator_post.backward()
                self.optimizer_discriminator.step()

                losses["losses_generator_post"].append(post_generator_loss.detach().cpu().item())
                losses["losses_generator_pre"].append(pre_generator_loss.detach().cpu().item())
                losses["losses_pre_to_post_cycle"].append(pre_to_post_cycle_loss.detach().cpu().item())
                losses["losses_post_to_pre_cycle"].append(post_to_pre_cycle_loss.detach().cpu().item())
                losses["losses_discriminator_pre_real"].append(loss_discriminator_real_pre.detach().cpu().item())
                losses["losses_discriminator_post_real"].append(loss_discriminator_real_post.detach().cpu().item())
                losses["losses_discriminator_pre_synthetic"].append(loss_discriminator_synthetic_pre.detach().cpu().item())
                losses["losses_discriminator_post_synthetic"].append(loss_discriminator_synthetic_post.detach().cpu().item())
                if self.add_identity_loss:
                    losses["losses_identity_post"].append(identity_loss_post.detach().cpu().item())
                    losses["losses_identity_pre"].append(identity_loss_pre.detach().cpu().item())

            self.scheduler_generator.step()
            self.scheduler_discriminator.step()  
    
            self.save_results(epoch=epoch, 
                             losses=losses,
                             epoch_start_time=epoch_start_time)    
