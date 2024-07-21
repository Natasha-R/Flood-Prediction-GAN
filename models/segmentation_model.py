from models import data
from models import model_architectures

import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.optim import lr_scheduler
from torchmetrics.regression import MeanSquaredError
from torchmetrics.classification import BinaryAccuracy, BinaryF1Score, BinaryPrecision, BinaryRecall

device = "cuda" if torch.cuda.is_available() else "cpu"

class SegmentationModel():
    """
    A class encapsulating the attributes and functions for training and evaluating the flood segmentation model.
    """
    def __init__(self,
                 dataset_subset="usa",
                 data_path=None,
                 num_epochs=100,
                 train_on_all=False,
                 save_model_interval=0,
                 save_images_interval=0,
                 verbose=True,
                 pretrained_model_path=None,
                 train=False,
                 plot_mask_image=None,
                 use_test_data=False,
                 seed=47):
    
        if verbose:
            print("\nSetting up the flood segmentation model...") 
        
        self.num_epochs = num_epochs
        self.verbose = verbose
        self.save_model_interval = save_model_interval
        self.save_images_interval = save_images_interval
        self.data_path = data_path
        self.dataset_subset = dataset_subset
        self.train_on_all = train_on_all
        self.train = train
        self.pretrained_model_path = pretrained_model_path
        self.seed = seed
        self.starting_epoch = 1
        self.current_epoch = 1
        self.all_losses = []
        self.all_accuracies = []

        self.model = model_architectures.UNet().apply(self.initialise_weights).to(device)

        if self.pretrained_model_path:
            saved_model = torch.load(self.pretrained_model_path)
            self.current_epoch = saved_model["current_epoch"]
            self.num_epochs = saved_model["num_epochs"]
            self.model.load_state_dict(saved_model["model"])
            self.all_losses = saved_model["all_losses"]
            self.all_accuracies = saved_model["all_accuracies"]

        self.loss_func = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001, betas=(0.5, 0.999))
        self.scheduler = lr_scheduler.LambdaLR(self.optimizer, lr_lambda=self.lambda_rule)

        self.train_loader, self.val_loader, self.test_loader = data.create_masks_dataset(dataset_subset = self.dataset_subset,
                                                                                         path = self.data_path,
                                                                                         train_on_all = self.train_on_all)
        
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
    
    def create_path(self, save_type):
        """
        Defines an informative path string to save images or models to, 
        containing information about the model and dataset.
        """
        file_types = {"image":".png", "figure":".png", "model":".pth.tar", "metric":".csv"}
        file_type = file_types[save_type]
        current_time = str(datetime.now())[:-7].replace(' ', '-').replace(':', '-')
        path = (f"{self.data_path}/{save_type}s/"
        f"SegmentationModel_epoch{self.current_epoch if self.train else self.current_epoch-1}_"
        f"{self.dataset_subset}Data_date{current_time}{file_type}")
        return path
    
    def save_results(self, epoch, losses, accuracies, epoch_start_time):
        """
        Save the current losses, model, and sample outputs, at given epochs.
        """
        self.current_epoch = epoch
        self.all_losses.append(np.mean(losses))
        self.all_accuracies.append(np.mean(accuracies))

        if self.verbose:
            epoch_end_time = time.time()
            print((f"Epoch {epoch} ({epoch_end_time - epoch_start_time :.2f} seconds) | "
                   f"Loss = {self.all_losses[-1]:.2f} | "
                   f"Accuracy = {self.all_accuracies[-1]:.2f}"))

        if self.save_model_interval != 0 and ((epoch % self.save_model_interval) == 0):
            saved_model = {
                "current_epoch": epoch + 1,
                "num_epochs": self.num_epochs,
                "model": self.model.state_dict(),
                "all_losses":self.all_losses,
                "all_accuracies":self.all_accuracies}
            model_path = self.create_path(save_type="model")
            print(f"Saving flood segmentation model to {model_path}")
            torch.save(saved_model, model_path)

        if self.save_images_interval != 0 and ((epoch % self.save_images_interval) == 0):
            self.plot_sample_images(num_images=10, use_test_data=False)
            self.plot_loss()

    def calculate_metrics(self, use_test_data=False):
        """
        Calculate metrics for the model.
        """
        train_loader, val_loader, test_loader = data.create_masks_dataset(dataset_subset=self.dataset_subset,
                                                                            path=self.data_path,
                                                                            train_on_all=self.train_on_all)
        dataloader = test_loader if use_test_data else val_loader
        all_true_flood_mask = None
        all_predicted_flood_mask = None
        metrics = {"MSE":MeanSquaredError().to(device),
                    "Accuracy":BinaryAccuracy().to(device),
                    "F1_Flood":BinaryF1Score().to(device),
                    "Precision_Flood":BinaryPrecision().to(device),
                    "Recall_Flood":BinaryRecall().to(device),
                    "F1_No_Flood":BinaryF1Score().to(device),
                    "Precision_No_Flood":BinaryPrecision().to(device),
                    "Recall_No_Flood":BinaryRecall().to(device)}
        metrics_results = {metric: list() for metric in list(metrics.keys())}
        
        print("\nCalculating metrics...")
        for input_image, true_mask, image_name in tqdm(dataloader, desc="Images", leave=False):

            input_image = input_image.to(device)
            true_mask = self.tensor_to_mask(true_mask, predicted=False).to(device)
            predicted_mask  = self.tensor_to_mask(self.model(input_image), predicted=True).to(device)
            flat_true_mask = true_mask.detach().clone().squeeze().flatten().squeeze()
            flat_predicted_mask = predicted_mask.detach().clone().squeeze().flatten().squeeze()

            all_true_flood_mask = torch.cat((all_true_flood_mask, flat_true_mask), dim=0).to(device) if not all_true_flood_mask==None else flat_true_mask
            all_predicted_flood_mask = torch.cat((all_predicted_flood_mask, flat_predicted_mask), dim=0).to(device) if not all_predicted_flood_mask==None else flat_predicted_mask

        for metrics_name in ["MSE", "Accuracy", "F1_Flood", "Precision_Flood", "Recall_Flood"]:
            metrics_results[metrics_name].append(metrics[metrics_name](all_predicted_flood_mask, all_true_flood_mask).item())
        all_true_no_flood_mask = torch.abs(all_true_flood_mask-1)
        all_predicted_no_flood_mask = torch.abs(all_predicted_flood_mask-1)
        for metrics_name in ["F1_No_Flood", "Precision_No_Flood", "Recall_No_Flood"]:
            metrics_results[metrics_name].append(metrics[metrics_name](all_predicted_no_flood_mask, all_true_no_flood_mask).item())

        metrics_df = pd.DataFrame([(metrics_name, np.mean(metrics_results[metrics_name])) for metrics_name in metrics_results]).set_index(0).transpose()
        print(metrics_df)
        metrics_df.to_csv(self.create_path("metric"))

    def plot_loss(self):
        """
        Plot the training loss over the epochs.
        """
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 7))
        ax.tick_params(axis="both", which="major", labelsize=14)
        ax.set_xlabel("Epoch", fontsize=14)
        ax.set_ylabel("Loss", fontsize=14)
        ax.set_title("Training loss", fontsize=15)
        ax.plot(range(1, self.current_epoch+1 if self.train else self.current_epoch),
                self.all_losses,
                c="black",
                linewidth=2)
        losses_path = self.create_path(save_type="figure")
        print(f"Saving losses figure to {losses_path}")
        fig.savefig(losses_path, bbox_inches="tight")
    
    def plot_mask_image(self, path_to_image):
        """
        Plot and save the predicted segmentation mask for the given image.
        """
        image_name = path_to_image.split("/")[-1][:-4]
        input_image = torch.from_numpy(plt.imread(path_to_image)[:, :, :3].transpose(2, 0, 1)).unsqueeze(dim=0).to(device)
        predicted_mask = self.tensor_to_mask(self.model(input_image), predicted=True).squeeze().cpu().numpy()

        current_time = str(datetime.now())[:-7].replace(' ', '-').replace(':', '-')
        path_to_mask = f"{self.data_path}/images/SegmentationMask_{image_name}_{current_time}.png"
        print(f"\nSaving segmentation mask for '{image_name}' to {path_to_mask}")
        plt.imsave(path_to_mask, predicted_mask, vmin=0, vmax=1, cmap="gray")

    def plot_sample_images(self, num_images, use_test_data=False):
        """
        Plot 'num_images' random sample generator output images from the validation dataset.
        The randomness of the images can be controlled via the seed.
        """
        train_loader, val_loader, test_loader = data.create_masks_dataset(dataset_subset=self.dataset_subset,
                                                                                    path=self.data_path,
                                                                                    train_on_all=self.train_on_all,
                                                                                    batch_size=1)
        dataloader = test_loader if use_test_data else val_loader
        fig, axes = plt.subplots(nrows=num_images, ncols=3, figsize=(3 * 5, num_images * 5))
        for ax in axes.ravel():
            ax.set_axis_off()
        torch.manual_seed(self.seed)
        for i, (input_image, true_mask, image_name) in enumerate(dataloader):
            input_image = input_image.to(device)
            true_mask = true_mask.to(device)
            predicted_mask = self.model(input_image)
            input_image = np.clip(input_image.squeeze().cpu().detach().numpy().transpose(1, 2, 0), 0, 1)
            true_mask = self.tensor_to_mask(true_mask, predicted=False).cpu().numpy()
            predicted_mask = self.tensor_to_mask(predicted_mask, predicted=True).cpu().numpy()
            axes[i, 0].imshow(input_image, vmin=0, vmax=1)
            axes[i, 1].imshow(true_mask.squeeze(), vmin=0, vmax=1, cmap="gray")
            axes[i, 2].imshow(predicted_mask.squeeze(), vmin=0, vmax=1, cmap="gray")
            axes[i, 0].set_title(f"Input ({image_name[0]})")
            axes[i, 1].set_title("Ground Truth Mask")
            axes[i, 2].set_title("Predicted Mask")
            if i >= num_images-1:
                break
        fig.tight_layout()
        images_path = self.create_path("image")
        print("Saving sample images to", images_path)
        fig.savefig(images_path, bbox_inches="tight")
        plt.close();

    def tensor_to_mask(self, tensor, predicted=True):
        if predicted:
            return (torch.sigmoid(tensor.detach().clone()) > 0.5).float()
        else:
            return (tensor.detach().clone() > 0.5).float()
    
    def train_model(self):
        """
        Train the flood segmentation model.
        """
        for epoch in range(self.starting_epoch, self.num_epochs + 1):
            epoch_start_time = time.time()
            losses = []
            accuracies = []

            self.model.train()

            for input_image, true_mask, _ in tqdm(self.train_loader, desc="Iterations", leave=False, disable=not self.verbose):

                input_image = input_image.to(device)
                true_mask = true_mask.to(device)
                predicted_mask = self.model(input_image)

                loss = self.loss_func(predicted_mask, true_mask)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                losses.append(loss.item())

                accuracy = (self.tensor_to_mask(predicted_mask, predicted=True) == self.tensor_to_mask(true_mask, predicted=False)).sum().item() / torch.numel(predicted_mask)
                accuracies.append(accuracy)

            self.scheduler.step()
            self.save_results(epoch=epoch, losses=losses, accuracies=accuracies, epoch_start_time=epoch_start_time)