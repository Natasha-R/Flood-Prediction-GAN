from models import data
from models import model_architectures

import time
import numpy as np
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.optim import lr_scheduler
from torchmetrics.regression import MeanSquaredError, R2Score
from torchmetrics.segmentation import GeneralizedDiceScore, MeanIoU
from torchmetrics.classification import BinaryAccuracy, BinaryF1Score, BinaryPrecision, BinaryRecall

device = "cuda" if torch.cuda.is_available() else "cpu"

class SegmentationModel():
    """
    A class encapsulating the attributes and functions for training and evaluating the flood segmentation model.
    """
    def __init__(self,
                 dataset_subset,
                 data_path,
                 num_epochs,
                 train_on_all,
                 save_model_interval,
                 save_images_interval,
                 verbose,
                 pretrained_model_path,
                 train,
                 learning_rate,
                 batch_size,
                 seed):
    
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
        self.learning_rate = learning_rate
        self.batch_size = batch_size
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
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, betas=(0.5, 0.999))
        self.scheduler = lr_scheduler.LambdaLR(self.optimizer, lr_lambda=self.lambda_rule)

        self.train_loader, self.val_loader, self.test_loader = data.create_masks_dataset(dataset_subset = self.dataset_subset,
                                                                                         path = self.data_path,
                                                                                         train_on_all = self.train_on_all,
                                                                                         batch_size = self.batch_size)
        
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
        file_type = ".png" if save_type=="image" or save_type=="figure" else ".pth.tar"
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
                                                                            train_on_all=self.train_on_all,
                                                                            batch_size=1)
        dataloader = test_loader if use_test_data else val_loader

        metrics = {"MSE":MeanSquaredError().to(device),
                    "R2":R2Score().to(device),
                    "Accuracy":BinaryAccuracy().to(device),
                    "F1":BinaryF1Score().to(device),
                    "Precision":BinaryPrecision().to(device),
                    "Recall":BinaryRecall().to(device),
                    "IOU":MeanIoU(num_classes=1).to(device),
                    "Dice":GeneralizedDiceScore(num_classes=1).to(device)}
        metrics_results = {metric: list() for metric in list(metrics.keys())}
        
        print("\nCalculating metrics...")
        for input_image, true_mask, image_name in tqdm(dataloader, desc="Images", leave=False):

            input_image = input_image.to(device)
            true_mask = self.tensor_to_mask(true_mask, predicted=False).to(device)
            predicted_mask  = self.tensor_to_mask(self.model(input_image), predicted=True).to(device)
            flat_true_mask = true_mask.squeeze().flatten()
            flat_predicted_mask = predicted_mask.squeeze().flatten()
            int_true_mask = true_mask.int()
            int_predicted_mask = predicted_mask.int()

            for metrics_name in metrics:
                if metrics_name in ["IOU", "Dice"]:
                    pred = int_predicted_mask.detach().clone()
                    true = int_true_mask.detach().clone()
                else:
                    pred = flat_predicted_mask.detach().clone()
                    true = flat_true_mask.detach().clone()
                metrics_results[metrics_name].append(metrics[metrics_name](pred, true).item())

        for metrics_name in metrics_results:
            print(f"{metrics_name}: {np.mean(metrics_results[metrics_name]):.3f} +/- {np.std(metrics_results[metrics_name], ddof=1) / np.sqrt(len(metrics_results[metrics_name])):.3f}")

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