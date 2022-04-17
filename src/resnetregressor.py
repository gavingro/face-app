import os
import logging
from datetime import datetime

import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np

from .hyperparameters import (
    DEVICE,
    DROPOUT_RATE,
    LEARNING_RATE,
    L2_DECAY_RATE,
    BATCH_SIZE,
)
from .data import train_loader, test_loader


class ResNetRegressor(nn.Module):
    def __init__(self):
        super(ResNetRegressor, self).__init__()
        # Model for forward()
        # Pretrained resnet, excluding usual classification step layer
        resnet_layers = list(
            torch.hub.load(
                "pytorch/vision:v0.10.0", "resnet34", pretrained=True
            ).children()
        )[:-2]
        # Age Regression Layers
        regression_layers = [nn.MaxPool2d(2, 2), nn.Flatten()]
        regression_layers += [
            nn.BatchNorm1d(4608, affine=True, track_running_stats=True)
        ]
        regression_layers += [nn.Dropout(p=DROPOUT_RATE)]
        regression_layers += [nn.Linear(4608, 1024, bias=True), nn.ReLU(inplace=True)]
        regression_layers += [
            nn.BatchNorm1d(1024, affine=True, track_running_stats=True)
        ]
        regression_layers += [nn.Dropout(p=DROPOUT_RATE)]
        regression_layers += [nn.Linear(1024, 512, bias=True), nn.ReLU(inplace=True)]
        regression_layers += [nn.Linear(512, 16, bias=True), nn.ReLU(inplace=True)]
        regression_layers += [nn.Linear(16, 1)]
        total_layers = resnet_layers + regression_layers
        self.fullmodel = nn.Sequential(*total_layers)

        # Algo's for backward()
        self.criterion = nn.SmoothL1Loss()
        self.optimizer = torch.optim.Adam(
            self.parameters(), lr=LEARNING_RATE, weight_decay=L2_DECAY_RATE
        )

        # Metric Tracking for Saving/Loading the model.
        self.training_loss_records = []
        self.validation_loss_records = []
        self.total_epochs = 0

    def forward(self, input):
        """
        Gets a model output for our input tensor.

        Must have a batch size (or at least an extra dimension
        to the tensor specifying batch size)
        to make batchnorm work properly.

        Returns
        -------
        output : torch.tensor
            A tensor of our age predictions.

        """
        output = self.fullmodel(input).squeeze(1)
        return output

    def backward(self, x, ages):
        """
        Updates the model based on how poorly the
        model predicts the ages of the input tensor
        x.

        Returns
        -------
        loss : self.criterion
            Our loss after the backwards iteration, returned
            for display in our train() function.
        """
        outputs = self.forward(x)
        loss = self.criterion(outputs, ages)
        self.optimizer.zero_grad()  # avoid accumulatio
        if self.training:
            loss.backward()  # update params
            self.optimizer.step()  # continue
        return loss

    def fit(
        self,
        train_loader: torch.utils.data.DataLoader = train_loader,
        validation_loader: torch.utils.data.DataLoader = test_loader,
        num_epochs: int = 5,
        model_name: str = "resnet_age",
        save_every: int = None,
    ) -> None:
        """
        Continuously updates the model with backward()
        for each of the epochs based on some loader of
        training data.

        Automatically saves model checkpoints which improve
        the perfomance of the model as you go for future training
        to models/model_name/epoch_n.pth


        Parameters
        ----------
        train_loader : torch.utils.data.DataLoader, optional
            Dataset you want to train your model on, by default train_loader
        validation_loader : torch.utils.data.DataLoader, optional
            Dataset you want to check validity on.
            Does NOT affect weights of the model,
            and shouldn't be "seen" by batch-normalization,
            by default validation_loader due to
            model.eval().
        num_epochs : int, optional
            Number of epochs to train, by default 5
        model_name : str, optional
            Filename to save your model to, by default "resnet_age"
        save_every : int, optional
            If give, will save the model every N epochs. Else, it
            will save the model every time the validation score improves,
            by default None
        """
        # Make Folder for saving snapshots if doesn't exist:
        if not os.path.exists("models/" + model_name):
            os.makedirs("models/" + model_name)

        self.to(DEVICE)

        final_epoch = self.total_epochs + num_epochs
        for epoch in range(num_epochs):
            # Ensure model is in train mode.
            self.train()
            # Print Metrics based on total epochs trained,
            # not just this training cycle.
            logging.info(f"=== Epoch {self.total_epochs + 1} / {final_epoch} ===")
            # Instantialize variable for epoch
            training_loss = None
            validation_loss = None

            # Train with Training Set
            epoch_training_losses = np.empty(len(train_loader))
            for i, (inputs, labels) in enumerate(train_loader):
                # Fit model
                training_loss = self.backward(inputs, labels)
                epoch_training_losses[i] = training_loss
                # Print fit updates
                if (i + 1) % 50 == 0:
                    now = datetime.now()
                    nowstring = now.strftime("%D %H:%M:%S")
                    logging.debug(
                        f"{nowstring} - Step [{i + 1} / {len(train_loader)}]: training_loss = {training_loss:.8f}"
                    )

            # Increment total epochs trained after train data.
            self.total_epochs += 1
            # Get Validation Loss for Epoch
            # Avoid updating gradients.
            self.eval()
            epoch_validation_losses = np.empty(len(validation_loader))
            with torch.no_grad():
                for i, (inputs, labels) in enumerate(validation_loader):
                    predictions = self.forward(inputs)
                    validation_loss = self.criterion(predictions, labels)
                    epoch_validation_losses[i] = validation_loss

            # Calculate Training and Validation Losses
            # Each individual loss is a mean loss value for the batch,
            # so we can find the mean of them all on the same scale.
            epoch_training_loss = np.mean(epoch_training_losses)
            epoch_validation_loss = np.mean(epoch_validation_losses)
            logging.info(f"{epoch_training_loss=},\t {epoch_validation_loss=}")

            # Save Losses for Metrics
            self.training_loss_records.append(epoch_training_loss)
            self.validation_loss_records.append(epoch_validation_loss)

            # Save Model Snapshot if it out-performs other validation sets
            # Handles also the first iteration where there is only the current loss record.
            PATH = os.path.join(
                "models", model_name, ("epoch_" + str(self.total_epochs) + ".pth")
            )
            if save_every:
                if ((self.total_epochs) % save_every) == 0:
                    torch.save(
                        {
                            "epoch": self.total_epochs,
                            "model_state_dict": self.state_dict(),
                            "optimizer_state_dict": self.optimizer.state_dict(),
                            "training_loss": self.training_loss_records,
                            "validation_loss": self.validation_loss_records,
                        },
                        PATH,
                    )
            else:
                if (len(self.validation_loss_records) == 1) or (
                    epoch_validation_loss > max(self.validation_loss_records[:-1])
                ):
                    torch.save(
                        {
                            "epoch": self.total_epochs,
                            "model_state_dict": self.state_dict(),
                            "optimizer_state_dict": self.optimizer.state_dict(),
                            "training_loss": self.training_loss_records,
                            "validation_loss": self.validation_loss_records,
                        },
                        PATH,
                    )
        return

    def visualize_training(self, ax=None):
        """
        Plots the loss vs the epochs based on the
        loss_records from a train() call.

        Optionally pass in an plt.axis to add it
        to an ongoing figure.


        Parameters
        ----------
        ax : matplotlib axis, optional
            The axis you'd like to build the plot
            on (if any), by default None

        Returns
        -------
        matplotlib axis
            An axis object holding the loss over epochs
            plot.
        """
        if not ax:
            ax = plt.subplot()
        ax.plot(
            range(1, len(self.training_loss_records) + 1),
            self.training_loss_records,
            color="red",
            label="Training Loss",
        )
        ax.plot(
            range(1, len(self.validation_loss_records) + 1),
            self.validation_loss_records,
            color="blue",
            label="Validation Loss",
        )
        ax.legend()
        ax.set_title("Training and Validation Loss vs Epochs")
        return ax

    def score(
        self,
        training_loader: torch.utils.data.DataLoader = train_loader,
        testing_loader: torch.utils.data.DataLoader = test_loader,
        show=False,
        model_name="resnet_age",
    ) -> None:
        """
        Evaluates the RMSE of the model on both a training and testing dataset,
        creating a visualization of these predictions vs truth along with
        the models loss over epochs during training.

        If show is true, plots the image and prints metrics. If not,
        the figure will be saved to metrics/model_name/epoch_n.jpg.

        Parameters
        ----------
        training_loader : torch.utils.data.DataLoader
            The dataset that your model has trained on.
        testing_loader : torch.utils.data.DataLoader
            The dataset that your model will be tested on.
        show : bool, optional
            Prints out the figure if True, else saves it
            to a .jpg file if False, by default False
        model_name : str, optional
            "name" of model to save the jpg to in filepath,
            by default "resnet_age"
        """
        self.eval()
        self.to(DEVICE)
        with torch.no_grad():
            # Create Figures
            f, axd = plt.subplot_mosaic(
                [["upper", "upper"], ["lower_left", "lower_right"]], figsize=(10, 8)
            )
            plt.suptitle(
                f"Metrics for {LEARNING_RATE=}, {DROPOUT_RATE=}\n\
                {L2_DECAY_RATE=}, {BATCH_SIZE=} at epoch {self.total_epochs}"
            )

            # Make LOSS VS EPOCHS Plot:
            axd["upper"] = self.visualize_training(axd["upper"])

            # Make TRAINING DATA plot
            axd["lower_left"].set_ylabel("Observed Ages")
            axd["lower_left"].set_xlabel("Predicted Ages")

            # Get predictions from model, and
            # compare with true labels.
            # Make MSE for each batch, then
            # average them.
            batch_mses = np.empty(len(training_loader))
            for i, (x, labels) in enumerate(training_loader):
                outputs = self.forward(x)
                axd["lower_left"].scatter(outputs, labels)
                batch_mse = torch.sum(torch.pow(labels - outputs, 2)) / len(x)
                batch_mses[i] = batch_mse
                # Cap at certain amount of points to save time and reduce clutter
                if i > 50:
                    break
            training_rmse = (np.sum(batch_mses) / len(training_loader)) ** (1 / 2)
            axd["lower_left"].set_title(
                f"Training RMSE (First 50 Batches): {training_rmse}"
            )

            # Make TESTING DATA plot
            axd["lower_right"].set_xlabel("Predicted Ages")

            # Get predictions from model, and
            # compare with true labels.
            # Make MSE for each batch, then
            # average them.
            batch_mses = np.empty(len(testing_loader))
            for i, (x, labels) in enumerate(testing_loader):
                outputs = self.forward(x)
                axd["lower_right"].scatter(outputs, labels)
                batch_mse = torch.sum(torch.pow(labels - outputs, 2)) / len(x)
                batch_mses[i] = batch_mse
            testing_rmse = (np.sum(batch_mses) / len(testing_loader)) ** (1 / 2)
            axd["lower_right"].set_title(f"Testing RMSE: {testing_rmse}")
            plt.tight_layout()

            # Display / save results
            if show:
                logging.info(f"RMSE: {training_rmse=}, {testing_rmse=}")
                plt.show()
                return
            else:
                if not os.path.exists("metrics/" + model_name):
                    os.makedirs("metrics/" + model_name)
                PATH = os.path.join(
                    "metrics",
                    (model_name),
                    ("epoch_" + str(self.total_epochs) + ".jpg"),
                )
                f.savefig(PATH, bbox_inches="tight")
                return
