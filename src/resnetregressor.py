import os
from datetime import datetime

import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np

from .hyperparameters import DROPOUT_RATE, LEARNING_RATE, L2_DECAY_RATE
from .data import train_loader, validation_loader


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
        self.optimizer.zero_grad()  # avoid accumulation
        loss.backward()  # update params
        self.optimizer.step()  # continue
        return loss

    def fit(
        self,
        train_loader=train_loader,
        validation_loader=validation_loader,
        num_epochs=5,
        model_name="resnet_age",
        save_every=1,
    ):
        """
        Continuously updates the model with backward()
        for each of the epochs based on some loader of
        training data.

        Automatically saves model checkpoints which improve
        the perfomance of the model as you go for future training.
        model_name must correspond to an existing folder models/.

        Returns
        -------
        self.loss_records : np.array
            A tuple of two (1 x num_epochs) vectors holding the record of
            losses measured at the end of each epoch.
        """
        # Make Folder for saving snapshots if doesn't exist:
        self.train()
        if not os.path.exists("models/" + model_name):
            os.makedirs("models/" + model_name)

        final_epoch = self.total_epochs + num_epochs
        for epoch in range(num_epochs):
            # Print Metrics based on total epochs trained,
            # not just this training cycle.
            print(f"=== Epoch {self.total_epochs + 1} / {final_epoch} ===")
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
                if (i + 1) % 25 == 0:
                    now = datetime.now()
                    nowstring = now.strftime("%D %H:%M:%S")
                    print(
                        f"{nowstring} - Step [{i + 1} / {len(train_loader)}]: training_loss = {training_loss:.8f}"
                    )

            # Increment total epochs trained after train data.
            self.total_epochs += 1

            # Get Validation Loss for Epoch
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
            print(f"{epoch_training_loss=},\t {epoch_validation_loss=}")

            # Save Losses for Metrics
            self.training_loss_records.append(epoch_training_loss)
            self.validation_loss_records.append(epoch_validation_loss)

            # Save Model Snapshot if it out-performs other validation sets
            # Handles also the first iteration where there is only the current loss record.

            # ACTUALLY Lets just save all epochs.
            # if (len(self.validation_loss_records) == 1)
            # or (epoch_validation_loss > max(self.validation_loss_records[:-1])):

            if ((self.total_epochs) % save_every) == 0:
                PATH = (
                    "models/"
                    + model_name
                    + "/"
                    + "epoch_"
                    + str(self.total_epochs)
                    + ".pth"
                )
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

        return self.training_loss_records, self.validation_loss_records

    def visualize_training(self):
        """
        Plots the loss vs the epochs based on the
        loss_records from a train() call.
        """
        ax = plt.subplot()
        ax.plot(
            range(len(self.training_loss_records)),
            self.training_loss_records,
            color="red",
            label="Training Loss",
        )
        ax.plot(
            range(len(self.validation_loss_records)),
            self.validation_loss_records,
            color="blue",
            label="Validation Loss",
        )
        ax.legend()
        ax.set_title("Training and Validation Loss vs Epochs")
        plt.show()
        return ax

    def score(self, input_loader):
        """
        Evaluates the accuracy of the trained model on
        a passed in data input loader.

        Returns
        -------
        MSE : A Float of the Mean Squared Error.
        """
        self.eval()
        with torch.no_grad():
            # Get predictions from model, and
            # compare with true labels.
            # Make MSE for each batch, then
            # average them.
            f = plt.figure()
            ax = plt.subplot()
            ax.set_ylabel("Observed Ages")
            ax.set_xlabel("Predicted Ages")
            batch_mses = np.empty(len(input_loader))
            for i, (x, labels) in enumerate(input_loader):
                outputs = self.forward(x)
                ax.scatter(outputs, labels)
                batch_mse = torch.sum(torch.pow(labels - outputs, 2)) / len(x)
                batch_mses[i] = batch_mse

            total_rmse = (np.sum(batch_mses) / len(input_loader)) ** (1 / 2)
            plt.show()
            # Display results for each class
            print(f"Total RMSE: {total_rmse}")
        return total_rmse
