import os
import logging
from datetime import datetime
from turtle import back

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


def backward(model, x, ages):
    """
    Updates the model based on how poorly the
    model predicts the ages of the input tensor
    x.

    Returns
    -------
    loss : model.criterion
        Our loss after the backwards iteration, returned
        for display in our train() function.
    """
    outputs = model(x)
    loss = model.criterion(outputs, ages)
    model.optimizer.zero_grad()  # avoid accumulatio
    if model.training:
        loss.backward()  # update params
        model.optimizer.step()  # continue
    return loss


def fit(
    model,
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
        If given, will save the model every N epochs. Else, it
        will save the model every time the validation score improves,
        by default None
    """
    # Make Folder for saving snapshots if doesn't exist:
    if not os.path.exists("models/" + model_name):
        os.makedirs("models/" + model_name)

    model.to(DEVICE)

    final_epoch = model.total_epochs + num_epochs
    for epoch in range(num_epochs):
        # Ensure model is in train mode.
        model.train()
        # Print Metrics based on total epochs trained,
        # not just this training cycle.
        logging.info(f"=== Epoch {model.total_epochs + 1} / {final_epoch} ===")
        # Instantialize variable for epoch
        training_loss = None
        validation_loss = None

        # Train with Training Set
        epoch_training_losses = np.empty(len(train_loader))
        for i, (inputs, labels) in enumerate(train_loader):
            # Fit model
            training_loss = backward(model, inputs, labels)
            epoch_training_losses[i] = training_loss
            # Print fit updates
            if (i + 1) % 25 == 0:
                now = datetime.now()
                nowstring = now.strftime("%D %H:%M:%S")
                logging.debug(
                    f"{nowstring} - Step [{i + 1} / {len(train_loader)}]: training_loss = {training_loss:.8f}"
                )

        # Increment total epochs trained after train data.
        model.total_epochs += 1
        # Get Validation Loss for Epoch
        # Avoid updating gradients.
        model.eval()
        epoch_validation_losses = np.empty(len(validation_loader))
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(validation_loader):
                validation_loss = backward(model, inputs, labels)
                epoch_validation_losses[i] = validation_loss

        # Calculate Training and Validation Losses
        # Each individual loss is a mean loss value for the batch,
        # so we can find the mean of them all on the same scale.
        epoch_training_loss = np.mean(epoch_training_losses)
        epoch_validation_loss = np.mean(epoch_validation_losses)
        logging.info(f"{epoch_training_loss=},\t {epoch_validation_loss=}")

        # Save Losses for Metrics
        model.training_loss_records.append(epoch_training_loss)
        model.validation_loss_records.append(epoch_validation_loss)

        # Save Model Snapshot if it out-performs other validation sets
        # Handles also the first iteration where there is only the current loss record.
        PATH = os.path.join(
            "models", model_name, ("epoch_" + str(model.total_epochs) + ".pth")
        )
        if save_every:
            if ((model.total_epochs) % save_every) == 0:
                torch.save(
                    {
                        "epoch": model.total_epochs,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": model.optimizer.state_dict(),
                        "training_loss": model.training_loss_records,
                        "validation_loss": model.validation_loss_records,
                    },
                    PATH,
                )
        else:
            if (len(model.validation_loss_records) == 1) or (
                epoch_validation_loss > max(model.validation_loss_records[:-1])
            ):
                torch.save(
                    {
                        "epoch": model.total_epochs,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": model.optimizer.state_dict(),
                        "training_loss": model.training_loss_records,
                        "validation_loss": model.validation_loss_records,
                    },
                    PATH,
                )
    return


def visualize_training(model, ax=None):
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
        range(1, len(model.training_loss_records) + 1),
        model.training_loss_records,
        color="red",
        label="Training Loss",
    )
    ax.plot(
        range(1, len(model.validation_loss_records) + 1),
        model.validation_loss_records,
        color="blue",
        label="Validation Loss",
    )
    ax.legend()
    ax.set_title("Training and Validation Loss vs Epochs")
    return ax


def score(
    model,
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
    model.eval()
    model.to(DEVICE)
    with torch.no_grad():
        # Create Figures
        f, axd = plt.subplot_mosaic(
            [["upper", "upper"], ["lower_left", "lower_right"]], figsize=(10, 8)
        )
        plt.suptitle(
            f"Metrics for {LEARNING_RATE=}, {DROPOUT_RATE=}\n\
            {L2_DECAY_RATE=}, {BATCH_SIZE=} at epoch {model.total_epochs}"
        )

        # Make LOSS VS EPOCHS Plot:
        axd["upper"] = visualize_training(model, axd["upper"])

        # Make TRAINING DATA plot
        axd["lower_left"].set_ylabel("Observed Ages")
        axd["lower_left"].set_xlabel("Predicted Ages")

        # Get predictions from model, and
        # compare with true labels.
        # Make MSE for each batch, then
        # average them.
        batch_mses = np.empty(len(training_loader))
        for i, (x, labels) in enumerate(training_loader):
            outputs = model.forward(x)
            axd["lower_left"].scatter(outputs, labels)
            batch_mse = torch.sum(torch.pow(labels - outputs, 2)) / len(x)
            batch_mses[i] = batch_mse
            # Cap at certain amount of points to save time and reduce clutter
            if i > 50:
                break
        training_rmse = (np.sum(batch_mses) / len(training_loader)) ** (1 / 2)
        axd["lower_left"].set_title(
            f"Training RMSE (First 50 Batches): {training_rmse:.5f}"
        )

        # Make TESTING DATA plot
        axd["lower_right"].set_xlabel("Predicted Ages")

        # Get predictions from model, and
        # compare with true labels.
        # Make MSE for each batch, then
        # average them.
        batch_mses = np.empty(len(testing_loader))
        for i, (x, labels) in enumerate(testing_loader):
            outputs = model.forward(x)
            axd["lower_right"].scatter(outputs, labels)
            batch_mse = torch.sum(torch.pow(labels - outputs, 2)) / len(x)
            batch_mses[i] = batch_mse
        testing_rmse = (np.sum(batch_mses) / len(testing_loader)) ** (1 / 2)
        axd["lower_right"].set_title(f"Testing RMSE: {testing_rmse:.5f}")
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
                ("epoch_" + str(model.total_epochs) + ".jpg"),
            )
            f.savefig(PATH, bbox_inches="tight")
            return


def parallelize(model):
    """
    Turns our resnetregressor into a data parallel
    nn model for training on distributed GPU's, while
    still keeping the same class attributes for our
    helper functions.

    Parameters
    ----------
    model : src.resnetregressor.ResNetRegressor
        Our ResNetRegressor Model to parallelize.

    Returns
    -------
    torch.nn.parallel.data_parallel.DataParallel
        Our parallelized model to continue to
        fit our parameters on.
    """
    # Parallelize but keep original object
    model, inner_model = nn.DataParallel(model), model
    # Persist custom attributes into new model.
    model.total_epochs = inner_model.total_epochs
    model.training_loss_records = inner_model.training_loss_records
    model.validation_loss_records = inner_model.validation_loss_records
    model.criterion = inner_model.criterion
    model.optimizer = inner_model.optimizer
    return model
