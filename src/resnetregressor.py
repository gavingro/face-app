import torch
from torch import nn

from .hyperparameters import (
    DROPOUT_RATE,
    LEARNING_RATE,
    L2_DECAY_RATE,
)


class ResNetRegressor(nn.Module):
    def __init__(self):
        super(ResNetRegressor, self).__init__()
        # Model for forward()
        # Pretrained resnet, excluding usual classification step layer
        resnet_layers = list(
            torch.hub.load(
                "pytorch/vision:v0.10.0",
                "resnet34",
                pretrained=True,
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
        regression_layers += [nn.Linear(16, 1), nn.ReLU(inplace=True)]
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
