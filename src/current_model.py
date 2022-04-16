import torch

from .resnetregressor import ResNetRegressor

MODEL_NAME = "resnet_age1"
LATEST_EPOCH = 9

model = ResNetRegressor()

# Load ongoing model
# checkpoint = torch.load("models/" + MODEL_NAME + "/epoch_" + str(LATEST_EPOCH) + ".pth")
# model.load_state_dict(checkpoint["model_state_dict"])
# model.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
# model.total_epochs = checkpoint["epoch"]
# model.training_loss_records = checkpoint["training_loss"]
# model.validation_loss_records = checkpoint["validation_loss"]
