import os

import torch

from .resnetregressor import ResNetRegressor
from .helper_func import parallelize
from .hyperparameters import DEVICE

MODEL_NAME = "resnet_age"
LATEST_EPOCH = 30

# # Setting the environment variable to use local resnet34 pretrained copy
# os.environ["TORCH_HOME"] = "models/resnet34_base"
# model = ResNetRegressor()

# # Load ongoing model
# checkpoint = torch.load(
#     "models/" + MODEL_NAME + "/epoch_" + str(LATEST_EPOCH) + ".pth", map_location=DEVICE
# )
# model.load_state_dict(checkpoint["model_state_dict"])
# model.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
# model.total_epochs = checkpoint["epoch"]
# model.training_loss_records = checkpoint["training_loss"]
# model.validation_loss_records = checkpoint["validation_loss"]

# Get smaller JIT model for predictions.
model = torch.jit.load(
    "models/" + MODEL_NAME + "/jit/epoch_" + str(LATEST_EPOCH) + ".pt"
)

if torch.cuda.device_count() > 1:
    model = parallelize(model)
