import os

import torch

from .helper_func import parallelize

MODEL_NAME = "resnet_age"
LATEST_EPOCH = 30

# Get smaller JIT model for predictions.
model = torch.jit.load(
    "models/" + MODEL_NAME + "/jit/epoch_" + str(LATEST_EPOCH) + ".pt"
)

if torch.cuda.device_count() > 1:
    model = parallelize(model)
