import os
import shutil
import gzip


import torch

from .helper_func import parallelize
from .hyperparameters import DEVICE

MODEL_NAME = "resnet_age"
LATEST_EPOCH = 30


def load_model_from_gzip(filename_in: str) -> torch.nn.Module:
    """
    Loads our neural network (as jit script) from a compressed gzip file.

    This is mostly to squeak under the 100mb filesize for github and
    Heroku.

    Parameters
    ----------
    filename_in : str
        filename of the .gzip file
    Returns
    -------
    torch.nn.module
        _description_
    """
    # Write temporary .pth file
    jit_filename = filename_in.replace(".tar.gz", ".pth")
    with gzip.open(filename_in, "rb") as fin, open(jit_filename, "wb") as fout:
        # Reads the file by chunks to avoid exhausting memory
        shutil.copyfileobj(fin, fout)

    # Load Model from .pth file
    model = torch.jit.load(jit_filename)

    # Remove the intermediate jit file
    if os.path.exists(jit_filename):
        os.remove(jit_filename)

    return model


# Load smaller JIT model for predictions.
model = load_model_from_gzip(
    "models/" + MODEL_NAME + "/jit/epoch_" + str(LATEST_EPOCH) + ".tar.gz"
)

if torch.cuda.device_count() > 1:
    model = parallelize(model)
