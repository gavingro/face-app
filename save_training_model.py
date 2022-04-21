import os
import shutil
import gzip

import torch

from src.current_training_model import training_model


def save_model_to_gzip(model: torch.nn.Module, model_name: str) -> None:
    """
    Saves our neural network (as jit script) to a compressed gzip file.

    This is mostly to squeak under the 100mb filesize for github and
    Heroku.

    Parameters
    ----------
    model : torch.nn.module
        Our model to save/compress.
    model_name : str
        The name of the model (affects filepath).
    """
    # Get Filenames
    latest_epoch = model.total_epochs
    jit_filename = "models/" + model_name + "/epoch_" + str(latest_epoch) + ".pth"
    filename_out = (
        "models/" + model_name + "/jit/epoch_" + str(latest_epoch) + ".tar.gz"
    )

    # Save to temporary jit file
    torch.jit.save(torch.jit.script(model), jit_filename)

    # Compress jit file to gzip
    with open(jit_filename, "rb") as fin, gzip.open(filename_out, "wb") as fout:
        # Reads the file by chunks to avoid exhausting memory
        shutil.copyfileobj(fin, fout)

    print(f"Saving {model_name} at Epoch {latest_epoch} to {jit_filename}")

    # Remove the intermediate jit file
    if os.path.exists(jit_filename):
        os.remove(jit_filename)


if __name__ == "__main__":
    save_model_to_gzip(training_model, model_name="resnet_age")
