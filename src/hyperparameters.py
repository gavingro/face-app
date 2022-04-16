import torch

# Hyperparameters for Model
BATCH_SIZE = 16
LEARNING_RATE = 0.00009
L2_DECAY_RATE = 0.0004
DROPOUT_RATE = 0.4

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
