import torch

# Hyperparameters for Model
BATCH_SIZE = 16
LEARNING_RATE = 0.0009
L2_DECAY_RATE = 0.0004
DROPOUT_RATE = 0.4

torch.manual_seed(0)

if torch.cuda.is_available():
    DEVICE = "cuda"
    torch.backends.cudnn.deterministic = True
else:
    DEVICE = "cpu"
