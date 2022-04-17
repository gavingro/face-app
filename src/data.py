import torchvision
import torch

from .faceagedataset import FaceAgeDataset
from .hyperparameters import BATCH_SIZE

# Inputs Transforms to make our input data look
# like what ResNet was trained on (224x224 normalized).
input_data_transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.Resize(256),
        torchvision.transforms.CenterCrop(
            224
        ),  # Some SMALL amount of Data Augmentation
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),
    ]
)

# Define Data with Dataset class
face_age_data = FaceAgeDataset(
    "data/UTKFace/ages-cropped.csv",
    "data/UTKFace/crop_part1",
    transform=input_data_transform,
)

# Make 85-15 train-test split.
# Skip Validation Set - not using it.
# train_data, validation_data, test_data = torch.utils.data.random_split(
train_data, test_data = torch.utils.data.random_split(
    face_age_data,
    [
        int(len(face_age_data) * 0.85),
        # int(len(face_age_data) * 0.05),
        int(len(face_age_data) * 0.15),
    ],
)

# Make Batch Loader Objects
train_loader = torch.utils.data.DataLoader(
    train_data,
    batch_size=BATCH_SIZE,
    shuffle=True,
)

# validation_loader = torch.utils.data.DataLoader(
#     validation_data, batch_size=BATCH_SIZE, shuffle=True
# )

test_loader = torch.utils.data.DataLoader(
    test_data, batch_size=BATCH_SIZE, shuffle=True
)

# MAKE DUMMY SMALL DATA FOR HYPERPARAMETER EXPLORATION
dummy_data = [face_age_data[i] for i in range(24)]

dummy_train, dummy_validation, dummy_test = torch.utils.data.random_split(
    dummy_data,
    [
        16,
        4,
        4,
    ],
)

dummy_train = torch.utils.data.DataLoader(dummy_train, batch_size=4, shuffle=True)
dummy_validation = torch.utils.data.DataLoader(
    dummy_validation, batch_size=4, shuffle=True
)
dummy_test = torch.utils.data.DataLoader(dummy_test, batch_size=4, shuffle=True)
