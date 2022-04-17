import logging

from src.data import train_loader, test_loader
from src.current_model import model

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, filename="resnet_age_trainer.log")
    model.fit(train_loader, test_loader, num_epochs=100, model_name="resnet_age",)
