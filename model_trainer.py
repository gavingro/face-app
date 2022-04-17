import logging

from src.data import train_loader, test_loader
from src.current_model import model
from src.helper_func import fit, score

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, filename="resnet_age_trainer.log")
    fit(model, train_loader, test_loader, num_epochs=100, model_name="resnet_age")
    score(model, train_loader, test_loader, model_name="resnet_age")
