import logging

# Load model first to set env variable before
# importing torchvision.
from src.current_training_model import training_model
from src.data import train_loader, test_loader
from src.helper_func import fit, score

if __name__ == "__main__":
    # Log to Console AND to file
    MODEL_NAME = "resnet_age"
    logging.basicConfig(
        level=logging.DEBUG,
        handlers=[logging.FileHandler(MODEL_NAME + ".log"), logging.StreamHandler()],
    )
    try:
        logging.debug("Beginning Model Training.")
        fit(
            training_model,
            train_loader,
            test_loader,
            num_epochs=100,
            model_name=MODEL_NAME,
        )
        logging.debug("Concluding Model Training.")
    except KeyboardInterrupt:
        logging.warning("Exiting Training Cycle due to Keyboard Interrupt.")
        logging.warning("Continuing with Model Scoring/Evaluation.")
    finally:
        logging.debug("Beginning Model Evaluation.")
        score(training_model, train_loader, test_loader, model_name=MODEL_NAME)
        logging.debug("Concluding Model Evaluation.")
