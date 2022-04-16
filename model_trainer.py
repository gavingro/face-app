from src.data import train_loader
from src.current_model import model

model.fit(train_loader, num_epochs=5, model_name="resnet_age1")
