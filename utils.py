import pickle
from models.MFModel import MFModel_light
def load_model(model: MFModel_light, model_path:str):
    with open(model_path, "rb") as f:
        hiddens = pickle.load(f)
    
    model.load_hiddens(hiddens["user"], hiddens["item"])