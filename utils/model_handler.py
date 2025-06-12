import torch
from big_model.model import FullModel
from big_model.single_inference import prepare_features
from big_model.utils import load_embeddings, get_device
import json


def get_full_model_preprocessor():
    """Create a preprocessor function with loaded dependencies"""
    # Load necessary data
    w2i, embedding_matrix = load_embeddings("skipgram_models/silvery200.pt")
    
    # Load vocab sizes from vocab file
    with open("data/train_vocab.json", 'r') as f:
        vocabs = json.load(f)
    
    domain_to_idx = vocabs['domain_vocab']
    tld_to_idx = vocabs['tld_vocab']
    user_to_idx = vocabs['user_vocab']
    
    def preprocess(post_dict):
        return prepare_features(post_dict, w2i, embedding_matrix, domain_to_idx, tld_to_idx, user_to_idx)
    
    return preprocess

def load_full_model(model_path: str) -> FullModel:
    """Load a FullModel from checkpoint"""
    device = get_device()
    
    # Load vocab sizes from vocab file
    with open("data/train_vocab.json", 'r') as f:
        vocabs = json.load(f)
    
    # Create model with correct parameters
    model = FullModel(
        vector_size_num=0,  # Set to 0 to match checkpoint dimensions
        vector_size_title=200,
        scale=3,
        domain_vocab_size=len(vocabs['domain_vocab']),
        tld_vocab_size=len(vocabs['tld_vocab']),
        user_vocab_size=len(vocabs['user_vocab'])
    ).to(device)

    # Load model weights
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model

class Predictor:
    def __init__(self, model, preprocessing_fn: callable):
        self.model = model
        self.preprocessing_fn = preprocessing_fn

    def predict(self, input_data: dict) -> float:
        processed_data = self.preprocessing_fn(input_data)
        #May need to modify if model is not preprocessed
        with torch.no_grad():
            prediction = 10 ** self.model(*processed_data) - 1

        return prediction.item()

def get_predictor(model_name: str):
    match model_name:
        case "full_model":
            model = load_full_model("big_model/models/20250612_190305/best_model_1.pth")
            return Predictor(model, get_full_model_preprocessor())
        case _:
            raise ValueError(f"Model {model_name} not found")
