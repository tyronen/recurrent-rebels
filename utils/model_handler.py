import pickle

import torch

from big_model.inference_preprocess import CACHE_FILE
from big_model.model import FullModel
from big_model import utils
import json
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S"
)

def load_user_data():
    with open(CACHE_FILE, "rb") as fh:
        cache = pickle.load(fh)

    # propagate globals so utils.process_row has the right reference frame
    utils.global_Tmin = cache["global_Tmin"]
    utils.global_Tmax = cache["global_Tmax"]

    logging.info(
        f"Loaded inference cache with {len(cache['user_features'])} users "
        f"from {CACHE_FILE}"
    )
    return cache["columns"], cache["user_features"]


def get_full_model_preprocessor():
    """Load necessary data"""
    w2i, embedding_matrix = utils.load_embeddings("skipgram_models/silvery200.pt")
    utils.global_w2i = w2i
    utils.global_embedding_matrix = embedding_matrix
    # Load vocab sizes from vocab file
    with open(utils.TRAINING_VOCAB_PATH, 'r') as f:
        vocabs = json.load(f)
    
    utils.global_domain_vocab = vocabs['domain_vocab']
    utils.global_tld_vocab = vocabs['tld_vocab']
    utils.global_user_vocab = vocabs['user_vocab']

def load_full_model(model_path: str) -> FullModel:
    """Load a FullModel from checkpoint"""
    device = utils.get_device()

    # Create model with correct parameters
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint["config"]
    model = FullModel(**config)

    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model


class Predictor:
    def __init__(self, model):
        self.model = model
        self.columns, self.user_features = load_user_data()

    def preprocess_input(self, data: dict) -> list[float]:
        # Get user features from memory (instant lookup)
        username = data['by']
        if username in self.user_features:
            row = self.user_features[username]
        else:
            # New user - all zeros
            row = {col: 0 for col in self.columns}

        row['by'] = data['by']
        row['title'] = data['title']
        row['url'] = data['url']
        row['time'] = data['time']
        return row

    def predict(self, input_data: dict) -> float:
        features_vec = self.preprocess_input(input_data)
        data = utils.process_row(features_vec)
        features_num = torch.tensor(data['features_num'], dtype=torch.float32)
        # Load title embeddings (precomputed)
        title_embeddings = torch.tensor(data['embedding'], dtype=torch.float32)

        # Load categorical indices
        domain_indices = torch.tensor(data['domain_idx'], dtype=torch.long)
        tld_indices = torch.tensor(data['tld_idx'], dtype=torch.long)
        user_indices = torch.tensor(data['user_idx'], dtype=torch.long)

        #May need to modify if model is not preprocessed
        with torch.no_grad():
            prediction = 10 ** self.model(features_num, title_embeddings, domain_indices, tld_indices, user_indices) - 1

        return prediction.item()

def get_predictor(model_path: str):
    model = load_full_model(model_path)
    get_full_model_preprocessor()
    return Predictor(model)
