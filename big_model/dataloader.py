import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import re


def time_features(value):
    # Convert timestamp to multiple features
    timestamp = pd.to_datetime(value)

    # Year as regular feature
    time_features  = [timestamp.year]

    # Hour as circular features (0-23)
    hour_angle = 2 * np.pi * timestamp.hour / 24
    dow_angle = 2 * np.pi * timestamp.dayofweek / 7
    day_angle = 2 * np.pi * timestamp.dayofyear / 365
    time_features.extend([
        np.sin(hour_angle),
        np.cos(hour_angle),
        np.sin(dow_angle),
        np.cos(dow_angle),
        np.sin(day_angle),
        np.cos(day_angle)
    ])
    return time_features


def extract_features(data):
    # process features as necessary, anything that is not text 
    
    #extract time features
    feature_values = []
    
    for col_name, value in data.items():
        if col_name == 'time':
            feature_values.extend(time_features(value))
        else:
            feature_values.append(value)
    
    return torch.tensor(feature_values, dtype=torch.float32)


class PostDataset(Dataset):
    def __init__(self, dataframe, embedding_matrix, w2i_dict):
        """
        dataframe: pandas DataFrame with user features, title (raw text), and score
        embedding_matrix: torch.Tensor of shape (vocab_size, embedding_dim)
        w2i_dict: dictionary mapping words to indices
        """
        self.df = dataframe
        self.embedding_matrix = embedding_matrix
        self.w2i_dict = w2i_dict
        
        # Select feature columns
        print(dataframe.dtypes)
        self.feature_cols = [col for col in dataframe.columns if col not in ['title', 'score', 'url']]


    def __len__(self):
        return len(self.df)
    
    def tokenize_title(self, title_text):
        """
        Converts raw title text into list of token indices using w2i_dict.
        Unknown words get index 0 (like your CBOW model does).
        """
        tokens = re.findall(r'[a-zA-Z]+', title_text.lower())  # split along any non-alphabetic characters
        token_indices = [self.w2i_dict.get(token, 0) for token in tokens]
        return token_indices
    
    def embed_title(self, title_tokens):
        """
        Given token indices, return embedded tensor.
        """
        token_indices = torch.tensor(title_tokens, dtype=torch.long)
        embedded = self.embedding_matrix[token_indices]  # shape: (seq_len, embedding_dim)
        avg_embedding = embedded.mean(dim=0)  # (embed_dim,) we currently take the mean in the embedding 
        return avg_embedding
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Features
        if self.feature_cols:
            data = row[self.feature_cols]
            features = extract_features(data) #return tensor
        else:
            features = torch.tensor([], dtype=torch.float32)

        # Title embedding
        title_tokens = self.tokenize_title(row['title'])
        title_embedding = self.embed_title(title_tokens)

        # Concatenate user features and averaged title embedding
        x = torch.cat([features, title_embedding], dim=0)

        # Target score
        score = torch.clamp(torch.tensor(row['score'], dtype=torch.float32), min=0)
        y = torch.log(score + 1)

        return x, y
