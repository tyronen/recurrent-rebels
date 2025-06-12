from urllib.parse import urlparse
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from utils import log_transform_plus1, time_transform
import tldextract

def extract_domain_tld(url_text: str):
    """
    Return (domain, tld) using the PSL via tldextract.
    Example: 'https://sub.example.co.uk/page'
             → ('sub.example', 'co.uk')
    """
    ext = tldextract.extract(url_text.lower())
    if not ext.domain:          # no hostname in the URL
        return '', ''
    domain = '.'.join(label for label in (ext.subdomain, ext.domain) if label)
    tld    = ext.suffix         # 'co.uk', 'com', 'dev', etc.
    return domain, tld


def extract_features(data, no_of_features=9):

    features = np.zeros(no_of_features)

    for col_name, value in data.items():
        if col_name == 'time':

            year, hour_angle, dow_angle, day_angle = time_transform(value, offset=2006)
            # for clarity downwards instead of one line
            features[0] = year / (2023 - 2006) # divide by max difference for easier values, need to make this a variable
            features[1] = np.sin(hour_angle)
            features[2] = np.cos(hour_angle)
            features[3] = np.sin(dow_angle)
            features[4] = np.cos(dow_angle)
            features[5] = np.sin(day_angle)
            features[6] = np.cos(day_angle)
        
        elif col_name == 'length_submitted':
            features[7] = log_transform_plus1(value)
        
        elif col_name == 'story_count':
            features[8] = log_transform_plus1(value) 
    
    return torch.tensor(features, dtype=torch.float32)


class PostDataset(Dataset):
    def __init__(self, dataframe, embedding_matrix, w2i_dict, ref_time=None, lambda_=None):
        """
        dataframe: pandas DataFrame with user features, title (raw text), and score
        embedding_matrix: torch.Tensor of shape (vocab_size, embedding_dim)
        w2i_dict: dictionary mapping words to indices
        """
        self.df = dataframe
        self.embedding_matrix = embedding_matrix
        self.w2i_dict = w2i_dict
        
        if ref_time is not None:
            self.ref_time = ref_time
            self.lambda_ = lambda_
            self.df['time'] = pd.to_datetime(self.df['time'])
            delta_t = (self.ref_time - self.df['time']) / np.timedelta64(30, 'D')  # months
            self.weights = np.exp(-self.lambda_ * delta_t)
        else:
            self.weights = None

        self.feature_cols = [col for col in dataframe.columns if col in ["time", "length_submitted", "story_count"]]

    def __len__(self):
        return len(self.df)
    
    def tokenize_title(self, title_text):
        """
        Converts raw title text into list of token indices using w2i_dict.
        Unknown words get index 0 (like your CBOW model does).
        """
        tokens = title_text.lower().split()  # simple whitespace tokenizer
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
        features = extract_features(data=row[self.feature_cols]) #return tensor
            
        # Title embedding
        title_tokens = self.tokenize_title(row['title'])
        title_embedding = self.embed_title(title_tokens)

        x = torch.cat([features, title_embedding], dim=0)

        # Target score
        score = torch.clamp(torch.tensor(row['score'], dtype=torch.float32), min=0)
        y = torch.log10(score + 1)

        if self.weights is not None:
            weight = self.weights.iloc[idx]
            return x, y, weight
        else:
            return x, y


class PrecomputedNPZDataset(Dataset):
    def __init__(self, npz_path, time_decay=None):
        data = np.load(npz_path)

        self.features = torch.tensor(data['features'], dtype=torch.float32)
        self.embeddings = torch.tensor(data['embeddings'], dtype=torch.float32)
        self.targets = torch.log10(torch.tensor(data['targets'], dtype=torch.float32) + 1)

        self.has_delta_t = 'delta_t' in data.files
        self.time_decay = time_decay

        if self.has_delta_t:
            self.delta_t = torch.tensor(data['delta_t'], dtype=torch.float32)

            if self.time_decay is not None:
                self.weights = torch.exp(-self.time_decay * self.delta_t)
            else:
                self.weights = None
        else:
            self.delta_t = None
            self.weights = None

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        x = torch.cat([self.features[idx], self.embeddings[idx]], dim=0)
        y = self.targets[idx]

        if self.weights is not None:
            w = self.weights[idx]
            return x, y, w
        else:
            return x, y


