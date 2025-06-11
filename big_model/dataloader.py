import torch
from torch.utils.data import Dataset


def extract_features(arr):
    # process features as necessary, anything that is not text 
    return arr 


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
        self.feature_cols = [col for col in dataframe.columns if col not in ['title', 'score']]

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
        if self.feature_cols:
            data = torch.tensor(row[self.feature_cols].values, dtype=torch.float32)
            features = extract_features(data)
        else:
            features = torch.tensor([], dtype=torch.float32)

        # Title embedding
        title_tokens = self.tokenize_title(row['title'])
        title_embedding = self.embed_title(title_tokens)

        # Concatenate user features and averaged title embedding
        x = torch.cat([features, title_embedding], dim=0)

        # Target score
        y = torch.log(torch.tensor(row['score'] + 1, dtype=torch.float32))

        return x, y
