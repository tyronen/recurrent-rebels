import torch
import torch.nn as nn
import numpy as np
import math
from typing import List, Union, Tuple, Dict, Any
import os



class TitleRegressionNN(nn.Module):
    """Neural network for predicting Hacker News story scores from titles."""
    
    def __init__(self, embedding_layer: nn.Module, embedding_dim: int, hidden_dim: int = 512):
        super().__init__()
        # Use the pretrained embedding weights from the CBOW model
        self.embeddings = embedding_layer
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 1)  # Output is a single numeric value

    def forward(self, input_indices: torch.Tensor) -> torch.Tensor:
        # input_indices: (batch_size, seq_len)
        embeds = self.embeddings(input_indices)  # (batch_size, seq_len, embedding_dim)
        avg_embeds = embeds.mean(dim=1)         # (batch_size, embedding_dim)
        x = self.fc1(avg_embeds)
        x = self.relu(x)
        x = self.fc2(x)
        return x.squeeze(-1)  # (batch_size,)


class HackerNewsScorePredictor:
    """Complete model wrapper for predicting Hacker News story scores."""
    
    def __init__(self, device: str = None):
        self.device = device if device else self._get_device()
        self.model = None
        self.word2idx = None
        self.idx2word = None
        self.embedding_dim = 100
        self.average_score = 1.5576210926494947  # From training data
        
    def _get_device(self) -> str:
        """Automatically detect the best available device."""
        if torch.backends.mps.is_available():
            return "mps"
        elif torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"
    
    def load_model(self, cbow_checkpoint_path: str = "title_regression_model/cbow_final_with_vocab.pt", 
                   regression_model_path: str = "title_regression_model/regression_model.pt") -> None:
        """
        Load the trained model from checkpoint files.
        
        Args:
            cbow_checkpoint_path: Path to the CBOW checkpoint with vocabulary
            regression_model_path: Optional path to regression model checkpoint
        """
        # Load CBOW checkpoint with vocabulary
        if not os.path.exists(cbow_checkpoint_path):
            raise FileNotFoundError(f"CBOW checkpoint not found: {cbow_checkpoint_path}")
            
        checkpoint = torch.load(cbow_checkpoint_path, map_location=torch.device(self.device))
        
        self.word2idx = checkpoint['word2idx']
        self.idx2word = checkpoint['idx2word']
        
        vocab_size = len(self.word2idx) 
        
        # Extract embedding weights directly from CBOW checkpoint
        cbow_state_dict = checkpoint['model_state_dict']
        embedding_weights = cbow_state_dict['embeddings.weight']
        
        # Create embedding layer with pretrained weights
        embedding_layer = nn.Embedding(vocab_size, self.embedding_dim, max_norm=1.0)
        embedding_layer.weight.data.copy_(embedding_weights)
        
        # Initialize regression model with pretrained embeddings
        self.model = TitleRegressionNN(embedding_layer, self.embedding_dim)
        
        # Load regression model weights if provided
        if regression_model_path and os.path.exists(regression_model_path):
            regression_checkpoint = torch.load(regression_model_path, map_location=torch.device(self.device))
            self.model.load_state_dict(regression_checkpoint['model_state_dict'])
        
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Model loaded successfully on {self.device}")
        print(f"Vocabulary size: {vocab_size}")
    
    def _preprocess_title(self, title: str) -> List[int]:
        """
        Convert a title string to word indices.
        
        Args:
            title: Raw title string
            
        Returns:
            List of word indices
        """
        if not isinstance(title, str):
            return []
            
        words = title.lower().split()
        indices = [self.word2idx[w] for w in words if w in self.word2idx]
        return indices
    
    def predict_single(self, title: str) -> Dict[str, Any]:
        """
        Predict score for a single title.
        
        Args:
            title: Title string to predict score for
            
        Returns:
            Dictionary with prediction results
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        indices = self._preprocess_title(title)
        
        if not indices:
            # If no words are in vocabulary, return average score
            return {
                'title': title,
                'predicted_log_score': self.average_score,
                'predicted_score': math.exp(self.average_score) - 1,
                'words_in_vocab': 0,
                'total_words': len(title.split()),
                'coverage': 0.0
            }
        
        # Convert to tensor and add batch dimension
        input_tensor = torch.tensor([indices], dtype=torch.long).to(self.device)
        
        with torch.no_grad():
            log_score_pred = self.model(input_tensor).cpu().item()
        
        # Convert back from log space
        score_pred = math.exp(log_score_pred) - 1
        
        total_words = len(title.split())
        coverage = len(indices) / total_words if total_words > 0 else 0.0
        
        return {
            'title': title,
            'predicted_log_score': log_score_pred,
            'predicted_score': max(0, score_pred),  # Ensure non-negative
            'words_in_vocab': len(indices),
            'total_words': total_words,
            'coverage': coverage
        }
    
    def predict_batch(self, titles: List[str]) -> List[Dict[str, Any]]:
        """
        Predict scores for multiple titles.
        
        Args:
            titles: List of title strings
            
        Returns:
            List of prediction dictionaries
        """
        return [self.predict_single(title) for title in titles]
    
    def get_word_embedding(self, word: str) -> np.ndarray:
        """
        Get the embedding vector for a specific word.
        
        Args:
            word: Word to get embedding for
            
        Returns:
            Embedding vector as numpy array
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        word_lower = word.lower()
        if word_lower not in self.word2idx:
            raise ValueError(f"Word '{word}' not in vocabulary")
        
        word_idx = self.word2idx[word_lower]
        with torch.no_grad():
            embedding = self.model.embeddings.weight[word_idx].cpu().numpy()
        
        return embedding
    
    def get_similar_words(self, word: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Find words with similar embeddings.
        
        Args:
            word: Target word
            top_k: Number of similar words to return
            
        Returns:
            List of (word, similarity_score) tuples
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        try:
            target_embedding = self.get_word_embedding(word)
        except ValueError:
            return []
        
        similarities = []
        with torch.no_grad():
            embeddings = self.model.embeddings.weight.cpu().numpy()
            
            # Compute cosine similarities
            target_norm = np.linalg.norm(target_embedding)
            
            for idx, embedding in enumerate(embeddings):
                if idx == self.word2idx[word.lower()]:
                    continue  # Skip the word itself
                
                embedding_norm = np.linalg.norm(embedding)
                if embedding_norm > 0 and target_norm > 0:
                    similarity = np.dot(target_embedding, embedding) / (target_norm * embedding_norm)
                    similarities.append((self.idx2word[idx], similarity))
        
        # Sort by similarity and return top_k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def save_regression_model(self, path: str) -> None:
        """
        Save the regression model state.
        
        Args:
            path: Path to save the model
        """
        if self.model is None:
            raise RuntimeError("Model not loaded.")
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'embedding_dim': self.embedding_dim,
            'device': self.device
        }, path)
        print(f"Regression model saved to {path}")


def load_predictor(cbow_checkpoint_path: str, regression_model_path: str = None, 
                  device: str = None) -> HackerNewsScorePredictor:
    """
    Convenience function to load a complete predictor.
    
    Args:
        cbow_checkpoint_path: Path to CBOW checkpoint
        regression_model_path: Optional path to regression model
        device: Device to use ('cpu', 'cuda', 'mps', or None for auto)
        
    Returns:
        Loaded HackerNewsScorePredictor instance
    """
    predictor = HackerNewsScorePredictor(device=device)
    predictor.load_model(cbow_checkpoint_path, regression_model_path)
    return predictor


# Example usage
if __name__ == "__main__":
    # Example of how to use the model
    predictor = load_predictor("cbow_final_with_vocab.pt")
    
    # Test predictions
    test_titles = [
        "Show HN: I built a new programming language",
        "Google announces new AI breakthrough",
        "Why I left my job at Facebook",
        "The future of cryptocurrency",
        "Building a startup in 2024"
    ]
    
    print("Predictions:")
    print("-" * 60)
    for result in predictor.predict_batch(test_titles):
        print(f"Title: {result['title']}")
        print(f"Predicted Score: {result['predicted_score']:.1f}")
        print(f"Vocabulary Coverage: {result['coverage']:.1%}")
        print("-" * 60) 