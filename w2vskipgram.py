import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import numpy as np
import random
from tqdm import tqdm
import math

# =============================================================================
# Configuration & Hyperparameters
# =============================================================================
# Data parameters
DATA_FILE = 'text8'
MIN_FREQ = 50       # Minimum word frequency to be included in vocabulary

# Model parameters
EMBEDDING_DIM = 300 # Dimensionality of the word embeddings
WINDOW_SIZE = 5     # Max context window size (words on each side of the center word)
NEG_SAMPLES = 5     # Number of negative samples for each positive sample

# Training parameters
BATCH_SIZE = 1024
LEARNING_RATE = 0.001
EPOCHS = 5

# =============================================================================
# Step 1: Data Preparation
# =============================================================================
print("Starting Step 1: Data Preparation...")

def preprocess(file_path, min_freq):
    """
    Reads text data, builds a vocabulary, and converts words to integer indices.
    """
    print("Reading and processing text data...")
    with open(file_path, 'r') as f:
        text = f.read().split()

    print(f"Original corpus has {len(text)} tokens.")

    # Count word frequencies and filter out rare words
    word_counts = Counter(text)
    # Remove words with frequency less than MIN_FREQ
    text = [word for word in text if word_counts[word] > min_freq]
    print(f"Corpus after filtering rare words has {len(text)} tokens.")

    # Build vocabulary
    vocab = Counter(text)
    int2word = {i: word for i, word in enumerate(vocab)}
    word2int = {word: i for i, word in int2word.items()}
    
    # Convert the entire text to integer indices
    int_words = [word2int[word] for word in text]
    
    return int_words, vocab, int2word, word2int

# Load and preprocess the data
int_words, vocab, int2word, word2int = preprocess(DATA_FILE, MIN_FREQ)
vocab_size = len(vocab)
print(f"Vocabulary size: {vocab_size}")


class SkipGramDataset(Dataset):
    """
    Optimized, memory-efficient PyTorch Dataset for Word2Vec.
    It generates skip-gram pairs on-the-fly instead of pre-generating them.
    """
    def __init__(self, text, window_size, vocab_size):
        self.text = text
        self.window_size = window_size
        self.vocab_size = vocab_size

        # For negative sampling, we need a distribution of word frequencies.
        # This distribution is based on the unigram frequency raised to the 3/4 power.
        word_counts = Counter(text)
        # Create a frequency array where the index corresponds to the word's integer id.
        freqs = np.zeros(vocab_size)
        for word_idx, count in word_counts.items():
            freqs[word_idx] = count
        
        freqs_pow = freqs**0.75
        self.word_dist = freqs_pow / np.sum(freqs_pow)
        self.word_dist_tensor = torch.from_numpy(self.word_dist).float()
        
        print("Memory-efficient SkipGramDataset created.")

    def __len__(self):
        # The length of the dataset is the number of words in our corpus.
        # Each word will be treated as a center word.
        return len(self.text)

    def __getitem__(self, idx):
        """
        Returns one (center_word, context_word) pair and negative samples.
        The context word is chosen randomly from the window of the center word.
        """
        center_word = self.text[idx]
        
        # Use a dynamic window size (subsampling) for each call, as in the original Word2Vec paper.
        # This gives more weight to closer words.
        dynamic_window = random.randint(1, self.window_size)
        
        # Get all possible context word indices within the dynamic window
        start_idx = max(0, idx - dynamic_window)
        end_idx = min(len(self.text), idx + dynamic_window + 1)
        possible_context_indices = [i for i in range(start_idx, end_idx) if i != idx]
        
        # Choose one context word randomly from the possible contexts
        # Handle the rare edge case where a word has no context
        if not possible_context_indices:
            context_word_idx = idx # Fallback to the word itself
        else:
            context_word_idx = random.choice(possible_context_indices)
        
        context_word = self.text[context_word_idx]

        # Generate negative samples using the pre-calculated distribution
        neg_samples = torch.multinomial(self.word_dist_tensor,
                                        NEG_SAMPLES, replacement=True)

        return torch.tensor(center_word), torch.tensor(context_word), neg_samples

# Create dataset and dataloader
dataset = SkipGramDataset(int_words, WINDOW_SIZE, vocab_size)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
print("Data Preparation complete.")

# =============================================================================
# Step 2: Model Definition
# =============================================================================
print("\nStarting Step 2: Model Definition...")

class SkipGramNegativeSampling(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super(SkipGramNegativeSampling, self).__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        
        # Embedding for center words
        self.in_embed = nn.Embedding(vocab_size, embed_dim)
        # Embedding for context words (and negative samples)
        self.out_embed = nn.Embedding(vocab_size, embed_dim)

        # Initialize embeddings with a uniform distribution
        self.in_embed.weight.data.uniform_(-1, 1)
        self.out_embed.weight.data.uniform_(-1, 1)

    def forward(self, center_words, context_words, neg_samples):
        """
        Forward pass for the skip-gram model with negative sampling.
        """
        center_embeds = self.in_embed(center_words) # (batch_size, embed_dim)
        context_embeds = self.out_embed(context_words) # (batch_size, embed_dim)
        neg_embeds = self.out_embed(neg_samples) # (batch_size, num_neg_samples, embed_dim)

        # Reshape for batch matrix multiplication
        center_embeds = center_embeds.unsqueeze(2) # (batch_size, embed_dim, 1)
        
        # Positive scores
        context_embeds = context_embeds.unsqueeze(1) # (batch_size, 1, embed_dim)
        pos_scores = torch.bmm(context_embeds, center_embeds).squeeze(2) # (batch_size, 1)
        
        # Negative scores
        neg_scores = torch.bmm(neg_embeds, center_embeds).squeeze(2) # (batch_size, num_neg_samples)
        
        return pos_scores, neg_scores

class NegativeSamplingLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pos_scores, neg_scores):
        """
        Calculates the negative sampling loss.
        """
        pos_loss = torch.log(torch.sigmoid(pos_scores))
        neg_loss = torch.log(torch.sigmoid(-neg_scores)).sum(1)

        return - (pos_loss + neg_loss).mean()

# Instantiate the model and loss function
model = SkipGramNegativeSampling(vocab_size, EMBEDDING_DIM)
criterion = NegativeSamplingLoss()
print("Model Definition complete.")

# =============================================================================
# Step 3: Training
# =============================================================================
print("\nStarting Step 3: Training...")

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model.to(device)
criterion.to(device)

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

for epoch in range(EPOCHS):
    total_loss = 0
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
    for center_words, context_words, neg_samples in progress_bar:
        # Move data to the appropriate device
        center_words, context_words, neg_samples = center_words.to(device), context_words.to(device), neg_samples.to(device)

        optimizer.zero_grad()
        pos_scores, neg_scores = model(center_words, context_words, neg_samples)
        loss = criterion(pos_scores, neg_scores)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1}/{EPOCHS}, Average Loss: {avg_loss:.4f}")

print("Training complete.")

# =============================================================================
# Step 4: Save Embeddings
# =============================================================================
print("\nStarting Step 4: Saving Embeddings...")

# We are interested in the input embeddings
embeddings = model.in_embed.weight.cpu().data

torch.save(embeddings, 'skipgram_embeddings.pt')

import json
with open('word2int.json', 'w') as f:
    json.dump(word2int, f)
with open('int2word.json', 'w') as f:
    json.dump(int2word, f)

print("Embeddings and vocabulary saved successfully.")
print("File 'skipgram_embeddings.pt' created.")
