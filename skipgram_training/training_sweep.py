import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import numpy as np
import random
from tqdm import tqdm
import math
import os
import json
import wandb

# =============================================================================
# Step 1: Define the W&B Sweep Configuration
# =============================================================================
# This dictionary defines the hyperparameter search space and strategy.
# The `wandb.agent` will use this to run different trials.
sweep_config = {
    'method': 'bayes',  # Can be 'random', 'grid', or 'bayes'
    'metric': {
        'name': 'validation_loss',
        'goal': 'minimize'
    },
    'parameters': {
        'embedding_dim': {
            'values': [100, 200, 500]
        },
        'learning_rate': {
            'distribution': 'log_uniform',
            'min': 1e-4,
            'max': 1e-2
        },
        'window_size': {
            'values': [2, 4]
        },
        'neg_samples': {
            'values': [5, 10, 15]
        },
        'batch_size': {
            'values': [512, 1024, 2048]
        }
    }
}


# =============================================================================
# Main Training Function (to be called by W&B agent)
# =============================================================================
def train():
    # Initialize a new W&B run
    wandb.init()
    config = wandb.config

    # --- Configuration & Hyperparameters from W&B ---
    DATA_FILE = 'text8'
    MIN_FREQ = 50
    VALIDATION_SPLIT = 0.05

    EMBEDDING_DIM = config.embedding_dim
    WINDOW_SIZE = config.window_size
    NEG_SAMPLES = config.neg_samples
    
    BATCH_SIZE = config.batch_size
    LEARNING_RATE = config.learning_rate
    EPOCHS = 5 # Set a max number of epochs per run
    
    # Use run-specific filenames to avoid conflicts
    CHECKPOINT_FILE = f'word2vec_checkpoint_{wandb.run.id}.pt'
    BEST_MODEL_FILE = f'word2vec_best_model_{wandb.run.id}.pt'
    
    EVALUATION_STEP = 2000
    EARLY_STOPPING_PATIENCE = 3

    # --- Data Preparation ---
    print("Starting Data Preparation...")
    with open(DATA_FILE, 'r') as f:
        text = f.read().split()
    word_counts = Counter(text)
    text = [word for word in text if word_counts[word] > MIN_FREQ]
    vocab = Counter(text)
    int2word = {i: word for i, word in enumerate(vocab)}
    word2int = {word: i for i, word in int2word.items()}
    int_words = [word2int[word] for word in text]
    vocab_size = len(vocab)
    print(f"Vocabulary size: {vocab_size}")

    split_idx = int(len(int_words) * (1 - VALIDATION_SPLIT))
    train_words = int_words[:split_idx]
    valid_words = int_words[split_idx:]

    class SkipGramDataset(Dataset):
        def __init__(self, text, window_size, vocab_size):
            self.text = text
            self.window_size = window_size
            self.vocab_size = vocab_size
            word_counts = Counter(text)
            freqs = np.zeros(vocab_size)
            for word_idx, count in word_counts.items():
                if word_idx < vocab_size: freqs[word_idx] = count
            freqs_pow = freqs**0.75
            self.word_dist_tensor = torch.from_numpy(freqs_pow / np.sum(freqs_pow)).float()
        def __len__(self): return len(self.text)
        def __getitem__(self, idx):
            center_word = self.text[idx]
            dynamic_window = random.randint(1, self.window_size)
            start_idx = max(0, idx - dynamic_window)
            end_idx = min(len(self.text), idx + dynamic_window + 1)
            possible_context_indices = [i for i in range(start_idx, end_idx) if i != idx]
            context_word = self.text[random.choice(possible_context_indices) if possible_context_indices else idx]
            neg_samples = torch.multinomial(self.word_dist_tensor, NEG_SAMPLES, replacement=True)
            return torch.tensor(center_word), torch.tensor(context_word), neg_samples

    train_dataset = SkipGramDataset(train_words, WINDOW_SIZE, vocab_size)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    valid_dataset = SkipGramDataset(valid_words, WINDOW_SIZE, vocab_size)
    valid_dataloader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, num_workers=2)

    # --- Model Definition ---
    class SkipGramNegativeSampling(nn.Module):
        def __init__(self, vocab_size, embed_dim):
            super().__init__()
            self.in_embed = nn.Embedding(vocab_size, embed_dim)
            self.out_embed = nn.Embedding(vocab_size, embed_dim)
            self.in_embed.weight.data.uniform_(-1, 1)
            self.out_embed.weight.data.uniform_(-1, 1)
        def forward(self, center, context, neg):
            c_emb = self.in_embed(center).unsqueeze(2)
            p_emb = self.out_embed(context).unsqueeze(1)
            n_emb = self.out_embed(neg)
            p_score = torch.bmm(p_emb, c_emb).squeeze(2)
            n_score = torch.bmm(n_emb, c_emb).squeeze(2)
            return p_score, n_score

    class NegativeSamplingLoss(nn.Module):
        def __init__(self): super().__init__()
        def forward(self, pos_scores, neg_scores):
            return - (torch.log(torch.sigmoid(pos_scores)) + torch.log(torch.sigmoid(-neg_scores)).sum(1)).mean()

    model = SkipGramNegativeSampling(vocab_size, EMBEDDING_DIM)
    criterion = NegativeSamplingLoss()

    # --- W&B Watch ---
    # This will log gradients and model topology
    wandb.watch(model, criterion, log="all", log_freq=1000)

    # --- Training Loop ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    global_step = 0
    best_validation_loss = float('inf')
    patience_counter = 0
    early_stop = False

    for epoch in range(EPOCHS):
        model.train()
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for center_words, context_words, neg_samples in progress_bar:
            center_words, context_words, neg_samples = center_words.to(device), context_words.to(device), neg_samples.to(device)

            optimizer.zero_grad()
            pos_scores, neg_scores = model(center_words, context_words, neg_samples)
            loss = criterion(pos_scores, neg_scores)
            loss.backward()
            optimizer.step()
            
            global_step += 1
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
            wandb.log({"train_loss": loss.item(), "global_step": global_step})
            
            if global_step % EVALUATION_STEP == 0:
                model.eval()
                total_val_loss = 0
                with torch.no_grad():
                    for val_data in valid_dataloader:
                        val_center, val_context, val_neg = (d.to(device) for d in val_data)
                        pos, neg = model(val_center, val_context, val_neg)
                        total_val_loss += criterion(pos, neg).item()
                
                avg_val_loss = total_val_loss / len(valid_dataloader)
                wandb.log({"validation_loss": avg_val_loss, "epoch": epoch})
                print(f"\nStep {global_step}: Validation Loss: {avg_val_loss:.4f}")

                if avg_val_loss < best_validation_loss:
                    best_validation_loss = avg_val_loss
                    patience_counter = 0
                    wandb.run.summary["best_validation_loss"] = best_validation_loss
                    torch.save({'model_state_dict': model.state_dict()}, BEST_MODEL_FILE)
                else:
                    patience_counter += 1
                
                if patience_counter >= EARLY_STOPPING_PATIENCE:
                    early_stop = True
                    break
                model.train()
        if early_stop: break

    # --- Save final artifacts to W&B ---
    if os.path.exists(BEST_MODEL_FILE):
        best_model_artifact = wandb.Artifact("best-word2vec-model", type="model")
        best_model_artifact.add_file(BEST_MODEL_FILE)
        wandb.log_artifact(best_model_artifact)

# =============================================================================
# Step 4: Initialize the Sweep
# =============================================================================
if __name__ == '__main__':
    # To run this:
    # 1. In your terminal, run `wandb login` and paste your API key.
    # 2. Run this python script: `python your_script_name.py`
    
    # Create the sweep
    sweep_id = wandb.sweep(sweep_config, project="word2vec-pytorch-sweep")

    # Start the sweep agent
    # The `count` parameter specifies how many trials to run.
    wandb.agent(sweep_id, train, count=10)
