import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import pandas as pd
import numpy as np

from dataloader import PostDataset  
from model import BigModel

import argparse
import os
from datetime import datetime
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser(description="Train the master NN.")
parser.add_argument('--items', default="data/items.parquet", help='Items data frame')
parser.add_argument('--embeddings', default="skipgram_models/silvery200.pt", help='Word2Vec embeddings')
parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
parser.add_argument("--epochs", type=int, default=10, help="Epochs")
parser.add_argument("--lr", type=float, default=1e-3, help="Batch size")
args = parser.parse_args()

# Hyperparameters
BATCH_SIZE = args.batch_size
EPOCHS = args.epochs
LEARNING_RATE = args.lr


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


DEVICE = get_device()


def load_data():
    rawdf = pd.read_parquet(args.items)
    has_score = rawdf.dropna(subset=["score"])
    has_title = has_score[has_score["title"].notnull()]
    has_title = has_title[has_title["title"].str.strip().astype(bool)]  # drop empty or whitespace-only
    numeric = has_title.select_dtypes(include=[np.number])
    return pd.concat([has_title[["title"]], numeric.drop(columns=["id"])], axis=1)

def load_embeddings():
    efile = torch.load(args.embeddings)
    embeddings = efile["embeddings"]
    word_to_ix = efile["word_to_ix"]
    word_to_ix['UNK'] = 0
    return word_to_ix, embeddings


if __name__ == '__main__':
    # Timestamp for this run
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Create directories
    model_dir = f'models/{timestamp}'
    log_dir = f'runs/{timestamp}'
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # TensorBoard writer
    writer = SummaryWriter(log_dir)
    global_step = 0

    # Prepare data
    df = load_data()
    w2i, embedding_matrix = load_embeddings()

    dataset = PostDataset(df, embedding_matrix, w2i)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Model
    sample_input, _ = dataset[0]
    input_dim = sample_input.shape[0]

    model = BigModel(vector_size=input_dim, scale=3).to(DEVICE)

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training loop
    for epoch in range(1, EPOCHS+1):
        model.train()
        total_loss = 0.0

        progress_bar = tqdm(enumerate(dataloader, 1), total=len(dataloader), desc=f"Epoch {epoch}", unit="batch", dynamic_ncols=True)

        for step, (batch_x, batch_y) in progress_bar:
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE).unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            avg_loss = total_loss / step
            progress_bar.set_postfix(loss=avg_loss)

            writer.add_scalar("Loss/Step", avg_loss, global_step)
            global_step += 1

        avg_epoch_loss = total_loss / len(dataloader)
        writer.add_scalar("Loss/Epoch", avg_epoch_loss, epoch)
        print(f"✓ Epoch {epoch} complete. Average Loss: {avg_epoch_loss:.4f}\n")

    writer.close()

    # Save model
    model_path = os.path.join(model_dir, f'weights_{timestamp}.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'w2i': w2i,  # saving vocab for reuse
        'embedding_matrix': embedding_matrix,  # saving embeddings as well
        'input_dim': input_dim
    }, model_path)

    print(f"Model saved to {model_path}")
