import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import pandas as pd
import numpy as np

from dataloader import PostDataset  
from model import BigModel

import os
from datetime import datetime
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

# Hyperparameters
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 1e-3
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def create_dummy_data():
    data = {
        'feature1': np.random.rand(1000),
        'feature2': np.random.rand(1000),
        'feature3': np.random.rand(1000),
        'title': ['this is a sample title'] * 1000,
        'score': np.random.rand(1000) * 100
    }
    return pd.DataFrame(data)

def create_dummy_embeddings():
    vocab = ['this', 'is', 'a', 'sample', 'title', 'UNK']
    w2i = {word: idx+1 for idx, word in enumerate(vocab)}
    w2i['UNK'] = 0 

    embedding_matrix = torch.randn(len(w2i), 50)
    return w2i, embedding_matrix


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
    df = create_dummy_data() 
    w2i, embedding_matrix = create_dummy_embeddings() 

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
