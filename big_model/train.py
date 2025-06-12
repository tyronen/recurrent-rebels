import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils import load_data, load_embeddings, get_device
from dataloader import PrecomputedNPZDataset
from model import BigModel
import os
from datetime import datetime
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import numpy as np

# Hyperparameters
BATCH_SIZE = 64
EPOCHS = 5
LEARNING_RATE = 1e-3
TIME_DECAY = 0.02
DEVICE = get_device()
TRAIN_FILE = "precomputed_npz/train.npz"
VAL_FILE = "precomputed_npz/val.npz"

if __name__ == '__main__':

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_dir = f'models/{timestamp}'
    log_dir = f'runs/{timestamp}'
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # TensorBoard writer
    writer = SummaryWriter(log_dir)

    train_dataset = PrecomputedNPZDataset(TRAIN_FILE, time_decay=TIME_DECAY)
    val_dataset = PrecomputedNPZDataset(VAL_FILE)

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Dataset sizes - Train: {len(train_dataset)}, Validation: {len(val_dataset)}")

    # Model
    sample_input, _, _= train_dataset[0]
    input_dim = sample_input.shape[0]
    print(f"Input dimension: {input_dim}")

    model = BigModel(vector_size=input_dim, scale=3).to(DEVICE)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    best_val_loss = float('inf')

    # Training loop
    for epoch in range(1, EPOCHS+1):
        # Training phase
        model.train()        
        progress_bar = tqdm(enumerate(train_dataloader, 1), total=len(train_dataloader), desc=f"Epoch {epoch}", unit="batch", dynamic_ncols=True)

        train_loss = 0
        train_steps = 0

        for step, (batch_x, batch_y, batch_w) in progress_bar:

            batch_x = batch_x.to(DEVICE)
            batch_y = batch_y.to(DEVICE).unsqueeze(1)
            batch_w = batch_w.to(DEVICE).unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(batch_x)

            losses = criterion(outputs, batch_y)
            weighted_loss = (losses * batch_w).mean()
            weighted_loss.backward()

            optimizer.step()

            train_loss += weighted_loss.item()
            train_steps += 1
            progress_bar.update()

        avg_train_loss = train_loss / train_steps

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_steps = 0
        predictions = []
        targets = []

        # todo: remove stopwords from the embeddings
        
        with torch.no_grad():
            for batch_x, batch_y in val_dataloader:
                batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE).unsqueeze(1)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
                val_steps += 1
                
                # Collect predictions for analysis
                predictions.extend(outputs.cpu().numpy().flatten())
                targets.extend(batch_y.cpu().numpy().flatten())

        avg_val_loss = val_loss / val_steps
    
        print(f'Training LOSS: Alpha {avg_train_loss}\n'
          f'Validation LOSS: Alpha {avg_val_loss} \n')
                
        # Calculate additional metrics
        mae = np.mean(np.abs(np.array(predictions) - np.array(targets)))
        
        writer.add_scalar("Loss/Train_Epoch", avg_train_loss, epoch)
        writer.add_scalar("Loss/Val_Epoch", avg_val_loss, epoch)
        writer.add_scalar("Metrics/MAE", mae, epoch)
        
        print(f"✓ Epoch {epoch} complete. Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model_path = os.path.join(model_dir, f'best_model_{epoch}.pth')
            torch.save({
                'model_state_dict': model.state_dict(),
                'input_dim': input_dim
            }, model_path)
            print(f"✓ Saved new best model at epoch {epoch} with val loss {avg_val_loss:.4f}")
    
    writer.close()

