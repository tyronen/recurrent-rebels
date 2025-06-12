import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from external_data import load_data, load_embeddings
from dataloader import PostDataset
from model import BigModel

import argparse
import os
from datetime import datetime
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser(description="Train the master NN.")
parser.add_argument('--items', default="data/posts.parquet", help='Items data frame')
parser.add_argument('--users', default="data/users.parquet", help='Ignored')
parser.add_argument('--embeddings', default="skipgram_models/silvery200.pt", help='Word2Vec embeddings')
parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
parser.add_argument("--epochs", type=int, default=10, help="Epochs")
parser.add_argument("--lr", type=float, default=1e-3, help="Learning Rate")
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
    df = load_data(args.items)
    w2i, embedding_matrix = load_embeddings(args.embeddings)

    dataset = PostDataset(df, embedding_matrix, w2i)
    
    # Split dataset into train/validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Dataset sizes - Train: {train_size}, Validation: {val_size}")

    # Model
    sample_input, _ = dataset[0]
    input_dim = sample_input.shape[0]
    print(f"Input dimension: {input_dim}")

    model = BigModel(vector_size=input_dim, scale=3).to(DEVICE)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training loop
    for epoch in range(1, EPOCHS+1):
        # Training phase
        model.train()
        train_loss = 0.0
        train_steps = 0
        
        # Monitor gradients and weights
        total_grad_norm = 0.0
        grad_norm_count = 0

        progress_bar = tqdm(enumerate(train_dataloader, 1), total=len(train_dataloader), desc=f"Epoch {epoch}", unit="batch", dynamic_ncols=True)

        for step, (batch_x, batch_y) in progress_bar:
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE).unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            
            # Monitor gradient norms before clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            total_grad_norm += grad_norm.item()
            grad_norm_count += 1
            
            optimizer.step()

            train_loss += loss.item()
            train_steps += 1
            avg_loss = train_loss / train_steps
            progress_bar.set_postfix(loss=avg_loss, grad_norm=f"{grad_norm:.3f}")

            writer.add_scalar("Loss/Train_Step", avg_loss, global_step)
            writer.add_scalar("Gradients/Norm", grad_norm, global_step)
            global_step += 1

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_steps = 0
        predictions = []
        targets = []
        
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

        avg_train_loss = train_loss / train_steps
        avg_val_loss = val_loss / val_steps
        avg_grad_norm = total_grad_norm / grad_norm_count
        
        # Convert to numpy for analysis
        predictions = np.array(predictions)
        targets = np.array(targets)
        
        # Calculate additional metrics
        mae = np.mean(np.abs(predictions - targets))
        pred_std = np.std(predictions)
        target_std = np.std(targets)
        
        writer.add_scalar("Loss/Train_Epoch", avg_train_loss, epoch)
        writer.add_scalar("Loss/Val_Epoch", avg_val_loss, epoch)
        writer.add_scalar("Gradients/Avg_Norm", avg_grad_norm, epoch)
        writer.add_scalar("Metrics/MAE", mae, epoch)
        writer.add_scalar("Metrics/Pred_Std", pred_std, epoch)
        writer.add_scalar("Metrics/Target_Std", target_std, epoch)
        
        # Monitor weight statistics
        for name, param in model.named_parameters():
            if param.requires_grad:
                writer.add_histogram(f"Weights/{name}", param, epoch)
                writer.add_scalar(f"Weights/{name}_mean", param.mean().item(), epoch)
                if param.numel() > 1:
                    writer.add_scalar(f"Weights/{name}_std", param.std().item(), epoch)
                else:
                    writer.add_scalar(f"Weights/{name}_std", 0.0, epoch)
        
        print(f"✓ Epoch {epoch} complete. Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        print(f"  MAE: {mae:.4f}, Pred Range: [{predictions.min():.2f}, {predictions.max():.2f}], Grad Norm: {avg_grad_norm:.4f}")
        
        # Sanity checks
        if np.isnan(predictions).any():
            print("🚨 NaN predictions detected!")
        if avg_grad_norm < 1e-6:
            print("⚠️  Very small gradients - possible vanishing gradient problem")
        if avg_grad_norm > 10:
            print("⚠️  Large gradients - possible exploding gradient problem")
        if pred_std < 0.01:
            print("⚠️  Very low prediction variance - model might not be learning")
        if avg_val_loss > avg_train_loss * 1.5:
            print("⚠️  Potential overfitting detected!")
        
        # Show a few sample predictions vs targets
        if epoch % 5 == 0:  # Every 5 epochs
            sample_indices = np.random.choice(len(predictions), min(5, len(predictions)), replace=False)
            print("  Sample predictions vs targets:")
            for i in sample_indices:
                print(f"    Pred: {predictions[i]:.3f}, Target: {targets[i]:.3f}, Diff: {abs(predictions[i] - targets[i]):.3f}")

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
