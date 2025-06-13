import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import sys
import os

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.utils import get_device
from big_model.dataloader import PrecomputedNPZDataset
from model import HenselModel
import os
from datetime import datetime
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import json
from torchinfo import summary
import wandb

# Constants
DEVICE = get_device()
TRAIN_FILE = "data/train.npz"
VAL_FILE = "data/val.npz"
VOCAB_FILE = "data/train_vocab.json"

def train(config=None):
    # Initialize wandb
    with wandb.init(config=config) as run:
        # Get hyperparameters from wandb config
        config = wandb.config
        
        # Create directories for model checkpoints and logs
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_dir = f'models/hensel/{timestamp}'
        log_dir = f'runs/hensel/{timestamp}'
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

        # Initialize TensorBoard writer
        writer = SummaryWriter(log_dir)

        # Load vocabulary sizes
        with open(VOCAB_FILE, 'r') as f:
            vocabs = json.load(f)

        # Initialize datasets
        train_dataset = PrecomputedNPZDataset(TRAIN_FILE, time_decay=None)
        val_dataset = PrecomputedNPZDataset(VAL_FILE, time_decay=None)

        train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4)
        val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4)

        print(f"Dataset sizes - Train: {len(train_dataset)}, Validation: {len(val_dataset)}")

        # Get sample batch for model configuration
        sample_batch = train_dataset[0]
        features_num_sample, title_emb_sample, *_ = sample_batch

        # Model configuration
        model_config = {
            'vector_size_title': title_emb_sample.shape[0],
            'vector_size_num': features_num_sample.shape[0],
            'scale': 3,
            'domain_vocab_size': len(vocabs['domain_vocab']),
            'tld_vocab_size': len(vocabs['tld_vocab']),
            'user_vocab_size': len(vocabs['user_vocab']),
            'hidden_size': config.hidden_size,
            'dropout_rate': config.dropout_rate
        }

        # Initialize model
        model = HenselModel(**model_config).to(DEVICE)

        # Generate model summary
        summary(model, input_data=(
            torch.randn(1, model_config['vector_size_num']).to(DEVICE),
            torch.randn(1, model_config['vector_size_title']).to(DEVICE),
            torch.randint(0, model_config['domain_vocab_size'], (1,)).to(DEVICE),
            torch.randint(0, model_config['tld_vocab_size'], (1,)).to(DEVICE),
            torch.randint(0, model_config['user_vocab_size'], (1,)).to(DEVICE),
        ))

        # Initialize loss function and optimizer
        criterion = nn.SmoothL1Loss(beta=5.0)
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=2
        )

        best_val_loss = float('inf')
        patience = 5
        patience_counter = 0

        # Training loop
        for epoch in range(1, config.num_epochs + 1):
            model.train()
            train_loss = 0
            train_steps = 0

            progress_bar = tqdm(train_dataloader, total=len(train_dataloader),
                              desc=f"Epoch {epoch} [Train]", unit="batch", dynamic_ncols=True)

            for batch in progress_bar:
                features_num, title_emb, domain_idx, tld_idx, user_idx, target = [b.to(DEVICE) for b in batch]
                
                optimizer.zero_grad()
                output = model(features_num, title_emb, domain_idx, tld_idx, user_idx)
                loss = criterion(output, target.unsqueeze(1)).mean()

                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                train_loss += loss.item()
                train_steps += 1
                progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

            avg_train_loss = train_loss / train_steps

            # Validation phase
            model.eval()
            val_loss = 0.0
            val_steps = 0
            predictions = []
            targets = []

            val_progress = tqdm(val_dataloader, total=len(val_dataloader),
                              desc=f"Epoch {epoch} [Val]", unit="batch", dynamic_ncols=True)

            with torch.no_grad():
                for batch in val_progress:
                    features_num, title_emb, domain_idx, tld_idx, user_idx, target = [b.to(DEVICE) for b in batch]

                    output = model(features_num, title_emb, domain_idx, tld_idx, user_idx)
                    loss = criterion(output, target.unsqueeze(1)).mean()

                    val_loss += loss.item()
                    val_steps += 1

                    predictions.extend(output.cpu().numpy().flatten())
                    targets.extend(target.cpu().numpy().flatten())

            avg_val_loss = val_loss / val_steps
            scheduler.step(avg_val_loss)

            # Calculate metrics
            predictions = np.array(predictions)
            targets = np.array(targets)
            mae = np.mean(np.abs(predictions - targets))

            # R² on log scale
            ss_res_log = np.sum((targets - predictions) ** 2)
            ss_tot_log = np.sum((targets - np.mean(targets)) ** 2)
            r2_log = 1 - ss_res_log / ss_tot_log

            # R² on real scale
            predictions_real = 10**predictions - 1
            targets_real = 10**targets - 1
            ss_res_real = np.sum((targets_real - predictions_real) ** 2)
            ss_tot_real = np.sum((targets_real - np.mean(targets_real)) ** 2)
            r2_real = 1 - ss_res_real / ss_tot_real

            # Log metrics to wandb
            wandb.log({
                "epoch": epoch,
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "mae": mae,
                "r2_log": r2_log,
                "r2_real": r2_real,
                "learning_rate": optimizer.param_groups[0]['lr']
            })

            # Log metrics to TensorBoard
            writer.add_scalar("Metrics/R2_log", r2_log, epoch)
            writer.add_scalar("Metrics/R2_real", r2_real, epoch)
            writer.add_scalar("Loss/Train_Epoch", avg_train_loss, epoch)
            writer.add_scalar("Loss/Val_Epoch", avg_val_loss, epoch)
            writer.add_scalar("Metrics/MAE", mae, epoch)
            writer.add_scalar("Learning_Rate", optimizer.param_groups[0]['lr'], epoch)

            print(f"✓ Epoch {epoch} complete.")
            print(f"  Train Loss: {avg_train_loss:.4f}")
            print(f"  Val Loss: {avg_val_loss:.4f}")
            print(f"  MAE: {mae:.4f}")
            print(f"  R2 real: {r2_real:.4f}")
            print(f"  R2 log: {r2_log:.4f}")

            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                model_path = os.path.join(model_dir, f'best_model_{epoch}.pth')
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'config': model_config,
                    'epoch': epoch,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': avg_val_loss
                }, model_path)
                print(f"✓ Saved new best model at epoch {epoch} with val loss {avg_val_loss:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping triggered after {epoch} epochs")
                    break

        writer.close()
        print("Training completed!")

if __name__ == '__main__':
    # For local testing without wandb sweep
    default_config = {
        'batch_size': 32,
        'learning_rate': 5e-5,
        'weight_decay': 1e-5,
        'hidden_size': 256,
        'dropout_rate': 0.2,
        'num_epochs': 10
    }
    train(default_config) 