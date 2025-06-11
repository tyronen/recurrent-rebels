import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

import pandas as pd
import numpy as np

from dataloader import PostDataset  
from model import BigModel

import argparse
import os
from datetime import datetime
from tqdm import tqdm
import wandb

parser = argparse.ArgumentParser(description="Train the master NN with W&B logging.")
parser.add_argument('--items', default="data/items.parquet", help='Items data frame')
parser.add_argument('--embeddings', default="skipgram_models/silvery200.pt", help='Word2Vec embeddings')
parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
parser.add_argument("--epochs", type=int, default=10, help="Epochs")
parser.add_argument("--lr", type=float, default=1e-3, help="Learning Rate")
parser.add_argument("--project", default="post-score-prediction", help="W&B project name")
parser.add_argument("--run_name", default=None, help="W&B run name")
parser.add_argument("--tags", nargs='+', default=[], help="W&B tags")
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
    
    # Initialize wandb
    run_name = args.run_name or f"run_{timestamp}"
    wandb.init(
        project=args.project,
        name=run_name,
        tags=args.tags,
        config={
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "learning_rate": LEARNING_RATE,
            "device": str(DEVICE),
            "model_scale": 3,
            "grad_clip_norm": 1.0,
            "train_split": 0.8,
            "items_file": args.items,
            "embeddings_file": args.embeddings,
        }
    )

    # Create directories
    model_dir = f'models/{timestamp}'
    os.makedirs(model_dir, exist_ok=True)

    # Prepare data
    df = load_data()
    w2i, embedding_matrix = load_embeddings()

    dataset = PostDataset(df, embedding_matrix, w2i)
    
    # Split dataset into train/validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Dataset sizes - Train: {train_size}, Validation: {val_size}")
    wandb.log({
        "dataset/train_size": train_size,
        "dataset/val_size": val_size,
        "dataset/total_size": len(dataset)
    })

    # Model
    sample_input, _ = dataset[0]
    input_dim = sample_input.shape[0]
    print(f"Input dimension: {input_dim}")

    model = BigModel(vector_size=input_dim, scale=3).to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Model parameters: {total_params:,}")
    wandb.log({
        "model/input_dim": input_dim,
        "model/total_params": total_params,
        "model/trainable_params": trainable_params
    })

    # Log model architecture
    wandb.watch(model, log="all", log_freq=100)

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training loop
    global_step = 0
    best_val_loss = float('inf')
    
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

            # Log every 10 steps to avoid overwhelming W&B
            if global_step % 10 == 0:
                wandb.log({
                    "train/loss_step": avg_loss,
                    "train/grad_norm": grad_norm,
                    "train/step": global_step
                })
            
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
        mse = np.mean((predictions - targets) ** 2)
        pred_std = np.std(predictions)
        target_std = np.std(targets)
        pred_min, pred_max = predictions.min(), predictions.max()
        target_min, target_max = targets.min(), targets.max()
        
        # Calculate correlation
        if len(predictions) > 1:
            correlation = np.corrcoef(predictions, targets)[0, 1]
        else:
            correlation = 0.0
        
        # Track best model
        is_best = avg_val_loss < best_val_loss
        if is_best:
            best_val_loss = avg_val_loss
        
        # Log epoch metrics
        epoch_metrics = {
            "epoch": epoch,
            "train/loss_epoch": avg_train_loss,
            "val/loss_epoch": avg_val_loss,
            "val/mae": mae,
            "val/mse": mse,
            "val/correlation": correlation,
            "gradients/avg_norm": avg_grad_norm,
            "predictions/std": pred_std,
            "predictions/min": pred_min,
            "predictions/max": pred_max,
            "targets/std": target_std,
            "targets/min": target_min,
            "targets/max": target_max,
            "model/is_best": is_best
        }
        
        # Log weight statistics
        for name, param in model.named_parameters():
            if param.requires_grad:
                epoch_metrics[f"weights/{name}_mean"] = param.mean().item()
                epoch_metrics[f"weights/{name}_std"] = param.std().item()
                epoch_metrics[f"weights/{name}_norm"] = param.norm().item()
        
        wandb.log(epoch_metrics)
        
        print(f"✓ Epoch {epoch} complete. Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        print(f"  MAE: {mae:.4f}, Correlation: {correlation:.4f}, Pred Range: [{pred_min:.2f}, {pred_max:.2f}], Grad Norm: {avg_grad_norm:.4f}")
        
        # Sanity checks
        warnings = []
        if np.isnan(predictions).any():
            warnings.append("NaN predictions detected!")
        if avg_grad_norm < 1e-6:
            warnings.append("Very small gradients - possible vanishing gradient problem")
        if avg_grad_norm > 10:
            warnings.append("Large gradients - possible exploding gradient problem")
        if pred_std < 0.01:
            warnings.append("Very low prediction variance - model might not be learning")
        if avg_val_loss > avg_train_loss * 1.5:
            warnings.append("Potential overfitting detected!")
        
        if warnings:
            warning_text = "; ".join(warnings)
            print(f"⚠️  {warning_text}")
            wandb.log({"warnings": warning_text})
        
        # Show a few sample predictions vs targets
        if epoch % 5 == 0:  # Every 5 epochs
            sample_indices = np.random.choice(len(predictions), min(5, len(predictions)), replace=False)
            print("  Sample predictions vs targets:")
            sample_data = []
            for i in sample_indices:
                pred, target = predictions[i], targets[i]
                diff = abs(pred - target)
                print(f"    Pred: {pred:.3f}, Target: {target:.3f}, Diff: {diff:.3f}")
                sample_data.append([epoch, pred, target, diff])
            
            # Log sample predictions as a table
            sample_table = wandb.Table(
                columns=["epoch", "prediction", "target", "abs_diff"],
                data=sample_data
            )
            wandb.log({"sample_predictions": sample_table})
        
        # Create prediction scatter plot every 10 epochs
        if epoch % 10 == 0:
            # Sample subset for plotting (to avoid huge plots)
            n_plot = min(1000, len(predictions))
            plot_indices = np.random.choice(len(predictions), n_plot, replace=False)
            plot_preds = predictions[plot_indices]
            plot_targets = targets[plot_indices]
            
            # Create scatter plot
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.scatter(plot_targets, plot_preds, alpha=0.6, s=20)
            
            # Perfect prediction line
            min_val = min(plot_targets.min(), plot_preds.min())
            max_val = max(plot_targets.max(), plot_preds.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, label='Perfect Prediction')
            
            ax.set_xlabel('Target Score')
            ax.set_ylabel('Predicted Score')
            ax.set_title(f'Predictions vs Targets (Epoch {epoch})')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            wandb.log({f"plots/predictions_vs_targets_epoch_{epoch}": wandb.Image(fig)})
            plt.close(fig)

    # Save model
    model_path = os.path.join(model_dir, f'weights_{timestamp}.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'w2i': w2i,  # saving vocab for reuse
        'embedding_matrix': embedding_matrix,  # saving embeddings as well
        'input_dim': input_dim,
        'config': wandb.config,
        'best_val_loss': best_val_loss
    }, model_path)

    print(f"Model saved to {model_path}")
    
    # Log model as artifact
    model_artifact = wandb.Artifact(
        name=f"model_{timestamp}",
        type="model",
        description=f"Trained BigModel - Best Val Loss: {best_val_loss:.4f}"
    )
    model_artifact.add_file(model_path)
    wandb.log_artifact(model_artifact)
    
    # Final summary
    wandb.summary.update({
        "final/best_val_loss": best_val_loss,
        "final/total_epochs": EPOCHS,
        "final/model_path": model_path,
        "final/total_params": total_params
    })
    
    wandb.finish()
    print("🎉 Training completed and logged to W&B!") 