import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils import load_data, load_embeddings, get_device
from dataloader import PostDataset
from model import BigModel
import os
from datetime import datetime
import optuna

# CONSTANTS
DEVICE = get_device()
EMBEDDING_FILE = "skipgram_models/silvery200.pt"
ITEMS_FILE = "data/posts.parquet"
EPOCHS = 2  # You can increase for better optimization

# Load and prepare data once globally
df = load_data(ITEMS_FILE)
df = df.drop(["by", "url", "text", "created", "karma", 'max_score', 'min_score',
              'mean_score', 'max_descendants', 'min_descendants', 'mean_descendants'], axis=1)
df = df.sort_values(by="time").reset_index(drop=True)

train_size = int(0.9 * len(df))
train_df = df.iloc[:train_size]
val_df = df.iloc[train_size:]

T_ref = train_df['time'].max()

############### we are taking a subset of train
reduced_t = int(0.95 * len(train_df))
train_df = train_df.iloc[reduced_t:]
###############
print(len(train_df), len(val_df))
w2i, embedding_matrix = load_embeddings(EMBEDDING_FILE)

sample_dataset = PostDataset(train_df, embedding_matrix, w2i, ref_time=T_ref, lambda_=0.02)
sample_input, _, _ = sample_dataset[0]
input_dim = sample_input.shape[0]

def objective(trial):
    # Suggest hyperparameters
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128, 256])
    time_decay = trial.suggest_uniform('time_decay', 0.0, 0.1)
    scale = trial.suggest_int('scale', 1, 5)

    # Prepare datasets with current time_decay
    train_dataset = PostDataset(train_df, embedding_matrix, w2i, ref_time=T_ref, lambda_=time_decay)
    val_dataset = PostDataset(val_df, embedding_matrix, w2i)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = BigModel(vector_size=input_dim, scale=scale).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(EPOCHS):
        model.train()
        for batch_x, batch_y, batch_w in train_dataloader:
            batch_x, batch_y, batch_w = batch_x.to(DEVICE), batch_y.to(DEVICE).unsqueeze(1), batch_w.to(DEVICE).unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(batch_x)
            losses = criterion(outputs, batch_y)
            weighted_loss = (losses * batch_w).mean()
            weighted_loss.backward()
            optimizer.step()

    # Validation loss
    model.eval()
    val_loss = 0.0
    val_steps = 0
    with torch.no_grad():
        for batch_x, batch_y in val_dataloader:
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE).unsqueeze(1)
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            val_loss += loss.item()
            val_steps += 1

    avg_val_loss = val_loss / val_steps
    return avg_val_loss

if __name__ == '__main__':
    storage = "sqlite:///optuna_total_model.db"
    study = optuna.create_study(study_name="hyperparam_opt", direction="minimize", storage=storage, load_if_exists=True)
    study.optimize(objective, n_trials=50)

    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    # Save best model (retrain with best params)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_dir = f'final_model/models/{timestamp}'
    os.makedirs(model_dir, exist_ok=True)

    best_lr = trial.params['learning_rate']
    best_bs = trial.params['batch_size']
    best_decay = trial.params['time_decay']
    best_scale = trial.params['scale']