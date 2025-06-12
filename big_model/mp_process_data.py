import os
import numpy as np
import pandas as pd
import torch
from utils import load_data, load_embeddings, log_transform_plus1, time_transform
from tqdm import tqdm
import multiprocessing as mp

# global variables to be shared across workers
global_embedding_matrix = None
global_w2i = None
global_Tmin = None
global_Tmax = None

def init_worker(embedding_matrix, w2i_dict, Tmin, Tmax):
    global global_embedding_matrix
    global global_w2i
    global global_Tmin
    global global_Tmax
    if isinstance(embedding_matrix, torch.Tensor):
        global_embedding_matrix = embedding_matrix.detach().clone()
    else:
        global_embedding_matrix = torch.tensor(embedding_matrix)
    global_w2i = w2i_dict
    global_Tmin = Tmin
    global_Tmax = Tmax


def extract_features(row):
    # Time features
    year, hour_angle, dow_angle, day_angle = time_transform(row['time'], offset=global_Tmin.year)
    year_norm = (year - global_Tmin.year) / (global_Tmax.year - global_Tmin.year)

    time_feats = [
        year_norm,
        np.sin(hour_angle),
        np.cos(hour_angle),
        np.sin(dow_angle),
        np.cos(dow_angle),
        np.sin(day_angle),
        np.cos(day_angle),
        log_transform_plus1(row['num_posts'])
    ]

    # Collect all remaining user features (everything except those handled above
    user_feature_names = [
        col for col in row.keys()
        if col not in ['time', 'title', 'score', 'url', 'num_posts']
    ]

    user_feats = [row[col] for col in user_feature_names]

    all_features = np.array(time_feats + user_feats, dtype=np.float32)
    return all_features


def tokenize_title(title_text):
    tokens = title_text.lower().split()
    token_indices = [global_w2i.get(token, 0) for token in tokens]
    return token_indices

def embed_title(token_indices):
    if len(token_indices) == 0:
        return torch.zeros(global_embedding_matrix.shape[1])
    token_indices = torch.tensor(token_indices, dtype=torch.long)
    embedded = global_embedding_matrix[token_indices]
    avg_embedding = embedded.mean(dim=0)
    return avg_embedding.numpy()

def process_row(row):
    feats = extract_features(row)
    tokens = tokenize_title(row['title'])
    emb = embed_title(tokens)
    target = np.clip(row['score'], 0, None)
    return feats, emb, target

def precompute_parallel(df, embedding_matrix, w2i_dict, Tmin=None, Tmax=None, ref_time=None, compute_delta_t=False, num_workers=None):
    df = df.reset_index(drop=True)
    df['time'] = pd.to_datetime(df['time'])

    if compute_delta_t:
        delta_t = (ref_time - df['time']) / np.timedelta64(30, 'D')
        delta_t = delta_t.to_numpy().astype(np.float32)
    else:
        delta_t = None

    with mp.Pool(processes=num_workers, initializer=init_worker, initargs=(embedding_matrix, w2i_dict, Tmin, Tmax)) as pool:
        results = list(tqdm(pool.imap(process_row, df.to_dict(orient='records')), total=len(df), desc="Precomputing"))

    # Unpack results
    all_features, all_embeddings, all_targets = zip(*results)

    all_features = np.stack(all_features).astype(np.float32)
    all_embeddings = np.stack(all_embeddings).astype(np.float32)
    all_targets = np.array(all_targets, dtype=np.float32)

    return all_features, all_embeddings, all_targets, delta_t

if __name__ == '__main__':
    EMBEDDING_FILE = "skipgram_models/silvery200.pt"
    FILEPATH = "data/posts.parquet"
    OUTPUT_DIR = "precomputed_npz"
    NUM_WORKERS = 20

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = pd.read_parquet(FILEPATH)
    df = df.drop(["id", "by", "url"], axis=1)
    df = df.sort_values(by="time").reset_index(drop=True)
    df = df.dropna()

    train_size = int(0.9 * len(df))
    train_df = df.iloc[:train_size]
    val_df = df.iloc[train_size:]
    T_ref = train_df['time'].max()

    Tmin = train_df['time'].min()
    Tmax = train_df['time'].max()

    w2i, embedding_matrix = load_embeddings(EMBEDDING_FILE)

    train_features, train_embeddings, train_targets, train_delta_t = precompute_parallel(
        train_df, embedding_matrix, w2i, Tmin=Tmin, Tmax=Tmax, ref_time=T_ref, compute_delta_t=True, num_workers=NUM_WORKERS)

    np.savez(os.path.join(OUTPUT_DIR, 'train.npz'),
             features=train_features,
             embeddings=train_embeddings,
             targets=train_targets,
             delta_t=train_delta_t)

    val_features, val_embeddings, val_targets, _ = precompute_parallel(
        val_df, embedding_matrix, w2i, Tmin=Tmin, Tmax=Tmax, compute_delta_t=False, num_workers=NUM_WORKERS)

    np.savez(os.path.join(OUTPUT_DIR, 'val.npz'),
             features=val_features,
             embeddings=val_embeddings,
             targets=val_targets)

    print("Precomputation finished!")
