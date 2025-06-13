import numpy as np
import torch
from datetime import datetime

def load_data(items_file, users_file):
    raw_items = pd.read_parquet(items_file)
    raw_users = pd.read_parquet(users_file)
    merged = pd.merge(raw_items, raw_users, on="by", how="left", suffixes=("", "_user"))
    has_score = merged.dropna(subset=["score"])
    has_title = has_score[has_score["title"].notnull()]
    has_title = has_title[has_title["title"].str.strip().astype(bool)]  # drop empty or whitespace-only
    return has_title.drop(columns=["id"])


def load_embeddings(embeddings_file):
    edict = torch.load(embeddings_file, weights_only=True)
    embeddings = edict["embeddings"]
    word_to_ix = edict["word_to_ix"]
    word_to_ix['UNK'] = 0
    return word_to_ix, embeddings

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def log_transform_plus1(x):
    if x <= 0:
        return x
    else:
        return np.log10(x+1)

def time_transform(time):
    if isinstance(time, (int, float)):
        timestamp = datetime.fromtimestamp(time)
    else:
        timestamp = time
            
    year = timestamp.year    
    hour_angle = 2 * np.pi * timestamp.hour / 24
    dow_angle = 2 * np.pi * timestamp.weekday() / 7
    day_angle = 2 * np.pi * (timestamp.timetuple().tm_yday - 1) / 365
    return year, hour_angle, dow_angle, day_angle

global global_Tmin
global global_Tmax

def extract_features(row):
    # Time features
    year, hour_angle, dow_angle, day_angle = time_transform(row['time'])
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
        if col not in ['by', 'time', 'title', 'score', 'url', 'num_posts']
    ]

    user_feats = [row[col] for col in user_feature_names]

    all_features = np.array(time_feats + user_feats, dtype=np.float32)
    return all_features
