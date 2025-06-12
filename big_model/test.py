import pandas as pd
import numpy as np
from utils import time_transform, log_transform_plus1

# Load dataframe
FILEPATH = "data/posts.parquet"
df = pd.read_parquet(FILEPATH)
df = df.drop(["id", "by", "url"], axis=1)
df = df.sort_values(by="time").reset_index(drop=True)
df = df.dropna()
print(df.keys())

# Compute Tmin and Tmax
Tmin = df['time'].min()
Tmax = df['time'].max()

# Pick a single row for debugging
row = df.iloc[0]

print(row)
exit()

# Compute time-based features
year, hour_angle, dow_angle, day_angle = time_transform(row['time'], offset=Tmin.year)
year_norm = (year - Tmin.year) / (Tmax.year - Tmin.year)

transformed_features = [
    year_norm,
    np.sin(hour_angle),
    np.cos(hour_angle),
    np.sin(dow_angle),
    np.cos(dow_angle),
    np.sin(day_angle),
    np.cos(day_angle),
    log_transform_plus1(row['length_submitted']),
    log_transform_plus1(row['story_count'])
]

# Collect user features (everything except time, title, score, length_submitted, story_count)
user_feature_names = [
    col for col in row.index
    if col not in ['time', 'title', 'score', 'length_submitted', 'story_count']
]

user_feats = [row[col] for col in user_feature_names]

# Combine all features
all_features = np.array(transformed_features + user_feats, dtype=np.float32)

# Inspect
print("Time features:", transformed_features)
print("User features:", user_feats)
print("Final feature vector shape:", all_features.shape)
print("All features vector:", all_features)
