import numpy as np
import pandas as pd
import torch


def load_data(items_file, users_file):
    raw_items = pd.read_parquet(items_file)
    raw_users = pd.read_parquet(users_file)
    merged = pd.merge(raw_items, raw_users, on="by", how="left", suffixes=("", "_user"))
    has_score = merged.dropna(subset=["score"])
    has_title = has_score[has_score["title"].notnull()]
    has_title = has_title[has_title["title"].str.strip().astype(bool)]  # drop empty or whitespace-only
    numeric = has_title.select_dtypes(include=[np.number])
    return pd.concat([has_title[["title"]], numeric.drop(columns=["id"])], axis=1)


def load_embeddings(embeddings_file):
    edict = torch.load(embeddings_file)
    embeddings = edict["embeddings"]
    word_to_ix = edict["word_to_ix"]
    word_to_ix['UNK'] = 0
    return word_to_ix, embeddings
