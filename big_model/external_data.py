import pandas as pd
import torch


def load_data(items_file):
    raw_items = pd.read_parquet(items_file)
    has_score = raw_items.dropna(subset=["score"])
    has_title = has_score[has_score["title"].notnull()]
    has_title = has_title[has_title["title"].str.strip().astype(bool)]  # drop empty or whitespace-only
    return has_title.drop(columns=["id", "by"])


def load_embeddings(embeddings_file):
    edict = torch.load(embeddings_file)
    embeddings = edict["embeddings"]
    word_to_ix = edict["word_to_ix"]
    word_to_ix['UNK'] = 0
    return word_to_ix, embeddings
