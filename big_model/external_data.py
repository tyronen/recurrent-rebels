import pandas as pd
import torch


def load_data(items_file):
    raw_items = pd.read_parquet(items_file)
    has_score = raw_items.dropna(subset=["score"])
    has_fields = has_score[has_score["title"].notnull() & has_score["url"].notnull()]
    has_fields = has_fields[has_fields["title"].str.strip().astype(bool)]  # drop empty or whitespace-only
    return has_fields.drop(columns=["id", "by"])


def load_embeddings(embeddings_file):
    edict = torch.load(embeddings_file)
    embeddings = edict["embeddings"]
    word_to_ix = edict["word_to_ix"]
    word_to_ix['UNK'] = 0
    return word_to_ix, embeddings
