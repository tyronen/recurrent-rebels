import pandas as pd
import argparse
import logging
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument("--items", default="data/items.parquet", help="Items file")
parser.add_argument("--posts", default="data/posts.parquet", help="Posts file")

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S"
)

def main(items_file, posts_file):
    logging.info(f"Reading {items_file}")
    items = pd.read_parquet(items_file)
    logging.info(f"Sorting items")
    items = items.sort_values("time").reset_index(drop=True)
    # items = items[items["dead"].isnull()]
    logging.info("Filtering for comments")
    comments = items[items["text"].notnull()].copy()
    logging.info("Calculating comment length")
    # comments["text_length"] = comments["text"].fillna("").str.len()
    comment_counts = comments.groupby("parent").size().to_dict()
    distinct_commenters = (
        comments.groupby("parent")["by"]
        .agg(lambda x: set(x.dropna()))
        .to_dict()
    )
    post_features = []

    logging.info("Starting loop")
    user_state = defaultdict(lambda: {"scores": [0], "lengths": [0], "kids": [0]})

    for row in items.itertuples(index=False):
        u = row.by
        if u not in user_state:
            user_state[u] = {"scores": [0], "lengths": [0], "kids": [0]}

        state = user_state[u]

        if row.title:  # story
            post_features.append({
                "id": row.id,
                "by": u,
                "time": row.time,
                "title": row.title,
                "url": row.url,
                "score": row.score,
                "num_comments": comment_counts.get(row.id, 0),
                "distinct_commenters": len(distinct_commenters.get(row.id, ())),
                "user_score_min": min(state["scores"]),
                "user_score_max": max(state["scores"]),
                "user_score_mean": sum(state["scores"]) / len(state["scores"]),
                "user_len_min": min(state["lengths"]),
                "user_len_max": max(state["lengths"]),
                "user_len_mean": sum(state["lengths"]) / len(state["lengths"]),
                "user_kids_min": min(state["kids"]),
                "user_kids_max": max(state["kids"]),
                "user_kids_mean": sum(state["kids"]) / len(state["kids"])
            })

        if row.score:
            state["scores"].append(row.score)
        elif row.text:
            state["lengths"].append(len(row.text))
            state["kids"].append(len(row.kids) if isinstance(row.kids, list) else 0)

    post_df = pd.DataFrame(post_features)
    logging.info(f"Writing {posts_file}")
    post_df.to_parquet(posts_file)
    logging.info(f"Wrote {post_df.shape[0]} posts")

if __name__ == "__main__":
    args = parser.parse_args()
    main(args.items, args.posts)