import pandas as pd
import argparse
import logging

from collections import defaultdict

SECONDS_PER_DAY = 24 * 60 * 60
SECONDS_PER_YEAR = 365.25 * SECONDS_PER_DAY

parser = argparse.ArgumentParser()
parser.add_argument("--items", default="data/items.parquet", help="Items file")
parser.add_argument("--posts", default="data/posts.parquet", help="Posts file")

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S"
)

def mean(x):
    return 0 if len(x) == 0 else sum(x) / len(x)

def percent(num, denom):
    return 0 if denom == 0 else 100.0 * num / denom

def comment_calculations(parent_map, comments):
    logging.info("Calculating depth and comment arrival times per story")
    depth_dict = defaultdict(int)
    depth_cache = {}
    story_comment_times = defaultdict(list)

    def _depth(cid):
        if cid in depth_cache:
            return depth_cache[cid]
        pid = parent_map.get(cid)
        if pid is None or pd.isna(pid):
            depth_cache[cid] = 0
            return 0
        d = _depth(pid) + 1
        depth_cache[cid] = d
        return d

    for row in comments.itertuples(index=False):
        d = _depth(row.id)
        root = row.id
        p = parent_map.get(root)
        while p is not None and not pd.isna(p):
            root = p
            p = parent_map.get(root)
        depth_dict[root] = max(depth_dict[root], d)
        story_comment_times[root].append(row.time)
    return depth_dict, story_comment_times




def get_story_author(parent_map, story_author_map, comment_id):
    # Ascend to the root story to see whose post this comment belongs to
    pid = parent_map.get(comment_id)
    while pid is not None and not pd.isna(pid):
        comment_id = pid
        pid = parent_map.get(comment_id)
    return story_author_map.get(comment_id)

def main(items_file, posts_file):
    logging.info(f"Reading {items_file}")
    items = pd.read_parquet(items_file)
    logging.info(f"Sorting items")
    items = items.sort_values("time").reset_index(drop=True)
    logging.info("Filtering for dead")
    not_dead = items[items["dead"].isnull()]
    logging.info("Building parent lookup map")
    parent_map = not_dead.set_index("id")["parent"].to_dict()
    logging.info("Filtering for stories")
    stories = not_dead[not_dead["type"] == "story"].set_index("id")
    logging.info("Building story author lookup map")
    story_author_map = stories["by"].to_dict()
    logging.info("Filtering for comments")
    comments = not_dead[not_dead["type"] == "comment"].copy()
    logging.info("Calculating comment length")
    comments["text_length"] = comments["text"].fillna("").str.len()
    logging.info("Grouping by parent")
    groups = comments.groupby("parent")
    logging.info("Calculating comment counts")
    comment_counts = groups.size().to_dict()
    logging.info("Calculating distinct commenters")
    distinct_commenter_counts = (
        comments.dropna(subset=["by"])
                .groupby("parent", sort=False)["by"]
                .nunique()
                .astype(int)
                .to_dict()
    )
    logging.info("Building story → time map")
    story_time_map = stories["time"].to_dict()

    depth_dict, story_comment_times = comment_calculations(parent_map, comments)

    logging.info("Computing first/10th comment delays")
    first_comment_delta = {}
    tenth_comment_delta = {}
    for sid, times in story_comment_times.items():
        times.sort()
        if len(times) >= 1:
            first_comment_delta[sid] = (times[0] - story_time_map.get(sid, times[0])).total_seconds()
        if len(times) >= 10:
            tenth_comment_delta[sid] = (times[9] - story_time_map.get(sid, times[9])).total_seconds()

    post_features = []

    logging.info("Starting loop")
    user_state = defaultdict(lambda: {
        "scores": [0],
        "comment_lengths": [0],
        "kids": [0],
        "descendants": [0],
        "depths": [0],
        "commenters": [0],
        "first_comment_times": [0],
        "tenth_comment_times": [0],
        "commented_story_ids": set(),
        "first_time": None,
        "last_time": None,
        "dead_posts": 0,
        "scores_above_1": 0
    })

    for row in items.itertuples(index=False):
        u = row.by
        if u not in user_state:
            user_state[u] = {
                "scores": [0],
                "comment_lengths": [0],
                "kids": [0],
                "descendants": [0],
                "depths": [0],
                "commenters": [0],
                "first_comment_times": [0],
                "tenth_comment_times": [0],
                "commented_story_ids": set(),
                "first_time": row.time,
                "last_time": row.time,
                "dead_posts": 0,
                "scores_above_1": 0
            }

        state = user_state[u]
        if state["first_time"] is None:
            state["first_time"] = row.time

        depth_this_item = depth_dict.get(row.id, 0)
        if row.type == "story":
            if row.dead:
                state["dead_posts"] += 1
                continue
            delta_first = (row.time - state["first_time"]).total_seconds()
            delta_last  = (row.time - state["last_time"]).total_seconds()
            elapsed_years = delta_first / SECONDS_PER_YEAR + 1
            days_since_first_post = delta_first / SECONDS_PER_DAY
            days_since_last_post  = delta_last  / SECONDS_PER_DAY
            num_posts = len(state["scores"])
            posts_per_year = num_posts / elapsed_years
            post_commenters = distinct_commenter_counts.get(row.id, 0)
            post_features.append({
                "id": row.id,
                "by": u,
                "time": row.time,
                "title": row.title,
                "url": row.url,
                "score": row.score,
                "num_posts": num_posts,
                "percent_posts_dead": percent(state["dead_posts"], state["dead_posts"] + num_posts),
                "percent_scores_above_1": percent(state["scores_above_1"], num_posts),
                "posts_per_year": posts_per_year,
                "days_since_first_post": days_since_first_post,
                "days_since_last_post": days_since_last_post,
                "user_depth_min": min(state["depths"]),
                "user_depth_max": max(state["depths"]),
                "user_depth_mean": mean(state["depths"]),
                "num_comments": comment_counts.get(row.id, 0),
                "descendants_min": min(state["descendants"]),
                "descendants_max": max(state["descendants"]),
                "descendants_mean": mean(state["descendants"]),
                "post_commenters_min": min(state["commenters"]),
                "post_commenters_max": max(state["commenters"]),
                "post_commenters_mean": mean(state["commenters"]),
                "first_comment_delay_min": min(state["first_comment_times"]),
                "first_comment_delay_max": max(state["first_comment_times"]),
                "first_comment_delay_mean": mean(state["first_comment_times"]),
                "tenth_comment_delay_min": min(state["tenth_comment_times"]),
                "tenth_comment_delay_max": max(state["tenth_comment_times"]),
                "tenth_comment_delay_mean": mean(state["tenth_comment_times"]),
                "other_posts_commented": len(state["commented_story_ids"]),
                "user_score_min": min(state["scores"]),
                "user_score_max": max(state["scores"]),
                "user_score_mean": mean(state["scores"]),
                "user_comment_len_min": min(state["comment_lengths"]),
                "user_comment_len_max": max(state["comment_lengths"]),
                "user_comment_len_mean": mean(state["comment_lengths"]),
                "user_kids_min": min(state["kids"]),
                "user_kids_max": max(state["kids"]),
                "user_kids_mean": mean(state["kids"])
            })

            if row.id in first_comment_delta:
                state["first_comment_times"].append(first_comment_delta[row.id])
            if row.id in tenth_comment_delta:
                state["tenth_comment_times"].append(tenth_comment_delta[row.id])
            if row.score > 1:
                state["scores_above_1"] += 1

            state["scores"].append(row.score or 0)
            state["descendants"].append(row.descendants or 0)
            state["commenters"].append(post_commenters)
            state["last_time"] = row.time
        elif row.type == "comment" and not row.dead and row.text:
            state["kids"].append(len(row.kids) if isinstance(row.kids, list) else 0)
            state["commenters"][-1] = state["commenters"][-1] + 0  # placeholder to keep list length accurate
            story_author = get_story_author(parent_map, story_author_map, row.id)
            if story_author is not None and story_author != u:
                state["commented_story_ids"].add(row.id)
        # Both stories and comments can have depths
        state["depths"].append(depth_this_item)


    post_df = pd.DataFrame(post_features)
    logging.info(f"Writing {posts_file}")
    post_df.to_parquet(posts_file)
    logging.info(f"Wrote {post_df.shape[0]} posts")

if __name__ == "__main__":
    args = parser.parse_args()
    main(args.items, args.posts)