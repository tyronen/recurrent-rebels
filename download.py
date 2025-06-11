import logging
import pandas as pd
from sqlalchemy import create_engine, text
import argparse

# Run pip install sqlalchemy psycopg2-binary before running this

db_url = "postgresql://sy91dhb:g5t49ao@178.156.142.230:5432/hd64m1ki"
titles_file = "data/titles.parquet"
comments_file = "data/comments.parquet"
items_file = "data/items.parquet"
users_file = "data/users.parquet"

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S"
)


def download_query(engine, query, outfile):
    total_rows = 0
    chunks = pd.read_sql_query(query, engine, chunksize=200000)
    all_chunks = []
    for chunk in chunks:
        all_chunks.append(chunk)
        total_rows += len(chunk)
        logging.info(f"⏳ Downloaded {total_rows} rows so far for {outfile}")
    df = pd.concat(all_chunks, ignore_index=True)
    df.to_parquet(outfile)
    logging.info(f"✅ Saved {outfile} ({len(df)} rows)")


def download_hn_data(args):
    """Download HN data from PostgreSQL"""

    # Create database connection
    logging.info("Connecting to database...")
    engine = create_engine(db_url)

    # Query 1: High-scoring titles
    titles_query = """
       SELECT title
       FROM hacker_news.items
       WHERE title IS NOT NULL
         AND dead IS NULL
         AND score > 2"""

    # Query 2: Long comments with children
    comments_query = """
         SELECT text
         FROM hacker_news.items
         WHERE LENGTH(text) > 2048
           AND CARDINALITY(KIDS) > 4
           AND dead IS NULL"""

    # Query 3: All items
    items_query = """
        SELECT id, "by", title, url, time, text, score 
        FROM hacker_news.items
        WHERE dead IS NULL AND type='story'"""

    # Query 4: Users
    users_query = """
        SELECT 
            "by", created, karma, CARDINALITY(submitted) as length_submitted,
            COUNT(*) as story_count,
            MAX(score) as max_score, MIN(score) as min_score, AVG(score) as mean_score,
            MAX(descendants) as max_descendants, MIN(descendants) as min_descendants, AVG(descendants) as mean_descendants
        FROM hacker_news.items i LEFT JOIN hacker_news.users u
        ON i."by" = u.id
        WHERE dead IS NULL AND type='story'
        GROUP BY "by", created, karma, CARDINALITY(submitted)"""

    if args.titles:
        download_query(engine, titles_query, titles_file)
    if args.comments:
        download_query(engine, comments_query, comments_file)
    if args.items:
        download_query(engine, items_query, items_file)
    if args.users:
        download_query(engine, users_query, users_file)
    logging.info(f"✅ All requested data saved")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--titles", action="store_true", help="Download titles data")
    parser.add_argument("--comments", action="store_true", help="Download comments data")
    parser.add_argument("--items", action="store_true", help="Download items data")
    parser.add_argument("--users", action="store_true", help="Download users data")
    args = parser.parse_args()
    download_hn_data(args)
