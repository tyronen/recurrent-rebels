import logging
import pandas as pd
from sqlalchemy import create_engine, text

# Run pip install sqlalchemy psycopg2-binary before running this

db_url = "postgresql://sy91dhb:g5t49ao@178.156.142.230:5432/hd64m1ki"
titles_file = "data/titles.parquet"
comments_file = "data/comments.parquet"
items_file = "data/items.parquet"

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


def download_hn_data():
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
                     WHERE length(text) > 2048
                       AND CARDINALITY(KIDS) > 4
                       AND dead IS NULL"""

    # Query 3: All items
    items_query = """
                    SELECT id, "by", title, url, time, text, score 
                    FROM hacker_news.items
                    WHERE dead IS NULL AND type='story'"""

    download_query(engine, titles_query, titles_file)
    download_query(engine, comments_query, comments_file)
    download_query(engine, items_query, items_file)
    logging.info(f"✅ All data saved")


if __name__ == "__main__":
    download_hn_data()
