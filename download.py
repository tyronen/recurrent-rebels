import os
import logging
import csv
from sqlalchemy import create_engine, text

# Run pip install sqlalchemy psycopg2-binary before running this

db_url = "postgresql://sy91dhb:g5t49ao@178.156.142.230:5432/hd64m1ki"
titles_file = "data/titles.txt"
comments_file = "data/comments.txt"
items_file = "data/items.txt"

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S"
)


def download_query(engine, query, outfile):
    name = os.path.splitext(os.path.basename(outfile))[0]
    logging.info(f"Downloading {name}...")
    count = 0

    with engine.connect().execution_options(stream_results=True) as conn:
        result = conn.execution_options(yield_per=100000).execute(text(query))
        with open(outfile, "w", encoding="utf-8", newline="") as f:
            writer = None
            for row in result:
                if not any(row):
                    continue
                if len(row) == 1:
                    f.write(str(row[0]) + "\n")
                else:
                    if writer is None:
                        writer = csv.writer(f)
                        writer.writerow(result.keys())
                    writer.writerow(row)
                count += 1
                if count % 200000 == 0:
                    logging.info(f"Downloaded {count} {name}...")

    logging.info(f"✅ Downloaded {count} {name} to {outfile}")


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
