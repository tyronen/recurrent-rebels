# download_hn_data.py
from sqlalchemy import create_engine, text

# Run pip install sqlalchemy psycopg2-binary before running this

db_url="postgresql://sy91dhb:g5t49ao@178.156.142.230:5432/hd64m1ki"
titles_file="titles.txt"
comments_file="comments.txt"

def download_hn_data():
    """Download HN titles and comments from PostgreSQL"""

    # Create database connection
    print("Connecting to database...")
    engine = create_engine(db_url)

    # Query 1: High-scoring titles
    titles_query = """
                   SELECT title
                   FROM hacker_news.items
                   WHERE title IS NOT NULL
                     AND dead IS NULL
                     AND score > 2 \
                   """

    # Query 2: Long comments with children
    comments_query = """
                     SELECT text
                     FROM hacker_news.items
                     WHERE length(text) > 2048
                       AND CARDINALITY(KIDS) > 4
                       AND dead IS NULL \
                     """

    # Download titles
    print("Downloading titles...")
    count = 0

    with engine.connect() as conn:
        result = conn.execute(text(titles_query))
        with open(titles_file, 'w', encoding='utf-8') as f:
            for row in result:
                if row[0]:  # Check if title is not None
                    f.write(row[0] + '\n')
                    count += 1
                    if count % 100000 == 0:
                        print(f"Downloaded {count} titles...")

    print(f"✅ Downloaded {count} titles to {titles_file}")

    # Download comments
    print("Downloading comments...")
    count = 0

    with engine.connect() as conn:
        result = conn.execute(text(comments_query))
        with open(comments_file, 'w', encoding='utf-8') as f:
            for row in result:
                if row[0]:  # Check if text is not None
                    f.write(row[0] + '\n')
                    count += 1
                    if count % 10000 == 0:
                        print(f"Downloaded {count} comments...")

    print(f"✅ Downloaded {count} comments to {comments_file}")
    print(f"✅ All data saved")


if __name__ == "__main__":
    download_hn_data()