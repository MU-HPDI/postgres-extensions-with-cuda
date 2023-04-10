import argparse
import psycopg2
import random
from datetime import datetime, timedelta
from dotenv import load_dotenv
import os

SMALL_INT_MAX = 32_767
SMALL_INT_MIN = -32_768

# Load environment variables from .env file
load_dotenv()

# Get database connection parameters from environment variables
dbname = os.getenv("DB_NAME")
user = os.getenv("DB_USER")
password = os.getenv("DB_PASSWORD")
host = os.getenv("DB_HOST")
port = os.getenv("DB_PORT")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num_records",
        type=int,
        default=10,
        help="number of random data points to generate",
    )
    parser.add_argument(
        "--array_length",
        type=int,
        default=1024,
        help="maximum number of values in each array",
    )
    args = parser.parse_args()

    # Set up database connection
    conn = psycopg2.connect(
        dbname=dbname, user=user, password=password, host=host, port=port
    )
    cur = conn.cursor()

    # try to create table
    try:
        cur.execute("CREATE TABLE example (tstmp TIMESTAMP, values smallint[]);")
    except:
        print("Table already exists")
        # rollback changes
        conn.rollback()

    # Generate random data and insert into database
    for i in range(args.num_records):
        # Generate a random timestamp between now and 7 days ago
        timestamp = datetime.now() - timedelta(
            days=random.randint(0, 7),
            minutes=random.randint(0, 59),
            seconds=random.randint(0, 59),
        )

        # Generate a random array of integers with length between 1 and 5
        values = [
            random.randint(SMALL_INT_MIN, SMALL_INT_MAX)
            for _ in range(args.array_length)
        ]

        # Construct and execute SQL statement to insert data into "example" table
        cur.execute(
            "INSERT INTO example (tstmp, values) VALUES (%s, %s)", (timestamp, values)
        )

    # Commit changes and close database connection
    conn.commit()
    cur.close()
    conn.close()

    print("Done")
