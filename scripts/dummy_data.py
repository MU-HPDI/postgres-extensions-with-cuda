import argparse
import psycopg2
import random
from datetime import datetime, timedelta
from dotenv import load_dotenv
import os

SMALL_INT_MIN = -32_768
SMALL_INT_MAX = 32_767

# Load environment variables from .env file
load_dotenv()

# Get database connection parameters from environment variables
dbname = os.getenv("PGDATABASE")
user = os.getenv("PGUSER")
password = os.getenv("PGPASSWORD")
host = os.getenv("PGHOST")
port = os.getenv("PGPORT")

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
    
    print(f"Inserting {args.num_records} records with array length {args.array_length} into example table")

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
        conn.rollback()
        
        
    records_inserted = 0

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
        records_inserted += 1

    # Commit changes and close database connection
    conn.commit()
    cur.close()
    conn.close()

    print(f"Inserted {records_inserted} records into example table")
    
