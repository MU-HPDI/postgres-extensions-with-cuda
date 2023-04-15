from dotenv import load_dotenv
import os
import pandas as pd
from sqlalchemy import create_engine

# Load environment variables from .env file
load_dotenv()

# Get database connection parameters from environment variables
dbname = os.getenv("PGDATABASE")
user = os.getenv("PGUSER")
password = os.getenv("PGPASSWORD")
host = os.getenv("PGHOST")
port = os.getenv("PGPORT")

if __name__ == "__main__":
    engine = create_engine(f"postgresql://{user}:{password}@{host}:{port}/{dbname}")

    # try to create table
    try:
        engine.execute(
            """
        CREATE TABLE IF NOT EXISTS bed_data (
            tstmp timestamp without time zone PRIMARY KEY,
            selected_filter smallint[]
        );
        """
        )
    except:
        print("Table already exists")

    df = pd.read_csv("scripts/bed_data.csv")

    try:
        df.to_sql(
            "bed_data",
            engine,
            if_exists="append",
            index=False,
        )
        print(f"Inserted {len(df)} records into bed_data table")
    except:
        print("No records inserted")


