import pandas as pd
import sys
import os

# Get the absolute path of the parent directory of this script
SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))

# Add the scripts directory to the Python path
sys.path.insert(0, os.path.join(SCRIPT_DIR, 'scripts'))

import db_funcs as db


if __name__ == "__main__":
    engine = db.get_engine()

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


