from dotenv import load_dotenv
import os
import pandas as pd
from sqlalchemy import create_engine
import psycopg2

# Load environment variables from .env file
load_dotenv()

# Get database connection parameters from environment variables
dbname = os.getenv("PGDATABASE")
user = os.getenv("PGUSER")
password = os.getenv("PGPASSWORD")
host = os.getenv("PGHOST")
port = os.getenv("PGPORT")

def get_engine(echo=False):
    return create_engine(f"postgresql://{user}:{password}@{host}:{port}/{dbname}", echo=echo)


def get_psycopg2_conn():
    return psycopg2.connect(
        dbname=dbname, user=user, password=password, host=host, port=port
    )

# def heart_rate_estimation(
#     engine = get_engine(),
#     table_name: str = "bed_data",
#     start_time: str = "2022-06-20 00:00:00",
#     end_time: str = "2022-06-20 01:00:00",
# ):
#     sql = f"""
#     SELECT 
#         * 
#     FROM 
#         heart_rate_estimation('{table_name}', '{start_time}'::timestamp, '{end_time}'::timestamp);
#     """
    
#     df = pd.read_sql(sql, engine)
    
#     return df