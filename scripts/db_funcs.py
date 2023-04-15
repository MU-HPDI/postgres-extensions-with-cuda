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

def heart_rate_estimation(
    engine = get_engine(),
    table_name: str = "bed_data",
    start_time: str = "2021-05-29 00:00:00",
    end_time: str = "2021-06-07 00:00:00",
    limit: int = 100,
    hardware: str = "GPU",
    version: str = "v1.0",
):
    sql = f"""
    SELECT 
        * 
    FROM 
        heart_rate_estimation('{table_name}', '{start_time}'::TIMESTAMP, '{end_time}'::TIMESTAMP, {limit}, '{hardware}', '{version}');
    """
    return sql
    # print(sql)
    
    # df = pd.read_sql(sql, engine)
    
    # return df

def get_timing_results(
    engine = get_engine(),
    experiment_version = "1.0",
) -> pd.DataFrame:
    sql = f"""
    SELECT 
        * 
    FROM 
        heart_rate_timings
    WHERE
        experiment_version = '{experiment_version}';
    """
    
    df = pd.read_sql(sql, engine)
    
    return df