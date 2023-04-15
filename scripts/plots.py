import pandas as pd
import sys
import os
import matplotlib.pyplot as plt

# Get the absolute path of the parent directory of this script
SCRIPT_DIR = os.path.abspath(os.path.dirname(__file__))

# Add the scripts directory to the Python path
sys.path.insert(0, os.path.join(SCRIPT_DIR, 'scripts'))

import db_funcs as db


def perform_experiment(engine, experiment_version):
    num_minutes_arr = [str(x) for x in range(60, 1800, 60)]
    num_minutes_arr.insert(0, "1")
    
    print(f"Running {experiment_version} experiment")
    
    
    for minute in num_minutes_arr:
        with engine.connect() as conn:
            print(f"Running {minute} minute experiment")
            conn.execute(db.heart_rate_estimation(engine, limit=minute, hardware="GPU", version=experiment_version))
            conn.execute(db.heart_rate_estimation(engine, limit=minute, hardware="CPU", version=experiment_version))
            conn.execute("COMMIT;")


def plot_times(df, outname):
    
    cpu_df = df[df["hardware"] == "CPU"]
    gpu_df = df[df["hardware"] == "GPU"]
    
    y_cpu = cpu_df["elapsed_time"].to_list()
    y_gpu = gpu_df["elapsed_time"].to_list()
    
    x_values = cpu_df["number_minutes"].to_list()
    x_values = [int(x) for x in x_values]
    
    ax = plt.gca()
    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("left")
    
    plt.plot(x_values, y_cpu, label="CPU", color="#31648C")
    plt.plot(x_values, y_gpu, label="GPU", color="#74B301")

    plt.xticks(x_values[::10])
    plt.yscale("log")
    outname += "_log_y"

    plt.xlabel("Number of Minutes")
    plt.ylabel("Execution Time (ms)")
    plt.legend(loc="lower right", fontsize=8)

    plt.savefig(f"{outname}.png")
    print(f"Saved plot to {outname}.png")

if __name__ == "__main__":
    engine = db.get_engine()
    experiment_version = "1.1"
    
    with engine.connect() as conn:
        sql = f"""
        SELECT 
            COUNT(*) as count
        FROM 
            heart_rate_timings
        WHERE 
            experiment_version = '{experiment_version}';
        """
        df = pd.read_sql(sql, engine)
        if df["count"][0] == 0:
            perform_experiment(engine, experiment_version)
        else:
            print(f"Experiment {experiment_version} has already been run, please change the version number")
            
            
    df_timings = db.get_timing_results(engine, experiment_version)
    
    plot_times(df_timings, "scripts/timings_results")