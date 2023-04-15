from scipy.signal import lfilter
import numpy as np
import pandas as pd
from scipy import signal
from statistics import mean
import time
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os 

load_dotenv()  # take environment variables from .env.

dbname = os.getenv("PGDATABASE")
user = os.getenv("PGUSER")
password = os.getenv("PGPASSWORD")
host = os.getenv("PGHOST")
port = os.getenv("PGPORT")

SLIDE = 500


def get_engine():
    return create_engine(f"postgresql://{user}:{password}@{host}:{port}/{dbname}")


def get_data(
    conn,
    start_time: str = "2022-06-20 00:00:00",
    end_time: str = "2022-06-20 01:00:00",
) -> pd.DataFrame:
    sql = f"""
    SELECT 
        *
    FROM 
        bed_data
    WHERE
        tstmp BETWEEN '{start_time}'::TIMESTAMP AND '{end_time}'::TIMESTAMP
    """
    return pd.read_sql(sql, conn)


def smooth(y, box_pts):
    box = np.ones(box_pts) / box_pts
    # y_smooth = np.convolve(y, box)
    y_smooth = np.convolve(y, box, mode="same")
    return y_smooth


def butter_lowpass_filter(data):
    # fs is 1Hz which is 100 samples/sec after hardware filtering
    # When using 4th order filtfilt, give filterorder = 2 because filtfilt applies a digital filter forward and backward to a signal. So give value 2 to 4th order.
    filterorder = 6  # software filtering 6 order filtering.

    b, a = signal.butter(filterorder, [0.014, 0.2], btype="bandpass")
    # b, a = signal.butter(
    #     N=filterorder, Wn=[0.014, 0.2], btype="bandpass")
    y = lfilter(b, a, data)

    return y


def heart_rate_estimation(input: list):
    # Compute the BTB function
    # Input: None
    # Output: BTB function
    # Path: computebtbfunction.py
    flipped_first_half = np.flip(input[:SLIDE])
    input_with_padding = np.concatenate((flipped_first_half, input))

    output_with_padding = butter_lowpass_filter(input_with_padding)
    # chunks = np.array_split(input_with_padding, 10)
    # output_with_padding = []
    # for chunk in chunks:
    #     output_with_padding.extend(butter_lowpass_filter(chunk))

    output_with_padding = np.array(output_with_padding)
    output = output_with_padding[SLIDE:]

    window_size = 30
    step = 1

    energy_list = []

    for i in range(0, len(output) - window_size + 1, step):
        energy_list.append(sum(output[i : i + window_size] ** 2))

    # print(f"{bcolors.BOLDCYAN}energy_list: {energy_list[:100]}{bcolors.RESET}")

    energy_list = np.array(energy_list)
    output_smooth = smooth(energy_list, 50)

    # print(f"{bcolors.MAGENTA}output_smooth: {output_smooth[:100]}{bcolors.RESET}")

    peaks_idxs = signal.find_peaks(output_smooth)[0]

    # print(f"{bcolors.BOLDMAGENTA}peaks_idxs: {peaks_idxs}{bcolors.RESET}")
    # print(f"{bcolors.BOLDMAGENTA}len(peaks_idxs): {len(peaks_idxs)}{bcolors.RESET}")

    temp_out = energy_list
    loc = []
    pks = []

    for i in range(len(peaks_idxs)):
        peak_value = peaks_idxs[i]
        if peak_value < 20:
            tmp_array = temp_out[peak_value : peak_value + 30]
            max_value = max(tmp_array)
            max_index = np.argmax(tmp_array)
            loc.append(peak_value + max_index)
        elif peak_value > len(temp_out) - 30:
            tmp_array = temp_out[peak_value - 20 : peak_value]
            max_value = max(tmp_array)
            max_index = np.argmax(tmp_array)
            loc.append(peak_value - 21 + max_index)
        else:
            tmp_array = temp_out[peak_value - 19 : peak_value + 30]
            max_value = max(tmp_array)
            max_index = np.argmax(tmp_array)

            loc.append(peak_value - 20 + max_index)
        pks.append(max_value)

    peaks_idxs = loc
    temp_out = input
    loc_new = []
    pksn = []

    for i in range(len(peaks_idxs)):
        peak_value = peaks_idxs[i]
        if peak_value < 40:
            tmp_array = temp_out[peak_value : peak_value + 30]
            max_value = max(tmp_array)
            max_index = np.argmax(tmp_array)
            loc_new.append(peak_value + max_index)
        elif peak_value > len(energy_list) - 40:
            tmp_array = temp_out[peak_value - 40 : peak_value]
            max_value = max(tmp_array)
            max_index = np.argmax(tmp_array)
            loc_new.append(peak_value - 41 + max_index)
        else:
            tmp_array = temp_out[peak_value - 39 : peak_value + 30]
            max_value = max(tmp_array)
            max_index = np.argmax(tmp_array)
            loc_new.append(peak_value - 40 + max_index)
        pksn.append(max_value)

    difference = np.diff(loc_new)
    remove_id = np.where(difference == 0)

    loc_new = np.delete(loc_new, remove_id)
    pksn = np.delete(pksn, remove_id)

    numerator = 100 / np.diff(loc_new)

    b2bhrfinal = numerator * 60
    if len(b2bhrfinal) == 0:
        return 0

    return mean(b2bhrfinal)


if __name__ == "__main__":
    engine = get_engine()
    import time

    start = time.time()

    df = get_data(engine)
    times = df["tstmp"]
    selected_filters = df["selected_filter"]
    estimations = []
    length_filters = []

    for selected_filter, tstmp in zip(selected_filters, times):
        try:
            estimation = heart_rate_estimation(selected_filter)
            estimations.append(estimation)
            length_filters.append(len(selected_filter))

            print(
                f"tstmp: {tstmp}, estimation: {estimation:.2f}, filter: {selected_filter[:5]}, length: {len(selected_filter)}"
            )

            estimation = round(estimation, 2)
        except Exception as e:
            estimations.append(0)
            length_filters.append(0)

