import numpy as np
import pandas as pd
from tsfresh import extract_features

custom_fc_parameters = {
#     "sum_values": None,
#     "median": None,
#     "mean": None,
#     "length": None,
#     "standard_deviation": None,
#     "variance": None,
#     "root_mean_square": None,
#     "maximum": None,
#     "absolute_maximum": None,
#     "minimum": None,
    "mean": None,
    "median": None,
    "standard_deviation": None,
    "variance": None,
    "maximum": None,
    "minimum": None,
    "root_mean_square": None,
    "absolute_maximum": None,
    "mean_abs_change": None,
    "number_peaks": [{"n": 5}],
    "skewness": None,
    "kurtosis": None,

    "fft_coefficient": [{"coeff":1,"attr":"abs"}],
    "fft_aggregated": [{"aggtype":"centroid"}]
}


def create_imu_windows(df, window_starts, window_size):

    df = df.set_index("time")
    windows = []

    for current in window_starts:

        window_df = df.loc[current:current + pd.Timedelta(seconds=window_size)]

        if len(window_df) > 0:
            window_df = window_df.copy()
            window_df["window_id"] = current
            windows.append(window_df)

    if not windows:
        return pd.DataFrame()

    return pd.concat(windows).reset_index()


def extract_imu_features(imu_windowed):

    imu_windowed["acc_rmssd"] = (
        imu_windowed["acc_mag"].diff()**2
    ).rolling(2).mean()**0.5

    imu_tsfresh = imu_windowed[[
        "window_id",
        "time",
        "acc_mag",
        "gyro_mag",
        "acc_energy",
        "gyro_energy",
        "movement_intensity"
    ]]

    features = extract_features(
        imu_tsfresh,
        column_id="window_id",
        column_sort="time",
        default_fc_parameters = custom_fc_parameters,
        disable_progressbar=False
    )

    return features