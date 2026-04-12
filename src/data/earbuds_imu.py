import pandas as pd
import numpy as np
from datetime import timedelta
from scipy.signal import butter, filtfilt


# -------------------------
# TIME HANDLING
# -------------------------
def convert_time(df):
    first = str(df['timestamp'].iloc[0])

    if any(c in first for c in ['-', '/', ':']):
        start_time = pd.to_datetime(first)
        ms = pd.to_numeric(df['timestamp'], errors='coerce')
        df['time'] = start_time + pd.to_timedelta(ms, unit='ms')
        df.loc[0, 'time'] = start_time
    else:
        ts = df['timestamp'].astype(float).astype('int64')
        df['time'] = pd.to_datetime(ts, unit='ms')

    return df


# -------------------------
# LOAD
# -------------------------
def load_imu(path):
    df = pd.read_csv(path)
    df = convert_time(df)
    df = df.sort_values('time').drop_duplicates('time')
    return df


# -------------------------
# UNIT CONVERSION
# -------------------------
def convert_units(df):
    df = df.copy()

    df['ax_mg'] = df['ax'] * 0.061
    df['ay_mg'] = df['ay'] * 0.061
    df['az_mg'] = df['az'] * 0.061

    df['gx_mdps'] = df['gx'] * 17.5
    df['gy_mdps'] = df['gy'] * 17.5
    df['gz_mdps'] = df['gz'] * 17.5

    return df


# -------------------------
# SIGNAL PROCESSING
# -------------------------
def compute_fs(df):
    dt = df['time'].diff().median()
    return 1 / dt.total_seconds()


def apply_cma(df, fs):
    df = df.copy()
    window_samples = int(60 * fs)

    for axis in ['ax_mg', 'ay_mg', 'az_mg']:
        df[f'{axis}_cma'] = df[axis].rolling(window_samples, center=True, min_periods=1).mean()
        df[f'{axis}_clean'] = df[axis] - df[f'{axis}_cma']

    return df


def apply_gyro_filter(df, fs):
    df = df.copy()

    b, a = butter(4, [0.5/(0.5*fs), 5.0/(0.5*fs)], btype='band')

    for axis in ['gx_mdps', 'gy_mdps', 'gz_mdps']:
        df[f'{axis}_clean'] = filtfilt(b, a, df[axis])

    return df


# -------------------------
# FEATURE COMPUTATION
# -------------------------
def compute_magnitude(df):
    df = df.copy()

    df["acc_mag"] = np.sqrt(
        df["ax_mg_clean"]**2 +
        df["ay_mg_clean"]**2 +
        df["az_mg_clean"]**2
    )

    df["gyro_mag"] = np.sqrt(
        df["gx_mdps_clean"]**2 +
        df["gy_mdps_clean"]**2 +
        df["gz_mdps_clean"]**2
    )

    df["acc_energy"] = df["acc_mag"] ** 2
    df["gyro_energy"] = df["gyro_mag"] ** 2

    df["movement_intensity"] = df["acc_mag"] * df["gyro_mag"]
    return df


# -------------------------
# SEGMENTATION
# -------------------------
def segment_by_activity(df, activity_df, window_minutes=5):

    segments = {}

    for _, row in activity_df.iterrows():

        activity = row["activity"]
        start = row["time"]
        end = start + timedelta(minutes=window_minutes)

        seg = df[(df["time"] >= start) & (df["time"] < end)]

        if len(seg) == 0:
            continue

        segments[activity] = seg

    return segments


# -------------------------
# MAIN PER-PARTICIPANT PIPELINE
# -------------------------
def process_imu_participant(pid, imu_path, activity_df):

    df = load_imu(imu_path)
    df = convert_units(df)

    segments = segment_by_activity(df, activity_df)

    processed_segments = {}

    for activity, seg in segments.items():

        fs = compute_fs(seg)

        seg = apply_cma(seg, fs)
        seg = apply_gyro_filter(seg, fs)

        # trim edges
        trim_start = seg['time'].min() + timedelta(minutes=1)
        trim_end = seg['time'].max() - timedelta(minutes=1)

        seg = seg[(seg['time'] >= trim_start) & (seg['time'] <= trim_end)]

        if len(seg) == 0:
            continue

        seg = compute_magnitude(seg)

        processed_segments[activity] = seg

    return processed_segments