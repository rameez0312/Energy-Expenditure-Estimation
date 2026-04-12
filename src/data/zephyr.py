import pandas as pd
import numpy as np
from pathlib import Path
from datetime import timedelta
from src.data.activity import build_activity_df


# -----------------------------
# LOAD DATA
# -----------------------------
def load_zephyr_data(path):
    df = pd.read_csv(path)
    df["time"] = pd.to_datetime(df["Time"], dayfirst=True)
    df = df.sort_values("time").drop_duplicates("time")
    return df


# -----------------------------
# ACTIVITY LABELING
# -----------------------------
# ACTIVITIES = [
#     "sitting", "standing", "cycling1",
#     "cycling2", "running1", "running2"
# ]

# COLUMNS = [
#     "Start_Sit", "Start_Stand", "Start_Cycle1",
#     "Start_Cycle2", "Start_Run1", "Start_Run2"
# ]


# def build_activity_df(pid, study_df):
#     row = study_df[study_df["Participant"] == pid].iloc[0]

#     df = pd.DataFrame({
#         "time": [row[c] for c in COLUMNS],
#         "activity": ACTIVITIES
#     })

#     df["time"] = pd.to_datetime(df["time"])
#     return df.sort_values("time")


def label_activity(df, activity_df):
    return pd.merge_asof(
        df.sort_values("time"),
        activity_df,
        on="time",
        direction="backward"
    )


def attach_activity_start(df, activity_df):
    df = df.sort_values("time")
    activity_starts = activity_df.rename(columns={"time": "activity_start"}).sort_values("activity_start")

    return pd.merge_asof(
        df,
        activity_starts,
        left_on="time",
        right_on="activity_start",
        by="activity",
        direction="backward"
    )


# -----------------------------
# CLEANING (HRV)
# -----------------------------
def clean_HRV_signal(df):
    df = df.copy()
    signal_col = "HRV"
    df[signal_col] = df[signal_col].replace(65535, np.nan)
    df.loc[df[signal_col] > 300, signal_col] = np.nan
    df.loc[df[signal_col] < 0, signal_col] = np.nan

    return df


# -----------------------------
# TRIMMING
# -----------------------------
def trim_edges_groupwise(df, trim_minutes=1):

    segments = []

    for act, seg in df.groupby("activity"):

        start = seg["time"].min() + timedelta(minutes=trim_minutes)
        end = seg["time"].max() - timedelta(minutes=trim_minutes)

        seg_trim = seg[
            (seg["time"] >= start) &
            (seg["time"] <= end)
        ].copy()

        segments.append(seg_trim)

    return pd.concat(segments, ignore_index=True)


# -----------------------------
# MAIN PIPELINE (PER PARTICIPANT)
# -----------------------------
def process_zephyr_participant(
    participant_id,
    zephyr_path,
    study_df,
    trim_hr=True
):

    df = load_zephyr_data(zephyr_path)
    activity_df = build_activity_df(participant_id, study_df)

    df = label_activity(df, activity_df)
    df = attach_activity_start(df, activity_df)

    # time since activity start
    df["time_since_start"] = df["time"] - df["activity_start"]

    if trim_hr:
        df = df[df["time_since_start"] <= pd.Timedelta(minutes=5)]

    # trim edges
    df = trim_edges_groupwise(df, trim_minutes=1)

    # clean signal
    df = clean_HRV_signal(df)

    df["participant"] = participant_id
    # df = df.rename(columns={"time": "time"})

    return df, activity_df