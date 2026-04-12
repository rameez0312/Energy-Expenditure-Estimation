import pandas as pd
import numpy as np
from datetime import timedelta

from src.data.activity import build_activity_df


def load_vo2_data(path):
    df = pd.read_csv(path)
    df["time"] = pd.to_datetime(df["Time"])
    return df.sort_values("time").drop_duplicates("time")


def label_activity(df, activity_df):
    return pd.merge_asof(
        df.sort_values("time"),
        activity_df.sort_values("time"),
        on="time",
        direction="backward"
    )


def attach_activity_start(df, activity_df):
    activity_starts = activity_df.rename(columns={"time": "activity_start"})

    return pd.merge_asof(
        df,
        activity_starts,
        left_on="time",
        right_on="activity_start",
        by="activity",
        direction="backward"
    )


def trim_edges_groupwise(df, trim_seconds=60):

    segments = []

    for act, seg in df.groupby("activity"):

        start = seg["time"].min() + timedelta(seconds=trim_seconds)
        end = seg["time"].max() - timedelta(seconds=trim_seconds)

        seg_trim = seg[
            (seg["time"] >= start) &
            (seg["time"] <= end)
        ]

        if not seg_trim.empty:
            segments.append(seg_trim)

    if not segments:
        return pd.DataFrame()

    return pd.concat(segments, ignore_index=True)


def process_vo2_participant(
    participant_id,
    vo2_path,
    study_df,
    vo2_col="VO2[mL/kg/min]",
    window_minutes=5
):

    df = load_vo2_data(vo2_path)
    activity_df = build_activity_df(participant_id, study_df)

    # #debug
    # print(df.columns)
    # print(activity_df.columns)

    df = label_activity(df, activity_df)
    df = attach_activity_start(df, activity_df)

    df["time_since_start"] = df["time"] - df["activity_start"]

    df = df[df["time_since_start"] <= pd.Timedelta(minutes=window_minutes)]

    # clean invalid values
    df.loc[df[vo2_col] <= 0, vo2_col] = np.nan

    # smoothing
    df["VO2_smooth"] = (
        df.groupby("activity")[vo2_col]
        .rolling(window=15, center=True)
        .mean()
        .reset_index(level=0, drop=True)
    )

    df["MET"] = df["VO2_smooth"] / 3.5

    df = trim_edges_groupwise(df)

    df["participant"] = participant_id

    return df, activity_df