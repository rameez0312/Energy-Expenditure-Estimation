import pandas as pd
import numpy as np
from tqdm import tqdm

from src.features.windowing import generate_window_starts
from src.features.imu_features import create_imu_windows, extract_imu_features
from src.features.zephyr_vo2_features import extract_zephyr_vo2_features


def align_modalities(imu_sub, zephyr_sub, vo2_sub):

    start_time = max(
        imu_sub["time"].min(),
        zephyr_sub["time"].min(),
        vo2_sub["time"].min()
    )

    end_time = min(
        imu_sub["time"].max(),
        zephyr_sub["time"].max(),
        vo2_sub["time"].max()
    )

    imu_sub = imu_sub[(imu_sub["time"] >= start_time) & (imu_sub["time"] <= end_time)]
    zephyr_sub = zephyr_sub[(zephyr_sub["time"] >= start_time) & (zephyr_sub["time"] <= end_time)]
    vo2_sub = vo2_sub[(vo2_sub["time"] >= start_time) & (vo2_sub["time"] <= end_time)]

    return imu_sub, zephyr_sub, vo2_sub, start_time, end_time


def generate_window_dataset(
    zephyr,
    imu,
    vo2,
    demo_df,
    window_size=None,
    step_size=None
):

    if step_size is None:
        step_size = window_size // 2

    all_windows = []
    participants = imu["participant"].unique()
    
    for pid in tqdm(participants, desc="Processing participants"):
        activities = imu["activity"].unique()
        
        for act in tqdm(activities, desc=f"{pid}", leave=False):
            imu_sub = imu[(imu["participant"] == pid) & (imu["activity"] == act)]
            zephyr_sub = zephyr[(zephyr["participant"] == pid) & (zephyr["activity"] == act)]
            vo2_sub = vo2[(vo2["participant"] == pid) & (vo2["activity"] == act)]

            if imu_sub.empty or zephyr_sub.empty or vo2_sub.empty:
                continue

            imu_sub, zephyr_sub, vo2_sub, start, end = align_modalities(
                imu_sub, zephyr_sub, vo2_sub
            )

            # -------------------------
            # INTERPOLATION
            # -------------------------
            for df_mod in [imu_sub, zephyr_sub, vo2_sub]:
                num_cols = df_mod.select_dtypes(include=[np.number]).columns
                df_mod[num_cols] = df_mod[num_cols].interpolate(limit=2)

            # -------------------------
            # CENTRAL WINDOW DEFINITION
            # -------------------------
            window_starts = generate_window_starts(
                start, end, window_size, step_size
            )

            if not window_starts:
                continue

            # -------------------------
            # IMU
            # -------------------------
            imu_windowed = create_imu_windows(
                imu_sub, window_starts, window_size
            )

            if imu_windowed.empty:
                continue

            imu_features = extract_imu_features(imu_windowed)

            # -------------------------
            # ZEPHYR + VO2
            # -------------------------
            zephyr_features, vo2_labels = extract_zephyr_vo2_features(
                zephyr_sub,
                vo2_sub,
                window_starts,
                window_size
            )

            imu_features = imu_features.reset_index().rename(columns={"index": "window_id"})

            merged = imu_features.merge(zephyr_features, on="window_id",how ="inner")
            merged = merged.merge(vo2_labels, on="window_id", how="inner")

            merged["participant"] = pid
            merged["activity"] = act

            merged = merged.merge(
                demo_df,
                left_on="participant",
                right_on="participant",
                how="left"
            ).drop(columns=["participant", "Comments"], errors="ignore")

            # Gender mapping
            if "Gender" in merged.columns:
                merged["Gender"] = merged["Gender"].map({"M": 0, "F": 1})

            all_windows.append(merged)

    if not all_windows:
        return pd.DataFrame()

    dataset = pd.concat(all_windows, ignore_index=True)
    dataset = dataset.dropna()

    return dataset