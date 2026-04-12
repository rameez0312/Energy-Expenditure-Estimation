import pandas as pd
from pathlib import Path

from src.data.earbuds_imu import process_imu_participant
from src.data.activity import build_activity_df


def run_imu_pipeline(base_dir, study_df):

    participants = [f"P{i:02d}" for i in range(1, 18) if i != 14]

    all_data = []

    for pid in participants:

        imu_path = Path(base_dir) / pid / "EARBUDS" / f"{pid}-imu-right.csv"

        activity_df = build_activity_df(pid, study_df)

        segments = process_imu_participant(pid, imu_path, activity_df)

        for activity, seg in segments.items():

            seg = seg.copy()
            seg["participant"] = pid
            seg["activity"] = activity

            all_data.append(seg)

    final_df = pd.concat(all_data, ignore_index=True)


    return final_df