import pandas as pd
from pathlib import Path
from src.data.zephyr import process_zephyr_participant


def run_zephyr_pipeline(base_dir, study_df):

    all_data = []
    activity_tables = {}

    participants = [f"P{i:02d}" for i in range(1, 18) if i != 14]

    for pid in participants:

        path = Path(base_dir) / pid / "ZEPHYR" / "summary.csv"

        df, act = process_zephyr_participant(
            participant_id=pid,
            zephyr_path=path,
            study_df=study_df
        )

        all_data.append(df)
        activity_tables[pid] = act

    df_all = pd.concat(all_data, ignore_index=True)
    # df_all = df_all.rename(columns={"timestamp": "time"})

    return df_all, activity_tables