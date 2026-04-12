import pandas as pd
from pathlib import Path

from src.data.MET_GT import process_vo2_participant


def run_vo2_pipeline(base_dir, study_df):

    participants = [f"P{i:02d}" for i in range(1, 18) if i != 14]

    all_data = []

    for pid in participants:

        path = Path(base_dir) / pid / "VO2" / "DataAverage.csv"

        df, _ = process_vo2_participant(
            participant_id=pid,
            vo2_path=path,
            study_df=study_df
        )

        all_data.append(df)

    return pd.concat(all_data, ignore_index=True)