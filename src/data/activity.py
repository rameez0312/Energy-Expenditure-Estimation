import pandas as pd

ACTIVITIES = [
    "sitting", "standing", "cycling1",
    "cycling2", "running1", "running2"
]

COLUMNS = [
    "Start_Sit", "Start_Stand", "Start_Cycle1",
    "Start_Cycle2", "Start_Run1", "Start_Run2"
]


def build_activity_df(pid, study_df):
    row = study_df[study_df["Participant"] == pid].iloc[0]

    df = pd.DataFrame({
        "time": [row[c] for c in COLUMNS],
        "activity": ACTIVITIES
    })

    df["time"] = pd.to_datetime(df["time"])
    return df.sort_values("time")