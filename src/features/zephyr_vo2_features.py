import pandas as pd
import numpy as np
from tqdm import tqdm

# For HR, BR and hrv feature extraction and VO2(MET) label mean.
def extract_zephyr_vo2_features(zephyr_df, vo2_df, window_starts, window_size):

    zephyr_df = zephyr_df.set_index("time")
    vo2_df = vo2_df.set_index("time")

    zephyr_features = []
    vo2_labels = []

    for current in tqdm(window_starts, desc="Processing Zephyr windows", leave=False):

        zephyr_window = zephyr_df.loc[current:current + pd.Timedelta(seconds=window_size)]
        vo2_window = vo2_df.loc[current:current + pd.Timedelta(seconds=window_size)]

        if len(zephyr_window) > 0 and len(vo2_window) > 0:

            zephyr_features.append({
                "window_id": current,

                "zephyr_mean": zephyr_window["HR"].mean(),
                "zephyr_std": zephyr_window["HR"].std(),
                "zephyr_min": zephyr_window["HR"].min(),
                "zephyr_max": zephyr_window["HR"].max(),

                "hr_slope": np.polyfit(
                    np.arange(len(zephyr_window)),
                    zephyr_window["HR"],
                    1
                )[0],

                "br_mean": zephyr_window["BR"].mean(),
                "br_std": zephyr_window["BR"].std(),

                "br_amp_mean": zephyr_window["BRAmplitude"].mean(),
                "br_amp_std": zephyr_window["BRAmplitude"].std(),

                "hrv_mean": zephyr_window["HRV"].median(),
                "hrv_std": zephyr_window["HRV"].std(),

                "ventilation_proxy":
                    zephyr_window["BR"].mean() *
                    zephyr_window["BRAmplitude"].mean()
            })

            vo2_labels.append({
                "window_id": current,
                "vo2_mean": vo2_window["MET"].mean()
            })

    return pd.DataFrame(zephyr_features), pd.DataFrame(vo2_labels)