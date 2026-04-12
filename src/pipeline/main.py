from pathlib import Path
import pandas as pd
from datetime import datetime

from src.pipeline.zephyr_pipeline import run_zephyr_pipeline
from src.pipeline.earbuds_imu_pipeline import run_imu_pipeline
from src.pipeline.MET_GT_pipeline import run_vo2_pipeline
from src.features.fusion import generate_window_dataset
from src.config import INTERIM_DIR, PROCESSED_DIR, VERSION, WINDOW_SIZE, STEP_SIZE, DEMOGRAPHICS_PATH, EXCLUDED_PARTICIPANTS


def run_full_pipeline(
    base_dir,
    study_df,
    save_intermediate=True,
    save_final=True
):

    # -------------------------
    # CREATE OUTPUT DIRS
    # -------------------------
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # -------------------------
    # RUN PIPELINES
    # -------------------------
    zephyr_df, _ = run_zephyr_pipeline(base_dir, study_df)
    imu_df = run_imu_pipeline(base_dir, study_df)
    vo2_df = run_vo2_pipeline(base_dir, study_df)

    # -------------------------
    # SAVE INTERMEDIATE (OPTIONAL)
    # -------------------------
    if save_intermediate:

        zephyr_path = INTERIM_DIR / f"zephyr_{VERSION}_{timestamp}.csv"
        imu_path = INTERIM_DIR / f"imu_{VERSION}_{timestamp}.csv"
        vo2_path = INTERIM_DIR / f"vo2_{VERSION}_{timestamp}.csv"

        zephyr_df.to_csv(zephyr_path, index=False)
        imu_df.to_csv(imu_path, index=False)
        vo2_df.to_csv(vo2_path, index=False)

        print(f"[INFO] Saved intermediate files:")
        print(f"  {zephyr_path}")
        print(f"  {imu_path}")
        print(f"  {vo2_path}")

    # -------------------------
    # FUSION
    # -------------------------

    def filter_participants(df):
        return df[~df["participant"].isin(EXCLUDED_PARTICIPANTS)]

    demo_df = pd.read_csv(DEMOGRAPHICS_PATH)
    demo_df = demo_df.rename(columns={"Participant": "participant"})

    demo_df = filter_participants(demo_df)
    zephyr_df = filter_participants(zephyr_df)
    imu_df = filter_participants(imu_df)
    vo2_df = filter_participants(vo2_df)

    print("[INFO] Starting fusion + windowing...")

    final_df = generate_window_dataset(
        zephyr_df,
        imu_df,
        vo2_df,
        demo_df,
        window_size= WINDOW_SIZE,
        step_size= STEP_SIZE
    )

    print("[INFO] Fusion complete")

    # -------------------------
    # SAVE FINAL
    # -------------------------
    if save_final:

        final_path = PROCESSED_DIR / f"final_dataset_{VERSION}_{timestamp}.csv"

        final_df.to_csv(final_path, index=False)

        print(f"[INFO] Saved final dataset: {final_path}")

        # -------------------------
        # SAVE METADATA
        # -------------------------
        meta_path = final_path.with_suffix(".meta.txt")

        with open(meta_path, "w") as f:
            f.write(f"version: {VERSION}\n")
            f.write(f"timestamp: {timestamp}\n")
            f.write(f"rows: {len(final_df)}\n")
            f.write(f"columns: {list(final_df.columns)}\n")

        print(f"[INFO] Metadata saved: {meta_path}")

    return final_df


















# from src.pipeline.zephyr_pipeline import run_zephyr_pipeline
# from src.pipeline.earbuds_imu_pipeline import run_imu_pipeline
# from src.pipeline.MET_GT_pipeline import run_vo2_pipeline
# from src.pipeline.fusion_pipeline import build_fusion_dataset


# def run_full_pipeline(base_dir, study_df, save = False):

#     zephyr_df = run_zephyr_pipeline(base_dir, study_df)
#     imu_df = run_imu_pipeline(base_dir, study_df)
#     vo2_df = run_vo2_pipeline(base_dir, study_df)

#     final_df = build_fusion_dataset(
#         zephyr_df,
#         imu_df,
#         vo2_df
#     )

#     if save:
#         output_path = Path("data/processed/final_dataset.csv")
#         output_path.parent.mkdir(parents=True, exist_ok=True)

#         final_df.to_csv(output_path, index=False)
#         print(f"Saved to {output_path}")

#     return final_df