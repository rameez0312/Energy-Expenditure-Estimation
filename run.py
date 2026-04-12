from pathlib import Path
import pandas as pd

from src.pipeline.main import run_full_pipeline
from src.config import BASE_DIR

if __name__ == "__main__":   
    # Load the study file
    study_df = pd.read_csv(BASE_DIR / "Study_Information.csv")
        
    final_df = run_full_pipeline(
        base_dir=BASE_DIR,
        study_df=study_df,
        save_intermediate=True,
        save_final=True,
    )
    print("[INFO] Pipeline complete!")
    print(final_df.shape)

