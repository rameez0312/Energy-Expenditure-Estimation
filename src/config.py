from pathlib import Path

BASE_DIR = Path("data/raw/dataset") 

# output dirs
INTERIM_DIR = Path("data/interim")
PROCESSED_DIR = Path("data/processed")

DEMOGRAPHICS_PATH = BASE_DIR / "Demographics.csv"

EXCLUDED_PARTICIPANTS = ["P14"]

# experiment settings
WINDOW_SIZE = 30
STEP_SIZE = 15
# version tag
VERSION = "v1"











# from dotenv import load_dotenv
# from loguru import logger

# # Load environment variables from .env file if it exists
# load_dotenv()

# # Paths
# PROJ_ROOT = Path(__file__).resolve().parents[1]
# logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

# DATA_DIR = PROJ_ROOT / "data"
# RAW_DATA_DIR = DATA_DIR / "raw"
# INTERIM_DATA_DIR = DATA_DIR / "interim"
# PROCESSED_DATA_DIR = DATA_DIR / "processed"
# EXTERNAL_DATA_DIR = DATA_DIR / "external"

# MODELS_DIR = PROJ_ROOT / "models"

# REPORTS_DIR = PROJ_ROOT / "reports"
# FIGURES_DIR = REPORTS_DIR / "figures"

# # If tqdm is installed, configure loguru with tqdm.write
# # https://github.com/Delgan/loguru/issues/135
# try:
#     from tqdm import tqdm

#     logger.remove(0)
#     logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
# except ModuleNotFoundError:
#     pass
