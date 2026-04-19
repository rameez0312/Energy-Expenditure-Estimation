
# Energy Expenditure Estimation


This project develops machine learning models to estimate energy expenditure (VO₂) from wearable sensor data, including heart rate (Zephyr), inertial measurement units (IMU), and respiratory signals. The models are validated using Leave-One-Subject-Out (LOSO) cross-validation to ensure generalization across individuals.

## Features

- **Data Sources**: Integrates physiological data from Zephyr HR monitors, earbud IMUs, and VO₂ measurements.
- **Feature Engineering**: Extracts windowed features from time-series data (HR, IMU, respiratory).
- **Models**: Supports linear regression, ridge, XGBoost, random forest, MLP, and custom PyTorch neural networks.
- **Validation**: LOSO cross-validation for robust evaluation.
- **Explainability**: SHAP analysis for model interpretability.
- **Visualization**: Plots for predictions, feature importance, and Bland-Altman analysis.

## Installation

### Prerequisites
- Python 3.11.9
- Conda (recommended for environment management)

### Setup
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd energy-expenditure-estimation
   ```

2. Create and activate the Conda environment:
   ```bash
   conda env create -f environment.yml
   conda activate EEE
   ```


## Usage

### Data Preparation
Place raw data files in `data/raw/` following the expected structure:
- `P01/`, `P02/`, etc., with subfolders for ZEPHYR, EARBUDS, VO2.
- Demographics file: `data/raw/Demographics.csv`.

### Running the Pipeline
Execute the full data processing and fusion pipeline:
```bash
python run.py
```
This generates intermediate and final datasets in `data/interim/` and `data/processed/`.


## Project Organization

```

├── data
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
│
└── src                <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes src a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling
    │   ├── __init__.py
    │   ├── predict.py          <- Code to run model inference with trained models
    │   └── train.py            <- Code to train models
    │
    ├── pipeline
    │   ├── main.py             <- Main pipeline script
    │   ├── zephyr_pipeline.py  <- Zephyr data processing
    │   ├── earbuds_imu_pipeline.py <- IMU data processing
    │   └── MET_GT_pipeline.py  <- VO2 data processing
    │
    └── data                    <- Data loading and processing modules
        ├── activity.py
        ├── earbuds_imu.py
        ├── MET_GT.py
        └── zephyr.py
```

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a feature branch: `git checkout -b feature-name`.
3. Commit your changes: `git commit -am 'Add feature'`.
4. Push to the branch: `git push origin feature-name`.
5. Submit a pull request.






