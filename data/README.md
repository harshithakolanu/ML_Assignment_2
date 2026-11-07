# Data
`raw/` receives the original UCI/Kaggle file. `processed/` receives the duplicate-cleaned dataset. `test_data.csv` is created only from the fixed stratified 20% holdout by `run_training.py`. The full public dataset is intentionally downloaded at execution time to preserve attribution and avoid silently altered values.
