# Medical Insurance Cost Prediction

## Project overview
This beginner-friendly project predicts medical insurance charges from a tabular dataset using two regression models: Linear Regression (baseline) and Random Forest Regressor (main). It includes simple preprocessing in pandas and saves plots and the trained model to disk.

## Dataset
Place `dataset.csv` in the project root with the following columns:
- age (int)
- sex (male/female)
- bmi (float)
- children (int)
- smoker (yes/no)
- region (N/E/S/W/NW/NE/SW/SE)
- charges (float, target)

## Install dependencies
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run training
```powershell
python .\src\train.py
```
This will:
- train both models
- print metrics for each
- save plots to `reports/`
- save the Random Forest model to `models/rf_model.joblib`

## Run prediction
```powershell
python .\src\predict.py
```
This uses a hardcoded sample input in the script and prints the predicted charges.

## Preprocessing and models (brief)
- Categorical columns are mapped to numeric values:
  - sex: male/female -> 0/1
  - smoker: yes/no -> 1/0
- region is one-hot encoded using `pd.get_dummies(..., drop_first=True)`.
- Linear Regression is the baseline model.
- Random Forest Regressor is the main model and is saved for later predictions.
