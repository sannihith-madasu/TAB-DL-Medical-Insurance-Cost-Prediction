from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib

# Use a non-GUI backend so plots save correctly on Windows without Tk/Tcl.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "dataset.csv"
REPORTS_DIR = ROOT_DIR / "reports"
MODELS_DIR = ROOT_DIR / "models"

REQUIRED_COLUMNS = {
    "age",
    "sex",
    "bmi",
    "children",
    "smoker",
    "region",
    "charges",
}


def exit_with_message(message: str) -> None:
    print(f"Error: {message}")
    sys.exit(1)


def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        exit_with_message(
            "dataset.csv not found in the project root. Please add the file and try again."
        )

    try:
        df = pd.read_csv(path)
    except Exception as exc:
        exit_with_message(f"Failed to read dataset.csv. Details: {exc}")

    missing_cols = REQUIRED_COLUMNS - set(df.columns)
    if missing_cols:
        exit_with_message(f"dataset.csv is missing columns: {sorted(missing_cols)}")

    return df


def preprocess(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    df = df.copy()

    df["sex"] = df["sex"].map({"male": 0, "female": 1})
    df["smoker"] = df["smoker"].map({"yes": 1, "no": 0})

    if df["sex"].isna().any():
        exit_with_message("Unexpected values in 'sex'. Expected male/female.")
    if df["smoker"].isna().any():
        exit_with_message("Unexpected values in 'smoker'. Expected yes/no.")

    df = pd.get_dummies(df, columns=["region"], drop_first=True)

    X = df.drop(columns=["charges"])
    y = df["charges"]

    return X, y


def regression_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    return {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2}


def print_metrics(model_name: str, metrics: dict) -> None:
    print(f"\n{model_name} metrics:")
    print(f"MAE: {metrics['MAE']:.2f}")
    print(f"MSE: {metrics['MSE']:.2f}")
    print(f"RMSE: {metrics['RMSE']:.2f}")
    print(f"R2: {metrics['R2']:.4f}")


def save_plots(df_raw: pd.DataFrame, y_test: pd.Series, rf_pred: np.ndarray) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # Charges distribution
    plt.figure(figsize=(8, 5))
    sns.histplot(df_raw["charges"], bins=30, kde=True, color="steelblue")
    plt.title("Charges Distribution")
    plt.xlabel("Charges")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "charges_distribution.png", dpi=150)
    plt.close()

    # Smoker vs Charges
    plt.figure(figsize=(7, 5))
    sns.boxplot(x="smoker", y="charges", data=df_raw, palette="Set2")
    plt.title("Smoker vs Charges")
    plt.xlabel("Smoker")
    plt.ylabel("Charges")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "smoker_vs_charges.png", dpi=150)
    plt.close()

    # Actual vs Predicted (Random Forest)
    plt.figure(figsize=(7, 5))
    sns.scatterplot(x=y_test, y=rf_pred, color="teal")
    min_val = min(y_test.min(), rf_pred.min())
    max_val = max(y_test.max(), rf_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], "r--")
    plt.title("Actual vs Predicted Charges (Random Forest)")
    plt.xlabel("Actual Charges")
    plt.ylabel("Predicted Charges")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "actual_vs_predicted_rf.png", dpi=150)
    plt.close()


def main() -> None:
    df_raw = load_data(DATA_PATH)
    X, y = preprocess(df_raw)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Baseline model
    linear_model = LinearRegression()
    linear_model.fit(X_train, y_train)
    linear_pred = linear_model.predict(X_test)

    # Main model
    rf_model = RandomForestRegressor(n_estimators=200, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)

    print("\nFirst 10 actual vs predicted (Random Forest):")
    comparison = pd.DataFrame(
        {
            "Actual": y_test.reset_index(drop=True),
            "Predicted": rf_pred,
        }
    )
    print(comparison.head(10).round(2).to_string(index=False))

    print_metrics("Linear Regression", regression_metrics(y_test, linear_pred))
    print_metrics("Random Forest", regression_metrics(y_test, rf_pred))

    feature_names = X.columns.tolist()
    importances = (
        pd.Series(rf_model.feature_importances_, index=feature_names)
        .sort_values(ascending=False)
    )

    print("\nRandom Forest feature importances:")
    for name, value in importances.items():
        print(f"{name}: {value:.4f}")

    save_plots(df_raw, y_test, rf_pred)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / "rf_model.joblib"
    joblib.dump({"model": rf_model, "feature_names": feature_names}, model_path)

    try:
        relative_model_path = model_path.relative_to(ROOT_DIR)
    except ValueError:
        relative_model_path = model_path

    print(f"\nSaved Random Forest model to {relative_model_path}")


if __name__ == "__main__":
    main()
