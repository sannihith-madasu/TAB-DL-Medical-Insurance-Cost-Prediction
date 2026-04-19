from pathlib import Path
import sys

import pandas as pd
import joblib

ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT_DIR / "models" / "rf_model.joblib"


def exit_with_message(message: str) -> None:
    print(f"Error: {message}")
    sys.exit(1)


def load_model(path: Path):
    if not path.exists():
        exit_with_message(
            "Model file not found. Run train.py first to create models/rf_model.joblib."
        )

    try:
        package = joblib.load(path)
    except Exception as exc:
        exit_with_message(f"Failed to load model file. Details: {exc}")

    if isinstance(package, dict) and "model" in package:
        model = package["model"]
        feature_names = package.get("feature_names")
    else:
        model = package
        feature_names = None

    if not feature_names:
        exit_with_message(
            "Model file is missing feature names. Re-run train.py to regenerate it."
        )

    return model, feature_names


def preprocess_input(sample: dict, feature_names: list[str]) -> pd.DataFrame:
    df = pd.DataFrame([sample])

    df["sex"] = df["sex"].map({"male": 0, "female": 1})
    df["smoker"] = df["smoker"].map({"yes": 1, "no": 0})

    if df["sex"].isna().any():
        exit_with_message("Unexpected value in 'sex'. Expected male/female.")
    if df["smoker"].isna().any():
        exit_with_message("Unexpected value in 'smoker'. Expected yes/no.")

    df = pd.get_dummies(df, columns=["region"], drop_first=True)

    # Align columns to the training feature set, filling missing dummies with 0.
    df = df.reindex(columns=feature_names, fill_value=0)

    return df


def main() -> None:
    model, feature_names = load_model(MODEL_PATH)

    # Sample input (edit this to test different values)
    sample_input = {
        "age": 35,
        "sex": "female",
        "bmi": 28.5,
        "children": 1,
        "smoker": "no",
        "region": "northwest",
    }

    X_input = preprocess_input(sample_input, feature_names)
    prediction = model.predict(X_input)[0]

    print("Sample input:")
    print(sample_input)
    print(f"Predicted charges: {prediction:.2f}")


if __name__ == "__main__":
    main()
