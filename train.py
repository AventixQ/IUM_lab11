import argparse
import pathlib
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

COLUMN_NAMES = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin",
    "BMI", "DiabetesPedigree", "Age", "Outcome"
]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--random_state", type=int, default=42)
    ap.add_argument("--c", type=float, default=1.0)
    args = ap.parse_args()

    csv_path = pathlib.Path(args.data_dir) / "diabetes.csv"
    df = pd.read_csv(csv_path, header=None, names=COLUMN_NAMES)

    X, y = df.drop("Outcome", axis=1), df["Outcome"]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=args.test_size,
        random_state=args.random_state, stratify=y
    )

    model = LogisticRegression(
        max_iter=500, solver="liblinear", C=args.c, random_state=args.random_state
    )
    model.fit(X_tr, y_tr)

    joblib.dump(model, "model.pkl")
    print("✅  model.pkl zapisany")

    print(f"Train acc: {accuracy_score(y_tr, model.predict(X_tr)):.4f}")
    print(f"Val   acc: {accuracy_score(y_te, model.predict(X_te)):.4f}")

if __name__ == "__main__":
    main()
