import argparse
import pathlib
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

COLUMN_NAMES = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin",
    "BMI", "DiabetesPedigree", "Age", "Outcome"
]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--random_state", type=int, default=42)
    args = ap.parse_args()

    csv_path = pathlib.Path(args.data_dir) / "diabetes.csv"
    df = pd.read_csv(csv_path, header=None, names=COLUMN_NAMES)

    X, y = df.drop("Outcome", axis=1), df["Outcome"]
    _, X_te, _, y_te = train_test_split(
        X, y, test_size=0.2,
        random_state=args.random_state, stratify=y
    )

    model = joblib.load("model.pkl")
    y_pred = model.predict(X_te)

    acc = accuracy_score(y_te, y_pred)
    report = classification_report(y_te, y_pred, digits=4)
    cm = confusion_matrix(y_te, y_pred)

    print(f"Test accuracy: {acc:.4f}\n")
    print(report)
    print("Confusion matrix:\n", cm)

    with open("evaluation.txt", "w") as f:
        f.write(f"Accuracy: {acc:.4f}\n\n{report}\nConfusion matrix:\n{cm}\n")
    print("File is ready")

if __name__ == "__main__":
    main()
