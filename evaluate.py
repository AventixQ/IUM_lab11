import argparse

print("Evaluation...")
parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", required=True)
args = parser.parse_args()

print(f"Evaluate on {args.data_dir}")
