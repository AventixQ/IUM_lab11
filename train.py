import argparse

print("Training...")
parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", required=True)
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--lr", type=float, default=0.001)
args = parser.parse_args()

print(f"Training on {args.data_dir} by {args.epochs} epochs with lr={args.lr}")