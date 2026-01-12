import argparse
from pathlib import Path
from typing import Optional
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def parse_args():
    parser = argparse.ArgumentParser(description="Plot training metrics.")
    parser.add_argument("--input_csv", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=None)
    return parser.parse_args()

def load_data(file_path: Path) -> pd.DataFrame:
    # TODO: Load CSV into Pandas DataFrame
    if not file_path.exists():
        raise FileNotFoundError(f"CSV file not found: {file_path}")

    df = pd.read_csv(file_path)
    return df

def setup_style():
    # TODO: Set seaborn theme
    sns.set_theme(style="whitegrid")
    sns.set_context("talk")

def plot_metrics(df: pd.DataFrame, output_path: Optional[Path]):
    """
    Generate and save plots for Loss, Accuracy, and F1.
    """
    if df is None: return
    if output_path is not None:
        output_path.mkdir(parents=True, exist_ok=True)
    # Create a figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # TODO: Plot Train/Val Loss
    axes[0, 0].plot(df["epoch"], df["train_loss"], label="Train Loss")
    axes[0, 0].plot(df["epoch"], df["val_loss"], label="Val Loss")
    axes[0, 0].set_title("Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend()
    # TODO: Plot Train/Val Accuracy
    axes[0, 1].plot(df["epoch"], df["train_acc"], label="Train Accuracy")
    axes[0, 1].plot(df["epoch"], df["val_acc"], label="Val Accuracy")
    axes[0, 1].set_title("Accuracy")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Accuracy")
    axes[0, 1].legend()
    # TODO: Plot Learning Rate
    if "lr" in df.columns:
        axes[1, 0].plot(df["epoch"], df["lr"])
        axes[1, 0].set_title("Learning Rate")
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_ylabel("LR")
    else:
        axes[1, 0].axis("off")
    axes[1, 1].axis("off")
    plt.tight_layout()
    if output_path is not None:
        plt.savefig(output_path / "training_metrics.png", dpi=150)
        print(f"Saved plot to {output_path / 'training_metrics.png'}")
    else:
        plt.show()

def main():
    args = parse_args()
    setup_style()
    df = load_data(args.input_csv)
    plot_metrics(df, args.output_dir)

if __name__ == "__main__":
    main()
