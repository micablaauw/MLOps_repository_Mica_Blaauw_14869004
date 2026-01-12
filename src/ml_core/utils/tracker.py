import csv
from pathlib import Path
from typing import Any, Dict
import yaml

# TODO: Add TensorBoard Support

class ExperimentTracker:
    def __init__(
        self,
        experiment_name: str,
        config: Dict[str, Any],
        base_dir: str = "experiments/results",
    ):
        self.run_dir = Path(base_dir) / experiment_name
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # TODO: Save config to yaml in run_dir
        config_path = self.run_dir / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)
        print(f"Saved config to {config_path}")
        self.csv_path = self.run_dir / "metrics.csv"
        self.csv_file = open(self.csv_path, "w", newline="")
        self.csv_writer = csv.writer(self.csv_file)
        
        # Header (TODO: add the rest of things we want to track, loss, gradients, accuracy etc.)
        self.csv_path = self.run_dir / "metrics.csv"
        self.csv_file = open(self.csv_path, "w", newline="")
        self.csv_writer = csv.writer(self.csv_file)
        self.metrics_keys = ["train_loss", "val_loss", "train_acc", "val_acc", "lr"]
        self.csv_writer.writerow(["epoch"])

    def log_metrics(self, epoch: int, metrics: Dict[str, float]):
        """
        Writes metrics to CSV (and TensorBoard).
        """
        # TODO: Write other useful metrics to CSV
        row = [epoch] + [metrics.get(k, None) for k in self.metrics_keys]
        self.csv_writer.writerow([row]) # Currently only logging epoch
        self.csv_file.flush()
        # TODO: Log to TensorBoard

    def get_checkpoint_path(self, filename: str) -> str:
        return str(self.run_dir / filename)

    def close(self):
        self.csv_file.close()
