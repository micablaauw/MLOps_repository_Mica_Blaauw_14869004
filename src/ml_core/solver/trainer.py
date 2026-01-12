import time
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..utils import ExperimentTracker, setup_logger


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        config: Dict[str, Any],
        device: str,
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.config = config
        self.device = device
        
        # TODO: Define Loss Function (Criterion)
        self.criterion = nn.CrossEntropyLoss()

        # TODO: Initialize ExperimentTracker
        self.tracker = ExperimentTracker(
            experiment_name=config["experiment_name"],
            config=config,
            base_dir=config.get("output_dir", "experiments/results")
        )
        self.logger = setup_logger()
        # TODO: Initialize metric calculation (like accuracy/f1-score) if needed

    def train_epoch(self, dataloader: DataLoader, epoch_idx: int) -> Tuple[float, float, float]:
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        # TODO: Implement Training Loop
        # 1. Iterate over dataloader
        # 2. Move data to device
        # 3. Forward pass, Calculate Loss
        # 4. Backward pass, Optimizer step
        # 5. Track metrics (Loss, Accuracy, F1)
        for inputs, targets in tqdm(dataloader, desc=f"Train Epoch {epoch_idx}"):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
        
        epoch_loss = running_loss / total
        epoch_acc = correct / total
        return epoch_loss, epoch_acc, 0.0

    def validate(self, dataloader: DataLoader, epoch_idx: int) -> Tuple[float, float, float]:
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        # TODO: Implement Validation Loop
        # Remember: No gradients needed here
        with torch.no_grad():
            for inputs, targets in tqdm(dataloader, desc=f"Val Epoch {epoch_idx}"):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        
        val_loss = running_loss / total
        val_acc = correct / total
        return val_loss, val_acc, 0.0

    def save_checkpoint(self, epoch: int, val_loss: float) -> None:
        # TODO: Save model state, optimizer state, and config
        checkpoint_path = self.tracker.get_checkpoint_path(f"checkpoint_epoch_{epoch}.pt")
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_loss": val_loss,
            "config": self.config
        }, checkpoint_path)
        self.logger.info(f"Saved checkpoint: {checkpoint_path}")

    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> None:
        epochs = self.config["training"]["epochs"]
        self.logger.info(f"Starting training for {epochs} epochs...")
        print(f"Starting training for {epochs} epochs...")
        
        for epoch in range(epochs):
            # TODO: Call train_epoch and validate
            # TODO: Log metrics to tracker
            # TODO: Save checkpoints
            train_loss, train_acc, train_f1 = self.train_epoch(train_loader, epoch)
            val_loss, val_acc, val_f1 = self.validate(val_loader, epoch)
            
            self.tracker.log_metrics(epoch, {
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_acc": train_acc,
                "val_acc": val_acc,
                "lr": self.optimizer.param_groups[0]["lr"]
            })
            
            self.save_checkpoint(epoch, val_loss)
	# Remember to handle the trackers properly
        self.tracker.close()
        self.logger.info("Training finished!")
