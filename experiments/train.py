import argparse
import torch
import torch.optim as optim
from ml_core.data.loader import get_dataloaders
from ml_core.models.mlp import MLP
from ml_core.solver.trainer import Trainer
from ml_core.utils import load_config, seed_everything, setup_logger

def main(args):
    #1 Load config & set seed
    config = load_config(args.config)
    seed_everything(config.get("seed", 42))

    #2 Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    #3 Setup logger
    logger = setup_logger("Experiment_Runner")
    logger.info(f"Using device: {device}")

    #4 Load data
    train_loader, val_loader = get_dataloaders(config)

    #5 Initialize model
    model = MLP(
        input_dim=int(torch.prod(torch.tensor(config["data"]["input_shape"]))),
        hidden_units=config["model"]["hidden_units"],
        dropout_rate=config["model"]["dropout_rate"],
        output_dim=config["model"]["num_classes"]
    ).to(device)

    #6 Optimizer
    optimizer_cfg = config.get("optimizer", {})
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["training"]["learning_rate"]
    )

    #7 Trainer & Fit
    trainer = Trainer(model, optimizer, config, device)
    trainer.fit(train_loader, val_loader)
    logger.info("Training finished!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Simple MLP on PCAM")
    parser.add_argument("--config", type=str, required=True, help="Path to config yaml")
    args = parser.parse_args()

    main(args)
