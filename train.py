"""
Usage:
    python train.py --config configs/infra/xbd/unet.yaml
    python train.py --config configs/infra/xbd/unet.yaml --override optimizer.learning_rate 0.0005 training.num_epochs 50
    python train.py --config configs/infra/xbd/unet.yaml --resume results/xbd/unet/latest.pth
"""
import argparse
from src.core.config import Config
from src.core.trainer import Trainer

def main():
    parser = argparse.ArgumentParser(description="Unified Disaster Trainer")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--override", nargs="*", default=[], help="Key-value pairs to override config")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint for resuming")
    args = parser.parse_args()

    config = Config.from_yaml(args.config)
    if args.override:
        config.apply_overrides(args.override)
    if args.resume:
        config.resume = args.resume

    trainer = Trainer(config)
    trainer.train()

if __name__ == "__main__":
    main()
