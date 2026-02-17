import argparse
from pathlib import Path

import torch
import yaml

from sac import SAC_DINO
from utils.env import create_environment
from utils.misc import load_config, set_seeds

parent_dir = Path(__file__).resolve().parent
device = "cuda" if torch.cuda.is_available() else "cpu"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arguments for SAC_DINO training")
    parser.add_argument(
        "--domain-name",
        type=str,
        default="cartpole",
        help="Domain name of the task in DM-Control Suite",
    )
    parser.add_argument(
        "--task-name",
        type=str,
        default="swingup",
        help="Task name of the task in DM-Control Suite",
    )
    parser.add_argument(
        "--model-config-path",
        type=str,
        default="configs/model_config.yml",
        help="Path to the model configuration YAML file",
    )
    parser.add_argument(
        "--train-config-path",
        type=str,
        default="configs/train_config.yml",
        help="Path to the training configuration YAML file",
    )

    args = parser.parse_args()
    domain_name = args.domain_name.lower()
    task_name = args.task_name.lower()
    model_config_path = args.model_config_path
    train_config_path = args.train_config_path

    hps = load_config(model_config_path)
    train_hps = load_config(train_config_path)
    set_seeds(getattr(train_hps, "seed", 42))

    env = create_environment(
        domain_name,
        task_name,
        train_hps.action_repeat,  # type:ignore
        train_hps.frame_stack,  # type:ignore
        train_hps.image_size,  # type:ignore
    )
    sac = SAC_DINO(env, hps, train_hps, device)
    sac.train()
