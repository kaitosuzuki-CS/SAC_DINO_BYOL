# SAC + DINO / BYOL for DM Control Suite

## Introduction

This repository contains implementations of **Soft Actor-Critic (SAC)** reinforcement learning agents augmented with self-supervised learning (SSL) objectives, specifically **Bootstrap Your Own Latent (BYOL)** and **DINO (Self-distillation with no labels)**. These methods are designed to improve sample efficiency and state representation learning for continuous control tasks in the **DeepMind Control Suite**.

The project explores how auxiliary self-supervised tasks can help an RL agent learn robust visual representations directly from pixel observations.

## Table of Contents

- [Project Overview](#project-overview)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)
- [Application Info](#application-info)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Clone the Repository](#clone-the-repository)
  - [Installation (Conda)](#installation-conda)
  - [Installation (Pip)](#installation-pip)
- [Project Files](#project-files)

## Project Overview

The core idea is to train an SAC agent where the encoder (which processes image observations) is jointly optimized with an RL objective (maximizing expected return) and an SSL objective (BYOL or DINO).

- **SAC (Soft Actor-Critic):** An off-policy actor-critic deep RL algorithm based on the maximum entropy reinforcement learning framework.
- **BYOL (Bootstrap Your Own Latent):** A self-supervised learning approach that learns representations by predicting previous versions of its own outputs (target network) without using negative pairs.
- **DINO:** A self-distillation approach where a student network predicts the output of a teacher network (momentum encoder), emphasizing local-to-global correspondence.

## Project Structure

```
SAC_DINO_BYOL/
├── configs/                # Configuration files for training and models
│   ├── byol_model_config.yml
│   ├── byol_train_config.yml
│   ├── dino_model_config.yml
│   └── dino_train_config.yml
├── model/                  # Neural network architectures
│   ├── actor.py            # Actor network implementation
│   ├── components.py       # Encoder (BYOL/DINO) and Projection Heads
│   └── critic.py           # Critic network implementation
├── utils/                  # Utility scripts
│   ├── crop.py             # Image cropping and augmentation
│   ├── env.py              # DM Control Suite environment wrapper
│   ├── misc.py             # Helper functions (config loading, seeding)
│   └── replay_buffer.py    # Experience replay buffer
├── sac_byol.py             # SAC agent with BYOL auxiliary task
├── sac_dino.py             # SAC agent with DINO auxiliary task
├── train.py                # Main training entry point
├── environment.yml         # Conda environment definition
└── requirements.txt        # Python dependencies
```

## Tech Stack

- **Language:** Python 3.12
- **Deep Learning Framework:** PyTorch 2.10, Torchvision 0.25
- **Simulation Environment:** DeepMind Control Suite (`dm_control`), MuJoCo
- **Configuration:** YAML
- **Numerical Computing:** NumPy

## Application Info

The project uses YAML configuration files to manage hyperparameters for both the model architecture and the training process. These are located in the `configs/` directory.

### Training

The main entry point for training is `train.py`. It accepts command-line arguments to specify the algorithm, environment domain, and task.

**Example Usage:**

```bash
python train.py --alg byol --domain-name cartpole --task-name swingup
```

**Arguments:**

- `--alg`: The training algorithm to use. Choices: `dino`, `byol`. (Default: `dino`)
- `--domain-name`: The domain name from DM Control Suite (e.g., `cartpole`, `cheetah`). (Default: `cartpole`)
- `--task-name`: The task name from DM Control Suite (e.g., `swingup`, `run`). (Default: `swingup`)
- `--model-config-path`: Path to the model configuration YAML file.
- `--train-config-path`: Path to the training configuration YAML file.

_Note: Ensure the correct configuration files are provided or the defaults in `train.py` match your intention._

## Getting Started

### Prerequisites

- Operating System: macOS (Darwin) or Linux recommended.
- **MuJoCo:** Required for `dm_control`. Ensure you have a valid MuJoCo installation and license (if applicable for your version, though newer versions are open source).

### Clone the Repository

  ```bash
  git clone https://github.com/kaitosuzuki-CS/SAC_DINO_BYOL.git
  cd SAC_DINO_BYOL
  ```

### Installation (Conda)

To set up the environment using Conda:

1.  **Create the environment:**
    ```bash
    conda env create -f environment.yml
    ```
2.  **Activate the environment:**
    ```bash
    conda activate sac_dino_byol
    ```

### Installation (Pip)

To install dependencies using Pip:

1.  **Create the environment:**

    ```bash
    conda create -n sac_dino_byol python=3.12
    ```

2.  **Activate the environment:**

    ```bash
    conda activate sac_dino_byol
    ```

3.  **Install requirements:**
    ```bash
    pip install -r requirements.txt
    ```

## Project Files

### Root Directory

- `train.py`: The main script to launch training experiments. It sets up the environment, initializes the agent (SAC_BYOL or SAC_DINO), and starts the training loop.
- `sac_byol.py`: Implementation of the `SAC_BYOL` class. Handles the training logic for SAC with the BYOL auxiliary loss.
- `sac_dino.py`: Implementation of the `SAC_DINO` class. Handles the training logic for SAC with the DINO auxiliary loss.

### `model/`

- `actor.py`: Defines the `Actor` class, typically a Gaussian policy network.
- `critic.py`: Defines the `SoftCritic` class (Q-functions).
- `components.py`: Contains the `EncoderBYOL`, `EncoderDINO`, and `ProjectionHead` classes used for feature extraction and the SSL projections.

### `utils/`

- `crop.py`: Implements the `Crop` class for random image cropping, which is crucial for generating the multiple views required by contrastive/self-supervised learning.
- `env.py`: Wraps the `dm_control` environment to handle pixel observations, action repeats, and frame stacking.
- `replay_buffer.py`: A standard replay buffer to store transitions `(state, action, reward, next_state, done)` for off-policy learning.
- `misc.py`: Miscellaneous utilities including the `HPS` class for hyperparameter management, `load_config`, and `set_seeds`.

### `configs/`

- `*_model_config.yml`: Hyperparameters for the neural network architectures (embedding dimensions, encoder types).
- `*_train_config.yml`: Hyperparameters for the training loop (learning rates, batch sizes, buffer capacity, update frequencies).
