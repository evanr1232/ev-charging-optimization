import argparse
import os
from collections import defaultdict

import gymnasium as gym
import pandas as pd
from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPOConfig
from tqdm import tqdm

from sustaingym.envs.evcharging import EVChargingEnv, GMMsTraceGenerator
import torch


# -----------------------------
# Constants
# -----------------------------
TRAIN_STEPS_PER_BATCH = 5000  # steps per batch
TOTAL_STEPS = 250_000  # by defaullt, 250_000 / 5000 = 50 epochs of training
SAVE_BASE_DIR = "logs"
NUM_WORKERS_ROLLOUT = 8  # Parallelize rollout generation


# -----------------------------
# Simple argument parser
# -----------------------------
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("-a", "--algo", default="ppo")
    parser.add_argument("-t", "--train_date_period", default="Summer 2021")
    parser.add_argument("-s", "--site", default="caltech")
    parser.add_argument("-r", "--seed", type=int, default=123)
    parser.add_argument("--lr", type=float, default=5e-4)

    return vars(parser.parse_args())


# -----------------------------
# Environment constructor
# -----------------------------
def make_env(dp: str, site: str, seed: int):
    """Always use GMM traces and flatten observations."""
    if dp != "Summer 2021":
        raise ValueError("This minimal script only supports Summer 2021.")

    date_range = ("2021-07-05", "2021-07-18")
    gen = GMMsTraceGenerator(site, date_range, seed=seed)
    return gym.wrappers.FlattenObservation(EVChargingEnv(gen))


# -----------------------------
# Train loop
# -----------------------------
def run_training(cfg: dict, save_dir: str):

    # register minimal env
    register_env(
        "evcharging",
        lambda config: make_env(cfg["train_date_period"], cfg["site"], cfg["seed"]),
    )

    train_config = (
        PPOConfig()
        .environment(env="evcharging")
        .rollouts(num_rollout_workers=NUM_WORKERS_ROLLOUT)
        .framework("torch")
    )

    algo = train_config.build(env="evcharging")

    # record training stats
    df_data = defaultdict(list)

    for i in tqdm(range(TOTAL_STEPS // TRAIN_STEPS_PER_BATCH)):
        results = algo.train()

        algo.save(save_dir)

        df_data["iter"].append(i)
        df_data["episode_reward_mean"].append(results["episode_reward_mean"])
        df_data["num_steps"].append((i + 1) * TRAIN_STEPS_PER_BATCH)

        # save neural network weights every 10 iterations
        if i % 10 == 0:
            policy = algo.get_policy()  # single-agent policy
            nn_cache = {
                "state_dict": policy.model.state_dict(),
                "optimizer": policy._optimizers[0].state_dict(),
            }
            torch_save_path = os.path.join(save_dir, f"nn_cache_iter_{i}.pt")
            torch.save(nn_cache, torch_save_path)
        # -------------------------------------------------

    pd.DataFrame(df_data).to_csv(os.path.join(save_dir, "train_results.csv"))
    print("Training complete.")
    return


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    cfg = parse_args()

    folder_name = (
        f'{cfg["site"]}_ppo_{cfg["train_date_period"]}_lr{cfg["lr"]}_seed{cfg["seed"]}'
    )
    save_dir = os.path.join(SAVE_BASE_DIR, folder_name)
    os.makedirs(save_dir, exist_ok=True)

    run_training(cfg, save_dir)
