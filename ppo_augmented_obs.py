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

import numpy as np
import pandas as pd


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

class MOERForecastWrapper(gym.ObservationWrapper):
    def __init__(self, env, error_table_path):
        super().__init__(env)
        
        # 1. Load your pre-computed table
        # We assume this is a dictionary or array keyed by 5-min intervals (0 to 287)
        # shape: (288, forecast_horizon)
        # For this example, I'll generate a random one
        self.forecast_horizon = 36  # Default for Summer 2021
        self.error_std_table = pd.read_csv(error_table_path, header = 0, index_col=0)

        # 2. Update the observation space
        # We need to tell RLLib the observation is now bigger.
        # The base env usually outputs a Dict. We assume we are wrapping BEFORE Flatten.
        
        # We need to know the shape of the original MOER forecast to append correctly.
        # In SustainGym, the obs is a Dictionary. We will add a new key.
        self.observation_space = env.observation_space
        
        # Define the space for our new error vector
        self.observation_space['moer_forecast_error'] = gym.spaces.Box(
            low=0, 
            high=np.inf, 
            shape=(self.forecast_horizon,), 
            dtype=np.float32
        )

    def timestep_to_index(self, ts):
        return int(np.round((ts * 60 * 24) / 5)) - 1

    def observation(self, obs):
        """
        This method is called every time the env returns an observation.
        """
        # 1. Get the time of day from the observation
        
        t_fraction = obs['timestep'] 
        time_index = self.timestep_to_index(t_fraction)
        
        # 2. Lookup the pre-computed error vector
        error_vector = self.error_std_table.iloc[time_index].values
        
        # 3. Add it to the observation dictionary
        # We return a copy so we don't mutate the original in place unexpectedly
        new_obs = obs.copy()
        new_obs['moer_forecast_error'] = error_vector
        
        return new_obs
    
def make_env(dp: str, site: str, seed: int):
    if dp != "Summer 2021":
        raise ValueError("This minimal script only supports Summer 2021.")

    date_range = ("2021-07-05", "2021-07-18")
    
    # 1. Base Environment
    gen = GMMsTraceGenerator(site, date_range, seed=seed)
    env = EVChargingEnv(gen)
    
    # This adds the error vector to the dictionary
    env = MOERForecastWrapper(env, error_table_path="moer_errors/error_std.csv")
    
    # 3. Flatten at the very end
    # FlattenObservation will automatically flatten the new 'moer_forecast_error' 
    # key along with the rest, making it visible to the PPO agent.
    return gym.wrappers.FlattenObservation(env)



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
                "optimizer": policy.optimizer()[0].state_dict(),
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
