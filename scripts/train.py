import argparse
from typing import Optional, Dict

import torch
from graphic_pomme_env.wrappers import NUM_STACK
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback

from scripts.agents import make_actor, ACTORS
from scripts.env import GraphicPommerEnv
from scripts.model import Extractor

policy_kwargs = dict(
    features_extractor_class=Extractor,
    features_extractor_kwargs=dict(features_dim=256),
)

parser = argparse.ArgumentParser(description="Train a PPO model")
parser.add_argument("--model", help="Path to your model .zip file")
make_env = lambda: GraphicPommerEnv(num_stack=NUM_STACK,
                                    start_pos=0,  # random
                                    opponent_actor=make_actor(ACTORS.simple),
                                    board="GraphicOVOCompact-v0")
checkpoint_callback = CheckpointCallback(save_freq=500000, save_path=f"./logs", name_prefix="PPO_")
eval_callback = EvalCallback(make_env(), best_model_save_path=f"./logs/best", log_path=f"./logs", eval_freq=100000)
callback = CallbackList([checkpoint_callback]) # , eval_callback])


def create_model(env, path: Optional[str] = None, policy_kwargs: Optional[Dict] = None):
    if path is not None:
        model = PPO.load(path, env)
    elif policy_kwargs is not None:
        model = PPO("CnnPolicy", env, n_steps=4096, ent_coef=0.0001, policy_kwargs=policy_kwargs, verbose=True)
    else:
        model = PPO("CnnPolicy", env, n_steps=4096, ent_coef=0.0001, verbose=True)
    return model


if __name__ == "__main__":
    args = parser.parse_args()
    env = make_vec_env(make_env, n_envs=2)
    model = create_model(env=env, path=args.model)
    model = model.learn(total_timesteps=50000000, callback=callback)
    model.save("PPO")
