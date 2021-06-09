from multiprocessing import freeze_support

from graphic_pomme_env.wrappers import NUM_STACK

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import SubprocVecEnv
from stable_baselines3.common.env_checker import check_env

# try to train against random agent who doesn't use bombs
from scripts.agents import make_actor, ACTORS
from scripts.env import GraphicPommerEnv

import sys
sys.setrecursionlimit(100000)

if __name__ == "__main__":
    actor = make_actor(ACTORS.simple)
    env_pom = GraphicPommerEnv(num_stack=NUM_STACK, start_pos=0, opponent_actor=actor,
                               board="GraphicOVOCompact-v0")
    check_env(env_pom)
    n_cpu = 6
    env = SubprocVecEnv([lambda: env_pom for i in range(n_cpu)])
    model = PPO("CnnPolicy", env, verbose=1, ent_coef=0.001)
    model = model.learn(total_timesteps=100000)  # num_update = total_timesteps // batch_size
    model.save("ppoCNN10k")
