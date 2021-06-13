import numpy as np
from graphic_pomme_env.wrappers import PommerEnvWrapperFrameSkip2, NUM_STACK, NUM_ACTIONS
from gym import spaces


class GraphicPommerEnv(PommerEnvWrapperFrameSkip2):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.metadata = {}
        self.action_space = spaces.Discrete(NUM_ACTIONS)
        self.observation_space = spaces.Box(low=0, high=255, shape=(NUM_STACK, 56, 48), dtype=np.uint8)
        self.reward_range = [-1, 1]

    def step(self, action):
        (obs, reward, done, info), _ = super().step(action)
        return np.array(obs), reward, done, info

    def reset(self):
        obs, _ = super().reset()
        return np.array(obs)

    def close(self):
        pass
