import gym
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class Extractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim):
        super(Extractor, self).__init__(observation_space, features_dim)

        self.features = nn.Sequential(
            nn.Conv2d(5, 16, kernel_size=3, stride=1),
            nn.LeakyReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1),
            nn.LeakyReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1),
            nn.LeakyReLU(),
            nn.AdaptiveAvgPool2d(2),
            nn.Flatten()
        )

    def forward(self, x):
        return self.features(x)
