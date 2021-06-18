import torch
import numpy as np
from enum import Enum
from pommerman.agents import BaseAgent, SimpleAgent, RandomAgent, RandomAgentNoBomb, SmartRandomAgent, \
    SmartRandomAgentNoBomb, StaticAgent, CautiousAgent


class ACTORS(Enum):
    idle = 0
    random = 1
    random_smart = 2
    random_nobomb = 3
    simple = 4
    simple_cautious = 5
    model = 6
    cautious = 7


def make_actor(kind, model=None):
    if kind == ACTORS.simple:
        return SimpleAgent()
    if kind == ACTORS.cautious:
        return CautiousAgent()
    elif kind == ACTORS.idle:
        return StaticAgent()
    elif kind == ACTORS.random:
        return RandomAgent()
    elif kind == ACTORS.random_smart:
        return SmartRandomAgent()
    elif kind == ACTORS.random_nobomb:
        return RandomAgentNoBomb()
    elif kind == ACTORS.random_smart_nobomb:
        return SmartRandomAgentNoBomb()
    elif kind == ACTORS.model:
        assert model is not None, "A model has to specified for the model actor"
        make_model_actor(model)


def make_model_actor(model):
    def model_actor(frame_stack):
        obs = torch.from_numpy(np.array(frame_stack.get_observation()))
        net_out = model(obs).detach().cpu().numpy()
        action = np.argmax(net_out)
        return action

    return model_actor


class ModelAgent(BaseAgent):
    def __init__(self):
        super().__init__()

    def act(self, obs, action_space):
        print(obs)
