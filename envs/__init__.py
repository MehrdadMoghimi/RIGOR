"""Custom environments and wrappers for the RIGOR project."""

import numpy as np
import gymnasium as gym
from gymnasium.error import Error as GymError

import ale_py
gym.register_envs(ale_py)

from .american_option_env import AmericanOptionEnv
from .GBWM import GBWMEnv
from .wrappers import StockAwareObservation, MultiDiscreteToDiscreteAction

__all__ = [
    "AmericanOptionEnv",
    "GBWMEnv",
    "StockAwareObservation",
    "MultiDiscreteToDiscreteAction",
]


def _register_env(env_id: str, env_cls, **kwargs):
    try:
        gym.spec(env_id)
    except GymError:
        gym.register(
            env_id, 
            entry_point=env_cls, 
            max_episode_steps=_max_episode_steps_dict.get(env_id, None),
            kwargs=kwargs
        )


_american_option_v1_kwargs = {
    "option": "put",
    "price_process": "OU",
    "kappa": 2.0,
    "sigma": 20.0,
    "theta": 100.0,
    "T": 1.0,
    "K": 100.0,
    "Ndt": 1000,
    "Nda": 2,
    "random_reset": False,
}


_gbwm_v1_kwargs = {
    "goals" : [(5, 100, 1000), (10, 150, 2000)],
}

_gbwm_v2_kwargs = {
    "goals" : [(15, 100, 1000), (30, 150, 1000)],
    "T" : 31,
    "dt" : 1,
    "mu" : np.array([0.0493, 0.0770, 0.0886]) / 3,
    "sigma" : np.array([[0.0017, -0.0017, -0.0021],
                        [-0.0017, 0.0396, 0.0309],
                        [-0.0021, 0.0309, 0.0392]]) / np.sqrt(3),
}

_gbwm_v3_kwargs = {
    "goals" : [(15, 100, 1000), (30, 150, 2000)],
    "T" : 31,
    "dt" : 1,
    "mu" : np.array([0.0493, 0.0770, 0.0886]) / 3,
    "sigma" : np.array([[0.0017, -0.0017, -0.0021],
                        [-0.0017, 0.0396, 0.0309],
                        [-0.0021, 0.0309, 0.0392]]) / np.sqrt(3),
}



_for_registration = (
    ("AmericanOptionEnv-v0", AmericanOptionEnv, {}),
    ("AmericanOptionEnv-v1", AmericanOptionEnv, _american_option_v1_kwargs),
    ("GBWMEnv-v0", GBWMEnv, {}),
    ("GBWMEnv-v1", GBWMEnv, _gbwm_v1_kwargs),
    ("GBWMEnv-v2", GBWMEnv, _gbwm_v2_kwargs),
    ("GBWMEnv-v3", GBWMEnv, _gbwm_v3_kwargs),
    ("WindyLunarLander-v0", "gymnasium.envs.box2d.lunar_lander:LunarLander", {"enable_wind": True}),
)

plot_names = {
    "AmericanOptionEnv-v0": "American Option",
    "AmericanOptionEnv-v1": "American Option",
    "GBWMEnv-v0": "GBWM",
    "GBWMEnv-v1": "GBWM",
    "GBWMEnv-v2": "GBWM",
    "GBWMEnv-v3": "GBWM",
    "WindyLunarLander-v0": "Windy Lunar Lander",
}


_max_episode_steps_dict = {
    "AmericanOptionEnv-v0": 10,
    "AmericanOptionEnv-v1": 1000,
    "GBWMEnv-v0": 11,
    "GBWMEnv-v1": 11,
    "GBWMEnv-v2": 31,
    "GBWMEnv-v3": 31,
    "WindyLunarLander-v0": 500,
}

for env_id, env_cls, kwargs in _for_registration:
    _register_env(env_id, env_cls, **kwargs)
