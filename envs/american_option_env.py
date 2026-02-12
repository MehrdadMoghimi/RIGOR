import gymnasium as gym
from gymnasium.spaces import Box, Discrete
import numpy as np


class AmericanOptionEnv(gym.Env):
    def __init__(self, 
                 option="put",
                 price_process="GBM",
                 mu=-0.3,
                 kappa=2,
                 sigma=0.3,
                 theta=1,
                 T=1,
                 K=1,
                 Ndt=10,
                 Nda=2,
                 random_reset=False):
        super(AmericanOptionEnv, self).__init__()

        self.params = {
            "option": option,  # call or put
            "price_process": price_process,  # GBM or OU
            "mu": mu,  # drift of the GBM process
            "kappa": kappa,  # kappa of the OU process
            "sigma": sigma,  # standard deviation of the OU process or volatility of the GBM process
            "theta": theta,  # mean-reversion level of the OU process or initial price of the GBM process
            "T": T,  # trading horizon
            "K": K,  # strike price of the option
            "Ndt": Ndt,  # number of periods
            "Nda": Nda,  # number of actions
            "random_reset": random_reset,  # reset
        }
        self.action_space = Discrete(self.params["Nda"])

        if self.params["price_process"] == "GBM":
            low = np.array(
                [self.params["theta"] - 5 * self.params["sigma"], 0],
                dtype=np.float32,
            )
            high = np.array(
                [self.params["theta"] + 5 * self.params["sigma"], self.params["Ndt"]],
                dtype=np.float32,
            )
            self.observation_space = Box(
                low=low,
                high=high,
                shape=(2,),
                dtype=np.float32,
            )
        elif self.params["price_process"] == "OU":
            low = np.array(
                [
                    self.params["theta"] - 6 * self.params["sigma"] / np.sqrt(2 * self.params["kappa"]),
                    0,
                ],
                dtype=np.float32,
            )
            high = np.array(
                [
                    self.params["theta"] + 6 * self.params["sigma"] / np.sqrt(2 * self.params["kappa"]),
                    self.params["Ndt"],
                ],
                dtype=np.float32,
            )
            self.observation_space = Box(
                low=low,
                high=high,
                shape=(2,),
                dtype=np.float32,
            )
        else:
            raise ValueError("Invalid price process")

    def step(self, action):
        assert self.action_space.contains(action), "Invalid action"

        self._time_step += 1

        if self._time_step == self.params["Ndt"] or action == 1:
            if self.params["option"] == "call":
                reward = np.max([0, self._stock_price - self.params["K"]])
            elif self.params["option"] == "put":
                reward = np.max([0, self.params["K"] - self._stock_price])
            else:
                raise ValueError("Invalid option type")
            terminated = True
        else:
            reward = 0
            terminated = False

        dt = self.params["T"] / self.params["Ndt"]

        if self.params["price_process"] == "GBM":
            self._stock_price *= np.exp(
                (self.params["mu"] - 0.5 * self.params["sigma"] ** 2) * dt
                + self.params["sigma"] * np.sqrt(dt) * np.random.normal()
            )
        elif self.params["price_process"] == "OU":
            eta = self.params["sigma"] * np.sqrt((1 - np.exp(-2 * self.params["kappa"] * dt)) / (2 * self.params["kappa"]))
            self._stock_price = (
                self.params["theta"] + (self._stock_price - self.params["theta"]) * np.exp(-self.params["kappa"] * dt)
                + eta * np.random.normal()
            )
        else:
            raise ValueError("Invalid price process")

        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, False, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if self.params["random_reset"]:
            self._stock_price = np.random.normal(self.params["theta"], self.params["sigma"])
            self._stock_price = np.min(
                [
                    np.max([self._stock_price, self.observation_space.low[0]]),
                    self.observation_space.high[0],
                ]
            )
            self._time_step = np.random.randint(0, self.params["Ndt"] - 1)
        else:
            self._stock_price = self.params["theta"]
            self._time_step = 0

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def _get_obs(self):
        return np.array([self._stock_price, self._time_step], dtype=np.float32)

    def _get_info(self):
        return {}

    def render(self, mode="human"):
        pass

    def close(self):
        pass
