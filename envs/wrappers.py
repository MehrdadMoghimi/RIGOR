import gymnasium as gym
from gymnasium.spaces import Box, Discrete, MultiDiscrete
import numpy as np


class StockAwareObservation(gym.Wrapper, gym.utils.RecordConstructorArgs):
    def __init__(self, env: gym.Env, **kwargs):
        gym.utils.RecordConstructorArgs.__init__(self)
        gym.Wrapper.__init__(self, env)
        assert isinstance(env.observation_space, Box)
        assert env.observation_space.dtype == np.float32
        
        # Extract parameters from kwargs with defaults
        gamma = kwargs.get('gamma', 1.0)
        initial_stock = kwargs.get('initial_stock', None)
        normalizer = kwargs.get('normalizer', 1.0)
        discount_factor = kwargs.get('discount_factor', 'exponential')
        hyperbolic_k = kwargs.get('hyperbolic_k', 0.05)

        assert initial_stock is not None, "initial_stock must be provided"
        self.env = env
        self.gamma = gamma
        self.normalizer = normalizer
        self.initial_stock = initial_stock/self.normalizer
        self.discount_factor = discount_factor
        horizon = env.spec.max_episode_steps
        # Set up discount function based on discount_factor
        if self.discount_factor == 'exponential':
            self.discount = lambda _: self.gamma
        elif self.discount_factor == 'hyperbolic':
            self.discount = lambda n: (1 + hyperbolic_k * n * horizon) / (1 + hyperbolic_k * (n * horizon + 1))
        else:
            raise NotImplementedError(f"Discount factor {self.discount_factor} not implemented")
        low = np.append(self.observation_space.low, [-np.inf])
        high = np.append(self.observation_space.high, [np.inf])
        self.observation_space = Box(low, high, dtype=np.float64)

    def observation(self, observation):
        return np.append(observation, [self.stock])

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        self.stock = self.initial_stock
        return self.observation(obs), info

    def step(self, action):
        obs, reward, done, truncation, info = self.env.step(action)
        if self.discount_factor == 'exponential':
            d = self.discount(None)  # Pass None as dummy argument
        elif self.discount_factor == 'hyperbolic':
            d = self.discount(obs[-1])  # assuming the last element of the observation is time step
        else:
            raise NotImplementedError(f"Discount factor {self.discount_factor} not implemented")
        self.stock = (self.stock + (reward / self.normalizer)) / d
        return self.observation(obs), reward, done, truncation, info

    def set_initial_stock(self, initial_stock: float) -> None:
        """
        Update the initial stock value used on reset.
        
        Args:
            initial_stock: The new initial stock value (unnormalized).
                          Will be divided by the normalizer internally.
        """
        self.initial_stock = initial_stock / self.normalizer


class MultiDiscreteToDiscreteAction(gym.ActionWrapper, gym.utils.RecordConstructorArgs):
    """Flattens a :class:`~gymnasium.spaces.MultiDiscrete` action space into a single :class:`~gymnasium.spaces.Discrete` space.

    The mapping enumerates the Cartesian product of the original action dimensions using C-order (last dimension varies fastest).
    """

    def __init__(self, env: gym.Env):
        gym.utils.RecordConstructorArgs.__init__(self)
        super().__init__(env)
        assert isinstance(env.action_space, MultiDiscrete), "Expected a MultiDiscrete action space"

        self._original_action_space: MultiDiscrete = env.action_space
        self._nvec = np.asarray(self._original_action_space.nvec, dtype=np.int64)
        self._shape = tuple(int(n) for n in self._nvec)
        total_actions = int(np.prod(self._nvec, dtype=np.int64))

        self.action_space = Discrete(total_actions)

    def action(self, action: int) -> np.ndarray:
        if not self.action_space.contains(action):
            raise gym.error.InvalidAction(f"Action {action} is outside the flattened Discrete space size {self.action_space.n}")

        multi = np.array(np.unravel_index(action, self._shape, order="C"), dtype=self._original_action_space.dtype)
        return multi

    def reverse_action(self, action: np.ndarray | list | tuple) -> int:
        multi = np.asarray(action, dtype=self._original_action_space.dtype).reshape(-1)
        if not self._original_action_space.contains(multi):
            raise gym.error.InvalidAction(f"Action {multi} is not valid for the original MultiDiscrete space {self._nvec}")

        return int(np.ravel_multi_index(multi.astype(np.int64), self._shape, order="C"))