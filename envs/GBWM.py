"""
Goal-Based Wealth Management RL Environment:

@date: September 2025
"""
# python libraries
import numpy as np
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Box

class GBWMEnv(gym.Env):
    def __init__(self,
                 portfolios=np.array([[0.9098, 0.0225, 0.0677],
                                    [0.8500, 0.0033, 0.1467],
                                    [0.7903, -0.0160, 0.2257],
                                    [0.7305, -0.0352, 0.3047],
                                    [0.6707, -0.0545, 0.3837],
                                    [0.6110, -0.0737, 0.4628],
                                    [0.5512, -0.0930, 0.5418],
                                    [0.4915, -0.1122, 0.6208],
                                    [0.4317, -0.1315, 0.6998],
                                    [0.3719, -0.1507, 0.7788],
                                    [0.3122, -0.1700, 0.8578],
                                    [0.2524, -0.1892, 0.9368],
                                    [0.1927, -0.2085, 1.0158],
                                    [0.1329, -0.2277, 1.0948],
                                    [0.0731, -0.2470, 1.1738]]),
                 mu=np.array([0.0493, 0.0770, 0.0886]),
                 sigma=np.array([[0.0017, -0.0017, -0.0021],
                                [-0.0017, 0.0396, 0.0309],
                                [-0.0021, 0.0309, 0.0392]]),
                 goals=[(5, 100, 1000), (10, 150, 1000)],
                 max_goals=1,
                 T=11,
                 dt=1,
                 w0=100):
        """constructor
        """
        super(GBWMEnv, self).__init__()

        # default parameters
        params = {
            'T' : T,
            'dt' : dt,
            'w0' : w0,
            'portfolios' : portfolios,
            'mu' : mu,
            'sigma' : sigma,
            'goals' : goals,
            'max_goals' : max_goals
        }

        self.params = params # list of all parameters
        self.T = params['T']  # length of trading horizon, in years
        self.dt = params['dt']  # time between periods, in years
        self.w0 = params['w0']  # initial wealth

        self.portfolios = params['portfolios']  # portfolios weights
        self.N_portfolios = len(self.portfolios)  # number of portfolios
        self.mu = params['mu']  # expected returns for the assets 
        self.sigma = params['sigma']  # covariance of returns for the assets

        self.goals = params['goals']  # list of all goals
        self.max_goals = params['max_goals']  # maximal number of goals (required for gym)

        obs_low = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        obs_high = np.array([20*self.w0, self.max_goals+1, self.T], dtype=np.float32)
        self.observation_space = Box(low=obs_low, high=obs_high, dtype=np.float32)
        self.action_space = MultiDiscrete([self.max_goals+1, self.N_portfolios])
        
    def reset(self, seed=None, options=None):
        """initialization of the environment

        Returns
        -------
        t, w, nbr_goals : initial time, wealth, number of available goals
        """
        super().reset(seed=seed)

        self.t = 0
        self.w = self.w0
        self.available_goals = [goal for goal in self.goals if goal[0] == self.t]

        obs = self._get_obs()
        info = self._get_info()

        return obs, info

    def step(self, action):
        """simulation engine

        Returns
        -------
        obs, reward, done, truncated, info : new state, reward, terminating boolean, truncated and info
        """
        reward = 0.0
        action_goal = int(action[0] - 1)
        action_pf = int(action[1])

        # fulfill goal(s)
        # NB: 2nd condition is an action_mask, where invalid actions lead to nothing
        if action_goal>-1 and len(self.available_goals)>action_goal and self.w>=self.available_goals[action_goal][1]:
            self.w = self.w - self.available_goals[action_goal][1]
            reward += self.available_goals[action_goal][2]

        # update time
        self.t = self.t+1
        terminated = self.t >= self.T
        self.available_goals = [goal for goal in self.goals if goal[0] == self.t]
        
        # update wealth according to chosen portfolio
        pf = self.portfolios[action_pf,:]
        mu_pf = pf @ self.mu
        sigma_pf = np.sqrt(pf @ self.sigma @ pf)
        dW = np.random.normal()
        self.w = self.w*np.exp((mu_pf - 0.5 * sigma_pf**2) * self.dt + sigma_pf * np.sqrt(self.dt) * dW)
        
        obs = self._get_obs()
        info = self._get_info()

        return obs, reward, terminated, False, info
    
    def _get_obs(self):
        return np.array([self.w, len(self.available_goals), self.t], dtype=np.float32)
    
    def _get_info(self):
        return {}
    
    def render(self, mode="human"):
        pass

    def close(self):
        pass

    def __repr__(self):
        return f"<GBWM instance>"

    def __str__(self):
        descrp = "*** Goal-Based Wealth Management RL Environment -- Full parameters ***\n"
        for key, val in self.params.items():
            descrp += '* {}: {}\n'.format(key, val)
        return descrp

