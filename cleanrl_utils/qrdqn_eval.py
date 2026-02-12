import random
from argparse import Namespace
from typing import Callable

import gymnasium as gym
import numpy as np
import torch
import math
import tqdm
from cleanrl_utils import agent_utils

def evaluate(
    model_path: str,
    make_env: Callable,
    env_id: str,
    eval_episodes: int,
    run_name: str,
    Model: torch.nn.Module,
    device: torch.device = torch.device("cpu"),
    epsilon: float = 0.05,
    seed: int = 0,
    capture_video: bool = False,
):
    
    torch.serialization.add_safe_globals([
        np._core.multiarray.scalar,
        np.dtype,
        np.dtypes.Float32DType,
    ])
    model_data = torch.load(model_path, map_location="cpu")
    args = Namespace(**model_data["args"])
    args_dict = vars(args)
    obs_time_idx = 0

    # Build env_kwargs based on saved arguments (algorithm-agnostic)
    env_kwargs = {}
    if "time_aware" in args_dict:
        env_kwargs["time_aware"] = args_dict["time_aware"]
        obs_time_idx += 1*env_kwargs["time_aware"]

    # Stock-aware environments need additional parameters
    if "initial_stock" in args_dict:
        initial_stock = args_dict.get("initial_stock")
        obs_time_idx += 1
        if initial_stock is None:
            raise ValueError("Saved model is missing 'initial_stock' required for StockAwareObservation.")
        stock_kwargs = {
            "gamma": args_dict.get("gamma"),
            "initial_stock": initial_stock,
            "normalizer": args_dict.get("reward_normalizer"),
            "discount_factor": args_dict.get("discount_factor"),
            "hyperbolic_k": args_dict.get("hyperbolic_k"),
        }
        env_kwargs.update({k: v for k, v in stock_kwargs.items() if v is not None})

    env = make_env(env_id, 0, 0, capture_video, run_name, **env_kwargs)()
    n_act = int(env.action_space.n)
    n_obs = int(math.prod(env.observation_space.shape))
    
    # Build model kwargs based on algorithm type
    if args.exp_name == "categorical" or args.exp_name == "categorical_mean_cvar":
        model_kwargs = {
            'n_obs': n_obs, 
            'n_act': n_act, 
            'n_atoms': args.n_atoms, 
            'v_min': args.v_min, 
            'v_max': args.v_max, 
            'device': device
        }
    else:
        # qrdqn, qrdqn_abs, qrdqn_mean_cvar all use n_quantiles (particles)
        model_kwargs = {'n_obs': n_obs, 'n_act': n_act, 'n_quantiles': args.n_quantiles, 'device': device}
    
    # Add qrdqn_mh specific kwargs if present
    if "n_gammas" in args_dict:
        model_kwargs['n_gammas'] = args_dict['n_gammas']
    
    model = Model(**model_kwargs)
    model.load_state_dict(model_data["model_weights"])
    model.eval()
    
    # Build get_action_kwargs based on algorithm type
    get_action_kwargs = {}
    if "reward_normalizer" in args_dict:
        get_action_kwargs['reward_normalizer'] = args_dict['reward_normalizer']

    if "mean_weight" in args_dict and "risk_level" in args_dict:
        mean_weight = args_dict['mean_weight']
        risk_level = args_dict['risk_level']
        
        # Both categorical_mean_cvar and qrdqn_mean_cvar now use weight_left and weight_right
        get_action_kwargs['weight_left'] = (1 - mean_weight) / risk_level + mean_weight
        get_action_kwargs['weight_right'] = mean_weight
        
        # categorical_mean_cvar uses n_quantiles for PMF to quantile approximation
        if args.exp_name == "categorical_mean_cvar":
            get_action_kwargs['n_quantiles'] = args_dict.get('n_atoms', 51) * 10
        
        get_action_kwargs['target_averaging'] = args_dict.get('target_averaging', False)
        get_action_kwargs['torch_generator'] = None  # Deterministic evaluation
    elif "n_gammas" in args_dict:
        eval_gammas = agent_utils.compute_eval_gamma_interval(args_dict['max_gamma'], args_dict['hyperbolic_k'], args_dict['n_gammas'])
        gammas = [math.pow(gamma, args_dict['hyperbolic_k']) for gamma in eval_gammas]
        assert max(gammas) <= args_dict['max_gamma']
        if args_dict['integral_estimate'] == 'lower':
            gamma_plus_one = eval_gammas + [1.]
            weights = [gamma_plus_one[i + 1] - gamma_plus_one[i] for i in range(args_dict['n_gammas'])]
        elif args_dict['integral_estimate'] == 'upper':
            gamma = eval_gammas
            weights = [gamma[i + 1] - gamma[i] for i in range(args_dict['n_gammas'] - 1)]
            weights = [0.] + weights
        else:
            raise ValueError(f"Unknown integral_estimate: {args_dict['integral_estimate']}")
        q_weights = torch.tensor(weights, device=device)
        gammas = torch.tensor(gammas, device=device)
        get_action_kwargs['acting_policy'] = args_dict['acting_policy']
        get_action_kwargs['time_consistent'] = args_dict['time_consistent']
        get_action_kwargs['gammas'] = gammas
        get_action_kwargs['weights'] = q_weights
        get_action_kwargs['horizon'] = env.spec.max_episode_steps if hasattr(env.spec, 'max_episode_steps') else None
    
    episodic_returns = []
    episodic_discounted_returns = []
    if env_id.startswith('GBWMEnv'):
        episodic_cash_goal1 = []
        episodic_achieve_goal1 = []
        episodic_achieve_goal2 = []
    
    # Setup discount factor for discounted returns
    if hasattr(args, 'discount_factor'):
        if args.discount_factor == 'hyperbolic':
            horizon = env.spec.max_episode_steps if hasattr(env.spec, 'max_episode_steps') else 1000
            hyperbolic_k = getattr(args, 'hyperbolic_k')
            discount_fn = lambda n: (1 + hyperbolic_k * n * horizon) / (1 + hyperbolic_k * (n * horizon + 1))
        elif args.discount_factor == 'exponential':
            gamma = getattr(args, 'gamma')
            discount_fn = lambda n: gamma
        else:
            discount_fn = lambda n: 1.0
    else:
        # Default to exponential with gamma
        gamma = getattr(args, 'gamma')
        discount_fn = lambda n: gamma
    
    # TRY NOT TO MODIFY: seeding
    #if seed != 0:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    python_rng = random.Random(seed)
    torch_rng = torch.Generator(device=device)
    torch_rng.manual_seed(seed)

    for episode in tqdm.tqdm(range(eval_episodes)):
        obs, _ = env.reset()
        done = False
        episode_return = 0.0
        episode_discounted_return = 0.0
        discount_cumulative = 1.0
        step = 0

        while not done:
            if python_rng.random() < epsilon:
                action = env.action_space.sample()
            else:
                obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).reshape(1, -1)
                action, _ = model.get_action(obs_tensor, **get_action_kwargs)
                action = int(action.item())

            next_obs, reward, terminated, truncated, _ = env.step(action)
            episode_return += reward
            episode_discounted_return += discount_cumulative * reward
            if env_id.startswith('GBWMEnv'):
                if next_obs[-1-obs_time_idx] == env.unwrapped.goals[0][0]:
                    episodic_cash_goal1.append(next_obs[0] > env.unwrapped.goals[0][1])
                if next_obs[-1-obs_time_idx] == env.unwrapped.goals[0][0]+1:
                    episodic_achieve_goal1.append(reward > 0)
                if next_obs[-1-obs_time_idx] == env.unwrapped.goals[1][0]+1:
                    episodic_achieve_goal2.append(reward > 0)
            
            # Update cumulative discount for next step
            discount_cumulative *= discount_fn(step / env.spec.max_episode_steps if hasattr(env.spec, 'max_episode_steps') else step / 1000)
            step += 1
            
            done = terminated or truncated
            obs = next_obs

        episodic_returns.append(episode_return)
        episodic_discounted_returns.append(episode_discounted_return)

    # Print statistics
    returns_array = np.array(episodic_returns)
    discounted_returns_array = np.array(episodic_discounted_returns)
    mean_return = np.mean(returns_array)
    std_return = np.std(returns_array)
    mean_discounted_return = np.mean(discounted_returns_array)
    std_discounted_return = np.std(discounted_returns_array)
    
    print(f"\n{'='*60}")
    print(f"Evaluation Results over {eval_episodes} episodes")
    print(f"{'='*60}")
    print(f"Undiscounted Mean Return: {mean_return:.4f} ± {std_return:.4f}")
    print(f"Discounted Mean Return:   {mean_discounted_return:.4f} ± {std_discounted_return:.4f}")
    if env_id.startswith('GBWMEnv'):
        print(f"{'='*60}")
        print(f"P(W(5) > 100): {np.mean(episodic_cash_goal1):.4f}")
        print(f"P(fulfill goal 1): {np.mean(episodic_achieve_goal1):.4f}")
        print(f"P(fulfill goal 2): {np.mean(episodic_achieve_goal2):.4f}")
    
    # Calculate and print CVaR and Mean-CVaR if applicable
    if args.exp_name == 'qrdqn_mean_cvar' or args.exp_name == 'categorical_mean_cvar':
        risk_level = getattr(args, 'risk_level')
        mean_weight = getattr(args, 'mean_weight')
        
        # Sort returns for CVaR calculation (use undiscounted returns)
        sorted_returns = np.sort(returns_array)
        sorted_discounted_returns = np.sort(discounted_returns_array)
        n_risk = max(1, int(np.ceil(risk_level * eval_episodes)))
        
        # CVaR (Conditional Value at Risk) - mean of worst (risk_level * 100)% returns
        cvar = np.mean(sorted_returns[:n_risk])
        cvar_discounted = np.mean(sorted_discounted_returns[:n_risk])
        
        # Mean-CVaR objective
        mean_cvar_objective = mean_weight * mean_return + (1 - mean_weight) * cvar
        mean_cvar_objective_discounted = mean_weight * mean_discounted_return + (1 - mean_weight) * cvar_discounted
        
        print(f"\nRisk Metrics (risk_level={risk_level:.2f}, mean_weight={mean_weight:.4f}):")
        print(f"  Undiscounted:")
        print(f"    CVaR (worst {risk_level*100:.0f}%): {cvar:.4f}")
        print(f"    Mean-CVaR Objective: {mean_cvar_objective:.4f}")
        print(f"    VaR (Value at Risk): {sorted_returns[n_risk-1]:.4f}")
        print(f"    Min Return: {np.min(returns_array):.4f}")
        print(f"    Max Return: {np.max(returns_array):.4f}")
        print(f"  Discounted:")
        print(f"    CVaR (worst {risk_level*100:.0f}%): {cvar_discounted:.4f}")
        print(f"    Mean-CVaR Objective: {mean_cvar_objective_discounted:.4f}")
        print(f"    VaR (Value at Risk): {sorted_discounted_returns[n_risk-1]:.4f}")
        print(f"    Min Return: {np.min(discounted_returns_array):.4f}")
        print(f"    Max Return: {np.max(discounted_returns_array):.4f}")

    print(f"{'='*60}\n")
    env.close()
    return episodic_returns, episodic_discounted_returns
