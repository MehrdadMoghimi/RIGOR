# This code is based on the implementation of the C51 algorithm in the CleanRL library.
# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/c51/#c51py
import math
import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
import tyro
from torch.utils.tensorboard import SummaryWriter

from cleanrl_utils.buffers import ReplayBuffer
from cleanrl_utils.config import parse_args_with_config
from cleanrl_utils.pretty_print import pretty_print_args

import envs as custom_envs
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = ""
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    evaluation_episodes: int = 1000
    """the number of evaluation episodes"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""
    dir: str = "runs"
    """the directory to store the model and log"""

    # Algorithm specific arguments
    env_id: str = "CartPole-v1"
    """the id of the environment"""
    total_timesteps: int = 500000
    """total timesteps of the experiments"""
    learning_rate: float = 1e-3
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    n_quantiles: int = 50
    """the number of quantiles"""
    huber_k: float = 0.1
    """the huber loss k (0 = l1, inf = l2)"""
    buffer_size: int = 100000
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.01
    """the target network update rate"""
    target_network_frequency: int = 500
    """the timesteps it takes to update the target network"""
    batch_size: int = 128
    """the batch size of sample from the reply memory"""
    start_e: float = 1
    """the starting epsilon for exploration"""
    end_e: float = 0.05
    """the ending epsilon for exploration"""
    exploration_fraction: float = 0.5
    """the fraction of `total-timesteps` it takes from start-e to go end-e"""
    learning_starts: int = 10000
    """timestep to start learning"""
    train_frequency: int = 10
    """the frequency of training"""
    time_aware: bool = False
    """if toggled, will add a time feature to the observation"""
    hyperbolic_k: float = 0.05
    """the hyperbolic discounting constant"""
    discount_factor: str = "exponential"
    """the discount factor (exponential, hyperbolic)"""
    initial_stock: float = 0.0
    """the initial stock in the stock-aware wrapper"""
    reward_normalizer: float = 1.0
    """the reward normalizing factor of the extended state space"""
    stock_min: float = -10.0
    """the minimum stock for searching optimal stock"""
    stock_max: float = 10.0
    """the maximum stock for searching optimal stock"""
    stock_grid_size: int = 2001
    """the number of stock values to search over when updating the initial stock"""
    stock_frequency: int = 1000
    """the frequency of updating stock in the stock-aware wrapper"""
    risk_level: float = 1.0
    """the risk level for cvar"""
    mean_weight: float = 0.0
    """the weight for the mean in the mean-cvar objective"""
    target_averaging: bool = False
    """if toggled, use policy averaging for target computation"""
    stock_editing: bool = False
    """if toggled, this experiment will use stock editing when adding to the replay buffer"""
    stock_editing_fanout: int = 49
    """the number of synthetic transitions to generate per real transition (N)"""

    

def make_env(env_id, seed, idx, capture_video, run_name, **env_kwargs):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if len(env.observation_space.shape) > 1:
            env = gym.wrappers.FlattenObservation(env)
        assert 'TimeLimit' in env.__str__(), "TimeLimit wrapper not applied."
        time_aware = env_kwargs.get('time_aware')
        env = gym.wrappers.TimeAwareObservation(env, normalize_time=True) if time_aware else env # time is the second to last element of the observation if time_aware
        env = custom_envs.StockAwareObservation(env, **env_kwargs) # stock is the last element of the observation
        if isinstance(env.action_space, gym.spaces.MultiDiscrete):
            env = custom_envs.MultiDiscreteToDiscreteAction(env)
        env.action_space.seed(seed)

        return env

    return thunk


# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    def __init__(self, n_obs, n_act, n_quantiles, device=None):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(n_obs, 120, device=device),
            nn.ReLU(),
            nn.Linear(120, 84, device=device),
            nn.ReLU(),
            nn.Linear(84, n_act * n_quantiles, device=device),
            nn.Unflatten(1, (n_act, n_quantiles)),
        )

    def forward(self, x):
        x = x.float()
        return self.network(x)
    
    def get_action(self, x, action=None, **kwargs):
        quantiles_all = self.forward(x) # (batch_size, n_act, n_quantiles)
        # If action is provided and we don't need policy average, just return quantiles for that action
        if action is not None:
            quantiles = quantiles_all[torch.arange(len(x)), action]
        else:
            # Compute Q-values with risk-sensitive weighting
            reward_normalizer = kwargs.get('reward_normalizer')
            weight_left = kwargs.get('weight_left')
            weight_right = kwargs.get('weight_right')
            generator = kwargs.get('torch_generator')

            stock_batch = reward_normalizer * x[:, -1].reshape(-1, 1, 1)
            sp_quantiles = quantiles_all + stock_batch
            zeros = torch.zeros_like(sp_quantiles)
            neg_part = torch.minimum(sp_quantiles, zeros)
            pos_part = torch.maximum(sp_quantiles, zeros)
            sp_quantiles = weight_left * neg_part + weight_right * pos_part
            q_values = sp_quantiles.mean(2)  # (batch_size, n_actions)

            # Find the maximum Q-value for each batch
            max_q_values = q_values.max(dim=1, keepdim=True)[0]  # (batch_size, 1)
            
            # Find all actions that have the maximum Q-value (within tolerance for tie-breaking)
            tolerance = 1e-8
            is_max_action = torch.abs(q_values - max_q_values) < tolerance  # (batch_size, n_actions)
            n_max_actions = is_max_action.sum(dim=1, keepdim=True).float()  # (batch_size, 1)
            
            # Create greedy policy: equal probability among all maximum actions
            greedy_policy = is_max_action.float() / n_max_actions  # (batch_size, n_actions)

            # Sample action from greedy policy (with ties broken uniformly)
            action = torch.multinomial(
                greedy_policy,
                num_samples=1,
                generator=generator
            ).squeeze(1) if generator is not None else torch.multinomial(greedy_policy, num_samples=1).squeeze(1)
            if kwargs.get('target_averaging'):
                # Return policy-averaged quantiles (for target computation)
                quantiles = (quantiles_all * greedy_policy.unsqueeze(-1)).sum(dim=1)  # (batch_size, n_quantiles)
            else:
                # Get quantiles for the action
                quantiles = quantiles_all[torch.arange(len(x)), action]
            
        return action, quantiles

# huber loss function
def huber(x, k=1.0):
    return x.abs() if k==0 else torch.where(x.abs() < k, 0.5 * x.pow(2), k * (x.abs() - 0.5 * k))

def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)

def update_initial_stock(
    initial_state: np.ndarray,
    q_network: nn.Module,
    stock_linspace: np.ndarray,
    device: torch.device,
    get_action_kwargs: dict,
    args: "Args",
) -> float:
    """
    Update the initial stock value by searching over a range of stock values
    and selecting the one that maximizes the risk-sensitive Q-value.
    
    Args:
        envs: The vectorized environment.
        q_network: The Q-network to evaluate stock values.
        stock_linspace: Array of stock values to search over.
        device: The torch device (CPU or CUDA).
        get_action_kwargs: Keyword arguments for get_action method (includes weights, normalizer, etc.).
        args: The experiment arguments.
    
    Returns:
        The optimal initial stock value (unnormalized).
    """
    
    # Create batch of initial states with different stock values
    initial_states = np.array(
        [[*initial_state[:-1], stock/args.reward_normalizer] for stock in stock_linspace], 
        dtype=np.float32
    )
    
    # Get quantiles for each stock value
    _, quantiles = q_network.get_action(
        torch.tensor(initial_states, device=device), 
        **get_action_kwargs
    )  # (batch_size, n_quantiles)
    
    # Apply risk-sensitive transformation
    stock_batch = torch.tensor(stock_linspace, device=device).reshape(-1, 1).repeat(1, args.n_quantiles)
    sp_quantiles = stock_batch + quantiles  # (batch_size, n_quantiles)
    
    # Apply asymmetric weighting
    weight_left = get_action_kwargs.get('weight_left')
    weight_right = get_action_kwargs.get('weight_right')
    sp_quantiles = (
        weight_left * torch.min(torch.tensor(0, device=device), sp_quantiles) + 
        weight_right * torch.max(torch.tensor(0, device=device), sp_quantiles)
    )
    
    # Compute mean to get Q-values
    stock_q_values = (-stock_batch + sp_quantiles).mean(dim=1)  # (batch_size,)
    
    # Select stock value that maximizes Q-value
    max_idx = np.argmax(stock_q_values.cpu().detach().numpy())
    #optimal_stock = stock_linspace[max_idx]
    q_idx = int(args.risk_level * args.n_quantiles) - 1
    optimal_stock = -quantiles[max_idx][q_idx].cpu().detach().numpy()
    
    return optimal_stock


def perform_stock_editing(
    obs: np.ndarray,
    real_next_obs: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    terminations: np.ndarray,
    infos: dict,
    args: "Args",
    horizon: int,
) -> tuple:
    """
    Generates N synthetic transitions from a single real transition batch using stock editing.
    This function is fully vectorized and correctly handles both exponential and hyperbolic discounting.

    Args:
        obs (np.ndarray): Observations from the envs, shape (num_envs, obs_dim).
        real_next_obs (np.ndarray): Next observations from the envs.
        actions (np.ndarray): Actions taken.
        rewards (np.ndarray): Rewards received.
        terminations (np.ndarray): Termination flags.
        infos (dict): Info dicts from the envs.
        args (Args): The experiment's arguments.
        horizon (int): The maximum episode length (from env.spec).

    Returns:
        A tuple containing the expanded and edited (obs, next_obs, actions, ...)
        ready to be added to the replay buffer.
    """
    if not args.stock_editing or args.stock_editing_fanout <= 0:
        return obs, real_next_obs, actions, rewards, terminations, infos

    num_envs, obs_dim = obs.shape
    fanout = args.stock_editing_fanout

    # --- 1. Extract original data ---
    # `original_n` is the normalized time t/H from the TimeAwareObservation wrapper.
    # `original_c_t` is the stock value c_t.
    original_n = obs[:, -2]  # Shape: (num_envs,)
    original_c_t = obs[:, -1] # Shape: (num_envs,)

    # --- 2. Calculate Inverse Cumulative and One-Step Discount Factors ---
    if args.discount_factor == "exponential":
        # Cumulative discount D(t) = γ^t. Inverse is γ⁻ᵗ.
        # Note: t = original_n * horizon
        t_steps = original_n * horizon
        inv_cumulative_discounts = np.power(args.gamma, -t_steps, dtype=np.float64)
        # One-step discount d(t) is always γ.
        one_step_discounts = np.full(num_envs, args.gamma, dtype=np.float64)

    elif args.discount_factor == "hyperbolic":
        # Cumulative discount D(t) = 1 / (1 + k*t). Inverse is 1 + k*t.
        # Note: t = original_n * horizon
        t_steps = original_n * horizon
        inv_cumulative_discounts = 1 + args.hyperbolic_k * t_steps
        # One-step discount d(t) = (1+k*t) / (1+k*(t+1))
        numerator = 1 + args.hyperbolic_k * t_steps
        denominator = 1 + args.hyperbolic_k * (t_steps + 1)
        one_step_discounts = numerator / denominator
        
    else:
        raise NotImplementedError(f"Discount factor '{args.discount_factor}' not implemented for stock editing.")

    # --- 3. Generate and apply the stock offset for c_t ---
    new_c0 = np.random.uniform(
        low=args.stock_min,
        high=args.stock_max,
        size=(num_envs, fanout),
    )
    # Offset formula: D(t)⁻¹ * (c'_₀ - c₀)
    stock_offset_t = inv_cumulative_discounts[:, np.newaxis] * (new_c0 - args.initial_stock)
    edited_c_t = original_c_t[:, np.newaxis] + stock_offset_t

    # --- 4. Calculate the edited next stocks c'_{t+1} ---
    # Formula: c'_{t+1} = (c'_t + r_{t+1} / normalizer) / d(t)
    normalized_rewards = rewards / args.reward_normalizer
    edited_c_t_plus_1 = (edited_c_t + normalized_rewards[:, np.newaxis]) / one_step_discounts[:, np.newaxis]

    # --- 5. Assemble the final batches for the replay buffer ---
    batch_obs = np.broadcast_to(obs[:, np.newaxis, :], (num_envs, fanout + 1, obs_dim)).copy()
    batch_obs[:, 1:, -1] = edited_c_t

    batch_next_obs = np.broadcast_to(real_next_obs[:, np.newaxis, :], (num_envs, fanout + 1, obs_dim)).copy()
    batch_next_obs[:, 1:, -1] = edited_c_t_plus_1

    flat_batch_obs = batch_obs.reshape(-1, obs_dim)
    flat_batch_next_obs = batch_next_obs.reshape(-1, obs_dim)

    flat_batch_actions = np.repeat(actions, fanout + 1)
    flat_batch_rewards = np.repeat(rewards, fanout + 1)
    flat_batch_terminations = np.repeat(terminations, fanout + 1)
    batch_infos = [info for info in infos for _ in range(fanout + 1)]

    return (
        flat_batch_obs,
        flat_batch_next_obs,
        flat_batch_actions,
        flat_batch_rewards,
        flat_batch_terminations,
        batch_infos,
    )

if __name__ == "__main__":
    # Parse CLI args and merge with environment-specific config
    args = parse_args_with_config(Args)
    
    # Pretty-print args before training starts
    pretty_print_args(args, title=f"Training Configuration ({args.env_id})")
    
    assert args.num_envs == 1, "vectorized envs are not supported at the moment"
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}__{args.learning_rate}lr"
    run_name += f"__{args.n_quantiles}"
    if args.discount_factor == "hyperbolic":
        run_name += f"__hyperbolic_k{args.hyperbolic_k}"
        assert args.hyperbolic_k > 0, "hyperbolic_k must be greater than 0 for hyperbolic discounting"
        assert args.time_aware, "time_aware must be True for hyperbolic discounting"
    elif args.discount_factor == "exponential":
        run_name += f"__exponential{args.gamma}"
        if args.time_aware:
            run_name += "__time"
    else:
        raise ValueError("Invalid discount factor. Choose 'exponential' or 'hyperbolic'.")
    run_name += f"__{args.mean_weight}__{args.risk_level}"

    if args.stock_editing: assert args.time_aware, "time_aware must be True when using stock editing"

    if args.track:
        import wandb
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"{args.dir}/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    # Fixed RNGs for reproducible sampling
    python_rng = random.Random(args.seed)
    torch_rng = torch.Generator(device=device)
    torch_rng.manual_seed(args.seed)

    stock_linspace = np.linspace(args.stock_min, args.stock_max, args.stock_grid_size) #unnormalized stock values

    # env setup
    env_kwargs = {
        'time_aware': args.time_aware,
        'gamma': args.gamma,
        'initial_stock': args.initial_stock,
        'normalizer': args.reward_normalizer,
        'discount_factor': args.discount_factor,
        'hyperbolic_k': args.hyperbolic_k
    }
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name, **env_kwargs) for i in range(args.num_envs)],
        autoreset_mode=gym.vector.AutoresetMode.SAME_STEP
    )
    horizon = envs.envs[0].spec.max_episode_steps
    print("Horizon:", horizon)

    # discount factor setup
    discount_vector = torch.ones((args.batch_size, 1), device=device)* args.gamma
    if args.discount_factor == "hyperbolic":
        discount = lambda n: (1 + args.hyperbolic_k * n * horizon) / (1 + args.hyperbolic_k * (n * horizon + 1))
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"
    n_act = int(envs.single_action_space.n)
    n_obs = int(math.prod(envs.single_observation_space.shape))

    q_network = QNetwork(n_obs=n_obs, n_act=n_act, n_quantiles=args.n_quantiles, device=device)
    optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate, eps=0.01 / args.batch_size)
    target_network = QNetwork(n_obs=n_obs, n_act=n_act, n_quantiles=args.n_quantiles, device=device)
    target_network.load_state_dict(q_network.state_dict())

    tau_hat = (2 * torch.arange(args.n_quantiles, device=device) + 1) / (2.0 * args.n_quantiles)
    tau_hat_reshaped = tau_hat.view(1, -1, 1).repeat(args.batch_size, 1, args.n_quantiles)

    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        n_envs=args.num_envs+args.stock_editing_fanout if args.stock_editing else args.num_envs,
        handle_timeout_termination=False,
    )
    start_time = time.time()
    initial_stock = args.initial_stock

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)
    initial_state = obs[0].copy()
    get_action_kwargs = {
        'reward_normalizer': args.reward_normalizer,
        'weight_left': (1 - args.mean_weight) / args.risk_level + args.mean_weight, # weight for the quantiles less than risk level
        'weight_right': args.mean_weight, # weight for the quantiles greater than risk level
        'target_averaging': args.target_averaging,
        'torch_generator': torch_rng
    }
    pbar = tqdm.tqdm(range(args.total_timesteps))
    for global_step in pbar:
        # ALGO LOGIC: put action logic here
        epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_step)
        if python_rng.random() < epsilon:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            actions, _ = q_network.get_action(torch.tensor(obs, device=device), **get_action_kwargs)
            actions = actions.cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        for i in range(envs.num_envs):       
            if terminations[i] or truncations[i]:
                real_next_obs[i] = infos["final_obs"][i]
                writer.add_scalar("charts/episodic_return", infos["final_info"]["episode"]["r"][i], global_step)
                writer.add_scalar("charts/episodic_length", infos["final_info"]["episode"]["l"][i], global_step)

        if args.stock_editing:
            (
                buffer_obs,
                buffer_next_obs,
                buffer_actions,
                buffer_rewards,
                buffer_terminations,
                buffer_infos
            ) = perform_stock_editing(obs, real_next_obs, actions, rewards, terminations, infos, args, horizon)
            
            # Add the entire batch (original + synthetic) to the replay buffer
            rb.add(buffer_obs, buffer_next_obs, buffer_actions, buffer_rewards, buffer_terminations, buffer_infos) 
        else:
            rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            if global_step % args.train_frequency == 0:
                data = rb.sample(args.batch_size)
                with torch.no_grad():
                    _, next_quantiles = target_network.get_action(data.next_observations, **get_action_kwargs) # (batch_size, n_quantiles)
                    discount_vector = discount_vector if args.discount_factor == "exponential" else discount(data.observations[:, -2]).unsqueeze(-1)
                    target_quantiles = data.rewards + discount_vector * next_quantiles * (1 - data.dones)

                _, old_quantiles = q_network.get_action(data.observations, action=data.actions.flatten()) # (batch_size, n_quantiles)

                # (batch_size, 1, n_quantiles) - (batch_size, n_quantiles, 1) = (batch_size, n_quantiles, n_quantiles)
                diff = target_quantiles.unsqueeze(-1).transpose(-1, -2) - old_quantiles.unsqueeze(-1)
                
                loss = (huber(diff, args.huber_k) * (tau_hat_reshaped - (diff.detach() < 0).float()).abs()).mean(2).sum(1)
                loss = loss.mean()

                if global_step % 100 == 0:
                    writer.add_scalar("losses/loss", loss.item(), global_step)
                    old_val = old_quantiles.mean(1)
                    writer.add_scalar("losses/q_values", old_val.mean().item(), global_step)
                    writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

                # optimize the model
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # update target network
            if global_step % args.target_network_frequency == 0:
                for target_network_param, q_network_param in zip(target_network.parameters(), q_network.parameters()):
                    target_network_param.data.copy_(
                        args.tau * q_network_param.data + (1.0 - args.tau) * target_network_param.data
                    )

            # update the stock
            if global_step % args.stock_frequency == 0:
                initial_stock = update_initial_stock(
                    initial_state, q_network, stock_linspace, device, get_action_kwargs, args
                )
                writer.add_scalar("rewards/initial_stock", initial_stock, global_step)
                args.initial_stock = initial_stock
                # Update initial stock in existing environment without recreating it
                for env in envs.envs:
                    env.unwrapped  # Access the base env
                    # Find the StockAwareObservation wrapper and update it
                    wrapper = env
                    while hasattr(wrapper, 'env'):
                        if hasattr(wrapper, 'set_initial_stock'):
                            wrapper.set_initial_stock(initial_stock)
                            break
                        wrapper = wrapper.env
                #obs, _ = envs.reset(seed=args.seed)
            
    model_path = f"{args.dir}/{run_name}/{args.exp_name}.cleanrl_model"
    if args.save_model:
        model_data = {
            "model_weights": q_network.state_dict(),
            "args": vars(args),
        }
        torch.save(model_data, model_path)
        print(f"model saved to {model_path}")
    if args.evaluation_episodes > 0:
        from cleanrl_utils.qrdqn_eval import evaluate

        episodic_returns, episodic_discounted_returns = evaluate(
            model_path,
            make_env,
            args.env_id,
            eval_episodes=args.evaluation_episodes,
            run_name=f"{run_name}-eval",
            Model=QNetwork,
            device=device,
            epsilon=0.0,
            seed=1234,
        )
        # Log episodic returns as histogram
        writer.add_histogram("eval/episodic_returns", np.array(episodic_returns), global_step=args.total_timesteps, bins='auto')
        # Also log summary statistics
        writer.add_scalar("eval/mean_return", np.mean(episodic_returns), args.total_timesteps)
        writer.add_scalar("eval/std_return", np.std(episodic_returns), args.total_timesteps)
        writer.add_scalar("eval/min_return", np.min(episodic_returns), args.total_timesteps)
        writer.add_scalar("eval/max_return", np.max(episodic_returns), args.total_timesteps)

    envs.close()
    writer.close()
