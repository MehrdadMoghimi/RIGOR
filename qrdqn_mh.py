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
from cleanrl_utils import agent_utils 
from pprint import pprint
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
    time_consistent: bool = False
    """if toggled, will use the time-consistent hyperbolic discounting"""
    n_gammas: int = 10
    """the number of discount factors to use for multi-horizon learning (1 for single horizon)"""
    max_gamma: float = 0.999
    """the maximum discount factor when using multi-horizon learning"""
    acting_policy: str = "hyperbolic"
    """the policy with which the agent will act.  One of ['hyperbolic', 'largest_gamma']"""
    integral_estimate:  str = "lower"
    """how to estimate the integral of the hyperbolic discounted q-values. One of ['lower', 'upper']"""


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
        env = gym.wrappers.TimeAwareObservation(env, normalize_time=True) if time_aware else env # time is the last element of the observation if time_aware
        if isinstance(env.action_space, gym.spaces.MultiDiscrete):
            env = custom_envs.MultiDiscreteToDiscreteAction(env)
        env.action_space.seed(seed)

        return env

    return thunk


# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    def __init__(self, n_obs, n_act, n_gammas, n_quantiles, device=None):
        super().__init__()
        self.n_gammas = n_gammas
        self.network = nn.Sequential(
            nn.Linear(n_obs, 120, device=device),
            nn.ReLU(),
            nn.Linear(120, 84, device=device),
            nn.ReLU(),
            nn.Linear(84, n_act * n_gammas * n_quantiles, device=device),
            nn.Unflatten(1, (n_act, n_gammas, n_quantiles)),
        )

    def forward(self, x):
        x = x.float()
        return self.network(x)
    
    def get_action(self, x, action=None, **kwargs):
        quantiles = self.forward(x) # (batch_size, n_act, n_gammas, n_quantiles)

        if action is None:
            acting_policy = kwargs.get('acting_policy', 'hyperbolic')
            time_consistent = kwargs.get('time_consistent')
            gammas = kwargs.get('gammas')
            weights = kwargs.get('weights')
            horizon = kwargs.get('horizon')
            q_values_gammas = quantiles.mean(3) # (batch_size, n_act, n_gammas)
            if acting_policy == 'hyperbolic':
                if time_consistent:
                    time_steps = x[:, -1] * horizon
                    weights = weights[None, :] * torch.pow(gammas[None, :], time_steps[:, None]) # (batch_size, n_gammas)
                    normalized_weights = weights / weights.sum(1, keepdim=True) # (batch_size, n_gammas)
                    # integrate over the gammas to get the hyperbolic q-values
                    q_values = (q_values_gammas * normalized_weights.unsqueeze(1)).sum(2) # (batch_size, n_act)
                else:
                    # integrate over the gammas to get the hyperbolic q-values
                    q_values = q_values_gammas.mul(weights).sum(2) # (batch_size, n_act)
            elif acting_policy == 'largest_gamma':
                q_values = q_values_gammas[:, :, -1] # (batch_size, n_act)
            else:
                raise ValueError("acting_policy must be one of ['hyperbolic', 'largest_gamma']")
            action = torch.argmax(q_values, 1)
        return action, quantiles[torch.arange(len(x)), action]

# huber loss function
def huber(x, k=1.0):
    return x.abs() if k==0 else torch.where(x.abs() < k, 0.5 * x.pow(2), k * (x.abs() - 0.5 * k))

def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


if __name__ == "__main__":
    # Parse CLI args and merge with environment-specific config
    args = parse_args_with_config(Args)
    
    # Pretty-print args before training starts
    pretty_print_args(args, title=f"Training Configuration ({args.env_id})")
    
    # assert args.num_envs == 1, "vectorized envs are not supported at the moment"
    if args.time_consistent: assert args.time_aware, "time_consistent requires time_aware to be True"
    if args.time_consistent: assert args.acting_policy == 'hyperbolic', "time_consistent requires hyperbolic acting_policy"

    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    run_name += f"__{args.n_quantiles}"
    run_name += f"__{args.n_gammas}_gammas__{args.integral_estimate}_estimate__{args.acting_policy}_policy"
    if args.time_consistent: 
        run_name += f"__time_consistent"
    else:
        if args.time_aware:
            run_name += "__time"

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

    # env setup
    env_kwargs = {
        'time_aware': args.time_aware
    }
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name, **env_kwargs) for i in range(args.num_envs)],
        autoreset_mode=gym.vector.AutoresetMode.SAME_STEP
    )
    horizon = envs.envs[0].spec.max_episode_steps
    print("Horizon:", horizon)

    # multi-horizon setup
    # These are the discount factors (gammas) used to estimate the integral.
    eval_gammas = agent_utils.compute_eval_gamma_interval(args.max_gamma, args.hyperbolic_k, args.n_gammas)
    # However, if we wish to estimate hyperbolic discounting with the form,
    #
    #      \Gamma_t =  1. / (1. + k * t)
    #
    # where we now have a coefficient k <= 1.0
    # we need consider the value functions for \gamma ^ k.  We refer to
    # these below as self.gammas, since these are the gammas actually being
    # learned via Bellman updates.
    gammas = [math.pow(gamma, args.hyperbolic_k) for gamma in eval_gammas]
    assert max(gammas) <= args.max_gamma

    if args.integral_estimate == 'lower':
        gamma_plus_one = eval_gammas + [1.]
        weights = [gamma_plus_one[i + 1] - gamma_plus_one[i] for i in range(args.n_gammas)]

    elif args.integral_estimate == 'upper':
        gamma = eval_gammas
        weights = [gamma[i + 1] - gamma[i] for i in range(args.n_gammas - 1)]
        weights = [0.] + weights
    q_weights = torch.tensor(weights, device=device)
    gammas = torch.tensor(gammas, device=device)
    pprint(f"eval_gammas: {eval_gammas}", width=200)
    pprint(f"gammas: {gammas}", width=200)
    pprint(f"q_weights: {q_weights}", width=200)
    pprint(f"weight_sum: {q_weights.sum()}", width=200)

    
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"
    n_act = int(envs.single_action_space.n)
    n_obs = int(math.prod(envs.single_observation_space.shape))

    q_network = QNetwork(n_obs=n_obs, n_act=n_act, n_gammas=args.n_gammas, n_quantiles=args.n_quantiles, device=device)
    optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate, eps=0.01 / args.batch_size)
    target_network = QNetwork(n_obs=n_obs, n_act=n_act, n_gammas=args.n_gammas, n_quantiles=args.n_quantiles, device=device)
    target_network.load_state_dict(q_network.state_dict())

    tau_hat = (2 * torch.arange(args.n_quantiles, device=device) + 1) / (2.0 * args.n_quantiles)
    tau_hat_reshaped = tau_hat.view(1, 1, -1, 1).repeat(args.batch_size, args.n_gammas, 1, args.n_quantiles)

    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        handle_timeout_termination=False,
        n_envs=args.num_envs,
    )
    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)
    get_action_kwargs = {
        'weights': q_weights,
        'acting_policy': args.acting_policy,
        'time_consistent': args.time_consistent,
        'gammas': gammas,
        'horizon': horizon
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
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            if global_step % args.train_frequency == 0:
                data = rb.sample(args.batch_size)
                with torch.no_grad():
                    _, next_quantiles = target_network.get_action(data.next_observations, **get_action_kwargs) # (batch_size, n_gammas, n_quantiles)
                    target_quantiles = data.rewards.unsqueeze(-1) + gammas[None, :, None] * next_quantiles * (1 - data.dones.unsqueeze(-1))

                _, old_quantiles = q_network.get_action(data.observations, data.actions.flatten()) # (batch_size, n_gammas, n_quantiles)

                # (batch_size, n_gammas, 1, n_quantiles) - (batch_size, n_gammas, n_quantiles, 1) = (batch_size, n_gammas, n_quantiles, n_quantiles)
                diff = target_quantiles.unsqueeze(-1).transpose(-1, -2) - old_quantiles.unsqueeze(-1)
                
                loss = (huber(diff, args.huber_k) * (tau_hat_reshaped - (diff.detach() < 0).float()).abs()).mean(3).sum(2)
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
            epsilon=args.end_e,
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
