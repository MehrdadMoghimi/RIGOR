# Decoupling Time and Risk: Risk-Sensitive RL with General Discounting

##  Quick Start

Create and activate a conda environment:

```bash
conda env create -f environment.yml
conda activate rigor
```

Run the QR-DQN algorithm:

```bash
python qrdqn.py --env-id CartPole-v1 --time-aware --discount-factor hyperbolic
```

## 📁 Project Structure

```
RIGOR/
├── qrdqn.py                      # Risk-neutral single-horizon QR-DQN
├── qrdqn_mh.py                   # Multi-horizon QR-DQN for standard envs
├── qrdqn_mh_atari.py             # Multi-horizon QR-DQN for Atari games
├── qrdqn_mean_cvar.py            # Stock-augmented risk-sensitive QR-DQN for Mean-CVaR
├── qrdqn_abs.py                  # Stock-augmented risk-sensitive QR-DQN for absolute value
├── envs/                         # Custom environments
│   ├── __init__.py
│   ├── american_option_env.py    # American option pricing environment
│   ├── GBWM.py                   # Goal-Based Wealth Management
│   └── wrappers.py               # Environment wrappers
├── cleanrl_utils/                # Utilities and helper functions
│   ├── agent_utils.py            # Helper functions for hyperbolic discounting
│   ├── atari_wrappers.py         # Atari-specific wrappers
│   ├── buffers.py                # Replay buffer implementation
│   ├── config.py                 # Configuration management
│   ├── plot_return_distribution.py  # Visualization tools
│   ├── pretty_print.py           # Pretty printing utilities
│   └── qrdqn_eval.py             # Evaluation utilities
├── configs/                      # YAML configuration files
│   ├── qrdqn.yaml
│   ├── qrdqn_mean_cvar.yaml
│   └── qrdqn_mh.yaml
├── files/                        # Media files (images, videos)
├── environment.yml               # Conda environment specification
├── requirements.txt              # Python dependencies
├── README.md                     # This file
└── SETUP.md                      # Detailed setup instructions
```

## 🔬 Algorithms

All algorithms are based on **QR-DQN (Quantile Regression DQN)**, a distributional reinforcement learning algorithm that learns the full distribution of returns.

### 1. `qrdqn.py` - Risk-Neutral Single-Horizon

**Description:** Standard QR-DQN implementation with support for both exponential and hyperbolic discounting. This is the **risk-neutral** version that optimizes expected returns.

**Key Features:**
- Single discount factor (single horizon)
- Exponential discounting (standard RL)
- Hyperbolic discounting
- Time-aware observations for non-stationary policies

**Key Arguments:**
- `--discount-factor {exponential,hyperbolic}`: Type of discounting (default: exponential)
- `--time-aware`: Enable time-augmented observations (required for hyperbolic)
- `--hyperbolic-k FLOAT`: Hyperbolic discounting constant (default: 0.05)
- `--gamma FLOAT`: Discount factor for exponential discounting (default: 0.99)
- `--n-quantiles INT`: Number of quantiles for return distribution (default: 50)

### 2. `qrdqn_mh.py` - Multi-Horizon (Standard Environments)

**Description:** Multi-horizon QR-DQN that learns multiple value functions with different discount factors and combines them to approximate hyperbolic discounting. Implements both time-consistent (our method) and time-inconsistent (Fedus et al.) variants.

**Key Features:**
- Multiple discount factors (multi-horizon learning)
- Time-consistent hyperbolic discounting (**our paper**)
- Time-inconsistent hyperbolic discounting (Fedus et al. baseline)
- Largest-gamma acting policy

**Key Arguments:**
- `--acting-policy {hyperbolic,largest_gamma}`: Policy for action selection (default: hyperbolic)
- `--time-consistent`: Enable time-consistent weights (our method)
- `--time-aware`: Enable time-augmented observations (required for time-consistent)
- `--n-gammas INT`: Number of discount factors (default: 10)
- `--max-gamma FLOAT`: Maximum discount factor (default: 0.999)
- `--hyperbolic-k FLOAT`: Hyperbolic discounting constant (default: 0.05)
- `--integral-estimate {lower,upper}`: Integration method (default: lower)

**Usage:**
```bash
# Time-consistent (our paper)
python qrdqn_mh.py --acting-policy hyperbolic --time-aware --time-consistent

# Time-inconsistent (Fedus et al.)
python qrdqn_mh.py --acting-policy hyperbolic

# Largest gamma policy
python qrdqn_mh.py --acting-policy largest_gamma
```

### 3. `qrdqn_mh_atari.py` - Multi-Horizon (Atari)

**Description:** Multi-horizon QR-DQN optimized for Atari environments. Uses convolutional neural networks and efficient ALE vector environments.

**Key Features:**
- CNN-based architecture for image observations
- Efficient ALE vector environment
- Time-consistent and time-inconsistent variants
- Same multi-horizon logic as `qrdqn_mh.py`

**Key Arguments:**
- Same as `qrdqn_mh.py` plus:
- Atari-specific hyperparameters (see configs)

**Usage:**
```bash
# Time-consistent on Breakout
python qrdqn_mh_atari.py --env-id BreakoutNoFrameskip-v4 --acting-policy hyperbolic --time-consistent --time-aware

# Time-inconsistent on Pong
python qrdqn_mh_atari.py --env-id PongNoFrameskip-v4 --acting-policy hyperbolic
```

### 4. `qrdqn_mean_cvar.py` - Stock-Augmented Risk-Sensitive

**Description:** Risk-sensitive QR-DQN using stock-augmented states. Supports Mean-CVaR objectives with both exponential and hyperbolic discounting.

**Key Features:**
- Stock augmentation for tracking accumulated rewards
- Risk-sensitive objectives (CVaR, Mean-CVaR)
- Target averaging for stable learning
- Stock editing for data augmentation
- Support for both exponential and hyperbolic discounting

**Key Arguments:**
- `--discount-factor {exponential,hyperbolic}`: Type of discounting
- `--risk-level FLOAT`: CVaR risk level (0 to 1, default: 1.0 for risk-neutral)
- `--mean-weight FLOAT`: Weight for mean in Mean-CVaR (default: 0.0)
- `--initial-stock FLOAT`: Initial stock value (default: 0.0)
- `--reward-normalizer FLOAT`: Reward normalization factor (default: 1.0)
- `--target-averaging`: Enable target averaging
- `--stock-editing`: Enable stock editing data augmentation
- `--stock-editing-fanout INT`: Number of synthetic transitions per real transition (default: 49)
- `--stock-min FLOAT`, `--stock-max FLOAT`: Stock value range
- `--stock-grid-size INT`: Grid size for stock optimization (default: 2001)

**Usage:**
```bash
# Risk-sensitive with CVaR
python qrdqn_mean_cvar.py --risk-level 0.8 --mean-weight 0.2

# With hyperbolic discounting
python qrdqn_mean_cvar.py --discount-factor hyperbolic --time-aware --risk-level 0.8

# With stock editing
python qrdqn_mean_cvar.py --stock-editing --target-averaging --risk-level 0.5
```

## 🛠 Features

- **YAML-based Configuration**: Algorithm-specific default hyperparameters per environment (`configs/qrdqn.yaml`, `configs/qrdqn_mean_cvar.yaml`)
- **Return Distribution Analysis**: Visualization tools for CDF/PDF comparison (`cleanrl_utils/plot_return_distribution.py`)
- **Stock Editing**: Data augmentation for mean-CVaR with synthetic transitions
- **Target Averaging**: Policy averaging for stable target computation in risk-sensitive RL
- CUDA support for GPU acceleration

## 📊 Supported Environments

### Standard Environments (via Gymnasium)
- **Classic Control**: CartPole, MountainCar, Acrobot, Pendulum
- **Box2D**: LunarLander, BipedalWalker
- **MuJoCo**: (with mujoco-py installation)

### Custom Environments
- **AmericanOptionEnv-v1**: American option pricing with Black-Scholes dynamics
- **GBWM (Goal-Based Wealth Management)**: Multi-goal portfolio optimization

### Atari Environments
- **50+ Atari Games**: Breakout, Pong, SpaceInvaders, Seaquest, etc.
- Uses efficient ALE (Arcade Learning Environment) with preprocessing

**Note:** Environment-specific hyperparameters are provided in the `configs/` directory.

## 🎯 Usage Examples

### Single-Horizon QR-DQN (`qrdqn.py`)

```bash
# Standard exponential discounting (risk-neutral)
python qrdqn.py --env-id CartPole-v1

# With time-aware observations
python qrdqn.py --env-id CartPole-v1 --time-aware

# Hyperbolic discounting (our paper)
python qrdqn.py --env-id CartPole-v1 --time-aware --discount-factor hyperbolic

# Custom hyperparameters
python qrdqn.py --env-id LunarLander-v2 --learning-rate 0.001 --batch-size 64 --n-quantiles 100

```

### Multi-Horizon QR-DQN (`qrdqn_mh.py`)

```bash
# Time-consistent hyperbolic (our paper)
python qrdqn_mh.py --env-id CartPole-v1 --acting-policy hyperbolic --time-aware --time-consistent

# Time-inconsistent hyperbolic (Fedus et al. baseline)
python qrdqn_mh.py --env-id CartPole-v1 --acting-policy hyperbolic

# Largest-gamma policy
python qrdqn_mh.py --env-id LunarLander-v2 --acting-policy largest_gamma

# Custom multi-horizon setup
python qrdqn_mh.py --n-gammas 20 --max-gamma 0.999 --hyperbolic-k 0.1
```

### Multi-Horizon Atari (`qrdqn_mh_atari.py`)

```bash
# Time-consistent on Breakout (reproducing paper results)
python qrdqn_mh_atari.py --env-id BreakoutNoFrameskip-v4 --time-consistent --time-aware

# Time-inconsistent baseline on Pong
python qrdqn_mh_atari.py --env-id PongNoFrameskip-v4 --acting-policy hyperbolic

# Different Atari game
python qrdqn_mh_atari.py --env-id SeaquestNoFrameskip-v4 --time-consistent --time-aware
```

### Risk-Sensitive Stock-Augmented (`qrdqn_mean_cvar.py`)

```bash
# CVaR with risk level 0.8 (risk-averse)
python qrdqn_mean_cvar.py --env-id AmericanOptionEnv-v1 --risk-level 0.8 --mean-weight 0.0

# Mean-CVaR objective (80% CVaR, 20% mean)
python qrdqn_mean_cvar.py --env-id AmericanOptionEnv-v1 --risk-level 0.8 --mean-weight 0.2

# With hyperbolic discounting
python qrdqn_mean_cvar.py --env-id AmericanOptionEnv-v1 \
  --discount-factor hyperbolic --time-aware --hyperbolic-k 0.05 \
  --risk-level 0.8 --mean-weight 0.2

# With target averaging (recommended for stability)
python qrdqn_mean_cvar.py --env-id AmericanOptionEnv-v1 \
  --risk-level 0.5 --mean-weight 0.5 --target-averaging

# With stock editing (data augmentation)
python qrdqn_mean_cvar.py --env-id AmericanOptionEnv-v1 \
  --risk-level 0.8 --mean-weight 0.2 --gamma 0.99 \
  --stock-editing --stock-editing-fanout 49 \
  --stock-min 0 --stock-max 20 --stock-grid-size 2001 \
  --target-averaging --time-aware

# Full example: hyperbolic + risk-sensitive + stock editing
python qrdqn_mean_cvar.py --env-id AmericanOptionEnv-v1 \
  --discount-factor hyperbolic --hyperbolic-k 0.05 --time-aware \
  --risk-level 0.8 --mean-weight 0.2 \
  --stock-editing --target-averaging \
  --total-timesteps 1000000 --n-quantiles 200 \
  --save-model --dir runs/option
```

### Common Options (All Algorithms)

```bash
# Training parameters
--total-timesteps INT        # Total training steps (default: 500000)
--learning-rate FLOAT        # Learning rate (default: 1e-3)
--batch-size INT             # Batch size (default: 128)
--buffer-size INT            # Replay buffer size (default: 100000)

# Exploration
--start-e FLOAT              # Starting epsilon (default: 1.0)
--end-e FLOAT                # Final epsilon (default: 0.05)
--exploration-fraction FLOAT  # Fraction of training for exploration decay (default: 0.5)

# Network & training
--n-quantiles INT            # Number of quantiles (default: 50)
--target-network-frequency INT  # Target network update frequency (default: 500)
--train-frequency INT        # Training frequency (default: 10)

# Logging & saving
--track                      # Enable Weights & Biases tracking
--wandb-project-name STR     # W&B project name
--save-model                 # Save trained model
--dir STR                    # Directory for runs (default: "runs")
--seed INT                   # Random seed (default: 1)

# Evaluation
--evaluation-episodes INT    # Number of evaluation episodes (default: 1000)
```

## � Research Results & Key Findings

**For a detailed discussion of our research, see our [blog post](https://mehrdadmoghimi.github.io/posts/2026/02/rigor/).**

In standard Reinforcement Learning (RL), the discount factor (γ) is often treated as a fixed parameter of the Markov Decision Process or a tunable hyperparameter for training stability. We typically default to **exponential discounting**, where the value of a reward decays by a constant factor at every time step.

While mathematically convenient, this standard formulation is restrictive. It limits our ability to model complex **time preferences** (how an agent values the future vs. the present) and **risk preferences** (how an agent handles uncertainty) independently.

In our recent [paper](https://arxiv.org/abs/2602.04131), **"Decoupling Time and Risk: Risk-Sensitive RL with General Discounting,"** we propose a unified framework that supports general discount functions and risk measures. By properly handling **time consistency** and tracking accumulated rewards, we show that we can capture more expressive behaviors, like preference reversals, and significantly improve performance in complex environments.

### The Problem with "Stationary" Hyperbolic Discounting

A major motivation for this work was to revisit **hyperbolic discounting**. Unlike exponential discounting, hyperbolic discounting models agents that are impatient in the short term but patient in the long term—a behavior observed in humans and animals.

A notable approach by **Fedus et al. (2019)** attempted to introduce hyperbolic discounting into Deep RL. They approximated the hyperbolic discount function as a weighted average of multiple exponential discount factors, but used **fixed weights** throughout the episode. By enforcing a stationary policy, they implicitly reset the agent's "time zero" at every step, leading to **time inconsistency**: the policy the agent plans at t=0 is not the policy it wants to execute at t=1.

### Our Solution: Time-Dependent Weights

We argue that to solve general discounting problems correctly, the agent must be explicitly aware of time, and the weights must evolve. In our **multi-horizon framework**, we show that as time t progresses, the effective contribution of each exponential discount factor changes. The weights should not be static constants, but rather time-dependent weights that vary with time.

<video controls autoplay loop muted playsinline width="100%">
  <source src="files/EvolvingWeightsExact.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

*Evolution of time-dependent weights in our multi-horizon framework. As time progresses, the contribution of each discount factor changes, ensuring time-consistency.*

### A Unified Framework for Time and Risk

Our contributions go beyond just fixing hyperbolic discounting. We introduce a broad framework called **RIGOR** (**RI**sk-sensitive RL under **G**eneral discounting **O**f **R**eturns) that decouples time and risk:

1. **Stock-Augmented Distributional RL:** We build on the idea of augmenting the state with a "stock" that tracks accumulated rewards, with an "Anytime Proxy" equation that guarantees the agent optimizes the global objective from any time step.

2. **General Discount Functions:** Our method supports any non-increasing discount function (hyperbolic, quasi-hyperbolic, etc.), not just exponential.

3. **OCE Risk Measures:** By operating on the full return distribution, we can optimize for **Optimized Certainty Equivalent (OCE)** risk measures, such as **Conditional Value at Risk (CVaR)** or **Entropic Risk**.

![OCE Risk Measures Utilities](files/utilities.png)

*Utility functions for common OCE risk measures. Our framework allows us to plug in different utility functions to shape the agent's risk profile, independent of the discount function.*

### Preference Reversals in Wealth Management

To demonstrate that our agent captures human-like time preferences, we tested it on a "Goal-Based Wealth Management" problem. We compared a standard Exponential agent against our Hyperbolic agent.

The results showed a clear **preference reversal**. When the "late goal" was more valuable, the Hyperbolic agent showed impatience for immediate rewards but patience for distant ones, shifting its probability of success in a way the Exponential agent could not capture.

![Goal-Based Wealth Management](files/gbwm-risk-E.png)

*Monte-Carlo probabilities of achieving goals. The shift in the red markers (Hyperbolic) compared to the blue (Exponential) illustrates the agent's non-linear time preference, capturing behaviors that standard RL misses.*

### Improving Performance in Atari

Finally, we evaluated whether fixing the theoretical inconsistency in Fedus et al. actually matters for performance. We compared our **Time-Consistent** agent against the **Time-Inconsistent** baseline across 50 Atari games.

The results were significant. By correctly modeling the non-stationary optimal policy and evolving the weights over time, our method achieved higher returns in **39 out of 50 games**, with a mean improvement of roughly **40%**.

![Atari Performance Improvement](files/improvement.png)

*Relative performance improvement of our Time-Consistent algorithm across 50 Atari games. The consistent positive trend demonstrates the benefits of maintaining time-consistency under hyperbolic discounting.*

### Key Takeaways

Discounting is a fundamental part of the problem definition. It encodes **time preference**, which is distinct from the **risk preference** encoded in the objective function.

By decoupling these two dimensions and ensuring our optimization remains time-consistent, we can build RL agents that are not only more expressive and robust but also perform better on complex control tasks.

**Read the full paper:** [arXiv:2602.04131](https://arxiv.org/abs/2602.04131)

**Detailed blog post:** [https://mehrdadmoghimi.github.io/posts/2026/02/rigor/](https://mehrdadmoghimi.github.io/posts/2026/02/rigor/)


## � References

If you use this code in your research, please cite:

```bibtex
@article{moghimi2026decoupling,
  title={Decoupling Time and Risk: Risk-Sensitive Reinforcement Learning with General Discounting},
  author={Moghimi, Mehrdad and Coache, Anthony and Ku, Hyejin},
  journal={arXiv preprint arXiv:2602.04131},
  year={2026}
}
```

## 🙏 Acknowledgments

This project builds upon:
- [CleanRL](https://github.com/vwxyzjn/cleanrl) - Clean implementations of RL algorithms