"""
Plot the CDF and PDF of return distribution for the initial state of a trained QR-DQN model.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path
import torch
import argparse
from argparse import Namespace
from pprint import pprint

# Add project root to path
project_root = Path.cwd().resolve()
if not (project_root / "envs").exists():
    project_root = project_root.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def load_model(model_path, device='cpu'):
    """Load a trained QR-DQN model."""
    print(f"Loading model from: {model_path}")
    model_data = torch.load(model_path, map_location=device)
    args = Namespace(**model_data["args"])
    
    # Debug: Print all args
    print("\nAll loaded args:")
    for key, value in vars(args).items():
        print(f"  {key}: {value}")
    
    # Get n_quantiles from args (only for non-categorical algorithms)
    n_quantiles = getattr(args, 'n_quantiles', None)

    # Create environment with proper kwargs
    if args.exp_name == "qrdqn":
        from qrdqn import QNetwork as QNetwork_qrdqn, make_env as make_env_qrdqn
        env_kwargs = {'time_aware': args.time_aware}
        env = make_env_qrdqn(args.env_id, 0, 0, False, "", **env_kwargs)()
    elif args.exp_name == "qrdqn_abs":
        from qrdqn_abs import QNetwork as QNetwork_qrdqn_abs, make_env as make_env_qrdqn_abs
        env_kwargs = {
            'time_aware': args.time_aware,
            'gamma': args.gamma,
            'initial_stock': args.initial_stock,
            'normalizer': args.reward_normalizer,
            'discount_factor': args.discount_factor,
            'hyperbolic_k': args.hyperbolic_k
        }
        env = make_env_qrdqn_abs(args.env_id, 0, 0, False, "", **env_kwargs)()
    elif args.exp_name == "qrdqn_mean_cvar":
        from qrdqn_mean_cvar import QNetwork as QNetwork_qrdqn_mean_cvar, make_env as make_env_qrdqn_mean_cvar
        env_kwargs = {
            'time_aware': args.time_aware,
            'gamma': args.gamma,
            'initial_stock': args.initial_stock,
            'normalizer': args.reward_normalizer,
            'discount_factor': args.discount_factor,
            'hyperbolic_k': args.hyperbolic_k
        }
        env = make_env_qrdqn_mean_cvar(args.env_id, 0, 0, False, "", **env_kwargs)()
    elif args.exp_name == "categorical":
        from categorical import QNetwork as QNetwork_categorical, make_env as make_env_categorical
        env_kwargs = {'time_aware': args.time_aware}
        env = make_env_categorical(args.env_id, 0, 0, False, "", **env_kwargs)()
    elif args.exp_name == "categorical_mean_cvar":
        from categorical_mean_cvar import QNetwork as QNetwork_categorical_mean_cvar, make_env as make_env_categorical_mean_cvar
        env_kwargs = {
            'time_aware': args.time_aware,
            'gamma': args.gamma,
            'initial_stock': args.initial_stock,
            'normalizer': args.reward_normalizer,
            'discount_factor': args.discount_factor,
            'hyperbolic_k': args.hyperbolic_k
        }
        env = make_env_categorical_mean_cvar(args.env_id, 0, 0, False, "", **env_kwargs)()
    else:
        raise ValueError(f"Unsupported algorithm: {args.exp_name}")
    
    pprint(f"Environment created: {args.env_id} with kwargs: {env_kwargs}")
    
    # Get environment dimensions
    obs_shape = env.observation_space.shape
    n_actions = env.action_space.n
    
    # Create model with correct QNetwork based on algorithm type
    if args.exp_name == "qrdqn":
        model = QNetwork_qrdqn(n_obs=int(np.prod(obs_shape)), n_act=int(n_actions), n_quantiles=int(n_quantiles), device=device)
    elif args.exp_name == "qrdqn_abs":
        model = QNetwork_qrdqn_abs(n_obs=int(np.prod(obs_shape)), n_act=int(n_actions), n_quantiles=int(n_quantiles), device=device)
    elif args.exp_name == "qrdqn_mean_cvar":
        model = QNetwork_qrdqn_mean_cvar(n_obs=int(np.prod(obs_shape)), n_act=int(n_actions), n_quantiles=int(n_quantiles), device=device)
    elif args.exp_name == "categorical":
        model = QNetwork_categorical(n_obs=int(np.prod(obs_shape)), n_act=int(n_actions), n_atoms=args.n_atoms, v_min=args.v_min, v_max=args.v_max, device=device)
    elif args.exp_name == "categorical_mean_cvar":
        model = QNetwork_categorical_mean_cvar(n_obs=int(np.prod(obs_shape)), n_act=int(n_actions), n_atoms=args.n_atoms, v_min=args.v_min, v_max=args.v_max, device=device)
    else:
        raise ValueError(f"Unsupported algorithm: {args.exp_name}")
    
    # Load weights
    model.load_state_dict(model_data["model_weights"])
    model.eval()
    
    print(f"Model loaded successfully!")
    print(f"  Algorithm: {args.exp_name}")
    print(f"  Observation shape: {obs_shape}")
    print(f"  Number of actions: {n_actions}")
    if args.exp_name == "categorical" or args.exp_name == "categorical_mean_cvar":
        print(f"  Number of atoms: {args.n_atoms}")
    else:
        print(f"  Number of quantiles: {n_quantiles}")
    
    # Prepare get_action_kwargs based on algorithm
    get_action_kwargs = {}
    if args.exp_name == "qrdqn_abs":
        get_action_kwargs = {'reward_normalizer': args.reward_normalizer}
    elif args.exp_name == "qrdqn_mean_cvar":
        # Debug: print the args to check if mean_weight and risk_level exist
        print(f"\nChecking args for qrdqn_mean_cvar:")
        print(f"  mean_weight: {getattr(args, 'mean_weight', 'NOT FOUND')}")
        print(f"  risk_level: {getattr(args, 'risk_level', 'NOT FOUND')}")
        
        mean_weight = args.mean_weight
        risk_level = args.risk_level
        weight_left = (1 - mean_weight) / risk_level + mean_weight
        weight_right = mean_weight
        
        print(f"  Calculated weight_left: {weight_left}")
        print(f"  Calculated weight_right: {weight_right}")
        
        get_action_kwargs = {
            'reward_normalizer': args.reward_normalizer,
            'weight_left': weight_left,
            'weight_right': weight_right,
            'target_averaging': getattr(args, 'target_averaging', False),
            'torch_generator': None
        }
    elif args.exp_name == "categorical_mean_cvar":
        # Debug: print the args to check if mean_weight and risk_level exist
        print(f"\nChecking args for categorical_mean_cvar:")
        print(f"  mean_weight: {getattr(args, 'mean_weight', 'NOT FOUND')}")
        print(f"  risk_level: {getattr(args, 'risk_level', 'NOT FOUND')}")
        
        # Same as qrdqn_mean_cvar: use weight_left/weight_right for quantile-based action selection
        mean_weight = getattr(args, 'mean_weight', 0.5)
        risk_level = getattr(args, 'risk_level', 0.5)
        weight_left = (1 - mean_weight) / risk_level + mean_weight
        weight_right = mean_weight
        
        get_action_kwargs = {
            'reward_normalizer': args.reward_normalizer,
            'weight_left': weight_left,
            'weight_right': weight_right,
            'n_quantiles': getattr(args, 'n_atoms', 51) * 10,  # 10x atoms for quantile approximation
            'target_averaging': getattr(args, 'target_averaging', False),
            'torch_generator': None
        }
    
    return model, env, args, get_action_kwargs


def get_initial_state_distribution(model, env, args, get_action_kwargs, device='cpu', seed=42):
    """Get the return distribution for the initial state."""
    # Reset environment to get initial observation
    obs, info = env.reset(seed=seed)
    print(f"\nInitial state info: {info}")
    print(f"Observation shape: {obs.shape}")
    print(f"Initial Observation: {obs}")
    
    # Convert observation to tensor
    obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    
    with torch.no_grad():
        # Get action and distribution for the initial state
        if args.exp_name == "categorical" or args.exp_name == "categorical_mean_cvar":
            action, pmfs = model.get_action(obs_tensor, **get_action_kwargs)
            action = action.item()
            pmfs = pmfs.squeeze(0).cpu().numpy()
            atoms = model.atoms.cpu().numpy()
            distribution = (atoms, pmfs)
            print(f"Selected action: {action}")
            print(f"Atoms range: [{atoms.min():.4f}, {atoms.max():.4f}]")
            print(f"PMF sum: {pmfs.sum():.4f}")
        else:
            action, quantiles = model.get_action(obs_tensor, **get_action_kwargs)
            action = action.item()
            quantiles = quantiles.squeeze(0).cpu().numpy()  # Shape: (n_quantiles,)
            distribution = quantiles
            print(f"Selected action: {action}")
            print(f"Quantiles shape: {quantiles.shape}")
            print(f"Quantiles range: [{quantiles.min():.4f}, {quantiles.max():.4f}]")
            print(f"Quantiles mean: {quantiles.mean():.4f}")
    
    return distribution, action, obs, info


def plot_distribution(distribution, action, args, output_file='return_distribution.png'):
    """Plot the return distribution."""
    plt.figure(figsize=(10, 6))
    
    if args.exp_name == "categorical" or args.exp_name == "categorical_mean_cvar":
        atoms, pmfs = distribution
        # Plot PMF
        plt.bar(atoms, pmfs, width=(atoms[1] - atoms[0]), alpha=0.7, label='Predicted PMF', color='blue', edgecolor='black')
        
        # Calculate and plot mean
        mean_val = np.sum(atoms * pmfs)
        plt.axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.2f}')
        
        plt.title(f'Return Distribution (Categorical) - {args.env_id}\nAction: {action}')
        plt.xlabel('Return')
        plt.ylabel('Probability')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save plot
        os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
        plt.savefig(output_file)
        print(f"Plot saved to {output_file}")
        plt.close()
        return plt.gcf()
    else:
        quantiles = distribution
        n_quantiles = len(quantiles)
        
        # Sort quantiles (should already be sorted, but just to be sure)
        quantiles_sorted = np.sort(quantiles)
        
        # Use midpoint quantile positions (tau_hat) as in QR-DQN
        # tau_hat = (2 * i + 1) / (2 * N) for i = 0, 1, ..., N-1
        tau_hat = (2 * np.arange(n_quantiles) + 1) / (2.0 * n_quantiles)
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot CDF using tau_hat
        ax1.step(quantiles_sorted, tau_hat, where='post', linewidth=2, color='blue', label='CDF')
        ax1.set_xlabel('Return Value', fontsize=12)
        ax1.set_ylabel('Cumulative Probability', fontsize=12)
        ax1.set_title(f'CDF of Return Distribution\n(Initial State, Action={action})', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Add statistics to CDF plot
        mean_return = np.mean(quantiles_sorted)
        median_return = np.median(quantiles_sorted)
        std_return = np.std(quantiles_sorted)
        
        # Calculate CVaR (conditional value at risk) at different risk levels
        cvar_95 = np.mean(quantiles_sorted[:int(0.05 * n_quantiles)])
        cvar_90 = np.mean(quantiles_sorted[:int(0.10 * n_quantiles)])
        
        stats_text = f'Mean: {mean_return:.2f}\nMedian: {median_return:.2f}\nStd: {std_return:.2f}\n'
        stats_text += f'CVaR(5%): {cvar_95:.2f}\nCVaR(10%): {cvar_90:.2f}'
        ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, 
                 fontsize=10, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Plot PDF (histogram approximation)
        # Use histogram to approximate PDF
        hist, bin_edges = np.histogram(quantiles_sorted, bins=min(50, n_quantiles // 2), density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_width = bin_edges[1] - bin_edges[0]
        
        ax2.bar(bin_centers, hist, width=bin_width, alpha=0.7, color='green', edgecolor='black', label='PDF (histogram)')
        
        # Also plot individual quantiles as vertical lines
        quantile_weights = np.ones(n_quantiles) / n_quantiles
        markerline, stemlines, baseline = ax2.stem(quantiles_sorted, quantile_weights, linefmt='red', markerfmt='ro', 
                                                    basefmt=' ', label='Quantile weights')
        # Set alpha on the returned objects instead of as a parameter
        plt.setp(markerline, alpha=0.3)
        plt.setp(stemlines, alpha=0.3)
        
        ax2.set_xlabel('Return Value', fontsize=12)
        ax2.set_ylabel('Probability Density', fontsize=12)
        ax2.set_title(f'PDF of Return Distribution\n(Initial State, Action={action})', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.legend()
        
        # Add algorithm info to the figure
        fig.suptitle(f'Algorithm: {args.exp_name} | Environment: {args.env_id} | Quantiles: {n_quantiles}', 
                     fontsize=12, y=1.02)
        
        plt.tight_layout()
        
        # Save figure
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"\n✓ Plot saved to: {output_file}")
        
        # Show plot
        plt.show()
        
        return fig


def plot_all_actions_distribution(model_path, seed=42, output_file='return_distribution_all_actions.png'):
    """Plot the return distribution for all actions from the initial state."""
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, env, args, get_action_kwargs = load_model(model_path, device=device)
    
    # Reset environment to get initial observation
    obs, info = env.reset(seed=seed)
    print(f"\nPlotting distributions for all actions...")
    print(f"Initial state info: {info}")
    print(f"Observation shape: {obs.shape}")
    print(f"Initial Observation: {obs}")
    
    # Convert observation to tensor
    obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    
    # Get number of actions
    n_actions = env.action_space.n
    action_names = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'STAY'][:n_actions]
    
    with torch.no_grad():
        # Get all quantiles for all actions
        quantiles_all = model.forward(obs_tensor)  # Shape: (1, n_actions, n_quantiles)
        quantiles_all = quantiles_all.squeeze(0).cpu().numpy()  # Shape: (n_actions, n_quantiles)
    
    n_quantiles = quantiles_all.shape[1]
    
    # Create subplots for each action
    fig, axes = plt.subplots(2, n_actions, figsize=(4 * n_actions, 8))
    if n_actions == 1:
        axes = axes.reshape(2, 1)
    
    cdf_probs = np.linspace(1.0 / (2 * n_quantiles), 1 - 1.0 / (2 * n_quantiles), n_quantiles)
    
    for action_idx in range(n_actions):
        quantiles = np.sort(quantiles_all[action_idx])
        
        # CDF plot
        ax_cdf = axes[0, action_idx]
        ax_cdf.step(quantiles, cdf_probs, where='post', linewidth=2, color='blue')
        ax_cdf.set_xlabel('Return Value', fontsize=10)
        ax_cdf.set_ylabel('Cumulative Probability', fontsize=10)
        ax_cdf.set_title(f'Action: {action_names[action_idx]}\nMean: {np.mean(quantiles):.2f}', 
                         fontsize=11, fontweight='bold')
        ax_cdf.grid(True, alpha=0.3)
        
        # PDF plot
        ax_pdf = axes[1, action_idx]
        hist, bin_edges = np.histogram(quantiles, bins=min(30, n_quantiles // 3), density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_width = bin_edges[1] - bin_edges[0]
        ax_pdf.bar(bin_centers, hist, width=bin_width, alpha=0.7, color='green', edgecolor='black')
        ax_pdf.set_xlabel('Return Value', fontsize=10)
        ax_pdf.set_ylabel('Probability Density', fontsize=10)
        ax_pdf.grid(True, alpha=0.3, axis='y')
    
    fig.suptitle(f'Return Distribution for All Actions\nAlgorithm: {args.exp_name} | Environment: {args.env_id}', 
                 fontsize=14, fontweight='bold', y=1.00)
    
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Plot saved to: {output_file}")
    
    plt.show()
    
    env.close()
    
    return fig


def evaluate_and_compare_distributions(model_path, eval_episodes=1000, epsilon=0.0, seed=42, 
                                       output_file=None):
    """
    Evaluate the model for N episodes and compare actual returns with quantile predictions.
    
    Args:
        model_path: Path to saved model
        eval_episodes: Number of episodes to evaluate
        epsilon: Epsilon for epsilon-greedy exploration (default 0.0 for greedy)
        seed: Random seed
        output_file: Output filename for the plot. If None, saves in the same directory as model_path
    
    Returns:
        actual_returns: Array of actual returns from evaluation
        predicted_quantiles: Predicted quantiles from initial state
        fig: Matplotlib figure
    """
    from cleanrl_utils.qrdqn_eval import evaluate
    
    # Determine output file path if not specified
    if output_file is None:
        model_dir = os.path.dirname(model_path)
        output_file = os.path.join(model_dir, 'return_distribution_comparison.png')
    
    print(f"\n{'='*60}")
    print(f"Evaluating model for {eval_episodes} episodes...")
    print(f"{'='*60}")
    
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, env, args, get_action_kwargs = load_model(model_path, device=device)
    
    # Get predicted quantiles from initial state
    obs, info = env.reset(seed=seed)
    obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    
    with torch.no_grad():
        if args.exp_name == "categorical" or args.exp_name == "categorical_mean_cvar":
            action, pmfs = model.get_action(obs_tensor, **get_action_kwargs)
            selected_action = action.item()
            pmfs = pmfs.squeeze(0).cpu().numpy()
            # Zero out probabilities below threshold and renormalize
            pmf_threshold = 1e-5
            pmfs[pmfs < pmf_threshold] = 0.0
            pmfs = pmfs / pmfs.sum()  # Renormalize to ensure sum = 1

            atoms = model.atoms.cpu().numpy()
            print(pmfs[pmfs > 0]*100.0)
            print(atoms[pmfs > 0])
            # Calculate mean and std from PMF
            mean_val = np.sum(atoms * pmfs)
            variance = np.sum(pmfs * (atoms - mean_val)**2)
            std_val = np.sqrt(variance)
            print(f"Initial state - Selected action: {selected_action}")
            print(f"Predicted Distribution: mean={mean_val:.4f}, std={std_val:.4f}")
            predicted_quantiles = None # Marker for categorical
        else:
            action, predicted_quantiles = model.get_action(obs_tensor, **get_action_kwargs)
            selected_action = action.item()
            predicted_quantiles = predicted_quantiles.squeeze(0).cpu().numpy()
            print(f"Initial state - Selected action: {selected_action}")
            print(f"Predicted quantiles: mean={predicted_quantiles.mean():.4f}, "
                  f"std={predicted_quantiles.std():.4f}, range=[{predicted_quantiles.min():.4f}, {predicted_quantiles.max():.4f}]")
    
    # Determine make_env and Model from args
    if args.exp_name == "qrdqn":
        from qrdqn import make_env, QNetwork as Model
    elif args.exp_name == "qrdqn_abs":
        from qrdqn_abs import make_env, QNetwork as Model
    elif args.exp_name == "qrdqn_mean_cvar":
        from qrdqn_mean_cvar import make_env, QNetwork as Model
    elif args.exp_name == "categorical":
        from categorical import make_env, QNetwork as Model
    elif args.exp_name == "categorical_mean_cvar":
        from categorical_mean_cvar import make_env, QNetwork as Model
    else:
        raise ValueError(f"Unsupported algorithm: {args.exp_name}")
    
    # Call the evaluate function from qrdqn_eval
    _, actual_returns = evaluate(
        model_path=model_path,
        make_env=make_env,
        env_id=args.env_id,
        eval_episodes=eval_episodes,
        run_name="",
        Model=Model,
        device=device,
        epsilon=epsilon,
    )
    
    actual_returns = np.array(actual_returns)
    
    if args.exp_name == "categorical" or args.exp_name == "categorical_mean_cvar":
        # For categorical, we have atoms and pmfs instead of quantiles
        # Convert PMF to quantiles for comparison by sampling or using CDF
        predicted_atoms = atoms
        predicted_pmfs = pmfs
        
        # Calculate statistics from PMF
        predicted_mean = np.sum(predicted_atoms * predicted_pmfs)
        predicted_variance = np.sum(predicted_pmfs * (predicted_atoms - predicted_mean)**2)
        predicted_std = np.sqrt(predicted_variance)
        
        # Convert PMF to CDF for comparison
        predicted_cdf = np.cumsum(predicted_pmfs)
        
        print(f"\n{'='*60}")
        print(f"Predicted Distribution Statistics (Categorical):")
        print(f"{'='*60}")
        print(f"  Mean: {predicted_mean:.4f}")
        print(f"  Std:  {predicted_std:.4f}")
        print(f"  Support: [{predicted_atoms.min():.4f}, {predicted_atoms.max():.4f}]")
        print(f"{'='*60}\n")
        
        # Plot comparison for categorical
        fig, axes = plt.subplots(2, 3, figsize=(20, 10))
        
        # Sort actual returns for CDF plotting
        actual_sorted = np.sort(actual_returns)
        actual_cdf_probs = np.arange(1, len(actual_sorted) + 1) / len(actual_sorted)

        
        # Plot 1: CDF Comparison
        ax1 = axes[0, 0]
        ax1.step(actual_sorted, actual_cdf_probs, where='post', linewidth=2, 
                 color='blue', label=f'Actual (N={eval_episodes})', alpha=0.8)
        ax1.step(predicted_atoms, predicted_cdf, where='post', linewidth=2, 
                 color='red', label=f'Predicted (C51)', alpha=0.8, linestyle='--')
        ax1.set_xlabel('Return Value', fontsize=12)
        ax1.set_ylabel('Cumulative Probability', fontsize=12)
        ax1.set_title('CDF: Actual vs Predicted', fontsize=13, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=9)
        
        # Plot 2: PDF Comparison
        ax2 = axes[0, 1]
        # Histogram for actual returns
        bins = np.linspace(min(actual_sorted.min(), predicted_atoms.min()),
                           max(actual_sorted.max(), predicted_atoms.max()), 50)
        ax2.hist(actual_sorted, bins=bins, density=True, alpha=0.6, color='blue', 
                 label=f'Actual (N={eval_episodes})', edgecolor='black')
        # Bar plot for predicted PMF
        bar_width = (predicted_atoms[1] - predicted_atoms[0]) * 0.8
        ax2.bar(predicted_atoms, predicted_pmfs / bar_width * 0.8, width=bar_width, 
                alpha=0.6, color='red', label='Predicted (C51)', edgecolor='black')
        ax2.set_xlabel('Return Value', fontsize=12)
        ax2.set_ylabel('Probability Density', fontsize=12)
        ax2.set_title('PDF: Actual vs Predicted', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.legend()
        
        # Plot 3: Quantile Function (Inverse CDF)
        ax3 = axes[0, 2]
        ax3.step(actual_cdf_probs, actual_sorted, where='post', linewidth=2,
                 color='blue', label=f'Actual (N={eval_episodes})', alpha=0.8)
        ax3.step(predicted_cdf, predicted_atoms, where='post', linewidth=2,
                 color='red', label='Predicted (C51)', alpha=0.8, linestyle='--')
        ax3.set_xlabel('Probability Level (τ)', fontsize=12)
        ax3.set_ylabel('Return Value (Quantile)', fontsize=12)
        ax3.set_title('Quantile Function: Actual vs Predicted', fontsize=13, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.legend(fontsize=9)
        
        # Plot 4: Q-Q Plot
        ax4 = axes[1, 0]
        # Interpolate to common quantile levels
        common_quantiles = np.linspace(0.01, 0.99, 50)
        actual_interp = np.interp(common_quantiles, actual_cdf_probs, actual_sorted)
        predicted_interp = np.interp(common_quantiles, predicted_cdf, predicted_atoms)
        
        ax4.scatter(predicted_interp, actual_interp, alpha=0.5, s=20)
        min_val = min(predicted_interp.min(), actual_interp.min())
        max_val = max(predicted_interp.max(), actual_interp.max())
        ax4.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect match')
        ax4.set_xlabel('Predicted Quantiles', fontsize=12)
        ax4.set_ylabel('Actual Quantiles', fontsize=12)
        ax4.set_title('Q-Q Plot', fontsize=13, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        ax4.axis('equal')
        
        # Plot 5: Error Analysis
        ax5 = axes[1, 1]
        stats_labels = ['Mean', 'Std', 'Min', 'Max']
        actual_stats = [actual_returns.mean(), actual_returns.std(), 
                       actual_returns.min(), actual_returns.max()]
        predicted_stats = [predicted_mean, predicted_std,
                          predicted_atoms.min(), predicted_atoms.max()]
        differences = [a - p for a, p in zip(actual_stats, predicted_stats)]
        
        colors_bars = ['green' if d >= 0 else 'red' for d in differences]
        bars = ax5.bar(stats_labels, differences, color=colors_bars, alpha=0.7, edgecolor='black')
        ax5.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax5.set_ylabel('Difference (Actual - Predicted)', fontsize=12)
        ax5.set_title('Statistical Differences', fontsize=13, fontweight='bold')
        ax5.grid(True, alpha=0.3, axis='y')
        
        for bar, diff in zip(bars, differences):
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height,
                    f'{diff:.2f}',
                    ha='center', va='bottom' if height >= 0 else 'top', fontsize=10)
        
        # Plot 6: Quantile-wise Error
        ax6 = axes[1, 2]
        common_taus = np.linspace(0.01, 0.99, 50)
        actual_quantiles_at_tau = np.interp(common_taus, actual_cdf_probs, actual_sorted)
        predicted_quantiles_at_tau = np.interp(common_taus, predicted_cdf, predicted_atoms)
        quantile_errors = actual_quantiles_at_tau - predicted_quantiles_at_tau
        
        ax6.plot(common_taus, quantile_errors, linewidth=2, color='purple')
        ax6.axhline(y=0, color='black', linestyle='--', linewidth=1)
        ax6.fill_between(common_taus, 0, quantile_errors, alpha=0.3, 
                         color='red' if quantile_errors.mean() < 0 else 'green')
        ax6.set_xlabel('Probability Level (τ)', fontsize=12)
        ax6.set_ylabel('Error (Actual - Predicted)', fontsize=12)
        ax6.set_title('Quantile-wise Prediction Error', fontsize=13, fontweight='bold')
        ax6.grid(True, alpha=0.3)
        
        # Overall title
        title_text = f'Return Distribution Comparison: Actual vs Predicted (Categorical/C51)\n'
        title_text += f'Algorithm: {args.exp_name} | Environment: {args.env_id} | Episodes: {eval_episodes}'
        fig.suptitle(title_text, fontsize=14, fontweight='bold', y=0.995)
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"✓ Comparison plot saved to: {output_file}")
        plt.show()
        
        env.close()
        return actual_returns, (predicted_atoms, predicted_pmfs), fig

    # The evaluate function already prints comprehensive statistics,
    # so we just need to print the predicted quantile statistics for comparison
    print(f"\n{'='*60}")
    print(f"Predicted Quantiles Statistics:")
    print(f"{'='*60}")
    print(f"  Mean: {predicted_quantiles.mean():.4f}")
    print(f"  Std:  {predicted_quantiles.std():.4f}")
    print(f"  Min:  {predicted_quantiles.min():.4f}")
    print(f"  Max:  {predicted_quantiles.max():.4f}")
    
    # Calculate CVaR and VaR metrics for predicted quantiles if applicable
    if args.exp_name == 'qrdqn_mean_cvar' or args.exp_name == 'categorical_mean_cvar':
        risk_level = getattr(args, 'risk_level')
        mean_weight = getattr(args, 'mean_weight')
        
        sorted_predicted = np.sort(predicted_quantiles)
        n_risk_predicted = max(1, int(np.ceil(risk_level * len(predicted_quantiles))))
        
        cvar_predicted = np.mean(sorted_predicted[:n_risk_predicted])
        mean_cvar_predicted = mean_weight * predicted_quantiles.mean() + (1 - mean_weight) * cvar_predicted
        
        # Calculate VaR for predicted quantiles (use tau_hat midpoint positions)
        predicted_tau_hat_for_var = (2 * np.arange(len(sorted_predicted)) + 1) / (2.0 * len(sorted_predicted))
        predicted_var = np.interp(risk_level, predicted_tau_hat_for_var, sorted_predicted)
        
        print(f"\nPredicted Risk Metrics (risk_level={risk_level:.2f}, mean_weight={mean_weight:.2f}):")
        print(f"  VaR (at {risk_level*100:.0f}%): {predicted_var:.4f}")
        print(f"  CVaR (worst {risk_level*100:.0f}%): {cvar_predicted:.4f}")
        print(f"  Mean-CVaR Objective: {mean_cvar_predicted:.4f}")
        
        # Calculate and print actual VaR and CVaR for comparison
        actual_var = np.quantile(actual_returns, risk_level)
        sorted_actual = np.sort(actual_returns)
        n_risk_actual = max(1, int(np.ceil(risk_level * len(actual_returns))))
        cvar_actual = np.mean(sorted_actual[:n_risk_actual])
        mean_cvar_actual = mean_weight * actual_returns.mean() + (1 - mean_weight) * cvar_actual
        
        print(f"\nActual Risk Metrics (risk_level={risk_level:.2f}, mean_weight={mean_weight:.2f}):")
        print(f"  VaR (at {risk_level*100:.0f}%): {actual_var:.4f}")
        print(f"  CVaR (worst {risk_level*100:.0f}%): {cvar_actual:.4f}")
        print(f"  Mean-CVaR Objective: {mean_cvar_actual:.4f}")
        
        print(f"\nVaR/CVaR Differences (Actual - Predicted):")
        print(f"  VaR Error: {actual_var - predicted_var:.4f}")
        print(f"  CVaR Error: {cvar_actual - cvar_predicted:.4f}")
        print(f"  Mean-CVaR Error: {mean_cvar_actual - mean_cvar_predicted:.4f}")
    
    # Print initial stock if available
    if hasattr(args, 'initial_stock') and args.initial_stock is not None:
        print(f"\nInitial Stock: {args.initial_stock}")
    
    print(f"{'='*60}\n")
    
    # Plot comparison
    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    
    # Sort for CDF plotting
    actual_sorted = np.sort(actual_returns)
    predicted_sorted = np.sort(predicted_quantiles)
    
    # CDF probabilities
    actual_cdf_probs = np.arange(1, len(actual_sorted) + 1) / len(actual_sorted)
    # Use tau_hat for predicted quantiles (midpoint positions)
    predicted_tau_hat = (2 * np.arange(len(predicted_sorted)) + 1) / (2.0 * len(predicted_sorted))
    
    # Plot 1: CDF Comparison with VaR markers
    ax1 = axes[0, 0]
    ax1.step(actual_sorted, actual_cdf_probs, where='post', linewidth=2, 
             color='blue', label=f'Actual (N={eval_episodes})', alpha=0.8)
    ax1.step(predicted_sorted, predicted_tau_hat, where='post', linewidth=2, 
             color='red', label=f'Predicted (N={len(predicted_sorted)})', alpha=0.8, linestyle='--')
    
    # Add VaR markers if applicable
    if args.exp_name == 'qrdqn_mean_cvar' or args.exp_name == 'categorical_mean_cvar':
        risk_level = getattr(args, 'risk_level')
        
        # Calculate VaR values
        actual_var = np.quantile(actual_returns, risk_level)
        predicted_var = np.interp(risk_level, predicted_tau_hat, predicted_sorted)
        
        # Draw vertical lines at VaR
        ax1.axvline(x=actual_var, color='blue', linestyle=':', linewidth=2, alpha=0.7,
                   label=f'Actual VaR({risk_level:.0%})={actual_var:.2f}')
        ax1.axvline(x=predicted_var, color='red', linestyle=':', linewidth=2, alpha=0.7,
                   label=f'Predicted VaR({risk_level:.0%})={predicted_var:.2f}')
        
        # Add initial stock line if available
        if hasattr(args, 'initial_stock') and args.initial_stock is not None:
            ax1.axvline(x=-args.initial_stock, color='green', linestyle='-.', linewidth=2, alpha=0.7,
                       label=f'Initial Stock={args.initial_stock:.2f}')
        
        # Draw horizontal line at risk level
        ax1.axhline(y=risk_level, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    
    ax1.set_xlabel('Return Value', fontsize=12)
    ax1.set_ylabel('Cumulative Probability', fontsize=12)
    ax1.set_title('CDF: Actual vs Predicted', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=9)
    
    # Plot 2: PDF Comparison
    ax2 = axes[0, 1]
    bins = np.linspace(min(actual_sorted.min(), predicted_sorted.min()),
                       max(actual_sorted.max(), predicted_sorted.max()), 50)
    ax2.hist(actual_sorted, bins=bins, density=True, alpha=0.6, color='blue', 
             label=f'Actual (N={eval_episodes})', edgecolor='black')
    ax2.hist(predicted_sorted, bins=bins, density=True, alpha=0.6, color='red', 
             label=f'Predicted (N={len(predicted_sorted)})', edgecolor='black')
    
    # Add VaR and initial stock lines if applicable
    if args.exp_name == 'qrdqn_mean_cvar' or args.exp_name == 'categorical_mean_cvar':
        risk_level = getattr(args, 'risk_level')
        actual_var = np.quantile(actual_returns, risk_level)
        predicted_var = np.interp(risk_level, predicted_tau_hat, predicted_sorted)
        
        ax2.axvline(x=actual_var, color='blue', linestyle=':', linewidth=2, alpha=0.5)
        ax2.axvline(x=predicted_var, color='red', linestyle=':', linewidth=2, alpha=0.5)
        
        if hasattr(args, 'initial_stock') and args.initial_stock is not None:
            ax2.axvline(x=-args.initial_stock, color='green', linestyle='-.', linewidth=2, alpha=0.5)
    
    ax2.set_xlabel('Return Value', fontsize=12)
    ax2.set_ylabel('Probability Density', fontsize=12)
    ax2.set_title('PDF: Actual vs Predicted', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.legend()
    
    # Plot 3: Quantile Function (Inverse CDF) with VaR lines
    ax3 = axes[0, 2]
    ax3.step(actual_cdf_probs, actual_sorted, where='post', linewidth=2,
             color='blue', label=f'Actual (N={eval_episodes})', alpha=0.8)
    ax3.step(predicted_tau_hat, predicted_sorted, where='post', linewidth=2,
             color='red', label=f'Predicted (N={len(predicted_sorted)})', alpha=0.8, linestyle='--')
    
    # Add VaR lines if applicable
    if args.exp_name == 'qrdqn_mean_cvar' or args.exp_name == 'categorical_mean_cvar':
        risk_level = getattr(args, 'risk_level')
        
        # Calculate VaR for actual returns (Value at Risk at risk_level)
        actual_var = np.quantile(actual_returns, risk_level)
        # Calculate VaR for predicted quantiles
        predicted_var = np.interp(risk_level, predicted_tau_hat, predicted_sorted)
        
        # Draw horizontal lines for VaR
        ax3.axhline(y=actual_var, color='blue', linestyle=':', linewidth=2, 
                   label=f'Actual VaR({risk_level:.0%})={actual_var:.2f}')
        ax3.axhline(y=predicted_var, color='red', linestyle=':', linewidth=2,
                   label=f'Predicted VaR({risk_level:.0%})={predicted_var:.2f}')
        
        # Add initial stock line if available
        if hasattr(args, 'initial_stock') and args.initial_stock is not None:
            ax3.axhline(y=-args.initial_stock, color='green', linestyle='-.', linewidth=2, alpha=0.7,
                       label=f'Initial Stock={args.initial_stock:.2f}')
        
        # Draw vertical line at risk level
        ax3.axvline(x=risk_level, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    
    ax3.set_xlabel('Probability Level (τ)', fontsize=12)
    ax3.set_ylabel('Return Value (Quantile)', fontsize=12)
    ax3.set_title('Quantile Function: Actual vs Predicted', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=9)
    
    # Plot 4: Q-Q Plot
    ax4 = axes[1, 0]
    # Interpolate to common quantile levels
    common_quantiles = np.linspace(0, 1, min(len(actual_sorted), len(predicted_sorted)))
    actual_interp = np.interp(common_quantiles, actual_cdf_probs, actual_sorted)
    predicted_interp = np.interp(common_quantiles, predicted_tau_hat, predicted_sorted)
    
    ax4.scatter(predicted_interp, actual_interp, alpha=0.5, s=20)
    # Add diagonal line for perfect match
    min_val = min(predicted_interp.min(), actual_interp.min())
    max_val = max(predicted_interp.max(), actual_interp.max())
    ax4.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect match')
    ax4.set_xlabel('Predicted Quantiles', fontsize=12)
    ax4.set_ylabel('Actual Quantiles', fontsize=12)
    ax4.set_title('Q-Q Plot', fontsize=13, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    ax4.axis('equal')
    
    # Plot 5: Error Analysis
    ax5 = axes[1, 1]
    # Calculate difference in statistics
    stats_labels = ['Mean', 'Std', 'Min', 'Max']
    actual_stats = [actual_returns.mean(), actual_returns.std(), 
                   actual_returns.min(), actual_returns.max()]
    predicted_stats = [predicted_quantiles.mean(), predicted_quantiles.std(),
                      predicted_quantiles.min(), predicted_quantiles.max()]
    differences = [a - p for a, p in zip(actual_stats, predicted_stats)]
    
    colors_bars = ['green' if d >= 0 else 'red' for d in differences]
    bars = ax5.bar(stats_labels, differences, color=colors_bars, alpha=0.7, edgecolor='black')
    ax5.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax5.set_ylabel('Difference (Actual - Predicted)', fontsize=12)
    ax5.set_title('Statistical Differences', fontsize=13, fontweight='bold')
    ax5.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, diff in zip(bars, differences):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height,
                f'{diff:.2f}',
                ha='center', va='bottom' if height >= 0 else 'top', fontsize=10)
    
    # Plot 6: Quantile-wise Error
    ax6 = axes[1, 2]
    # Interpolate predicted quantiles to match actual quantile positions
    common_taus = np.linspace(0.01, 0.99, 50)
    actual_quantiles_at_tau = np.interp(common_taus, actual_cdf_probs, actual_sorted)
    predicted_quantiles_at_tau = np.interp(common_taus, predicted_tau_hat, predicted_sorted)
    quantile_errors = actual_quantiles_at_tau - predicted_quantiles_at_tau
    
    ax6.plot(common_taus, quantile_errors, linewidth=2, color='purple')
    ax6.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax6.fill_between(common_taus, 0, quantile_errors, alpha=0.3, 
                     color='red' if quantile_errors.mean() < 0 else 'green')
    ax6.set_xlabel('Probability Level (τ)', fontsize=12)
    ax6.set_ylabel('Error (Actual - Predicted)', fontsize=12)
    ax6.set_title('Quantile-wise Prediction Error', fontsize=13, fontweight='bold')
    ax6.grid(True, alpha=0.3)
    
    # Add shaded regions for different risk levels
    if args.exp_name == 'qrdqn_mean_cvar' or args.exp_name == 'categorical_mean_cvar':
        risk_level = getattr(args, 'risk_level')
        ax6.axvspan(0, risk_level, alpha=0.1, color='red', label=f'CVaR region (α={risk_level:.2f})')
        ax6.legend()
    
    # Overall title with initial stock if available
    title_text = f'Return Distribution Comparison: Actual vs Predicted\n'
    title_text += f'Algorithm: {args.exp_name} | Environment: {args.env_id} | Episodes: {eval_episodes}'
    
    # Add initial stock information if available
    if hasattr(args, 'initial_stock') and args.initial_stock is not None:
        title_text += f' | Initial Stock: {args.initial_stock}'
    
    fig.suptitle(title_text, fontsize=14, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Comparison plot saved to: {output_file}")
    
    plt.show()
    
    env.close()
    
    return actual_returns, predicted_quantiles, fig


def main():
    parser = argparse.ArgumentParser(description="Plot return distribution from trained QR-DQN model")
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the trained model (.cleanrl_model file)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="return_distribution.png",
        help="Output plot filename"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for environment reset"
    )
    parser.add_argument(
        "--all-actions",
        action="store_true",
        help="Plot distributions for all actions"
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Evaluate model and compare actual returns with predicted quantiles"
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=1000,
        help="Number of episodes for evaluation (default: 1000)"
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.0,
        help="Epsilon for epsilon-greedy evaluation (default: 0.0 for greedy)"
    )
    
    args = parser.parse_args()
    
    # Check if model file exists
    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"Error: Model file not found: {model_path}")
        return
    
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, env, model_args, get_action_kwargs = load_model(str(model_path), device=device)
    
    if args.evaluate:
        # Evaluate and compare distributions - save in model directory
        evaluate_and_compare_distributions(
            model_path=str(model_path),
            eval_episodes=args.eval_episodes,
            epsilon=args.epsilon,
            seed=args.seed,
            output_file=None  # Will be saved in model directory
        )
    elif args.all_actions:
        # Plot distributions for all actions
        output_file = args.output.replace('.png', '_all_actions.png')
        plot_all_actions_distribution(
            model_path=str(model_path),
            seed=args.seed,
            output_file=output_file
        )
    else:
        # Get distribution for the selected action
        distribution, action, obs, info = get_initial_state_distribution(
            model, env, model_args, get_action_kwargs, device=device, seed=args.seed
        )
        
        # Plot the distribution
        plot_distribution(distribution, action, model_args, output_file=args.output)
    
    env.close()
    print("\nDone!")


if __name__ == "__main__":
    main()
