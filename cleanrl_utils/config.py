"""Config loader for environment-specific hyperparameter defaults."""

import inspect
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Optional, TypeVar

import yaml

# Type variable for generic dataclass support
T = TypeVar('T')


def _get_caller_script_name() -> Optional[str]:
    """
    Detect the name of the calling script (e.g., 'qrdqn' from 'qrdqn.py').
    
    Returns:
        Script name without .py extension, or None if not detectable
    """
    # First, check sys.argv[0] (most reliable when set correctly by tuner)
    if sys.argv and sys.argv[0].endswith('.py'):
        return Path(sys.argv[0]).stem
    
    # Fallback: Walk up the call stack to find the main script
    for frame_info in inspect.stack():
        frame_filename = frame_info.filename
        if frame_filename.endswith('.py') and '__main__' in (frame_info.code_context[0] if frame_info.code_context else ''):
            script_path = Path(frame_filename)
            return script_path.stem  # filename without extension
    
    return None


def load_env_config(env_id: str, config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load environment-specific defaults from YAML config.
    
    Automatically detects which config file to use based on the calling script:
    - qrdqn.py -> configs/qrdqn.yaml
    - qrdqn_mean_cvar.py -> configs/qrdqn_mean_cvar.yaml
    - etc.
    
    Args:
        env_id: The environment ID (e.g., "AmericanOptionEnv-v1")
        config_path: Optional explicit path to config file. If None, auto-detects.
    
    Returns:
        Dictionary of hyperparameter defaults for the environment.
        Falls back to "defaults" key if env_id not found.
    
    Example:
        >>> config = load_env_config("AmericanOptionEnv-v1")
        >>> print(config["learning_rate"])  # 5e-4
    """
    # Resolve path relative to repo root
    repo_root = Path(__file__).parent.parent
    
    # Auto-detect config file if not provided
    if config_path is None:
        script_name = _get_caller_script_name()
        if script_name:
            config_path = f"configs/{script_name}.yaml"
            print(f"ℹ️  Auto-detected config file: {config_path}")
        else:
            # Fallback to generic config
            config_path = "configs/envs.yaml"
            print(f"⚠️  Could not auto-detect script. Using fallback: {config_path}")
    
    config_file = repo_root / config_path
    
    if not config_file.exists():
        print(f"⚠️  Config file not found: {config_file}. Using built-in defaults.")
        return {}
    
    try:
        with open(config_file, "r") as f:
            all_configs = yaml.safe_load(f)
    except Exception as e:
        print(f"⚠️  Error reading config file: {e}. Using built-in defaults.")
        return {}
    
    # Return environment-specific config, or fall back to "defaults"
    if env_id in all_configs:
        print(f"✓ Loaded config for '{env_id}' from {config_file.name}")
        return all_configs[env_id]
    elif "defaults" in all_configs:
        print(f"ℹ️  No config for '{env_id}', using defaults from {config_file.name}.")
        return all_configs["defaults"]
    else:
        print(f"⚠️  No 'defaults' found in config. Proceeding with Args defaults.")
        return {}


def _normalize_arg_name(arg: str) -> str:
    """Normalize argument name: strip --, handle --no- prefix, replace - with _."""
    arg = arg.split('=')[0].lstrip('-')  # Remove -- and anything after =
    arg = arg.replace('-', '_')  # Normalize hyphens to underscores
    return arg


def _get_cli_provided_args() -> set:
    """
    Detect which arguments were explicitly provided on the command line.
    
    Handles:
    - --arg-name value and --arg_name value (both normalized to arg_name)
    - --arg=value format
    - --no-arg-name for boolean flags (maps to arg_name)
    """
    cli_provided_args = set()
    for arg in sys.argv[1:]:
        if arg.startswith('--'):
            normalized = _normalize_arg_name(arg)
            # Handle --no-xxx boolean flags: --no-foo maps to field "foo"
            if normalized.startswith('no_'):
                # Add both the no_ version and the base field name
                base_name = normalized[3:]  # Remove 'no_' prefix
                cli_provided_args.add(base_name)
                cli_provided_args.add(normalized)  # Keep no_ version too for edge cases
            else:
                cli_provided_args.add(normalized)
    return cli_provided_args


def _convert_type(value: Any, expected_type: type, field_name: str) -> Any:
    """Convert a value to the expected type with proper error handling."""
    try:
        # Handle Optional types and other generic types
        origin = getattr(expected_type, '__origin__', None)
        if origin is not None:
            # For Optional[X], Union[X, None], etc., try to get the actual type
            args = getattr(expected_type, '__args__', ())
            if args:
                # Use the first non-None type
                for arg in args:
                    if arg is not type(None):
                        expected_type = arg
                        break
        
        if expected_type == bool:
            # Handle various bool representations from YAML
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                return value.lower() in ('true', '1', 'yes', 'on')
            return bool(value)
        elif expected_type == int:
            return int(value)
        elif expected_type == float:
            return float(value)
        elif expected_type == str:
            return str(value)
        else:
            return value
    except (ValueError, TypeError) as e:
        print(f"⚠️  Warning: Could not convert {field_name}={value} to {expected_type}: {e}")
        return None  # Signal conversion failure


def parse_args_with_config(args_class: type[T]) -> T:
    """
    Parse CLI arguments and merge with environment-specific config from YAML.
    
    Priority: CLI > YAML config > dataclass defaults
    
    This function:
    1. Parses CLI arguments using tyro
    2. Loads environment-specific config based on --env-id
    3. Merges them with CLI taking precedence over YAML config
    4. Ensures proper type conversion from YAML values
    
    Handles:
    - Both --arg-name and --arg_name (normalized to same field)
    - Boolean --no-xxx flags to set xxx=False
    - Proper type conversion from YAML strings
    - --no-config flag to skip config file loading entirely
    
    Args:
        args_class: The dataclass type (e.g., Args)
    
    Returns:
        Instance of args_class with merged configuration
        
    Example:
        >>> from dataclasses import dataclass
        >>> @dataclass
        ... class Args:
        ...     env_id: str = "CartPole-v1"
        ...     learning_rate: float = 1e-3
        ...     save_model: bool = True
        >>> args = parse_args_with_config(Args)
        >>> # CLI: --no-save-model -> save_model=False (CLI wins)
        >>> # YAML: learning_rate: 5e-4 -> learning_rate=5e-4 (if not in CLI)
        >>> # CLI: --no-config -> skip config file, use only defaults + CLI
    """
    import tyro
    from dataclasses import fields
    
    # Check if user wants to skip config loading
    skip_config = '--no-config' in sys.argv
    if skip_config:
        # Remove --no-config from argv so tyro doesn't see it
        sys.argv = [arg for arg in sys.argv if arg != '--no-config']
        print("ℹ️  Skipping config file (--no-config flag detected)")
    
    # Detect which arguments were explicitly passed on the command line
    cli_provided_args = _get_cli_provided_args()
    
    # First pass: parse CLI to get all arguments (tyro handles all the parsing)
    cli_args = tyro.cli(args_class)
    
    # Skip config loading if requested
    if skip_config:
        return cli_args
    
    # Load env-specific config based on parsed env_id
    env_config = load_env_config(cli_args.env_id)
    
    if not env_config:
        # No config found, just return parsed args
        return cli_args
    
    # Get field types for proper type conversion
    field_types = {f.name: f.type for f in fields(args_class)}
    
    # Get default values from dataclass
    default_args = args_class()
    
    # Build merged dict: start with defaults, apply config, then CLI overrides
    merged_dict = {}
    
    for field_name in vars(default_args):
        cli_value = getattr(cli_args, field_name)
        default_value = getattr(default_args, field_name)
        
        # Check if this argument was explicitly provided on CLI
        was_cli_provided = field_name in cli_provided_args
        
        # Priority: CLI > config > default
        if was_cli_provided:
            # CLI takes highest priority - use tyro's parsed value
            merged_dict[field_name] = cli_value
        elif field_name in env_config:
            # Config takes second priority - convert type from YAML
            config_value = env_config[field_name]
            if field_name in field_types:
                converted = _convert_type(config_value, field_types[field_name], field_name)
                if converted is not None:
                    merged_dict[field_name] = converted
                else:
                    # Conversion failed, fall back to default
                    merged_dict[field_name] = default_value
            else:
                merged_dict[field_name] = config_value
        else:
            # Use dataclass default
            merged_dict[field_name] = default_value
    
    # Create final args with merged values
    final_args = replace(cli_args, **merged_dict)
    
    return final_args


def merge_configs(cli_args: Dict[str, Any], env_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge CLI arguments with environment config, with CLI taking precedence.
    
    Args:
        cli_args: Arguments from tyro.cli() (may be incomplete dict or have sys.argv)
        env_config: Environment-specific defaults from YAML
    
    Returns:
        Merged configuration where CLI args override env_config
    
    Example:
        >>> env_config = {"learning_rate": 5e-4, "batch_size": 64}
        >>> cli_args = {"learning_rate": 1e-3}  # user override
        >>> merged = merge_configs(cli_args, env_config)
        >>> merged["learning_rate"]  # 1e-3 (CLI wins)
        >>> merged["batch_size"]  # 64 (from env_config)
    """
    # Start with env config as base
    merged = env_config.copy()
    
    # Override with any CLI args that were explicitly provided
    if cli_args:
        for key, value in cli_args.items():
            if value is not None:  # Only override if user provided a value
                merged[key] = value
    
    return merged


def apply_config_to_dataclass(dataclass_type: type, env_id: str, cli_args: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Convenience function: load env config and merge with CLI args.
    
    Args:
        dataclass_type: The Args dataclass type (for reference only, unused)
        env_id: The environment ID
        cli_args: CLI arguments from tyro.cli()
    
    Returns:
        Dictionary of all parameters ready to instantiate the dataclass
    """
    env_config = load_env_config(env_id)
    merged = merge_configs(cli_args or {}, env_config)
    return merged


__all__ = [
    "load_env_config",
    "parse_args_with_config",
    "merge_configs",
    "apply_config_to_dataclass",
]
