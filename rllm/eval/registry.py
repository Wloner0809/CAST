"""Environment configuration registry for unified evaluation.

This module provides a registry pattern for managing environment configurations,
allowing easy registration and retrieval of environment-specific settings.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Type


@dataclass
class EnvConfig:
    """Configuration for an environment.

    Attributes:
        env_class: The environment class to instantiate.
        agent_class: The agent class to use with this environment.
        env_args: Default arguments for environment initialization.
        agent_args: Default arguments for agent initialization.
        data_loader: Function to load data for this environment.
        metrics: List of metrics to compute for this environment.
        requires_multithread_safe: Whether the environment requires thread-safe execution.
        default_max_steps: Default maximum steps per episode.
        description: Human-readable description of the environment.
        success_threshold: Reward threshold for success determination.
            A trajectory is successful if reward > success_threshold.
            Default 0.0 (any positive reward). For Sokoban, use 5.0.
    """

    env_class: Type
    agent_class: Type
    env_args: dict[str, Any] = field(default_factory=dict)
    agent_args: dict[str, Any] = field(default_factory=dict)
    data_loader: Callable | None = None
    metrics: list[str] = field(default_factory=lambda: ["pass_at_k", "success_rate"])
    requires_multithread_safe: bool = True
    default_max_steps: int = 50
    description: str = ""
    success_threshold: float = 0.0


class EnvConfigRegistry:
    """Registry for environment configurations.

    This class provides a centralized registry for managing environment
    configurations, allowing easy registration and retrieval of
    environment-specific settings.

    Example:
        >>> from rllm.eval.registry import EnvConfigRegistry, EnvConfig
        >>> EnvConfigRegistry.register("my_env", EnvConfig(
        ...     env_class=MyEnv,
        ...     agent_class=MyAgent,
        ...     data_loader=load_my_data,
        ... ))
        >>> config = EnvConfigRegistry.get("my_env")
    """

    _configs: dict[str, EnvConfig] = {}

    @classmethod
    def register(cls, name: str, config: EnvConfig) -> None:
        """Register an environment configuration.

        Args:
            name: Unique identifier for the environment.
            config: EnvConfig instance with environment settings.
        """
        cls._configs[name] = config

    @classmethod
    def get(cls, name: str) -> EnvConfig:
        """Get an environment configuration.

        Args:
            name: Environment identifier.

        Returns:
            EnvConfig instance for the specified environment.

        Raises:
            ValueError: If the environment is not registered.
        """
        if name not in cls._configs:
            available = ", ".join(cls._configs.keys()) if cls._configs else "none"
            raise ValueError(
                f"Environment '{name}' not registered. Available environments: {available}"
            )
        return cls._configs[name]

    @classmethod
    def list_envs(cls) -> list[str]:
        """List all registered environments.

        Returns:
            List of registered environment names.
        """
        return list(cls._configs.keys())

    @classmethod
    def is_registered(cls, name: str) -> bool:
        """Check if an environment is registered.

        Args:
            name: Environment identifier.

        Returns:
            True if the environment is registered, False otherwise.
        """
        return name in cls._configs

    @classmethod
    def unregister(cls, name: str) -> None:
        """Unregister an environment configuration.

        Args:
            name: Environment identifier to remove.
        """
        if name in cls._configs:
            del cls._configs[name]

    @classmethod
    def clear(cls) -> None:
        """Clear all registered configurations."""
        cls._configs.clear()

    @classmethod
    def get_all(cls) -> dict[str, EnvConfig]:
        """Get all registered configurations.

        Returns:
            Dictionary mapping environment names to their configurations.
        """
        return cls._configs.copy()
