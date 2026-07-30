"""Data structures for world modeling output from exploration.

This module provides data structures for storing and serializing
the world modeling output from the ExploreAgent.
"""

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class WorldModeling:
    """Data structure for world modeling output.

    Attributes:
        content: Free-form text content from <world_modeling> tags.
        exploration_trajectory: Steps taken during exploration.
        env_name: Name of the environment explored.
        total_steps: Total number of exploration steps taken.
        metadata: Additional metadata about the exploration.
    """

    content: str
    exploration_trajectory: list[dict] = field(default_factory=list)
    env_name: str = ""
    total_steps: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation of the world modeling.
        """
        return {
            "content": self.content,
            "exploration_trajectory": self.exploration_trajectory,
            "env_name": self.env_name,
            "total_steps": self.total_steps,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WorldModeling":
        """Create WorldModeling from dictionary.

        Args:
            data: Dictionary with world modeling data.

        Returns:
            WorldModeling instance.
        """
        return cls(
            content=data.get("content", ""),
            exploration_trajectory=data.get("exploration_trajectory", []),
            env_name=data.get("env_name", ""),
            total_steps=data.get("total_steps", 0),
            metadata=data.get("metadata", {}),
        )

    @classmethod
    def from_agent_response(
        cls,
        response: str,
        exploration_trajectory: list[dict] | None = None,
        env_name: str = "",
    ) -> "WorldModeling":
        """Extract WorldModeling from agent response.

        Args:
            response: Agent response text containing <world_modeling> tags.
            exploration_trajectory: Optional exploration trajectory.
            env_name: Name of the environment.

        Returns:
            WorldModeling instance with extracted content.
        """
        # Extract content between <world_modeling></world_modeling> tags
        pattern = r"<world_modeling>(.*?)</world_modeling>"
        match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)

        content = match.group(1).strip() if match else ""
        trajectory = exploration_trajectory or []

        return cls(
            content=content,
            exploration_trajectory=trajectory,
            env_name=env_name,
            total_steps=len(trajectory),
        )

    def is_valid(self) -> bool:
        """Check if the world modeling has valid content.

        Returns:
            True if content is non-empty.
        """
        return bool(self.content.strip())

    def get_summary(self) -> str:
        """Get a brief summary of the world modeling.

        Returns:
            First 200 characters of content or full content if shorter.
        """
        if not self.content:
            return "(No world modeling content)"
        if len(self.content) <= 200:
            return self.content
        return self.content[:200] + "..."

    def __str__(self) -> str:
        """String representation."""
        return (
            f"WorldModeling(env={self.env_name}, "
            f"steps={self.total_steps}, "
            f"content_length={len(self.content)})"
        )


@dataclass
class ExplorationResult:
    """Result of an exploration episode.

    Attributes:
        task_id: Identifier for the exploration task.
        world_modeling: The world modeling output.
        trajectory: Full trajectory from the agent.
        success: Whether exploration completed successfully.
        error: Error message if exploration failed.
    """

    task_id: str
    world_modeling: WorldModeling | None
    trajectory: Any = None  # Trajectory object
    success: bool = True
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation.
        """
        return {
            "task_id": self.task_id,
            "world_modeling": self.world_modeling.to_dict() if self.world_modeling else None,
            "trajectory": self.trajectory.to_dict() if self.trajectory else None,
            "success": self.success,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExplorationResult":
        """Create ExplorationResult from dictionary.

        Args:
            data: Dictionary with exploration result data.

        Returns:
            ExplorationResult instance.
        """
        from rllm.agents.agent import Trajectory

        world_modeling = None
        if data.get("world_modeling"):
            world_modeling = WorldModeling.from_dict(data["world_modeling"])

        trajectory = None
        if data.get("trajectory"):
            trajectory = Trajectory.from_dict(data["trajectory"])

        return cls(
            task_id=data.get("task_id", ""),
            world_modeling=world_modeling,
            trajectory=trajectory,
            success=data.get("success", True),
            error=data.get("error"),
        )
