# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Result structure for beam search algorithm.

This module provides dataclasses for capturing beam search execution
results and statistics.
"""

from dataclasses import dataclass, field
from typing import Generic, List, Optional, TypeVar

from pyrit.algorithms.beam_search.beam_node import BeamNode

StateT = TypeVar("StateT")
ActionT = TypeVar("ActionT")


@dataclass
class BeamSearchStatistics:
    """
    Statistics collected during beam search execution.

    Tracks various metrics about the search process for analysis
    and debugging purposes.
    """

    # Total number of nodes instantiated during search.
    total_nodes_created: int = 0

    # Number of nodes that had their children generated.
    total_nodes_expanded: int = 0

    # Number of nodes removed during beam narrowing.
    total_nodes_pruned: int = 0

    # Deepest level explored in the search tree.
    max_depth_reached: int = 0

    # Beam width at each iteration.
    beam_width_history: List[int] = field(default_factory=list)

    # Best score at each iteration.
    score_history: List[float] = field(default_factory=list)

    # Number of iterations actually executed.
    iterations_completed: int = 0

    # Whether search terminated before max_depth.
    early_termination: bool = False

    # Human-readable reason for search termination.
    termination_reason: str = ""

    # Number of times beam width was expanded.
    expansion_events: int = 0

    # Number of times beam width was contracted.
    contraction_events: int = 0


@dataclass
class BeamSearchResult(Generic[StateT, ActionT]):
    """
    Result of a beam search execution.

    Contains the best node found, success status, and execution statistics.

    Type Parameters:
        StateT: The type of state in the search space.
        ActionT: The type of action that transitions between states.
    """

    # The highest-scoring node found during search.
    best_node: Optional[BeamNode[StateT, ActionT]] = None

    # Whether the search found a successful terminal state.
    success: bool = False

    # All leaf nodes at search termination (for analysis).
    final_beam: List[BeamNode[StateT, ActionT]] = field(default_factory=list)

    # Execution statistics and metrics.
    statistics: BeamSearchStatistics = field(default_factory=BeamSearchStatistics)

    @property
    def best_score(self) -> float:
        """
        Get the score of the best node.

        Returns:
            float: The best node's score, or 0.0 if no best node exists.
        """
        return self.best_node.score if self.best_node else 0.0

    @property
    def best_action_sequence(self) -> List[ActionT]:
        """
        Get the action sequence leading to the best node.

        Returns:
            List[ActionT]: Ordered list of actions from root to best node.
        """
        return self.best_node.get_action_sequence() if self.best_node else []

    @property
    def best_state(self) -> Optional[StateT]:
        """
        Get the state of the best node.

        Returns:
            Optional[StateT]: The best node's state, or None if no best node.
        """
        return self.best_node.state if self.best_node else None

    @property
    def depth_reached(self) -> int:
        """
        Get the depth of the best node.

        Returns:
            int: The best node's depth, or 0 if no best node exists.
        """
        return self.best_node.depth if self.best_node else 0

    def get_all_terminal_nodes(self) -> List[BeamNode[StateT, ActionT]]:
        """
        Get all nodes that reached terminal state.

        Returns:
            List[BeamNode[StateT, ActionT]]: All terminal nodes in final beam.
        """
        return [node for node in self.final_beam if node.is_terminal]

    def get_top_k_nodes(self, k: int) -> List[BeamNode[StateT, ActionT]]:
        """
        Get the top k nodes by score from the final beam.

        Args:
            k (int): Number of nodes to return.

        Returns:
            List[BeamNode[StateT, ActionT]]: Top k nodes sorted by score descending.
        """
        sorted_beam = sorted(self.final_beam, key=lambda n: n.score, reverse=True)
        return sorted_beam[:k]

    def __str__(self) -> str:
        """
        Return string representation of the result.

        Returns:
            str: Summary of the search result.
        """
        return (
            f"BeamSearchResult(success={self.success}, "
            f"best_score={self.best_score:.3f}, "
            f"depth={self.depth_reached}, "
            f"iterations={self.statistics.iterations_completed})"
        )

    __repr__ = __str__
