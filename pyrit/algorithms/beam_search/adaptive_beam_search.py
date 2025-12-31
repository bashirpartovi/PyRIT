# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Adaptive Beam Search Algorithm.

This module provides a generic, reusable beam search implementation with
adaptive beam width that can be applied to various search problems.

The algorithm is agnostic to the domain - it operates on generic states
and actions, with domain-specific behavior injected via callback functions.
"""

from __future__ import annotations

import asyncio
import logging
from typing import (
    Awaitable,
    Callable,
    Generic,
    List,
    Optional,
    Tuple,
    TypeVar,
)

from pyrit.algorithms.beam_search.beam_node import BeamNode
from pyrit.algorithms.beam_search.beam_search_config import (
    BeamContractionStrategy,
    BeamExpansionStrategy,
    BeamSearchConfig,
)
from pyrit.algorithms.beam_search.beam_search_result import (
    BeamSearchResult,
    BeamSearchStatistics,
)

logger = logging.getLogger(__name__)

StateT = TypeVar("StateT")
ActionT = TypeVar("ActionT")

# Type aliases for callback functions
ExpandFunc = Callable[[StateT], Awaitable[List[Tuple[ActionT, StateT]]]]
ScoreFunc = Callable[[StateT], Awaitable[float]]
TerminalFunc = Callable[[StateT, float], bool]
IterationCallback = Callable[[int, List[BeamNode[StateT, ActionT]]], Awaitable[None]]


class AdaptiveBeamSearch(Generic[StateT, ActionT]):
    """
    Generic adaptive beam search algorithm.

    This algorithm explores a state space using beam search with dynamically
    adjusting beam width based on search progress and resource constraints.

    The algorithm is agnostic to the domain - it operates on generic states
    and actions, with domain-specific behavior injected via callback functions.

    Type Parameters:
        StateT: The type representing a state in the search space.
        ActionT: The type representing an action/transition between states.
    """

    def __init__(self, *, config: Optional[BeamSearchConfig] = None) -> None:
        """
        Initialize the adaptive beam search algorithm.

        Args:
            config (Optional[BeamSearchConfig]): Configuration for the beam search.
                Defaults to BeamSearchConfig() with default values.
        """
        self._config = config or BeamSearchConfig()
        self._current_beam_width = self._config.initial_beam_width
        self._plateau_counter = 0
        self._best_score_seen = float("-inf")
        self._previous_best_score = float("-inf")

    async def search_async(
        self,
        *,
        initial_state: StateT,
        expand_fn: ExpandFunc[StateT, ActionT],
        score_fn: ScoreFunc[StateT],
        is_terminal_fn: Optional[TerminalFunc[StateT]] = None,
        on_iteration_complete: Optional[IterationCallback[StateT, ActionT]] = None,
    ) -> BeamSearchResult[StateT, ActionT]:
        """
        Execute adaptive beam search from an initial state.

        Args:
            initial_state (StateT): The starting state for the search.
            expand_fn (ExpandFunc): Async function that generates (action, new_state)
                pairs from a state.
            score_fn (ScoreFunc): Async function that scores a state (higher = better).
            is_terminal_fn (Optional[TerminalFunc]): Optional function to check if a state
                is terminal. Signature: (state, score) -> bool. Defaults to score-threshold check.
            on_iteration_complete (Optional[IterationCallback]): Optional callback invoked
                after each iteration with (iteration_number, current_beam).

        Returns:
            BeamSearchResult[StateT, ActionT]: The search result containing best node
                and statistics.
        """
        # Reset internal state for new search
        self._current_beam_width = self._config.initial_beam_width
        self._plateau_counter = 0
        self._best_score_seen = float("-inf")
        self._previous_best_score = float("-inf")

        # Initialize statistics
        stats = BeamSearchStatistics()

        # Default terminal function uses score threshold
        if is_terminal_fn is None:
            is_terminal_fn = self._default_terminal_fn

        # Create and score root node
        root_score = await score_fn(initial_state)
        root: BeamNode[StateT, ActionT] = BeamNode(
            state=initial_state,
            score=root_score,
            depth=0,
        )
        stats.total_nodes_created = 1
        self._best_score_seen = root_score

        logger.debug(f"Root node created with score {root_score:.3f}")

        # Check if initial state is already terminal
        if self._check_terminal_condition(
            node=root,
            is_terminal_fn=is_terminal_fn,
            stats=stats,
        ):
            stats.early_termination = True
            stats.termination_reason = "Initial state is terminal"
            return BeamSearchResult(
                best_node=root,
                success=True,
                final_beam=[root],
                statistics=stats,
            )

        # Initialize beam with root
        current_beam: List[BeamNode[StateT, ActionT]] = [root]
        best_node = root

        # Main search loop
        for iteration in range(self._config.max_depth):
            stats.iterations_completed = iteration + 1
            stats.beam_width_history.append(len(current_beam))
            stats.score_history.append(best_node.score)
            stats.max_depth_reached = max(stats.max_depth_reached, iteration + 1)

            logger.debug(
                f"Iteration {iteration + 1}/{self._config.max_depth}, "
                f"beam_width={len(current_beam)}, "
                f"best_score={best_node.score:.3f}"
            )

            # Expand all nodes in current beam
            all_children: List[BeamNode[StateT, ActionT]] = []

            for node in current_beam:
                if node.is_terminal:
                    # Terminal nodes don't expand but stay in consideration
                    all_children.append(node)
                    continue

                children = await self._expand_node_async(
                    node=node,
                    expand_fn=expand_fn,
                    score_fn=score_fn,
                    stats=stats,
                )
                all_children.extend(children)

            # Handle case where no children were generated
            if not all_children:
                logger.debug("No children generated, terminating search")
                stats.early_termination = True
                stats.termination_reason = "No valid expansions available"
                break

            # Check for terminal states among children
            terminal_node = self._find_best_terminal_node(
                nodes=all_children,
                is_terminal_fn=is_terminal_fn,
                stats=stats,
                min_depth=self._config.min_depth_before_termination,
            )

            if terminal_node is not None:
                stats.early_termination = True
                stats.termination_reason = f"Terminal state found with score {terminal_node.score:.3f}"
                return BeamSearchResult(
                    best_node=terminal_node,
                    success=True,
                    final_beam=all_children,
                    statistics=stats,
                )

            # Update beam width adaptively
            self._update_beam_width(stats=stats)

            # Prune to maintain beam width
            current_beam = self._prune_beam(
                nodes=all_children,
                stats=stats,
            )

            # Update best node
            if current_beam:
                iteration_best = max(current_beam, key=lambda n: n.score)
                if iteration_best.score > best_node.score:
                    best_node = iteration_best
                    self._plateau_counter = 0
                else:
                    self._plateau_counter += 1

                self._previous_best_score = self._best_score_seen
                self._best_score_seen = max(self._best_score_seen, iteration_best.score)

            # Invoke iteration callback if provided
            if on_iteration_complete is not None:
                await on_iteration_complete(iteration + 1, current_beam)

            # Check if beam is empty (shouldn't happen after pruning, but safety check)
            if not current_beam:
                stats.early_termination = True
                stats.termination_reason = "Beam became empty"
                break

        # Search completed without finding terminal state
        if not stats.early_termination:
            stats.termination_reason = f"Reached maximum depth ({self._config.max_depth})"

        return BeamSearchResult(
            best_node=best_node,
            success=False,
            final_beam=current_beam,
            statistics=stats,
        )

    def _default_terminal_fn(self, state: StateT, score: float) -> bool:
        """
        Check if score meets or exceeds the success threshold.

        Args:
            state (StateT): The state to check (unused in default implementation).
            score (float): The score of the state.

        Returns:
            bool: True if score meets or exceeds success threshold.
        """
        return score >= self._config.success_threshold

    async def _expand_node_async(
        self,
        *,
        node: BeamNode[StateT, ActionT],
        expand_fn: ExpandFunc[StateT, ActionT],
        score_fn: ScoreFunc[StateT],
        stats: BeamSearchStatistics,
    ) -> List[BeamNode[StateT, ActionT]]:
        """
        Expand a node by generating and scoring its children.

        Args:
            node (BeamNode): The node to expand.
            expand_fn (ExpandFunc): Function to generate successor states.
            score_fn (ScoreFunc): Function to score states.
            stats (BeamSearchStatistics): Statistics to update.

        Returns:
            List[BeamNode]: List of child nodes created.
        """
        try:
            expansions = await expand_fn(node.state)
            node.is_expanded = True
            stats.total_nodes_expanded += 1
        except Exception as e:
            logger.warning(f"Expansion failed for node {node.node_id}: {e}")
            return []

        if not expansions:
            return []

        children: List[BeamNode[StateT, ActionT]] = []

        # Score all children concurrently
        score_tasks = [score_fn(state) for _, state in expansions]
        scores = await asyncio.gather(*score_tasks, return_exceptions=True)

        for (action, state), score in zip(expansions, scores):
            if isinstance(score, BaseException):
                logger.warning(f"Scoring failed for child state: {score}")
                continue

            child = node.add_child(
                state=state,
                action=action,
                score=score,
            )
            stats.total_nodes_created += 1
            children.append(child)

        return children

    def _check_terminal_condition(
        self,
        *,
        node: BeamNode[StateT, ActionT],
        is_terminal_fn: TerminalFunc[StateT],
        stats: BeamSearchStatistics,
    ) -> bool:
        """
        Check if a node satisfies the terminal condition.

        Args:
            node (BeamNode): The node to check.
            is_terminal_fn (TerminalFunc): Function to determine terminal status.
            stats (BeamSearchStatistics): Statistics (unused but kept for consistency).

        Returns:
            bool: True if the node is terminal.
        """
        is_terminal = is_terminal_fn(node.state, node.score)
        if is_terminal:
            node.is_terminal = True
        return is_terminal

    def _find_best_terminal_node(
        self,
        *,
        nodes: List[BeamNode[StateT, ActionT]],
        is_terminal_fn: TerminalFunc[StateT],
        stats: BeamSearchStatistics,
        min_depth: int,
    ) -> Optional[BeamNode[StateT, ActionT]]:
        """
        Find the best terminal node among candidates.

        Args:
            nodes (List[BeamNode]): Candidate nodes to check.
            is_terminal_fn (TerminalFunc): Function to determine terminal status.
            stats (BeamSearchStatistics): Statistics to update.
            min_depth (int): Minimum depth before allowing termination.

        Returns:
            Optional[BeamNode]: Best terminal node, or None if none found.
        """
        terminal_nodes: List[BeamNode[StateT, ActionT]] = []

        for node in nodes:
            if node.depth >= min_depth and self._check_terminal_condition(
                node=node,
                is_terminal_fn=is_terminal_fn,
                stats=stats,
            ):
                terminal_nodes.append(node)

        if not terminal_nodes:
            return None

        return max(terminal_nodes, key=lambda n: n.score)

    def _should_expand_beam(self) -> bool:
        """
        Determine if beam width should increase.

        Returns:
            bool: True if beam should be expanded.
        """
        if self._current_beam_width >= self._config.max_beam_width:
            return False

        strategy = self._config.expansion_strategy

        if strategy == BeamExpansionStrategy.FIXED:
            return False

        if strategy == BeamExpansionStrategy.SCORE_THRESHOLD:
            return self._best_score_seen >= self._config.expansion_score_threshold

        if strategy == BeamExpansionStrategy.SCORE_IMPROVEMENT:
            improvement = self._best_score_seen - self._previous_best_score
            return improvement >= self._config.expansion_improvement_threshold

        return False

    def _should_contract_beam(self) -> bool:
        """
        Determine if beam width should decrease.

        Returns:
            bool: True if beam should be contracted.
        """
        if self._current_beam_width <= self._config.min_beam_width:
            return False

        strategy = self._config.contraction_strategy

        if strategy == BeamContractionStrategy.FIXED:
            return False

        if strategy == BeamContractionStrategy.SCORE_PLATEAU:
            return self._plateau_counter >= self._config.plateau_patience

        # RATE_LIMIT strategy is handled externally via signal_rate_limit()
        return False

    def _update_beam_width(self, *, stats: BeamSearchStatistics) -> None:
        """
        Update beam width based on current strategy.

        Args:
            stats (BeamSearchStatistics): Statistics to update.
        """
        if self._should_expand_beam():
            self._current_beam_width = min(
                self._current_beam_width + 1,
                self._config.max_beam_width,
            )
            stats.expansion_events += 1
            logger.debug(f"Beam expanded to {self._current_beam_width}")

        elif self._should_contract_beam():
            self._current_beam_width = max(
                self._current_beam_width - 1,
                self._config.min_beam_width,
            )
            stats.contraction_events += 1
            self._plateau_counter = 0  # Reset counter after contraction
            logger.debug(f"Beam contracted to {self._current_beam_width}")

    def _prune_beam(
        self,
        *,
        nodes: List[BeamNode[StateT, ActionT]],
        stats: BeamSearchStatistics,
    ) -> List[BeamNode[StateT, ActionT]]:
        """
        Prune nodes to maintain beam width constraint.

        Args:
            nodes (List[BeamNode]): Candidate nodes to prune.
            stats (BeamSearchStatistics): Statistics to update.

        Returns:
            List[BeamNode]: Top-k nodes by score.
        """
        if len(nodes) <= self._current_beam_width:
            return nodes

        # Sort by score descending
        sorted_nodes = sorted(nodes, key=lambda n: n.score, reverse=True)

        # Keep top beam_width nodes
        kept = sorted_nodes[: self._current_beam_width]
        pruned_count = len(nodes) - len(kept)
        stats.total_nodes_pruned += pruned_count

        logger.debug(f"Pruned {pruned_count} nodes, keeping {len(kept)}")

        return kept

    def signal_rate_limit(self) -> None:
        """
        Signal that a rate limit was encountered.

        This method can be called by external code to trigger beam contraction
        when using the RATE_LIMIT contraction strategy.
        """
        if self._config.contraction_strategy == BeamContractionStrategy.RATE_LIMIT:
            if self._current_beam_width > self._config.min_beam_width:
                self._current_beam_width = max(
                    self._current_beam_width - 1,
                    self._config.min_beam_width,
                )
                logger.debug(f"Rate limit signal received, beam contracted to {self._current_beam_width}")

    @property
    def current_beam_width(self) -> int:
        """
        Get the current beam width.

        Returns:
            int: Current beam width.
        """
        return self._current_beam_width
