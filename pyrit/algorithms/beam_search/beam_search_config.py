# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Configuration for adaptive beam search algorithm.

This module provides configuration dataclasses and enums for controlling
beam search behavior, including adaptive width strategies.
"""

from dataclasses import dataclass
from enum import Enum


class BeamExpansionStrategy(Enum):
    """Strategy for when to expand beam width."""

    # Expand when best score exceeds the expansion threshold.
    SCORE_THRESHOLD = "score_threshold"

    # Expand when score improves by at least the improvement threshold.
    SCORE_IMPROVEMENT = "score_improvement"

    # Never adapt beam width, always use initial width.
    FIXED = "fixed"


class BeamContractionStrategy(Enum):
    """Strategy for when to contract beam width."""

    # Contract when scores stop improving for plateau_patience iterations.
    SCORE_PLATEAU = "score_plateau"

    # Contract when rate limiting is detected (requires scheduler integration).
    RATE_LIMIT = "rate_limit"

    # Never contract beam width.
    FIXED = "fixed"


@dataclass(frozen=True)
class BeamSearchConfig:
    """
    Configuration for adaptive beam search.

    This configuration controls the beam width adaptation, search depth,
    and termination conditions for the beam search algorithm.

    All fields are immutable after construction (frozen=True).
    """

    # Starting beam width for the search.
    initial_beam_width: int = 3

    # Minimum beam width (never go below this).
    min_beam_width: int = 1

    # Maximum beam width (never exceed this).
    max_beam_width: int = 10

    # Maximum number of iterations (search depth).
    max_depth: int = 10

    # Strategy for when to expand beam width.
    expansion_strategy: BeamExpansionStrategy = BeamExpansionStrategy.SCORE_THRESHOLD

    # Score threshold that triggers beam expansion (for SCORE_THRESHOLD strategy).
    expansion_score_threshold: float = 0.5

    # Minimum score improvement that triggers expansion (for SCORE_IMPROVEMENT strategy).
    expansion_improvement_threshold: float = 0.1

    # Strategy for when to contract beam width.
    contraction_strategy: BeamContractionStrategy = BeamContractionStrategy.SCORE_PLATEAU

    # Iterations without score improvement before contracting beam.
    plateau_patience: int = 2

    # Whether to generate extra candidates without executing them.
    enable_speculative_expansion: bool = True

    # Number of extra candidates to generate speculatively.
    speculative_candidates: int = 3

    # Score threshold to consider the search successful.
    success_threshold: float = 0.8

    # Minimum depth before allowing early termination.
    min_depth_before_termination: int = 1

    def __post_init__(self) -> None:
        """
        Validate configuration values after initialization.

        Raises:
            ValueError: If configuration values are invalid or inconsistent.
        """
        if self.initial_beam_width < self.min_beam_width:
            raise ValueError(
                f"initial_beam_width ({self.initial_beam_width}) must be >= " f"min_beam_width ({self.min_beam_width})"
            )

        if self.initial_beam_width > self.max_beam_width:
            raise ValueError(
                f"initial_beam_width ({self.initial_beam_width}) must be <= " f"max_beam_width ({self.max_beam_width})"
            )

        if self.min_beam_width < 1:
            raise ValueError(f"min_beam_width ({self.min_beam_width}) must be >= 1")

        if self.max_depth < 1:
            raise ValueError(f"max_depth ({self.max_depth}) must be >= 1")

        if not 0.0 <= self.success_threshold <= 1.0:
            raise ValueError(f"success_threshold ({self.success_threshold}) must be between 0.0 and 1.0")

        if not 0.0 <= self.expansion_score_threshold <= 1.0:
            raise ValueError(
                f"expansion_score_threshold ({self.expansion_score_threshold}) " "must be between 0.0 and 1.0"
            )

        if self.expansion_improvement_threshold < 0.0:
            raise ValueError(
                f"expansion_improvement_threshold ({self.expansion_improvement_threshold}) " "must be >= 0.0"
            )

        if self.plateau_patience < 1:
            raise ValueError(f"plateau_patience ({self.plateau_patience}) must be >= 1")

        if self.speculative_candidates < 0:
            raise ValueError(f"speculative_candidates ({self.speculative_candidates}) must be >= 0")

        if self.min_depth_before_termination < 0:
            raise ValueError(f"min_depth_before_termination ({self.min_depth_before_termination}) " "must be >= 0")
