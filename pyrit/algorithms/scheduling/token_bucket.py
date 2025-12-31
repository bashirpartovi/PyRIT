# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Token bucket algorithm implementation for rate limiting.

This module provides a token bucket that can be used to implement
rate limiting with burst capacity.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Optional


@dataclass
class TokenBucketConfig:
    """Configuration for a token bucket."""

    # Maximum number of tokens the bucket can hold.
    capacity: float = 10.0

    # Tokens added per second.
    refill_rate: float = 1.0

    # Starting tokens. Defaults to capacity if None.
    initial_tokens: Optional[float] = None

    def __post_init__(self) -> None:
        """
        Validate configuration values.

        Raises:
            ValueError: If any configuration value is invalid.
        """
        if self.capacity <= 0:
            raise ValueError(f"capacity ({self.capacity}) must be > 0")
        if self.refill_rate <= 0:
            raise ValueError(f"refill_rate ({self.refill_rate}) must be > 0")
        if self.initial_tokens is not None and self.initial_tokens < 0:
            raise ValueError(f"initial_tokens ({self.initial_tokens}) must be >= 0")


class TokenBucket:
    """
    Token bucket implementation for rate limiting.

    The token bucket algorithm allows for controlled bursts while
    maintaining an average rate limit. Tokens are consumed when
    requests are made and refill over time.
    """

    def __init__(
        self,
        *,
        capacity: float = 10.0,
        refill_rate: float = 1.0,
        initial_tokens: Optional[float] = None,
    ) -> None:
        """
        Initialize the token bucket.

        Args:
            capacity (float): Maximum tokens the bucket can hold. Defaults to 10.0.
            refill_rate (float): Tokens added per second. Defaults to 1.0.
            initial_tokens (Optional[float]): Starting tokens. Defaults to capacity.

        Raises:
            ValueError: If capacity or refill_rate is not positive.
        """
        if capacity <= 0:
            raise ValueError(f"capacity ({capacity}) must be > 0")
        if refill_rate <= 0:
            raise ValueError(f"refill_rate ({refill_rate}) must be > 0")

        self._capacity = capacity
        self._refill_rate = refill_rate
        self._tokens = initial_tokens if initial_tokens is not None else capacity
        self._last_refill = time.monotonic()
        self._lock = asyncio.Lock()

    def _refill(self) -> None:
        """
        Refill tokens based on elapsed time.

        This method should be called before any token operation.
        """
        now = time.monotonic()
        elapsed = now - self._last_refill
        tokens_to_add = elapsed * self._refill_rate

        self._tokens = min(self._capacity, self._tokens + tokens_to_add)
        self._last_refill = now

    def try_acquire(self, tokens: float = 1.0) -> bool:
        """
        Attempt to acquire tokens without blocking.

        Args:
            tokens (float): Number of tokens to acquire. Defaults to 1.0.

        Returns:
            bool: True if tokens were acquired, False otherwise.

        Raises:
            ValueError: If tokens is not positive.
        """
        if tokens <= 0:
            raise ValueError(f"tokens ({tokens}) must be > 0")

        self._refill()

        if self._tokens >= tokens:
            self._tokens -= tokens
            return True
        return False

    async def acquire_async(self, tokens: float = 1.0) -> None:
        """
        Acquire tokens, waiting if necessary.

        Args:
            tokens (float): Number of tokens to acquire. Defaults to 1.0.

        Raises:
            ValueError: If tokens is not positive or exceeds capacity.
        """
        if tokens <= 0:
            raise ValueError(f"tokens ({tokens}) must be > 0")
        if tokens > self._capacity:
            raise ValueError(f"tokens ({tokens}) cannot exceed capacity ({self._capacity})")

        async with self._lock:
            while True:
                self._refill()

                if self._tokens >= tokens:
                    self._tokens -= tokens
                    return

                # Calculate wait time
                tokens_needed = tokens - self._tokens
                wait_time = tokens_needed / self._refill_rate

                # Wait for tokens to become available
                await asyncio.sleep(wait_time)

    @property
    def tokens_available(self) -> float:
        """
        Get current number of available tokens.

        Returns:
            float: Available tokens after refill calculation.
        """
        self._refill()
        return self._tokens

    def time_until_available(self, tokens: float = 1.0) -> float:
        """
        Calculate time until specified tokens will be available.

        Args:
            tokens (float): Number of tokens needed. Defaults to 1.0.

        Returns:
            float: Seconds until tokens will be available. Returns 0.0
                if tokens are already available.

        Raises:
            ValueError: If tokens is not positive or exceeds capacity.
        """
        if tokens <= 0:
            raise ValueError(f"tokens ({tokens}) must be > 0")
        if tokens > self._capacity:
            raise ValueError(f"tokens ({tokens}) cannot exceed capacity ({self._capacity})")

        self._refill()

        if self._tokens >= tokens:
            return 0.0

        tokens_needed = tokens - self._tokens
        return tokens_needed / self._refill_rate

    def update_rate(self, *, new_rate: float) -> None:
        """
        Update the refill rate.

        This can be used for adaptive rate limiting based on
        observed rate limit responses.

        Args:
            new_rate (float): New tokens per second.

        Raises:
            ValueError: If new_rate is not positive.
        """
        if new_rate <= 0:
            raise ValueError(f"new_rate ({new_rate}) must be > 0")

        # Refill with old rate before changing
        self._refill()
        self._refill_rate = new_rate

    def update_capacity(self, *, new_capacity: float) -> None:
        """
        Update the bucket capacity.

        Args:
            new_capacity (float): New maximum capacity.

        Raises:
            ValueError: If new_capacity is not positive.
        """
        if new_capacity <= 0:
            raise ValueError(f"new_capacity ({new_capacity}) must be > 0")

        self._capacity = new_capacity
        # Clamp current tokens to new capacity
        self._tokens = min(self._tokens, new_capacity)

    def reset(self) -> None:
        """Reset the bucket to full capacity."""
        self._tokens = self._capacity
        self._last_refill = time.monotonic()

    @property
    def capacity(self) -> float:
        """
        Get the bucket capacity.

        Returns:
            float: Maximum tokens the bucket can hold.
        """
        return self._capacity

    @property
    def refill_rate(self) -> float:
        """
        Get the current refill rate.

        Returns:
            float: Tokens added per second.
        """
        return self._refill_rate

    @property
    def utilization(self) -> float:
        """
        Get current utilization as a fraction.

        Returns:
            float: Current tokens divided by capacity (0.0 to 1.0).
        """
        self._refill()
        return self._tokens / self._capacity

    def __str__(self) -> str:
        """
        Return string representation of the bucket.

        Returns:
            str: Summary of bucket state.
        """
        return f"TokenBucket(tokens={self.tokens_available:.2f}/{self._capacity}, " f"rate={self._refill_rate}/s)"

    __repr__ = __str__
