# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Rate-limit-aware scheduler for async operations.

This module provides a scheduler that manages async request execution
with rate limiting, priority queuing, and adaptive pacing.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import (
    Any,
    Awaitable,
    Callable,
    List,
    Optional,
    TypeVar,
)

from pyrit.algorithms.scheduling.request_priority import (
    PrioritizedRequest,
    RequestPriority,
    create_prioritized_request,
)
from pyrit.algorithms.scheduling.token_bucket import TokenBucket

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class SchedulerStatistics:
    """Statistics collected during scheduler operation."""

    # Total number of requests processed.
    total_requests: int = 0

    # Number of requests that completed successfully.
    successful_requests: int = 0

    # Number of requests that failed.
    failed_requests: int = 0

    # Number of rate limit errors encountered.
    rate_limit_events: int = 0

    # Cumulative time spent waiting for rate limits.
    total_wait_time_seconds: float = 0.0

    # Number of low-priority requests dropped due to capacity.
    dropped_requests: int = 0

    # Current effective request rate.
    current_rate: float = 0.0

    # Maximum queue size observed.
    peak_queue_size: int = 0

    # Average request latency.
    average_latency_seconds: float = 0.0

    _latency_sum: float = field(default=0.0, repr=False)
    _latency_count: int = field(default=0, repr=False)

    def record_latency(self, latency: float) -> None:
        """
        Record a request latency measurement.

        Args:
            latency (float): Latency in seconds.
        """
        self._latency_sum += latency
        self._latency_count += 1
        if self._latency_count > 0:
            self.average_latency_seconds = self._latency_sum / self._latency_count


@dataclass
class RateLimitConfig:
    """Configuration for the rate-limit-aware scheduler."""

    # Target request rate.
    requests_per_second: float = 1.0

    # Maximum burst size allowed.
    burst_capacity: int = 10

    # Maximum number of parallel requests.
    max_concurrent: int = 5

    # Whether to adjust rate based on errors.
    adaptive_pacing: bool = True

    # Multiplier applied to rate on rate limit error (reduces rate).
    backoff_factor: float = 0.5

    # Multiplier applied to rate after success period (increases rate).
    recovery_factor: float = 1.1

    # Successful requests before attempting rate recovery.
    recovery_threshold: int = 10

    # Minimum rate (never go below this).
    min_rate: float = 0.1

    # Maximum queue size before dropping low-priority requests.
    max_queue_size: int = 100

    # Timeout for individual requests in seconds.
    request_timeout: float = 60.0

    def __post_init__(self) -> None:
        """
        Validate configuration values.

        Raises:
            ValueError: If any configuration value is invalid.
        """
        if self.requests_per_second <= 0:
            raise ValueError(f"requests_per_second ({self.requests_per_second}) must be > 0")
        if self.burst_capacity < 1:
            raise ValueError(f"burst_capacity ({self.burst_capacity}) must be >= 1")
        if self.max_concurrent < 1:
            raise ValueError(f"max_concurrent ({self.max_concurrent}) must be >= 1")
        if not 0 < self.backoff_factor < 1:
            raise ValueError(f"backoff_factor ({self.backoff_factor}) must be between 0 and 1")
        if self.recovery_factor <= 1:
            raise ValueError(f"recovery_factor ({self.recovery_factor}) must be > 1")
        if self.min_rate <= 0:
            raise ValueError(f"min_rate ({self.min_rate}) must be > 0")


class RateLimitAwareScheduler:
    """
    Scheduler for async operations with rate limiting and priority queuing.

    This scheduler manages the execution of async requests while respecting
    rate limits, prioritizing important requests, and adapting to rate limit
    errors through backoff and recovery.
    """

    def __init__(self, *, config: Optional[RateLimitConfig] = None) -> None:
        """
        Initialize the rate-limit-aware scheduler.

        Args:
            config (Optional[RateLimitConfig]): Scheduler configuration.
                Defaults to RateLimitConfig() with default values.
        """
        self._config = config or RateLimitConfig()
        self._original_rate = self._config.requests_per_second

        self._token_bucket = TokenBucket(
            capacity=float(self._config.burst_capacity),
            refill_rate=self._config.requests_per_second,
        )

        self._queue: List[PrioritizedRequest[Any]] = []
        self._active_count = 0
        self._consecutive_successes = 0
        self._paused = False
        self._pause_until: Optional[float] = None

        self._statistics = SchedulerStatistics(current_rate=self._config.requests_per_second)

        self._lock = asyncio.Lock()
        self._semaphore = asyncio.Semaphore(self._config.max_concurrent)

    async def schedule_async(
        self,
        coroutine_factory: Callable[[], Awaitable[T]],
        *,
        priority: RequestPriority = RequestPriority.NORMAL,
        metadata: Optional[dict[str, Any]] = None,
    ) -> T:
        """
        Schedule a single request for execution.

        Args:
            coroutine_factory (Callable[[], Awaitable[T]]): Factory that creates
                the coroutine to execute.
            priority (RequestPriority): Request priority. Defaults to NORMAL.
            metadata (Optional[dict[str, Any]]): Optional metadata. Defaults to None.

        Returns:
            T: The result of the coroutine.

        Raises:
            asyncio.TimeoutError: If the request times out.
            Exception: Any exception raised by the coroutine.
        """
        request = create_prioritized_request(
            coroutine_factory=coroutine_factory,
            priority=priority,
            metadata=metadata,
        )

        return await self._execute_request_async(request)

    async def schedule_batch_async(
        self,
        requests: List[PrioritizedRequest[T]],
    ) -> List[T]:
        """
        Schedule a batch of requests for execution.

        Requests are executed in priority order, respecting rate limits
        and concurrency constraints.

        Args:
            requests (List[PrioritizedRequest[T]]): Requests to execute.

        Returns:
            List[T]: Results in the same order as input requests.
        """
        if not requests:
            return []

        # Create tasks for all requests
        tasks = [asyncio.create_task(self._execute_request_async(req)) for req in requests]

        # Wait for all to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Re-raise first exception if any
        for result in results:
            if isinstance(result, Exception):
                raise result

        return list(results)  # type: ignore

    async def _execute_request_async(
        self,
        request: PrioritizedRequest[T],
    ) -> T:
        """
        Execute a single request with rate limiting.

        Args:
            request (PrioritizedRequest[T]): The request to execute.

        Returns:
            T: The result of the request.

        Raises:
            asyncio.CancelledError: If the request was cancelled.
            asyncio.TimeoutError: If the request times out.
        """
        if request.cancelled:
            raise asyncio.CancelledError("Request was cancelled")

        # Wait for pause to end if active
        await self._wait_for_unpause_async()

        # Acquire rate limit token
        wait_start = time.monotonic()
        await self._token_bucket.acquire_async()
        wait_time = time.monotonic() - wait_start

        if wait_time > 0:
            self._statistics.total_wait_time_seconds += wait_time

        # Acquire concurrency semaphore
        async with self._semaphore:
            self._active_count += 1
            self._statistics.total_requests += 1

            start_time = time.monotonic()

            try:
                # Execute with timeout
                result = await asyncio.wait_for(
                    request.coroutine_factory(),
                    timeout=self._config.request_timeout,
                )

                self._statistics.successful_requests += 1
                self._on_success()

                latency = time.monotonic() - start_time
                self._statistics.record_latency(latency)

                return result

            except asyncio.TimeoutError:
                self._statistics.failed_requests += 1
                logger.warning(f"Request {request.request_id} timed out")
                raise

            except Exception as e:
                self._statistics.failed_requests += 1
                # Check if this looks like a rate limit error
                if self._is_rate_limit_error(e):
                    self._on_rate_limit()
                raise

            finally:
                self._active_count -= 1

    def _is_rate_limit_error(self, error: Exception) -> bool:
        """
        Check if an exception appears to be a rate limit error.

        Args:
            error (Exception): The exception to check.

        Returns:
            bool: True if this looks like a rate limit error.
        """
        error_str = str(error).lower()
        rate_limit_indicators = [
            "rate limit",
            "rate_limit",
            "ratelimit",
            "429",
            "too many requests",
            "throttl",
        ]
        return any(indicator in error_str for indicator in rate_limit_indicators)

    def _on_success(self) -> None:
        """Handle successful request completion."""
        if not self._config.adaptive_pacing:
            return

        self._consecutive_successes += 1

        # Attempt rate recovery after sustained success
        if self._consecutive_successes >= self._config.recovery_threshold:
            current_rate = self._token_bucket.refill_rate
            new_rate = min(
                current_rate * self._config.recovery_factor,
                self._original_rate,
            )

            if new_rate > current_rate:
                self._token_bucket.update_rate(new_rate=new_rate)
                self._statistics.current_rate = new_rate
                logger.debug(f"Rate recovered to {new_rate:.2f}/s")

            self._consecutive_successes = 0

    def _on_rate_limit(self) -> None:
        """Handle rate limit error."""
        self._statistics.rate_limit_events += 1
        self._consecutive_successes = 0

        if not self._config.adaptive_pacing:
            return

        # Reduce rate
        current_rate = self._token_bucket.refill_rate
        new_rate = max(
            current_rate * self._config.backoff_factor,
            self._config.min_rate,
        )

        self._token_bucket.update_rate(new_rate=new_rate)
        self._statistics.current_rate = new_rate
        logger.warning(f"Rate limit hit, reducing rate to {new_rate:.2f}/s")

    def report_rate_limit(self) -> None:
        """
        Manually report a rate limit event.

        This can be called by external code when rate limits are
        detected through other means.
        """
        self._on_rate_limit()

    def report_success(self) -> None:
        """
        Manually report a successful operation.

        This can be called by external code to contribute to
        rate recovery tracking.
        """
        self._on_success()

    async def pause_async(self, seconds: float) -> None:
        """
        Pause all scheduling for a duration.

        Args:
            seconds (float): Duration to pause in seconds.
        """
        self._paused = True
        self._pause_until = time.monotonic() + seconds
        logger.debug(f"Scheduler paused for {seconds:.1f}s")
        await asyncio.sleep(seconds)
        self._paused = False
        self._pause_until = None

    async def _wait_for_unpause_async(self) -> None:
        """Wait until the scheduler is unpaused."""
        while self._paused and self._pause_until is not None:
            remaining = self._pause_until - time.monotonic()
            if remaining > 0:
                await asyncio.sleep(min(remaining, 0.1))
            else:
                self._paused = False
                self._pause_until = None

    def get_current_capacity(self) -> float:
        """
        Get current available capacity as a fraction.

        Returns:
            float: Available capacity from 0.0 (none) to 1.0 (full).
        """
        return self._token_bucket.utilization

    def get_statistics(self) -> SchedulerStatistics:
        """
        Get scheduler statistics.

        Returns:
            SchedulerStatistics: Current statistics snapshot.
        """
        return self._statistics

    def reset_statistics(self) -> None:
        """Reset all statistics to zero."""
        self._statistics = SchedulerStatistics(current_rate=self._token_bucket.refill_rate)

    @property
    def active_requests(self) -> int:
        """
        Get number of currently executing requests.

        Returns:
            int: Active request count.
        """
        return self._active_count

    @property
    def current_rate(self) -> float:
        """
        Get current request rate.

        Returns:
            float: Current tokens per second.
        """
        return self._token_bucket.refill_rate

    @property
    def is_paused(self) -> bool:
        """
        Check if scheduler is currently paused.

        Returns:
            bool: True if paused.
        """
        return self._paused

    def __str__(self) -> str:
        """
        Return string representation of the scheduler.

        Returns:
            str: Summary of scheduler state.
        """
        return (
            f"RateLimitAwareScheduler(rate={self.current_rate:.2f}/s, "
            f"active={self._active_count}, "
            f"paused={self._paused})"
        )

    __repr__ = __str__
