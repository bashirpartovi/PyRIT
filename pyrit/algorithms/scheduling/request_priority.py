# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Request priority definitions for rate-limit-aware scheduling.

This module provides priority levels and a wrapper class for
prioritized async requests.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import IntEnum
from typing import Any, Awaitable, Callable, Generic, Optional, TypeVar

T = TypeVar("T")


class RequestPriority(IntEnum):
    """
    Priority levels for request scheduling.

    Lower values indicate higher priority. Requests are processed
    in priority order when resources are constrained.
    """

    # Highest priority. Primary attack path, must execute.
    CRITICAL = 0

    # High priority. Promising branches that show good scores.
    HIGH = 1

    # Normal priority. Standard exploration requests.
    NORMAL = 2

    # Low priority. Speculative candidates.
    LOW = 3

    # Lowest priority. Pre-fetching, can be dropped if needed.
    BACKGROUND = 4


@dataclass(order=True)
class PrioritizedRequest(Generic[T]):
    """
    A wrapper for async requests with priority and metadata.

    This class is orderable by (priority, created_at) to support
    priority queue operations.

    Type Parameters:
        T: The return type of the async operation.
    """

    # Fields used for ordering (order=True uses these in sequence)
    priority: RequestPriority = field(compare=True)
    created_at: datetime = field(compare=True, default_factory=datetime.utcnow)

    # Fields not used for ordering
    request_id: str = field(compare=False, default_factory=lambda: str(uuid.uuid4()))
    coroutine_factory: Callable[[], Awaitable[T]] = field(compare=False, default=None)  # type: ignore
    metadata: dict[str, Any] = field(compare=False, default_factory=dict)
    cancelled: bool = field(compare=False, default=False)

    def cancel(self) -> None:
        """
        Mark this request as cancelled.

        Cancelled requests will be skipped during execution.
        """
        self.cancelled = True

    @property
    def age_seconds(self) -> float:
        """
        Get the age of this request in seconds.

        Returns:
            float: Seconds since the request was created.
        """
        return (datetime.utcnow() - self.created_at).total_seconds()

    def __str__(self) -> str:
        """
        Return string representation of the request.

        Returns:
            str: Summary of the request.
        """
        return (
            f"PrioritizedRequest(id={self.request_id[:8]}..., "
            f"priority={self.priority.name}, "
            f"age={self.age_seconds:.1f}s, "
            f"cancelled={self.cancelled})"
        )

    __repr__ = __str__


def create_prioritized_request(
    *,
    coroutine_factory: Callable[[], Awaitable[T]],
    priority: RequestPriority = RequestPriority.NORMAL,
    metadata: Optional[dict[str, Any]] = None,
) -> PrioritizedRequest[T]:
    """
    Create a PrioritizedRequest with the given parameters.

    Args:
        coroutine_factory (Callable[[], Awaitable[T]]): A callable that returns
            the coroutine to execute. Must be a factory (not the coroutine itself)
            to allow for retry logic.
        priority (RequestPriority): The priority level. Defaults to NORMAL.
        metadata (Optional[dict[str, Any]]): Optional metadata. Defaults to None.

    Returns:
        PrioritizedRequest[T]: The created request wrapper.
    """
    return PrioritizedRequest(
        priority=priority,
        coroutine_factory=coroutine_factory,
        metadata=metadata or {},
    )
