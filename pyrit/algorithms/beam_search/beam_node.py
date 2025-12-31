# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
Beam search node data structure.

This module provides a generic node class for beam search algorithms,
supporting tree-structured exploration with parent-child relationships.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Generic, List, Optional, TypeVar

StateT = TypeVar("StateT")
ActionT = TypeVar("ActionT")


@dataclass
class BeamNode(Generic[StateT, ActionT]):
    """
    Represents a node in the beam search tree.

    Each node encapsulates a state, the action that led to it, and
    maintains parent-child relationships for tree traversal.

    Type Parameters:
        StateT: The type of state this node holds.
        ActionT: The type of action that transitions between states.
    """

    state: StateT
    action: Optional[ActionT] = None
    score: float = 0.0
    depth: int = 0
    node_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    parent: Optional["BeamNode[StateT, ActionT]"] = field(default=None, repr=False)
    children: List["BeamNode[StateT, ActionT]"] = field(default_factory=list, repr=False)
    is_expanded: bool = False
    is_terminal: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_child(
        self,
        *,
        state: StateT,
        action: ActionT,
        score: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "BeamNode[StateT, ActionT]":
        """
        Create and add a child node.

        Args:
            state (StateT): The state for the child node.
            action (ActionT): The action that led to this state.
            score (float): Initial score for the child. Defaults to 0.0.
            metadata (Optional[Dict[str, Any]]): Optional metadata for the child. Defaults to None.

        Returns:
            BeamNode[StateT, ActionT]: The newly created child node.
        """
        child: BeamNode[StateT, ActionT] = BeamNode(
            state=state,
            action=action,
            score=score,
            depth=self.depth + 1,
            parent=self,
            metadata=metadata or {},
        )
        self.children.append(child)
        return child

    def get_path_from_root(self) -> List["BeamNode[StateT, ActionT]"]:
        """
        Get the path from the root to this node.

        Returns:
            List[BeamNode[StateT, ActionT]]: Ordered list of nodes from root to self.
        """
        path: List[BeamNode[StateT, ActionT]] = []
        current: Optional[BeamNode[StateT, ActionT]] = self

        while current is not None:
            path.append(current)
            current = current.parent

        return list(reversed(path))

    def get_action_sequence(self) -> List[ActionT]:
        """
        Get the sequence of actions from root to this node.

        Returns:
            List[ActionT]: Ordered list of actions (excludes root's None action).
        """
        path = self.get_path_from_root()
        return [node.action for node in path if node.action is not None]

    def get_ancestors(self) -> List["BeamNode[StateT, ActionT]"]:
        """
        Get all ancestor nodes (excluding self).

        Returns:
            List[BeamNode[StateT, ActionT]]: List of ancestors from parent to root.
        """
        ancestors: List[BeamNode[StateT, ActionT]] = []
        current = self.parent

        while current is not None:
            ancestors.append(current)
            current = current.parent

        return ancestors

    def get_siblings(self) -> List["BeamNode[StateT, ActionT]"]:
        """
        Get sibling nodes (other children of the same parent).

        Returns:
            List[BeamNode[StateT, ActionT]]: List of sibling nodes, empty if root.
        """
        if self.parent is None:
            return []

        return [child for child in self.parent.children if child.node_id != self.node_id]

    def is_root(self) -> bool:
        """
        Check if this node is the root node.

        Returns:
            bool: True if this node has no parent.
        """
        return self.parent is None

    def is_leaf(self) -> bool:
        """
        Check if this node is a leaf node.

        Returns:
            bool: True if this node has no children.
        """
        return len(self.children) == 0

    def __str__(self) -> str:
        """
        Return string representation of the node.

        Returns:
            str: A string summarizing the node's key attributes.
        """
        return (
            f"BeamNode(id={self.node_id[:8]}..., "
            f"depth={self.depth}, "
            f"score={self.score:.3f}, "
            f"terminal={self.is_terminal})"
        )

    __repr__ = __str__
