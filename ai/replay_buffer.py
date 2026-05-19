"""Prioritized Experience Replay buffer with SumTree."""
from __future__ import annotations

import numpy as np


class _SumTree:
    """Binary tree where each leaf stores a priority and each parent stores
    the sum of its children. Allows O(log N) proportional sampling."""

    __slots__ = ("capacity", "tree", "data", "write_idx", "count")

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.data: list[object | None] = [None] * capacity
        self.write_idx = 0
        self.count = 0

    def total(self) -> float:
        return float(self.tree[0])

    def add(self, priority: float, data: object) -> None:
        idx = self.write_idx + self.capacity - 1
        self.data[self.write_idx] = data
        self._update(idx, priority)
        self.write_idx = (self.write_idx + 1) % self.capacity
        self.count = min(self.count + 1, self.capacity)

    def _update(self, tree_idx: int, priority: float) -> None:
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        while tree_idx > 0:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def update(self, tree_idx: int, priority: float) -> None:
        self._update(tree_idx, priority)

    def get(self, cumulative: float) -> tuple[int, float, object]:
        """Walk down the tree to find the leaf whose cumulative priority
        interval contains *cumulative*. Returns (tree_idx, priority, data)."""
        idx = 0
        while True:
            left = 2 * idx + 1
            right = left + 1
            if left >= len(self.tree):
                break
            if cumulative <= self.tree[left]:
                idx = left
            else:
                cumulative -= self.tree[left]
                idx = right
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]


class PrioritizedReplayBuffer:
    """Proportional Prioritized Experience Replay (Schaul et al. 2016)."""

    def __init__(
        self,
        capacity: int,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_end: float = 1.0,
        beta_anneal_steps: int = 100_000,
        eps: float = 1e-5,
    ) -> None:
        self.tree = _SumTree(capacity)
        self.alpha = alpha
        self.beta = beta_start
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_anneal_steps = beta_anneal_steps
        self.eps = eps
        self.max_priority = 1.0
        self._step = 0

    def __len__(self) -> int:
        return self.tree.count

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        priority = self.max_priority ** self.alpha
        self.tree.add(priority, (state, action, reward, next_state, done))

    def sample(
        self, batch_size: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[int]]:
        """Sample a batch proportional to priorities.

        Returns
        -------
        states, actions, rewards, next_states, dones, weights, tree_indices
        """
        n = len(self)
        segment = self.tree.total() / batch_size

        states, actions, rewards, next_states, dones = [], [], [], [], []
        weights = np.empty(batch_size, dtype=np.float32)
        tree_indices: list[int] = []

        # Anneal beta
        self.beta = min(
            self.beta_end,
            self.beta_start + (self.beta_end - self.beta_start) * self._step / max(self.beta_anneal_steps, 1),
        )
        self._step += 1

        min_prob = (self.eps ** self.alpha) / self.tree.total() if self.tree.total() > 0 else 1e-8
        max_weight = (n * min_prob) ** (-self.beta) if min_prob > 0 else 1.0

        for i in range(batch_size):
            low = segment * i
            high = segment * (i + 1)
            cumulative = np.random.uniform(low, high)
            tree_idx, priority, data = self.tree.get(cumulative)
            tree_indices.append(tree_idx)

            prob = priority / max(self.tree.total(), 1e-8)
            w = (n * prob) ** (-self.beta)
            weights[i] = w / max(max_weight, 1e-8)

            s, a, r, ns, d = data  # type: ignore[misc]
            states.append(s)
            actions.append(a)
            rewards.append(r)
            next_states.append(ns)
            dones.append(d)

        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
            weights,
            tree_indices,
        )

    def update_priorities(self, tree_indices: list[int], td_errors: np.ndarray) -> None:
        for idx, td in zip(tree_indices, td_errors):
            priority = (abs(td) + self.eps) ** self.alpha
            self.max_priority = max(self.max_priority, priority)
            self.tree.update(idx, priority)
