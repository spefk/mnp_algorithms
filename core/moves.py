from __future__ import annotations

import copy
import random
from abc import abstractmethod, ABCMeta
from itertools import product
from typing import cast, List, Iterable, Protocol

import numpy as np

from .structures import PartialSolution


class Move_T(Protocol):
    @abstractmethod
    def __call__(self, ps: PartialSolution, **kwargs) -> PartialSolution:
        raise NotImplementedError()


class AbstractMover(metaclass=ABCMeta):
    """ Base class for classes that perform some moves in PartialSolution """
    def __init__(self, moves: Iterable[Move_T]):
        self.moves: List[Move_T] = list(moves)

    @abstractmethod
    def move(self, ps: PartialSolution, **kwargs) -> PartialSolution:
        """ Changes some PartialSolution inplace """
        raise NotImplementedError

    def move_n(self, ps: PartialSolution, n: int, **kwargs) -> PartialSolution:
        """ Apply class move N times """
        for _ in range(n):
            self.move(ps, **kwargs)
        return ps

    def __add__(self, other: AbstractMover) -> AbstractMover:
        if isinstance(other, AbstractMover):
            return AbstractMover(
                copy.deepcopy(self.moves) + copy.deepcopy(other.moves)
            )
        elif isinstance(other, list):
            return AbstractMover(
                copy.deepcopy(self.moves + copy.deepcopy(other))
            )
        else:
            raise TypeError("Can add to AbstractMover instance "
                            "only AbstractMover instances or lists.")

    def __radd__(self, other: AbstractMover):
        if isinstance(other, AbstractMover):
            return AbstractMover(
                copy.deepcopy(other.moves) + copy.deepcopy(self.moves)
            )
        elif isinstance(other, list):
            return AbstractMover(
                copy.deepcopy(copy.deepcopy(other) + self.moves)
            )
        else:
            raise TypeError("Can radd to AbstractMover instance "
                            "only AbstractMover instances or lists.")


class RandomMover(AbstractMover):
    """ Applies random move of self.moves """
    def move(self, ps: PartialSolution, **kwargs) -> PartialSolution:
        return random.choice(self.moves)(ps)


class ConsequentMover(AbstractMover):
    """ Applies all moves in self.moves consequently """
    def move(self, ps: PartialSolution, **kwargs) -> PartialSolution:
        for _move in self.moves:
            _move(ps)
        return ps


def greedy_transp_one(ps: PartialSolution, **_) -> PartialSolution:
    """ One greedy step with random element """
    idx = random.randint(0, ps.n - 1)
    ps.reset_item(idx)
    ps.put_item(idx, cast(int, np.argmin(ps.sums)))
    return ps


def greedy_transp_many(ps: PartialSolution, max_to_transp=None, **_) -> PartialSolution:
    """ One greedy step with random element """
    max_to_transp = max_to_transp or ps.n
    assert max_to_transp <= ps.n, "Cant transp that many"
    idx_list = random.sample(range(ps.n), random.randint(0, max_to_transp - 1))

    for idx in idx_list:
        ps.reset_item(idx)

    for value, idx in sorted(((ps.cost[idx], idx) for idx in idx_list), reverse=True):
        ps.put_item(idx, cast(int, np.argmin(ps.sums)))

    return ps


def reassign_one(ps: PartialSolution, i: int = -1, j: int = -1, **_) -> PartialSolution:
    """ One element reassignment """
    if i == -1 or j == -1:
        raise ValueError("Element and Set")
    ps.put_item(i, j)
    return ps


def max_to_min(ps: PartialSolution, **_) -> PartialSolution:
    """ Random element from max-sum set to min-sum set """
    max_idx = cast(int, np.argmax(ps.sums))
    min_idx = cast(int, np.argmin(ps.sums))
    elem_idx = random.choice(ps.get_index_list(max_idx))
    ps.put_item(elem_idx, min_idx)
    return ps


def tabu_transp_one(ps: PartialSolution, seen=None, **_) -> PartialSolution:
    if seen is None:
        seen = []

    best_d = 0
    i = random.randint(0, ps.n - 1)
    best_j = None
    for j in range(ps.m):
        if ps.hash_put(i, j) not in seen:
            d = ps.delta_put(i, j)
            if d > best_d:
                best_j = j
                best_d = d
    if best_j:
        ps.put_item(i, best_j)

    return ps


def tabu_swap_min_max(ps: PartialSolution, seen=None, **_) -> PartialSolution:
    if seen is None:
        seen = []

    max_idx = cast(int, np.argmax(ps.sums))
    min_idx = cast(int, np.argmin(ps.sums))
    max_indices = ps.get_index_list(max_idx)
    min_indices = ps.get_index_list(min_idx)
    best_d = 0
    best_pair = None
    for i, j in product(max_indices, min_indices):
        if ps.hash_swap(i, j) not in seen:
            d = ps.delta_swap(i, j)
            if d > best_d:
                best_d = d
                best_pair = i, j
    if best_pair:
        ps.put_item(best_pair[0], min_idx)
        ps.put_item(best_pair[1], max_idx)

    return ps
