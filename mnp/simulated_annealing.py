import logging
import copy
import random

from functools import partial
from typing import Callable, Collection

import numpy as np

from core import MNPAlgorithm
from .utility import MNPState


logger = logging.getLogger(__name__)


def transition_greedy_one(partitioning: MNPState):
    """One greedy step with random element."""
    _s_idx = random.randint(0, len(partitioning) - 1)
    if partitioning[_s_idx]:
        _elem_idx = random.randint(0, len(partitioning[_s_idx]) - 1)
        _elem = partitioning[_s_idx][_elem_idx]
        del(partitioning[_s_idx][_elem_idx])
        _idx = np.argmin(list(map(sum, partitioning)))  # type: ignore
        partitioning[_idx].append(_elem)  # type: ignore


def transition_greedy_n(n=1):
    def _func(partitioning: MNPState):
        f"""Make transition_greedy_one {n} times"""
        for _ in range(n):
            transition_greedy_one(partitioning)

    return _func


def transition_greedy_many(n=1):
    def _func(partitioning: MNPState):
        f"""Extract {n} random elements from partitioning and
        put them back in greedy manner."""
        _elements = []
        for _ in range(n):
            _s_idx = random.randint(0, len(partitioning) - 1)
            if partitioning[_s_idx]:
                _elem_idx = random.randint(0, len(partitioning[_s_idx]) - 1)
                _elem = partitioning[_s_idx][_elem_idx]
                del(partitioning[_s_idx][_elem_idx])
                _elements.append(_elem)
        _elements.sort(reverse=True)
        _sums = [sum(s) for s in partitioning]
        for elem in _elements:
            _min_idx = np.argmin(_sums)
            _sums[_min_idx] += elem
            partitioning[_min_idx].append(elem)  # type: ignore

    return _func


def temperature_div(t_cur, step=None, **_):
    step = step or 1
    return t_cur / step


def temperature_div_log(t_cur, step=None, **_):
    step = step or 1
    return t_cur / np.log2(step)


class SimulatedAnnealing(MNPAlgorithm):
    """Simulated Annealing algorithm implementation
    """
    transition_greedy_one = staticmethod(transition_greedy_one)
    transition_greedy_n = staticmethod(transition_greedy_n)
    temperature_div = staticmethod(temperature_div)
    temperature_div_log = staticmethod(temperature_div_log)

    def __init__(
            self,
            start_partitioning: MNPState,
            temperature_func: Callable,
            transition_func: Callable,
            t_max=10**3,
            t_min=0,
            max_iter=10**4,
            **kwargs,
    ):
        super(SimulatedAnnealing, self).__init__(**kwargs)

        self.start_sol = copy.deepcopy(start_partitioning)
        self.transition_func = transition_func
        self.temperature_func = temperature_func
        self.t_max = t_max
        self.t_min = t_min
        self.max_iter = max_iter

    def run(self, data: Collection):
        t_cur = self.t_max
        partitioning_cur = self.start_sol
        partitioning_best = partitioning_cur
        _perfect = self.get_perfect_mnp_value(data)
        _of = partial(self.calculate_sum_diff_sq, perfect=_perfect)
        of_cur = _of(partitioning=partitioning_cur)
        of_best = of_cur
        _iter = 1
        while _iter <= self.max_iter and t_cur > self.t_min:
            partitioning_new = copy.deepcopy(partitioning_cur)
            self.transition_func(partitioning=partitioning_new)
            of_new = _of(partitioning=partitioning_new)
            logger.debug(f'NEW OF {of_new}')

            if of_cur > of_new or random.random() < np.exp(-(of_new - of_cur) / t_cur):
                of_cur = of_new
                partitioning_cur = partitioning_new
            if of_best > of_new:
                of_best = of_new
                partitioning_best = partitioning_new  # refs will be okay since we dont change partitioning_cur

            _iter += 1
            t_cur = self.temperature_func(t_cur=t_cur, t_min=self.t_min, t_max=self.t_max, step=_iter)

            logger.debug(f'SIMULATED ANNEALING: STATE: T={t_cur}, ITERATION={_iter}\n')

        return partitioning_best
