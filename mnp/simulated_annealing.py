import logging
import copy
import random

from functools import partial
from typing import Callable, Collection

import numpy as np

from core import MNPAlgorithm
from .utility import MNPState


logger = logging.getLogger(__name__)


def transition_greedy_one(state: MNPState):
    """One greedy step with random element."""
    _s_idx = random.randint(0, len(state) - 1)
    if state[_s_idx]:
        _elem_idx = random.randint(0, len(state[_s_idx]) - 1)
        _elem = state[_s_idx][_elem_idx]
        del(state[_s_idx][_elem_idx])
        _idx = np.argmin(list(map(sum, state)))
        state[_idx].append(_elem)


def transition_greedy_n(n=1):
    def _func(state: MNPState):
        f"""Make transition_greedy_one {n} times"""
        for _ in range(n):
            transition_greedy_one(state)

    return _func


def transition_greedy_many(n=1):
    def _func(state: MNPState):
        f"""Extract {n} random elements from state and
        put them back in greedy manner."""
        _elements = []
        for _ in range(n):
            _s_idx = random.randint(0, len(state) - 1)
            if state[_s_idx]:
                _elem_idx = random.randint(0, len(state[_s_idx]) - 1)
                _elem = state[_s_idx][_elem_idx]
                del(state[_s_idx][_elem_idx])
                _elements.append(_elem)
        _elements.sort(reverse=True)
        _sums = [sum(s) for s in state]
        for elem in _elements:
            _min_idx = np.argmin(_sums)
            _sums[_min_idx] += elem
            state[_min_idx].append(elem)

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
            start_state: MNPState = None,
            temperature_func: Callable = None,
            transition_func: Callable = transition_greedy_one,
            t_max=10**3,
            t_min=0,
            max_iter=10**4,
            **kwargs,
    ):
        super(SimulatedAnnealing, self).__init__(**kwargs)
        if not start_state:
            raise ValueError(f"Start solution must be defined for {self.__class__},"
                             f" instead got {start_state}.")
        self.start_sol = copy.deepcopy(start_state)
        self.transition_func = transition_func
        self.temperature_func = temperature_func
        self.t_max = t_max
        self.t_min = t_min
        self.max_iter = max_iter

    def execute_algorithm(self, data: Collection):
        t_cur = self.t_max
        state_cur = self.start_sol
        state_best = state_cur
        _perfect = self.get_perfect_mnp(data)
        _of = partial(self.calculate_sum_diff_sq, perfect=_perfect)
        of_cur = _of(state=state_cur)
        of_best = of_cur
        _iter = 1
        while _iter <= self.max_iter and t_cur > self.t_min:
            state_new = copy.deepcopy(state_cur)
            self.transition_func(state=state_new)
            of_new = _of(state=state_new)
            logger.debug(f'NEW OF {of_new}')

            if of_cur > of_new or random.random() < np.exp(-(of_new - of_cur) / t_cur):
                of_cur = of_new
                state_cur = state_new
            if of_best > of_new:
                of_best = of_new
                state_best = state_new  # refs will be okay since we dont change state_cur

            _iter += 1
            t_cur = self.temperature_func(t_cur=t_cur, t_min=self.t_min, t_max=self.t_max, step=_iter)

            logger.debug(f'SIMULATED ANNEALING: STATE: T={t_cur}, ITERATION={_iter}\n')

        return of_best, state_best