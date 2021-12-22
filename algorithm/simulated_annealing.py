import copy
import logging
import random
from typing import Callable

import numpy as np

from core import AbstractSolver, AbstractMove, PartialSolution, Instance_T, Solution_T

logger = logging.getLogger(__name__)


def temperature_div(t_cur, step=None, **_):
    step = step or 1
    return t_cur / step


def temperature_div_log(t_cur, step=None, **_):
    step = step or 1
    return t_cur / np.log2(step)


class SimulatedAnnealing(AbstractSolver):
    """ Simulated Annealing algorithm """

    def __init__(
            self,
            ps: PartialSolution,
            move: AbstractMove,
            temperature_func: Callable,
            t_max=10**3,
            t_min=0,
            max_iter=10**4,
            **kwargs,
    ):
        if not ps.is_full:
            raise ValueError("Stating solution must be full assignment!")

        super(SimulatedAnnealing, self).__init__(**kwargs)

        self.ps = copy.deepcopy(ps)
        self.move = move
        self.temperature_func = temperature_func
        self.t_max = t_max
        self.t_min = t_min
        self.max_iter = max_iter

    def solve(self, data: Instance_T, m: int) -> PartialSolution:
        t_cur = self.t_max
        ps_cur = self.ps
        of_cur = ps_cur.mse
        ps_best = copy.deepcopy(ps_cur)
        of_best = ps_best.mse
        _iter = 1
        while _iter <= self.max_iter and t_cur > self.t_min:
            ps_new = copy.deepcopy(ps_cur)
            self.move.move(ps_new)

            of_new = ps_new.mse
            logger.debug(f'NEW OF {of_new}')

            if of_cur > of_new or random.random() < np.exp(-(of_new - of_cur) / t_cur):
                ps_cur = ps_new
                of_cur = of_new

            if of_best > of_cur:
                of_best = of_cur
                ps_best = ps_cur

            _iter += 1
            t_cur = self.temperature_func(
                t_cur=t_cur,
                t_min=self.t_min,
                t_max=self.t_max,
                step=_iter,
            )

            logger.debug(f'SIMULATED ANNEALING: STATE: T={t_cur}, ITERATION={_iter}\n')

        return ps_best