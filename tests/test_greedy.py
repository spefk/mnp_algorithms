import pytest
import numpy as np
import logging

from source.core import WMFileSource, ExternalSource
from source.mnp import MNP, GreedyAlgorithm


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def test_greedy():
    wms = WMFileSource(
        filepath='tests/instances/data.txt',
        item_type=float,
        to_collection=(lambda x: np.array(list(x))),
    )

    ext_s_one = ExternalSource(
        data=[8, 6, 7, 4, 5],
        item_type=float,
        to_collection=(lambda x: np.array(list(x))),
    )

    ext_s = ExternalSource(
        data=[1, 3, 4, 5, 17, 21],
        item_type=float,
        to_collection=(lambda x: np.array(list(x))),
    )

    solver = MNP(source=ext_s_one, algorithm=GreedyAlgorithm(number_of_sets=3))
    _of, _sets = solver.solve_problem()
    logger.warning(f'\nOF: {_of},\nSETS: {_sets}')

    solver = MNP(source=ext_s, algorithm=GreedyAlgorithm(number_of_sets=3))
    _of, _sets = solver.solve_problem()
    logger.warning(f'\nOF: {_of},\nSETS: {_sets}')

    solver = MNP(source=wms, algorithm=GreedyAlgorithm(number_of_sets=20))
    _of, _sets = solver.solve_problem()
    logger.warning(f'\nOF: {_of},\nSETS: {_sets}')
