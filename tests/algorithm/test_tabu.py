from collections import deque

import numpy as np

from algorithm import TranspOneTabu
from core import PartialSolution


def test_move():
    ps = PartialSolution([10, 30, 20], 2)
    ps.put_item(0, 0)
    ps.put_item(1, 0)
    ps.put_item(2, 0)
    assert np.array_equal(ps.get_index_list(0), [0, 1, 2])
    seen = deque()
    TranspOneTabu.move(ps, seen=seen)
    assert np.array_equal(ps.get_index_list(0), [0, 2])
    assert np.array_equal(ps.get_index_list(1), [1])

    ps = PartialSolution([10, 30, 20], 2)
    ps.put_item(0, 0)
    ps.put_item(1, 0)
    ps.put_item(2, 0)
    assert np.array_equal(ps.get_index_list(0), [0, 1, 2])
    seen = deque([ps.hash_put(1, 1)])
    TranspOneTabu.move(ps, seen=seen)
    assert np.array_equal(ps.get_index_list(0), [0, 1])
    assert np.array_equal(ps.get_index_list(1), [2])
