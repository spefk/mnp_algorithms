import copy

import numpy as np

from core.structures import PartialSolution


def test_assignment():
    data = [1, 2, 3, 4, 5]
    ps = PartialSolution(data, 3)
    assert ps.sums.size == 3
    assert np.array_equal(ps.sums, np.array([0, 0, 0], dtype=int))
    for x in range(len(data)):
        assert ps.find_item(x) is None
    ps.put_item(3, 1)
    assert ps.find_item(3) == 1
    assert ps.sums[1] == 4
    ps.reset_item(3)
    assert ps.find_item(3) is None
    assert ps.sums[1] == 0
    ps.put_item(3, 1)
    ps.put_item(3, 2)
    assert ps.find_item(3) == 2
    assert ps.sums[1] == 0
    assert ps.sums[2] == 4

    ps2 = PartialSolution(data, 3)
    ps2.put_item(3, 2)
    assert hash(ps) == hash(ps2)

    data_2 = [2, 2]
    ps3 = PartialSolution(data_2, 2)
    assert ps3.squared_error == np.sqrt(8)
    ps3.put_item(0, 0)
    assert ps3.squared_error == 2


def test_deepcopy():
    ps = PartialSolution([1, 2, 3], 2)
    ps.put_item(0, 0)
    assert np.array_equal(ps.get_index_list(0), [0])
    ps2 = copy.deepcopy(ps)
    ps2.put_item(1, 0)
    assert np.array_equal(ps.get_index_list(0), [0])
    assert np.array_equal(ps2.get_index_list(0), [0, 1])


def test_from_sol():
    data = [[10, 100], [15], [1, 2, 3]]
    ps = PartialSolution.from_solution(data)
    assert np.array_equal(ps.sums, [110, 15, 6])
    assert np.array_equal(ps.get_index_list(0), [0, 1])
    assert np.array_equal(ps.get_index_list(1), [2])
    assert np.array_equal(ps.solution, [[10, 100], [15], [1, 2, 3]])
    assert ps.is_full

