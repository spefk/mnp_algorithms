import pytest

from flow.network import Node, Arc, ResidualNetwork, generate_random_rnetwork_bernoulli


def assert_cap(a, src, dst, flow, cap):
    assert a.flow == flow
    assert a.available_capacity == cap - flow
    assert a.available_capacity_from(src) == cap - flow
    assert a.available_capacity_residual == flow
    assert a.available_capacity_from(dst) == flow


def test_arc():
    n1 = Node(1)
    n2 = Node(2)
    a = Arc(src=n1, dst=n2, capacity=10)
    a.increase_flow_from(n1, 7)
    assert_cap(a, n1, n2, 7, 10)
    a.increase_flow_from(n1, 3)
    assert_cap(a, n1, n2, 10, 10)
    a.increase_flow_from(n2, 5)
    assert_cap(a, n1, n2, 5, 10)


def get_test_rn_1():
    return ResidualNetwork(
        s_no=0,
        t_no=3,
        nodes=[(0,), (1,), (2,), (3,)],
        arcs=[(0, 1, 20), (0, 2, 5), (1, 2, 10), (2, 3, 15), (1, 3, 10)],
    )


def get_test_rn_2():
    return ResidualNetwork(
        s_no=0,
        t_no=2,
        nodes=[(0,), (1,), (2,)],
        arcs=[(0, 1, 20), (1, 2, 5)],
    )


def test_network_creation():
    rn = get_test_rn_1()

    assert len(rn.nodes) == 4
    assert len(rn.arcs) == 5
    assert rn.residual_nodes_reachable_from(rn.nodes[0]) == [rn.nodes[1], rn.nodes[2]]
    assert rn.residual_nodes_reachable_from(rn.nodes[1]) == [rn.nodes[2], rn.nodes[3]]
    for a in rn.residual_arcs_reachable_from(rn.nodes[0]):
        a.flow = a.capacity
    assert rn.residual_nodes_reachable_from(rn.nodes[0]) == []


def test_network_increase_flow():
    rn = get_test_rn_1()
    path_1 = [
        (rn.nodes[0], rn.arcs_dict[rn.nodes[0], rn.nodes[1]][0]),
        (rn.nodes[1], rn.arcs_dict[rn.nodes[1], rn.nodes[2]][0]),
    ]
    rn.increase_flow_by_path(path_1)
    _test_arcs_1 = [
        rn.arcs_dict[rn.nodes[0], rn.nodes[1]][0],
        rn.arcs_dict[rn.nodes[1], rn.nodes[2]][0],
    ]
    for a in _test_arcs_1:
        assert a.flow == 10
    for a in rn.arcs:
        if a not in _test_arcs_1:
            assert a.flow == 0
    path_2 = [
        (rn.nodes[0], rn.arcs_dict[rn.nodes[0], rn.nodes[1]][0]),
        (rn.nodes[1], rn.arcs_dict[rn.nodes[1], rn.nodes[3]][0]),
    ]
    rn.increase_flow_by_path(path_2)
    _test_arcs_2 = [
        rn.arcs_dict[rn.nodes[0], rn.nodes[1]][0],
        rn.arcs_dict[rn.nodes[1], rn.nodes[3]][0],
    ]
    assert rn.arcs_dict[rn.nodes[0], rn.nodes[1]][0].flow == 20
    assert rn.arcs_dict[rn.nodes[1], rn.nodes[2]][0].flow == 10
    assert rn.arcs_dict[rn.nodes[1], rn.nodes[3]][0].flow == 10
    for a in rn.arcs:
        if (a not in _test_arcs_1) and (a not in _test_arcs_2):
            assert a.flow == 0


def test_random_rnetwork_bernoulli():
    for _ in range(10):
        rn = generate_random_rnetwork_bernoulli(node_n=50, threshold=0.1, min_cap=10, max_cap=100)
        assert len(rn.nodes) == 50
        for a in rn.arcs:
            assert 10 <= a.capacity <= 100
