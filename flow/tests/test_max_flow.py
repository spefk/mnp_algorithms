from typing import Type

import pytest

from flow.max_flow import FordFulkerson, EdmondsKarp, MaxFlow
from flow.network import ResidualNetwork, generate_random_rnetwork_bernoulli
from tests.test_network import get_test_rn_1, get_test_rn_2


@pytest.mark.parametrize(
    'algorithm',
    [
        FordFulkerson,
        EdmondsKarp,
    ]
)
def test_basic_max_flow(algorithm: Type[MaxFlow]):
    mf = algorithm()
    rn = ResidualNetwork(
        s_no=0,
        t_no=1,
        nodes=[(0,), (1,), (200,), (300,)],
        arcs=[(0, 1, 20)],
    )
    mf.process_network(rn, 1000)
    assert rn.arcs[0].flow == 20

    rn = get_test_rn_1()
    mf.process_network(rn, 1000)

    def _assert_nodes_flow(id1, id2, val):
        assert rn.arcs_dict[rn.nodes[id1], rn.nodes[id2]][0].flow == val

    _assert_nodes_flow(0, 1, 20)
    _assert_nodes_flow(0, 2, 5)
    _assert_nodes_flow(1, 2, 10)
    _assert_nodes_flow(1, 3, 10)
    _assert_nodes_flow(2, 3, 15)

    rn = get_test_rn_2()
    mf.process_network(rn, 0)
    for a in rn.arcs:
        assert a.flow == 0
    mf.process_network(rn, 10)
    for a in rn.arcs:
        assert a.flow == 5


def test_on_random():
    ek = EdmondsKarp()
    for _ in range(50):
        rn = generate_random_rnetwork_bernoulli(node_n=20, threshold=0.15, min_cap=0, max_cap=0)
        ek.process_network(rn)
        assert rn.flow_value == 0

    for _ in range(10):
        rn = generate_random_rnetwork_bernoulli(node_n=30, threshold=1, min_cap=10, max_cap=10)
        ek.process_network(rn)
        assert rn.flow_value == 29 * 10
