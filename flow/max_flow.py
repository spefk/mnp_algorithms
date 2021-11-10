import logging

from abc import abstractmethod, ABCMeta
from typing import Optional, Tuple, Dict
from collections import defaultdict, deque

from flow.network import Node, Arc, ResidualNetwork, ResidualPath


logger = logging.getLogger(__name__)


class MaxFlow(metaclass=ABCMeta):
    def __init__(self):
        pass

    def process_network(self, network: ResidualNetwork, max_iter=1000) -> None:
        logger.debug("Preparing MaxFlow search.")
        for _ in range(max_iter):
            path = self.get_increasing_path(network)
            if path is None:
                break
            network.increase_flow_by_path(path)

    @abstractmethod
    def get_increasing_path(self, network: ResidualNetwork) -> Optional[ResidualPath]:
        # returns a list of tuples [(node=source, arc=(source, _))..],
        # that form a path in residual network, or None.
        raise NotImplementedError()


class FordFulkerson(MaxFlow):
    def get_increasing_path(self, network: ResidualNetwork) -> Optional[ResidualPath]:
        # DFS
        logger.debug("Preparing Ford-Fulkerson.")
        visited: Dict[Node, bool] = defaultdict(lambda: False)

        def _dfs(node: Node):
            logger.debug(f"\tVisiting node {node}.")
            for a in network.residual_arcs_reachable_from(node):
                n = a.dst if a.dst is not node else a.src
                logger.debug(f"\tDiscovering node {n}.")
                if n is network.t:
                    return [(node, a)]
                if not visited[n]:
                    visited[n] = True
                    _res = _dfs(n)
                    if _res is not None:
                        _res.append((node, a))
                        return _res
            return None
        return _dfs(network.s)


class EdmondsKarp(MaxFlow):
    def get_increasing_path(self, network: ResidualNetwork) -> Optional[ResidualPath]:
        # BFS
        logger.debug("Preparing Edmonds-Karp.")
        discovered: Dict[Node, bool] = defaultdict(lambda: False)
        ancestor_data: Dict[Node, Optional[Tuple[Node, Arc]]] = defaultdict(lambda: None)
        queue: deque = deque([network.s])

        def _bfs():
            while len(queue) > 0:
                node = queue.popleft()
                logger.debug(f"\tVisiting node {node}.")
                for a in network.residual_arcs_reachable_from(node):
                    n = a.dst if a.dst is not node else a.src
                    logger.debug(f"\tDiscovering node {n}.")
                    if not discovered[n]:
                        discovered[n] = True
                        ancestor_data[n] = (node, a)
                        queue.append(n)
                    if n is network.t:
                        return

        _bfs()

        _node = network.t
        if ancestor_data[_node] is None:
            return None
        else:
            # path retrieval
            _answer = []
            while _node is not network.s:
                _data = ancestor_data[_node]
                _answer.append(_data)
                if _data is None:
                    raise RuntimeError("Something went wrong.")
                else:
                    _node = _data[0]
            return _answer
