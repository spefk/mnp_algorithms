import logging

from abc import abstractmethod, ABCMeta
from random import randint, random
from itertools import product
from typing import List, Optional, Tuple, Dict
from collections import defaultdict, deque
from dataclasses import dataclass


logger = logging.getLogger(__name__)


@dataclass(unsafe_hash=True, order=True)
class Node:
    id: int


@dataclass(order=True)
class Arc:
    src: Node
    dst: Node
    capacity: int

    flow: int = 0

    @property
    def available_capacity(self) -> int:
        return self.capacity - self.flow

    @property
    def available_capacity_residual(self) -> int:
        return self.flow

    def available_capacity_from(self, start_node: Node):
        if start_node is self.src:
            return self.available_capacity
        elif start_node is self.dst:
            return self.available_capacity_residual
        else:
            raise ValueError(f"Node '{start_node.id}' is not one of ends of arc '{str(self)}'.")

    def increase_flow_from(self, start_node: Node, value: int):
        if start_node is self.src:
            self.flow += value
        elif start_node is self.dst:
            self.flow -= value
        else:
            raise ValueError(f"Node '{start_node.id}' is not one of ends of arc '{str(self)}'.")

    def __str__(self):
        return f"({self.src}->{self.dst})"


class Network:
    def __init__(self, s_no: int, t_no: int, nodes: List[Tuple[int]], arcs: List[Tuple[int, int, int]]):
        self.nodes: Dict[int, Node] = dict()
        self.arcs: List[Arc] = list()
        self.arcs_dict: Dict[Tuple[Node, Node], List[Arc]] = defaultdict(list)
        self.incidence: Dict[Node, List[Arc]] = defaultdict(list)
        for n in nodes:
            self.add_node(*n)
        for a in arcs:
            self.add_arc(*a)
        self.s: Node = self.nodes[s_no]
        self.t: Node = self.nodes[t_no]

    def add_node(self, node_id: int, skip_exist=True) -> None:
        logger.debug(f"Adding node {node_id}.")
        if node_id in self.nodes:
            if not skip_exist:
                raise ValueError(f"Node with id={node_id} is already in self.nodes.")
        else:
            self.nodes[node_id] = Node(id=node_id)

    def add_arc(self, src_id: int, dst_id: int, capacity: int):
        logger.debug(f"Adding arc ({src_id}->{dst_id}) with capacity={capacity}.")
        if src_id not in self.nodes:
            self.add_node(src_id)
        if dst_id not in self.nodes:
            self.add_node(dst_id)
        a = Arc(src=self.nodes[src_id], dst=self.nodes[dst_id], capacity=capacity)
        self.arcs.append(a)
        self.arcs_dict[self.nodes[src_id], self.nodes[dst_id]].append(a)
        self.incidence[self.nodes[src_id]].append(a)
        self.incidence[self.nodes[dst_id]].append(a)

    @property
    def flow_value(self):
        return sum([
            (a.flow if a.src is self.s else -a.flow)
            for a in self.incidence[self.s]
        ])


ResidualPath = List[Tuple[Node, Arc]]


class ResidualNetwork(Network):
    def residual_nodes_reachable_from(self, node: Node) -> List[Node]:
        return [
            (a.dst if node is not a.dst else a.src)
            for a in self.incidence[node]
            if a.available_capacity_from(node) > 0
        ]

    def residual_arcs_reachable_from(self, node: Node) -> List[Arc]:
        return [
            a for a in self.incidence[node]
            if a.available_capacity_from(node) > 0
        ]

    @staticmethod
    def increase_flow_by_path(path: ResidualPath):
        value = min([p.available_capacity_from(src) for src, p in path])
        for src, p in path:
            p.increase_flow_from(src, value)


def generate_random_rnetwork_bernoulli(node_n=10, threshold=0.2, min_cap=1, max_cap=100) -> ResidualNetwork:
    node_numbers = list(range(node_n))
    nodes = [(n,) for n in node_numbers]
    arcs = [
        (u, v, randint(min_cap, max_cap))
        for u, v in product(node_numbers, node_numbers)
        if random() < threshold
    ]
    return ResidualNetwork(
        s_no=0,
        t_no=node_n-1,
        nodes=nodes,
        arcs=arcs,
    )
