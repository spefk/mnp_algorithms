from typing import List, Optional
from heapq import heappop, heappush


class PartialSet:
    def __init__(self, m: int = 2, first_element: Optional[int] = None):
        self.m = m
        self.sets: List[List[int]] = [[] for _ in range(m)]
        self.errors: List[int] = [0 for _ in range(m)]
        if first_element:
            self.sets[-1].append(first_element)
            self.errors[-1] = first_element

    @property
    def error(self) -> int:
        return sum(self.errors)

    def _ground_errors(self) -> None:
        _m = min(self.errors)
        self.errors = [x - _m for x in self.errors]

    def _reorder(self) -> None:
        self.errors, self.sets = map(
            list,
            zip(*[[x, y] for x, y in sorted(zip(self.errors, self.sets))])
        )

    def __or__(self, other):
        # merges two PartialPlace in memory of first one (self)
        if self.m != other.m:
            raise ValueError("Wrong operands! Number of sets in operands is not equal.")

        for x, y in zip(self.sets, reversed(other.sets)):
            x.extend(y)
        self.errors = [x + y for x, y in zip(self.errors, reversed(other.errors))]
        self._ground_errors()
        self._reorder()
        return self

    def __lt__(self, other):
        # for heapq lesser is one with greater error
        return self.error > other.error

    def __repr__(self):
        return f">> PartialSet >>\nid: {id(self)}\nsets: {self.sets},\nsums: {self.errors},\nerror: {self.error}."

    def __str__(self):
        return self.__repr__()


def KarmarkarKarp(data: List[int], m: int = 2):
    if len(data) == 0:
        return PartialSet(m=m)

    queue: List[PartialSet] = []
    for x in data:
        heappush(queue, PartialSet(m=m, first_element=x))

    while len(queue) > 1:
        heappush(queue, heappop(queue) | heappop(queue))

    return queue[0]


if __name__ == '__main__':
    print(KarmarkarKarp([1, 2, 3, 4, 5, 6, 7], m=2))
    print(KarmarkarKarp([1, 2, 3, 4, 5, 6, 7, 8], m=2))
    print(KarmarkarKarp([1, 2, 3, 4, 5, 6, 7, 8, 9], m=2))
    print(KarmarkarKarp([1, 2, 3, 4, 5, 6, 7, 8, 9], m=4))
    print(KarmarkarKarp(range(0, 100, 3), m=7))
    print(KarmarkarKarp(range(0, 1000), m=7))
