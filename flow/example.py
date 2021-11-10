import logging

import numpy as np

from time import time
from typing import Type
from functools import wraps

from flow.max_flow import MaxFlow, FordFulkerson, EdmondsKarp
from flow.network import generate_random_rnetwork_bernoulli


logger = logging.getLogger(__name__)


def timeit(func):
    @wraps(func)
    def _func(*args, **kwargs):
        t = time()
        _res = func(*args, **kwargs)
        return time() - t, _res
    return _func


def random_run(algorithm: Type[MaxFlow], node_n: int, threshold: int, min_cap: int, max_cap: int) -> None:
    mf = algorithm()
    rn = generate_random_rnetwork_bernoulli(node_n=node_n, threshold=0.15, min_cap=0, max_cap=100)
    mf.process_network(rn)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    tests_num = 100
    params = (50, 0.005, 10, 100)

    ff_mean = np.mean([timeit(random_run)(FordFulkerson, *params)[0] for _ in range(tests_num)])
    ek_mean = np.mean([timeit(random_run)(EdmondsKarp, *params)[0] for _ in range(tests_num)])
    logger.info(f"Ford-Fulkerson mean time on random instances: {ff_mean:.3f}")
    logger.info(f"Edmonds-Karp mean time on random instances: {ek_mean:.3f}")
