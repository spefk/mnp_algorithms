from abc import abstractmethod, ABC
from .data_source import DataSource
from .algorithm import Algorithm


class Solver(ABC):
    """Base solver class. Puts together DataSource and Algorithm objects.
    Any derived class should implement method solve_problem."""
    def __init__(
        self,
        source: DataSource = None,
        algorithm: Algorithm = None,
    ):
        self.source = source
        self.algorithm = algorithm

    def set_params(
        self,
        source: DataSource = None,
        algorithm: Algorithm = None,
    ):
        self.source = source or self.source
        self.algorithm = algorithm or self.algorithm

    @abstractmethod
    def solve_problem(self):
        raise NotImplementedError()
