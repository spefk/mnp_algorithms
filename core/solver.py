from abc import abstractmethod, ABCMeta
from .data_source import DataSource
from .algorithm import Algorithm


class Solver(metaclass=ABCMeta):
    """
    Base solver class.

    Puts together DataSource and Algorithm objects.
    """

    def __init__(
        self,
        source: DataSource = None,
        algorithm: Algorithm = None,
    ):
        self.source = source
        self.algorithm = algorithm

    @abstractmethod
    def solve_problem(self):
        raise NotImplementedError()


class MNPSolver(Solver):
    """General solver interface for MNP."""

    def __init__(self, **kwargs):
        super(MNPSolver, self).__init__(**kwargs)
        self.of = None
        self.sets = None

    def solve_problem(self):
        self.of, self.sets = self.algorithm.run(self.source.get_data())
        return self.of, self.sets

    def __str__(self):
        if self.of is None:
            _str = 'NOT SOLVED'
        else:
            _str = f'OF: {self.of}' \
                   + '\nSETS:\n' \
                   + '\n'.join(f'SET #{i}: {s}' for i, s in enumerate(self.sets)) \
                   + '\nSET SUMS: ' \
                   + ' '.join(list(map(lambda x: str(sum(x)), self.sets))) \
                   + '\n'
        return _str

    def __repr__(self):
        return 'MNP SOLVER INSTANCE\n' \
               + f'SOURCE: {self.source}\n'\
               + f'ALGORITHM: {self.algorithm}\n'\
               + self.__str__()
