import logging

from source.core.solver import Solver


class MNP(Solver):
    """General class for solving MNP."""
    def __init__(self, **kwargs):
        super(MNP, self).__init__(**kwargs)
        self.of = None
        self.sets = None

    def solve_problem(self):
        self.of, self.sets = self.algorithm.run(self.source.get_data())
        return self.of, self.sets

    def __str__(self):
        if self.of is None:
            _str = 'NOT SOLVED'
        else:
            _str = f'\nOF: {self.of}' \
                   + '\nSETS:\n' \
                   + '\n'.join(f'SET #{i}: {s}' for i, s in enumerate(self.sets)) \
                   + '\nSET SUMS: ' \
                   + ' '.join(list(map(lambda x: str(sum(x)), self.sets)))
        return _str

    def __repr__(self):
        return '\nMNP SOLVER INSTANCE\n' \
               + f'SOURCE: {self.source}\n'\
               + f'ALGORITHM: {self.algorithm}\n'\
               + self.__str__()
