import pytest
import logging

from collections import deque
from core.data_source import PythonSource


logger = logging.getLogger(__name__)


@pytest.mark.parametrize(
    'mapper,gatherer,data,expected',
    [
        (int, list, [], []),
        (int, list, [1, 2, 3, 4, 5], [1, 2, 3, 4, 5]),
        (int, list, ('1', '2', '3', '4', '5'), [1, 2, 3, 4, 5]),
        (float, set, ('1', '1', '1', '1', '1'), {1.}),
        (str, deque, iter(('This', 'is', 'Some', 'text', 123)), deque(['This', 'is', 'Some', 'text', '123'])),
    ]
)
def test_python_source_id(mapper, gatherer, data, expected):
    ps = PythonSource(data, mapper=mapper, gatherer=gatherer)
    assert ps.get_data() == expected
    _iter = iter(expected)
