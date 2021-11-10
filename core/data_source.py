import re
import os.path
import logging

from typing import (
    Callable, Iterable, Collection,
    Generic, TypeVar, Union,
)
from abc import ABCMeta, abstractmethod


logger = logging.getLogger(__name__)


_R = TypeVar('_R')
_V = TypeVar('_V')
Gatherer = Union[Callable[[Iterable[_V]], Collection[_V]], type]
Mapper = Union[Callable[[_R], _V], type]


class DataSource(Generic[_R, _V], metaclass=ABCMeta):
    """ Base interface for data source. """

    def __init__(
        self,
        gatherer: Gatherer[_V] = list,
        mapper: Mapper[_R, _V] = int
     ):
        """
        :param gatherer - function that gather elements to collection;
        :param mapper - type to cast items before gathering them to Collection;
        """

        self._gatherer = gatherer
        self._mapper = mapper

    def get_data(self) -> Collection[_V]:
        return self._gatherer(self._map_data_iter())

    @abstractmethod
    def _map_data_iter(self) -> Iterable[_V]:
        raise NotImplementedError()


class PythonSource(DataSource):
    """ Class to work with python data. """

    def __init__(self, data: Iterable[_R], **kwargs):
        super(PythonSource, self).__init__(**kwargs)
        self._data = data

    def _map_data_iter(self) -> Iterable[_V]:
        return map(self._mapper, self._data)


class FileSource(DataSource[str, _V]):
    """
    Class to work with external files.

    :param _strip_chars - chars that are to be stripped from string;
    :param _split_chars_re - regular expression to define splitting sequences;
    """

    _strip_chars = ' \n'
    _split_chars_re = '[, \n]+'

    def __init__(self, filepath: str, **kwargs):
        if not os.path.isfile(filepath):
            raise ValueError(f'File {filepath} not exist!')

        super(FileSource, self).__init__(**kwargs)
        self.filepath = filepath

    def _map_data_iter(self) -> Iterable[_V]:
        return self._parse_from_file()

    def _parse_from_file(self) -> Iterable[_V]:
        with open(self.filepath, 'r') as file:
            logger.info(f"Parsing file '{self.filepath}'.")
            _out = file.read()
        _out_seq = re.split(self._split_chars_re, _out.strip(self._strip_chars))
        return map(self._mapper, _out_seq)


class WMFileSource(FileSource):
    """ Class to work with dumped Wolfram Mathematica Lists. """
    _strip_chars = '{} \n'
    _split_chars_re = '[, \n]+'
