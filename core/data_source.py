import re
import os.path

from typing import Callable, Iterable, Collection, Iterator
from abc import abstractmethod, ABCMeta


"""Classes to have unified data access interface."""


class DataSource(metaclass=ABCMeta):
    """
    Base abstract class for all data sources.
    """

    def __init__(
        self,
        to_collection: Callable[[Iterable], Collection] = list,
        item_type: type = int
     ):
        """
        param item_type: type to cast items before gathering them to Collection;
        param to_collection: Callable[[Iterable], Collection], function that gather
        elements to collection;
        """

        self.to_collection = to_collection
        self.item_type = item_type

    def get_data(self):
        return self.to_collection(self.get_data_iter())

    @abstractmethod
    def get_data_iter(self) -> Iterator:
        return self.get_data().__iter__()


class PythonSource(DataSource):
    """
    Class to work with python data.
    """

    def __init__(self, data: Iterable = None, **kwargs):
        """param data: some iterable data."""
        if not data:
            raise ValueError(f'Data must be set for PythonSource instance.')
        super(PythonSource, self).__init__(**kwargs)
        self.data = data

    def get_data_iter(self) -> Iterator:
        return self.data.__iter__()


class FileSource(DataSource):
    """
    Class to work with file sources.

    attribute _strip_chars: str, chars that are to be stripped from string;
    attribute _split_chars_re: str, regular expression to define splitting sequences;
    """

    _strip_chars = ' \n'
    _split_chars_re = '[, \n]+'

    def __init__(self, filepath: str = None, **kwargs):
        """param filepath: str, path to a file."""

        if not filepath:
            raise ValueError(f'Parameter filepath must be set for FileSource instance.')
        if not os.path.isfile(filepath):
            raise ValueError(f'File {filepath} not exist!')

        super(FileSource, self).__init__(**kwargs)
        self.filepath = filepath

    def get_data_iter(self) -> Iterator:
        return self._parse_from_file()

    def _parse_from_file(self):
        """
        Parses file <self.filepath>, mapping <self.item_type> parsed elements.
        """

        with open(self.filepath, 'r') as file:
            _out = file.read()
        _out = re.split(self._split_chars_re, _out.strip(self._strip_chars))
        return map(self.item_type, _out)


class WMFileSource(FileSource):
    """Class to work with dumped Wolfram Mathematica Lists."""
    _strip_chars = '{} \n'
    _split_chars_re = '[, \n]+'
