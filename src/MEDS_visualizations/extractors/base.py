import copy
from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from ..types import DF_T, PLOT_DATA_T

EXTRACTOR_T = TypeVar("EXTRACTOR_T", bound="BaseExtractor")


class BaseExtractor(Generic[DF_T, PLOT_DATA_T], ABC):
    """A base class for data extractors used for visualizing MEDS datasets.

    A user must overwrite three methods to leverage the extractor:
      - `_extract`: Extract relevant data from a shard of data.
      - `_merge`: Merge outputs from multiple extractors into a unified representation.
    """

    is_fit: bool

    def __init__(self):
        self.is_fit = False

    @abstractmethod
    def _extract(self, shard: DF_T, code_metadata: DF_T | None = None):
        """Extract relevant data from the full dataset."""
        pass

    def _fit(self, shard: DF_T, code_metadata: DF_T | None = None) -> EXTRACTOR_T:
        self._extract(shard)
        self.is_fit = True
        return self

    def fit(self, data: DF_T | list[DF_T], code_metadata: DF_T | None = None) -> EXTRACTOR_T:
        """Fit the extractor to the data, fitting all internal parameters needed across all shards."""

        if self.is_fit:
            raise ValueError("Extractor has already been fit. Please create a new instance.")

        if not isinstance(data, list):
            data = [data]

        if len(data) == 0:
            raise ValueError("No data provided to fit the extractor.")

        extracted = [copy.deepcopy(self)._fit(shard, code_metadata) for shard in data[1:]]

        self._fit(data[0], code_metadata)
        self.merge(*extracted, code_metadata=code_metadata)

        return self

    @abstractmethod
    def _merge(self, other: EXTRACTOR_T, code_metadata: DF_T | None = None):
        pass

    def merge(self, *others: EXTRACTOR_T, code_metadata: DF_T | None = None) -> EXTRACTOR_T:
        """Merge outputs from multiple shards into a final representation."""

        if len(others) == 0:
            return self

        for other in others:
            if not isinstance(other, self.__class__):
                raise TypeError(f"Cannot merge {self.__class__.__name__} with {other.__class__.__name__}")
            if not other.is_fit:
                raise RuntimeError("Cannot merge unfit extractors.")
            self._merge(other, code_metadata=code_metadata)

        return self

    @property
    @abstractmethod
    def plot_data(self) -> PLOT_DATA_T:
        """Return the data to be plotted."""
        pass
