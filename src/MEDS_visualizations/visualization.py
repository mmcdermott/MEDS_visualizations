from typing import Generic

from .extractors import BaseExtractor
from .plotters import BasePlotter
from .types import DF_T, FIG_T, PLOT_DATA_T


class Visualization(Generic[DF_T, PLOT_DATA_T, FIG_T]):
    def __init__(self, extractor: BaseExtractor[DF_T, PLOT_DATA_T], plotter: BasePlotter[PLOT_DATA_T, FIG_T]):
        self.extractor = extractor
        self.plotter = plotter

        if self.extractor.is_fit:
            raise ValueError("Extractor has already been fit. Please create a new instance.")

    def render(self, shards: list[DF_T]) -> FIG_T:
        self.extractor = self.extractor.fit(shards)
        return self.plotter.render(self.extractor.plot_data)
