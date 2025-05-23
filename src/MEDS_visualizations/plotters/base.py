from abc import ABC, abstractmethod
from typing import Generic

from ..types import FIG_T, PLOT_DATA_T


class BasePlotter(Generic[PLOT_DATA_T, FIG_T], ABC):
    @abstractmethod
    def render(self, plot_data: PLOT_DATA_T) -> FIG_T:
        """Render a Plotly figure from extracted data."""
        pass
