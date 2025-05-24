from collections.abc import Sequence
from datetime import timedelta
from typing import Any

import plotly.graph_objects as go
import polars as pl

from .base import BasePlotter

SECONDS_IN_DAY = 60 * 60 * 24


class DensityHeatmap(BasePlotter[pl.DataFrame, go.Figure]):
    def __init__(self, x: str, y: str):
        self.x = x
        self.y = y

    def normalize_plot_data(self, arr: Sequence[Any]):
        if isinstance(arr[0], timedelta):
            arr = [td.total_seconds() / SECONDS_IN_DAY for td in arr]
        return arr

    def render(self, plot_data: pl.DataFrame) -> go.Figure:
        x = self.normalize_plot_data(plot_data[self.x].to_list())
        y = self.normalize_plot_data(plot_data[self.y].to_list())

        return go.Figure(
            data=go.Histogram2d(
                x=x,
                y=y,
                coloraxis="coloraxis",
            )
        ).update_layout(coloraxis={"colorscale": "Viridis"}, title="2D Measurement Density")
