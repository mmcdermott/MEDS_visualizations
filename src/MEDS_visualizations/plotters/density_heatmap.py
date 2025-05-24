from collections.abc import Sequence
from datetime import timedelta
from typing import Any

import plotly.graph_objects as go
import polars as pl

from .base import BasePlotter

SECONDS_IN_DAY = 60 * 60 * 24

def remove_outliers(
    plot_data: pl.DataFrame,
    outliers_thresh: float = 0.01,
    outlier_cols: list[str] | None = None,
) -> pl.DataFrame:
    ub = 1 - outliers_thresh/2
    lb = outliers_thresh/2

    if outlier_cols is None:
        outlier_cols = plot_data.columns

    filters = []

    for c in outlier_cols:
        num_col = None
        if plot_data.schema[c].is_numeric():
            num_col = pl.col(c)
        elif isinstance(plot_data.schema[c], pl.datatypes.Datetime):
            num_col = pl.col(c).dt.epoch(time_unit="d")
        else:
            continue

        filters.append(
            (num_col <= num_col.quantile(ub)) & (num_col >= num_col.quantile(lb))
        )

    return plot_data.filter(pl.all_horizontal(filters))

class DensityHeatmap(BasePlotter[pl.DataFrame, go.Figure]):
    def __init__(self, x: str, y: str, outliers_thresh: float=0.01, outlier_cols: list[str] | None = None):
        self.x = x
        self.y = y
        self.outliers_thresh = outliers_thresh
        self.outlier_cols = outlier_cols

    def normalize_plot_data(self, arr: Sequence[Any]):
        if isinstance(arr[0], timedelta):
            arr = [td.total_seconds() / SECONDS_IN_DAY for td in arr]
        return arr

    def render(self, plot_data: pl.DataFrame) -> go.Figure:
        if len(plot_data) <= 0:
            raise ValueError("Empty plot data!")

        plot_data = remove_outliers(plot_data, self.outliers_thresh, self.outlier_cols)
        if len(plot_data) <= 0:
            raise ValueError("Plot data is all outliers!")

        x = self.normalize_plot_data(plot_data[self.x].to_list())
        y = self.normalize_plot_data(plot_data[self.y].to_list())

        return go.Figure(
            data=go.Histogram2d(
                x=x,
                y=y,
                coloraxis="coloraxis",
            )
        ).update_layout(coloraxis={"colorscale": "Viridis"}, title="2D Measurement Density")
