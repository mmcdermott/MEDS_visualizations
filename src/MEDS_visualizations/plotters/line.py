import plotly.graph_objects as go
import polars as pl

from .base import BasePlotter


class Line(BasePlotter[pl.DataFrame, go.Figure]):
    x: str
    y: str

    def __init__(self, x: str, y: str):
        self.x = x
        self.y = y

    def render(self, plot_data: pl.DataFrame) -> go.Figure:
        return go.Figure(
            go.Scatter(x=plot_data[self.x].to_list(), y=plot_data[self.y].to_list(), mode="lines")
        ).update_layout(title=f"{self.y} over {self.x}", xaxis_title=self.x, yaxis_title=self.y)
