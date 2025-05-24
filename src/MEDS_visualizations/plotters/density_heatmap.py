import plotly.graph_objects as go
import polars as pl

from .base import BasePlotter


class DensityHeatmap(BasePlotter[pl.DataFrame, go.Figure]):
    def __init__(self, x: str, y: str, z: str = "n_measurements"):
        self.x = x
        self.y = y
        self.z = z

    def render(self, plot_data: pl.DataFrame) -> go.Figure:
        return go.Figure(
            data=go.Histogram2d(
                x=plot_data[self.x].to_list(),
                y=plot_data[self.y].to_list(),
                z=plot_data[self.z].to_list(),
                coloraxis="coloraxis",
            )
        ).update_layout(coloraxis={"colorscale": "Viridis"}, title="2D Measurement Density")
