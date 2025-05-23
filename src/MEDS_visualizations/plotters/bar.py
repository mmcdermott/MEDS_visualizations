import plotly.graph_objects as go
import polars as pl

from .base import BasePlotter


class Bar(BasePlotter):
    top_k: int
    x_col: str | None
    y_cols: list[str] | None

    def __init__(self, top_k: int = 20, x_col: str | None = None, y_cols: list[str] | None = None):
        self.top_k = top_k
        self.x_col = x_col
        self.y_cols = y_cols

    def render(self, plot_data: pl.DataFrame) -> go.Figure:
        # Default to first column for X and all others for Y
        columns = plot_data.columns
        if not columns:
            raise ValueError("Empty dataframe passed to BarPlotter")

        x_col = self.x_col or columns[0]
        y_cols = self.y_cols or [c for c in columns if c != x_col]

        df = plot_data.head(self.top_k)

        fig = go.Figure()
        for y_col in y_cols:
            fig.add_trace(go.Bar(x=df[x_col].to_list(), y=df[y_col].to_list(), name=y_col))

        fig.update_layout(
            title=f"Bar Plot of {', '.join(y_cols)} by {x_col}",
            xaxis_title=x_col,
            yaxis_title="Values",
            barmode="group",
        )
        return fig
