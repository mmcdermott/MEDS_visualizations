from typing import TypeVar

import plotly.graph_objects as go
import polars as pl

DF_T = TypeVar("DF_T", default=pl.DataFrame)
FIG_T = TypeVar("FIG_T", default=go.Figure)
PLOT_DATA_T = TypeVar("PLOT_DATA_T", default=pl.DataFrame)
