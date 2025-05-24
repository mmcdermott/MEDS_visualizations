from __future__ import annotations

import polars as pl

from .base import BaseExtractor
from .meds_pl_mixin import MEDSColumns


class ActiveSubjectsOverTime(BaseExtractor[pl.DataFrame, pl.DataFrame], MEDSColumns):
    """Estimates active subject count over a uniform time grid based on start/end times.

    TODO(mmd): Exclude birth events?

    Args:
        time_resolution: Either a polars duration string (e.g., '1d', '6h', '10m') to produce a uniform time
            grid at that frequency, the string "full" to use all unique time bounds (to see full granularity
            increases), or an integer as the number of unique time points to use in the grid (which will be
            chosen randomly from the unique time boundS).

    Examples:
        >>> shard = pl.DataFrame({
        ...     "subject_id": [
        ...         1, 1, 1, 1, 1, 1,
        ...         2, 2, 2, 2,
        ...     ],
        ...     "time": [
        ...         datetime(2020, 1, 1),
        ...         datetime(2020, 1, 1),
        ...         datetime(2020, 1, 2),
        ...         datetime(2020, 1, 3),
        ...         datetime(2020, 1, 5),
        ...         datetime(2020, 1, 8),
        ...         datetime(2020, 1, 4),
        ...         datetime(2020, 1, 4),
        ...         datetime(2020, 1, 4),
        ...         datetime(2020, 1, 10),
        ...     ],
        ... })
        >>> extractor = ActiveSubjectsOverTime(time_resolution="full")
        >>> _ = extractor.fit(shard)
        >>> extractor.plot_data
        shape: (4, 2)
        ┌─────────────────────┬─────────────────┐
        │ Time                ┆ Active Subjects │
        │ ---                 ┆ ---             │
        │ datetime[μs]        ┆ i64             │
        ╞═════════════════════╪═════════════════╡
        │ 2020-01-01 00:00:00 ┆ 1               │
        │ 2020-01-04 00:00:00 ┆ 2               │
        │ 2020-01-08 00:00:00 ┆ 1               │
        │ 2020-01-10 00:00:00 ┆ 0               │
        └─────────────────────┴─────────────────┘
    """

    time_resolution: str | int
    time_bounds: pl.DataFrame | None
    _seed: int

    def __init__(self, time_resolution: str | int = "1y", seed: int = 0):
        super().__init__()
        self.time_resolution = time_resolution
        self.time_bounds = None
        self._seed = seed

    def _extract(self, shard: pl.DataFrame):
        self.time_bounds = (
            shard.group_by(self.SUBJECT_ID)
            .agg(self.TIME.min().alias("min_time"), self.TIME.max().alias("max_time"))
            .select("min_time", "max_time")
        )

    def _merge(self, other: ActiveSubjectsOverTime, code_metadata: pl.DataFrame | None = None):
        self.time_bounds = pl.concat([self.time_bounds, other.time_bounds], how="vertical")

    @property
    def all_time_bounds(self) -> pl.DataFrame:
        """Returns the time bounds for all subjects in the dataset."""
        if self.time_bounds is None:
            raise ValueError("Time bounds have not been computed yet.")
        return pl.concat([self.time_bounds["min_time"], self.time_bounds["max_time"]]).unique()

    @property
    def min_time(self) -> pl.Series:
        """Returns the minimum time bound across all subjects."""
        if self.time_bounds is None:
            raise ValueError("Time bounds have not been computed yet.")
        return self.time_bounds["min_time"].min()

    @property
    def max_time(self) -> pl.Series:
        """Returns the maximum time bound across all subjects."""
        if self.time_bounds is None:
            raise ValueError("Time bounds have not been computed yet.")
        return self.time_bounds["max_time"].max()

    @property
    def plot_data(self) -> pl.DataFrame:
        match self.time_resolution:
            case "full":
                # Use all unique time bounds
                grid = self.all_time_bounds.sort()
            case int() as n_samples:
                # Randomly sample unique time bounds
                self._time_grid = self.all_time_bounds.sample(n=n_samples, random_state=self._seed)
            case str():
                # Use polars duration string to create a uniform time grid
                if not pl.duration(self.time_resolution).is_valid():
                    raise ValueError(f"Invalid time resolution: {self.time_resolution}")

                grid = pl.datetime_range(
                    low=self.min_time, high=self.max_time, interval=self.time_resolution, eager=True
                )

        # For each grid time t, compute active subjects
        active_counts = [
            ((self.time_bounds["min_time"] <= t) & (self.time_bounds["max_time"] > t)).sum() for t in grid
        ]

        return pl.DataFrame(
            {
                "Time": grid,
                "Active Subjects": active_counts,
            }
        )
