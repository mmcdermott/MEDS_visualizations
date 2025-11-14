from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from .base import BaseExtractor
from .meds_pl_mixin import MEDSColumns

if TYPE_CHECKING:
    from datetime import datetime


class MeasurementsInDuration(BaseExtractor[pl.DataFrame, pl.DataFrame], MEDSColumns):
    """Extracts data to visualize what duration of time is captured by a certain measurement window.

    This extractor is used to visualize the duration of time captured by a certain measurement window from a
    given index date across subjects.

    Args:
        index_date: The date from which to calculate the measurement window.
        min_offset: The minimum offset from the index date to include in the output. If None, no minimum
            offset is applied.
        max_offset: The maximum offset from the index date to include in the output. If None, no maximum
            offset is applied.
        filter_to_active_subjects: If True, only include subjects who have measurements on both sides of the
            index date are included. If False, all subjects are included.

    Attributes:
        index_date: The date from which to calculate the measurement window.
        min_offset: The minimum offset from the index date to include in the output.
        max_offset: The maximum offset from the index date to include in the output.
        filter_to_active_subjects: If True, only include subjects who have measurements on both sides of the
            index date are included.
        offset_deltas: A DataFrame containing the offsets and durations of measurements for each subject.

    Yields Plot Data:
        This extractor yields a DataFrame containing the offsets and durations of measurements for each
        subject-measurement. There will be two columns: "offset" and "delta". The "offset" column contains the
        number of measurements from the last measurement observed less than or equal to the index date, under
        the original shard ordering, and the "delta" column contains the duration from the time at that
        measurement to the index date (which may be zero). Both columns could be negative, zero, or positive.

    Examples:
        >>> _ = pl.Config.set_tbl_rows(15)
        >>> shard = pl.DataFrame({
        ...     "subject_id": [
        ...         1, 1, 1, 1, 1, 1,
        ...         2, 2, 2, 2,
        ...         3, 3,
        ...         4, 4,
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
        ...         datetime(1980, 1, 1),
        ...         datetime(1990, 1, 1),
        ...         datetime(2023, 1, 1),
        ...         datetime(2023, 1, 10),
        ...     ],
        ... })
        >>> shard
        shape: (14, 2)
        ┌────────────┬─────────────────────┐
        │ subject_id ┆ time                │
        │ ---        ┆ ---                 │
        │ i64        ┆ datetime[μs]        │
        ╞════════════╪═════════════════════╡
        │ 1          ┆ 2020-01-01 00:00:00 │
        │ 1          ┆ 2020-01-01 00:00:00 │
        │ 1          ┆ 2020-01-02 00:00:00 │
        │ 1          ┆ 2020-01-03 00:00:00 │
        │ 1          ┆ 2020-01-05 00:00:00 │
        │ 1          ┆ 2020-01-08 00:00:00 │
        │ 2          ┆ 2020-01-04 00:00:00 │
        │ 2          ┆ 2020-01-04 00:00:00 │
        │ 2          ┆ 2020-01-04 00:00:00 │
        │ 2          ┆ 2020-01-10 00:00:00 │
        │ 3          ┆ 1980-01-01 00:00:00 │
        │ 3          ┆ 1990-01-01 00:00:00 │
        │ 4          ┆ 2023-01-01 00:00:00 │
        │ 4          ┆ 2023-01-10 00:00:00 │
        └────────────┴─────────────────────┘

    At its base, a `MeasurementsInDuration` extractor just needs an index date from which measurement deltas
    are calculated and returns a dataframe of all measurement offsets and what times are reached at those
    offsets across all subjects in the dataset who are active at the index date:

        >>> extractor = MeasurementsInDuration(index_date=datetime(2020, 1, 4))
        >>> extractor.fit(shard).plot_data
        shape: (10, 2)
        ┌────────┬─────────────────────┐
        │ offset ┆ time_reached        │
        │ ---    ┆ ---                 │
        │ i32    ┆ datetime[μs]        │
        ╞════════╪═════════════════════╡
        │ -3     ┆ 2020-01-01 00:00:00 │
        │ -2     ┆ 2020-01-01 00:00:00 │
        │ -1     ┆ 2020-01-02 00:00:00 │
        │ 0      ┆ 2020-01-03 00:00:00 │
        │ 1      ┆ 2020-01-05 00:00:00 │
        │ 2      ┆ 2020-01-08 00:00:00 │
        │ -2     ┆ 2020-01-04 00:00:00 │
        │ -1     ┆ 2020-01-04 00:00:00 │
        │ 0      ┆ 2020-01-04 00:00:00 │
        │ 1      ┆ 2020-01-10 00:00:00 │
        └────────┴─────────────────────┘
        >>> extractor = MeasurementsInDuration(index_date=datetime(1985, 1, 4))
        >>> extractor.fit(shard).plot_data
        shape: (2, 2)
        ┌────────┬─────────────────────┐
        │ offset ┆ time_reached        │
        │ ---    ┆ ---                 │
        │ i32    ┆ datetime[μs]        │
        ╞════════╪═════════════════════╡
        │ 0      ┆ 1980-01-01 00:00:00 │
        │ 1      ┆ 1990-01-01 00:00:00 │
        └────────┴─────────────────────┘

    You can constrain the extractor to only include measurements within a certain offset range from the index
    date:

        >>> extractor = MeasurementsInDuration(index_date=datetime(2020, 1, 4), min_offset=-1, max_offset=1)
        >>> extractor.fit(shard).plot_data
        shape: (6, 2)
        ┌────────┬─────────────────────┐
        │ offset ┆ time_reached        │
        │ ---    ┆ ---                 │
        │ i32    ┆ datetime[μs]        │
        ╞════════╪═════════════════════╡
        │ -1     ┆ 2020-01-02 00:00:00 │
        │ 0      ┆ 2020-01-03 00:00:00 │
        │ 1      ┆ 2020-01-05 00:00:00 │
        │ -1     ┆ 2020-01-04 00:00:00 │
        │ 0      ┆ 2020-01-04 00:00:00 │
        │ 1      ┆ 2020-01-10 00:00:00 │
        └────────┴─────────────────────┘

    You can also allow it to include patients who are not active at the index date (though by default they are
    not included):

        >>> extractor = MeasurementsInDuration(
        ...     index_date=datetime(2020, 1, 4), filter_to_active_subjects=False
        ... )
        >>> extractor.fit(shard).plot_data
        shape: (14, 2)
        ┌────────┬─────────────────────┐
        │ offset ┆ time_reached        │
        │ ---    ┆ ---                 │
        │ i32    ┆ datetime[μs]        │
        ╞════════╪═════════════════════╡
        │ -3     ┆ 2020-01-01 00:00:00 │
        │ -2     ┆ 2020-01-01 00:00:00 │
        │ -1     ┆ 2020-01-02 00:00:00 │
        │ 0      ┆ 2020-01-03 00:00:00 │
        │ 1      ┆ 2020-01-05 00:00:00 │
        │ 2      ┆ 2020-01-08 00:00:00 │
        │ -2     ┆ 2020-01-04 00:00:00 │
        │ -1     ┆ 2020-01-04 00:00:00 │
        │ 0      ┆ 2020-01-04 00:00:00 │
        │ 1      ┆ 2020-01-10 00:00:00 │
        │ -1     ┆ 1980-01-01 00:00:00 │
        │ 0      ┆ 1990-01-01 00:00:00 │
        │ 1      ┆ 2023-01-01 00:00:00 │
        │ 2      ┆ 2023-01-10 00:00:00 │
        └────────┴─────────────────────┘
    """

    index_date: datetime
    min_offset: int | None
    max_offset: int | None
    filter_to_active_subjects: bool
    offset_deltas: pl.DataFrame | None

    def __init__(
        self,
        index_date: datetime,
        min_offset: int | None = None,
        max_offset: int | None = None,
        filter_to_active_subjects: bool = True,
    ):
        super().__init__()
        self.index_date = index_date
        self.min_offset = min_offset
        self.max_offset = max_offset
        self.filter_to_active_subjects = filter_to_active_subjects
        self.offset_deltas = None

    def _extract(self, shard: pl.DataFrame, code_metadata: pl.DataFrame | None = None):
        if self.filter_to_active_subjects:
            shard = shard.filter(
                (self.TIME.max().over(self.SUBJECT_ID) >= self.index_date)
                & (self.TIME.min().over(self.SUBJECT_ID) <= self.index_date)
            )

        row_idx = "__idx"

        shard = (
            shard.filter(self.TIME.is_not_null())
            .with_row_index(row_idx)
            .with_columns(pl.col(row_idx).cast(pl.Int32).name.keep())
        )

        with_deltas = shard.with_columns((self.TIME - self.index_date).alias("delta"))

        subj_indices = with_deltas.group_by(self.SUBJECT_ID).agg(
            pl.col(row_idx).first().alias("first_row_per_subj"),
            (self.TIME.search_sorted(self.index_date, side="right").first().cast(pl.Int32)).alias(
                "per_subj_index_offset"
            ),
        )

        self.offset_deltas = with_deltas.join(
            subj_indices, on=self.SUBJECT_ID, how="left", maintain_order="left"
        ).select(
            "delta",
            (pl.col(row_idx) - pl.col("first_row_per_subj") - pl.col("per_subj_index_offset") + 1).alias(
                "offset"
            ),
        )

        if self.min_offset is not None:
            self.offset_deltas = self.offset_deltas.filter(pl.col("offset") >= self.min_offset)
        if self.max_offset is not None:
            self.offset_deltas = self.offset_deltas.filter(pl.col("offset") <= self.max_offset)

    def _merge(self, other: MeasurementsInDuration, code_metadata: pl.DataFrame | None = None):
        self.offset_deltas = pl.concat([self.offset_deltas, other.offset_deltas], how="vertical")

    @property
    def plot_data(self) -> pl.DataFrame:
        return self.offset_deltas.select("offset", (pl.col("delta") + self.index_date).alias("time_reached"))
