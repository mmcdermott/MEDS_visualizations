from __future__ import annotations

from typing import ClassVar

import polars as pl
import polars.selectors as cs

from .base import BaseExtractor
from .meds_pl_mixin import MEDSColumns


class CodeFrequency(BaseExtractor, MEDSColumns):
    """An extractor that counts the frequency of each code in a given dataset."""

    as_proportions: bool
    sort_by: str
    code_counts: pl.DataFrame | None
    n_subjects: int | None
    n_events: int | None
    n_measurements: int | None
    n_codes: int | None

    CODE_AGG_EXPRS: ClassVar[dict[str, pl.Expr]] = {
        "code/n_occurrences": pl.len().alias("code/n_occurrences"),
        "code/n_subjects": MEDSColumns.SUBJECT_ID.n_unique().alias("code/n_subjects"),
        "code/n_events": MEDSColumns.TIME.n_unique().alias("code/n_events"),
    }

    GLOBAL_AGG_EXPRS: ClassVar[dict[str, pl.Expr]] = {
        "n_subjects": MEDSColumns.SUBJECT_ID.n_unique().alias("n_subjects"),
        "n_events": MEDSColumns.TIME.n_unique().alias("n_events"),
        "n_measurements": pl.len().alias("n_measurements"),
    }

    def __init__(self, as_proportions: bool = True, sort_by: str = "n_occurrences"):
        super().__init__()
        self.as_proportions = as_proportions
        self.sort_by = sort_by
        self.code_counts = None
        self.n_subjects = None
        self.n_events = None
        self.n_measurements = None

    def _extract(self, shard: pl.DataFrame, code_metadata: pl.DataFrame | None = None):
        self.code_counts = shard.group_by("code").agg(**self.CODE_AGG_EXPRS)
        global_aggs = shard.select(**self.GLOBAL_AGG_EXPRS)

        for agg_name in self.GLOBAL_AGG_EXPRS:
            setattr(self, agg_name, global_aggs[agg_name].item())

    def _merge(self, other: CodeFrequency, code_metadata: pl.DataFrame | None = None):
        for agg_name in self.GLOBAL_AGG_EXPRS:
            setattr(self, agg_name, getattr(self, agg_name) + getattr(other, agg_name))

        self.code_counts = (
            pl.concat([self.code_counts, other.code_counts], how="vertical")
            .group_by(self.CODE)
            .agg(cs.starts_with("code/").sum().name.keep())
        )

    @property
    def proportion_exprs(self) -> dict[str, pl.Expr]:
        return {
            "code/n_occurrences": (pl.col("code/n_occurrences") / self.n_measurements),
            "code/n_subjects": (pl.col("code/n_subjects") / self.n_subjects),
            "code/n_events": (pl.col("code/n_events") / self.n_events),
        }

    @property
    def plot_data(self) -> pl.DataFrame:
        if self.as_proportions:
            df = self.code_counts.with_columns(**self.proportion_exprs)
        else:
            df = self.code_counts

        return df.select(
            self.CODE,
            cs.starts_with("code/").name.map(lambda x: x.removeprefix("code/")),
        ).sort(by=self.sort_by, descending=True)
