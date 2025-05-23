from typing import ClassVar

import polars as pl
from meds import DataSchema


class MEDSColumns:
    """A Mixin class to add polars column expressions for the MEDS schema as class variables."""

    CODE: ClassVar[pl.Expr] = pl.col(DataSchema.code_name)
    SUBJECT_ID: ClassVar[pl.Expr] = pl.col(DataSchema.subject_id)
    TIME: ClassVar[pl.Expr] = pl.col(DataSchema.time_name)
