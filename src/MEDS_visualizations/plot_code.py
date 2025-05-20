import polars as pl
from tqdm.auto import tqdm
from meds import (
    subject_id_field, prediction_time_field, time_field, code_field,
    subject_splits_filepath,
    tuning_split, train_split, held_out_split
)
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass
from functools import cached_property
from matplotlib import pyplot as plt
import numpy as np
import re
import polars.selectors as cs
from sklearn.metrics import roc_auc_score
from datetime import timedelta

SECONDS_PER_DAY = 60 * 60 * 24
MICROSECONDS_PER_DAY = 1_000_000 * SECONDS_PER_DAY

@dataclass
class TrajectorySpec:
    model_root: Path
    index_name: str
    data_root: Path

    @property
    def index_root(self) -> Path:
        return self.data_root / "index_dataframes"

    @cached_property
    def index_df(self) -> pl.DataFrame:
        return pl.read_parquet(self.index_root / self.index_name, use_pyarrow=True)

    @cached_property
    def subject_splits(self) -> pl.DataFrame:
        return pl.read_parquet(self.data_root / subject_splits_filepath, use_pyarrow=True)

    @cached_property
    def subjects_by_split(self) -> dict[str, set[int]]:
        out = {}
        for sp in self.subject_splits.select(pl.col("split").unique())["split"]:
            out[sp] = set(self.subject_splits.filter(pl.col("split") == sp)[subject_id_field])
        return out

    @cached_property
    def index_df_by_split(self) -> dict[str, pl.DataFrame]:
        out = {}
        for sp, subj in self.subjects_by_split.items():
            out[sp] = self.index_df.filter(pl.col(subject_id_field).is_in(subj))
        return out
    
    def real_data(self, split: str) -> pl.DataFrame:
        data_dir = (self.data_root / "data")
        data_fp_by_shard = {
            fp.relative_to(data_dir).with_suffix("").as_posix(): fp for fp in data_dir.rglob("*.parquet")
        }

        if any(shard.startswith(f"{split}/") for shard in data_fp_by_shard):
            data_fp_by_shard = {shard: fp for shard, fp in data_fp_by_shard.items() if shard.startswith(f"{split}/")}
            do_filter = False
        else:
            do_filter = True
            filter_expr = pl.col(subject_id_field).is_in(self.subjects_by_split[split])

        dfs = {}
        for shard, fp in data_fp_by_shard.items():
            df = pl.read_parquet(fp, use_pyarrow=True)
            if do_filter:
                df = df.filter(filter_expr)

            dfs[shard] = df

        return pl.concat(dfs.values(), how="vertical")

    def real_futures(self, split: str) -> pl.DataFrame:
        index_df = self.index_df_by_split[split].select(subject_id_field, prediction_time_field)

        return (
            self.real_data(split)
            .join(index_df, on=subject_id_field, how="inner", coalesce=True)
            .filter(pl.col(time_field) >= pl.col(prediction_time_field))
        )

    def real_histories(self, split: str) -> pl.DataFrame:
        index_df = self.index_df_by_split[split].select(subject_id_field, prediction_time_field)

        return (
            self.real_data(split)
            .join(index_df, on=subject_id_field, how="inner", coalesce=True)
            .filter(pl.col(time_field) < pl.col(prediction_time_field))
        )

    @cached_property
    def _raw_trajectories(self) -> dict[str, pl.DataFrame]:
        """The raw, unaltered trajectories, read from disk."""
        return read_trajectories(self.model_root, self.index_name)

    @cached_property
    def trajectories(self) -> dict[str, pl.DataFrame]:
        out = {}

        # Check index subject uniqueness
        n_task_samples_per_subject = self.index_df.group_by(subject_id_field).agg(pl.len().alias("n_samples"))
        if n_task_samples_per_subject.select((pl.col("n_samples") > 1).any()).item():
            raise ValueError(
                "Can't align task index to trajectories for indices with more than one sample per subject! "
                f"Got\n{n_task_samples_per_subject.filter(pl.col('n_samples') > 1)}."
            )
        idx_df = self.index_df.select(subject_id_field, prediction_time_field)

        for k, v in tqdm(list(self._raw_trajectories.items())):
            split = k.split("/")[0]
            df = v.join(idx_df, on=subject_id_field, how="left", maintain_order="left")
            cols = [c for c in v.columns if c not in {subject_id_field, prediction_time_field, "_task_sample_id", f"orig_{subject_id_field}"}]
            out[k] = df.select(subject_id_field, prediction_time_field, *cols)

        return out

    def __getitem__(self, k: str) -> Path | str | pl.DataFrame | dict[str, pl.DataFrame]:
        if hasattr(self, k):
            return getattr(self, k)

        if k == "index_df":
            return self.index_df

        if k in self.trajectories:
            return self.trajectories[k]

        raise ValueError(f"Unrecognized key {k}! Valid trajectory keys are {list(self.trajectories.keys())}")

@dataclass
class PlotSpec:
    split: str = tuning_split
    time_span: timedelta = timedelta(days=5*365)
    n_bins: int = 100

    def __post_init__(self):
        if not isinstance(self.time_span, timedelta) or self.time_span <= timedelta(0):
            raise ValueError(f"time_span must be a positive time delta; got {self.time_span}")

        if not isinstance(self.n_bins, int) or self.n_bins <= 0:
            raise ValueError(f"n_bins must be a positive integer; got {self.n_bins}")

    @property
    def span_days(self) -> float:
        return (self.time_span / timedelta(seconds=1)) / SECONDS_PER_DAY

    @cached_property
    def time_bins_days(self) -> np.ndarray:
        return np.linspace(start=0, stop=self.span_days, num=self.n_bins)

    @cached_property
    def time_bins_labels(self) -> list[str]:
        labels = ["ERROR"]
        for i in range(self.n_bins - 1):
            left = self.time_bins_days[i]
            right = self.time_bins_days[i+1]
            labels.append(f"[{left:.2f}d,{right:.2f}d)")
        labels.append(f">{self.time_bins_days[-1]}d")
        return labels

    @property
    def binned_time_delta_expr(self) -> pl.Expr:
        time_delta = pl.col(time_field) - pl.col(prediction_time_field)
        time_delta_days = time_delta.dt.total_microseconds() / MICROSECONDS_PER_DAY
        return (
            time_delta_days
            .cut(self.time_bins_days, labels=self.time_bins_labels, left_closed=True)
        )

    def matching_measurements(self, df: pl.DataFrame) -> pl.DataFrame:
        id_fields = [subject_id_field]
        if prediction_time_field in df.collect_schema():
            id_fields.append(prediction_time_field)
            
        return (
            df.filter(~pl.col(code_field).str.starts_with("TIMELINE//"))
            .group_by(*id_fields, time_field, maintain_order=True)
            .agg(pl.len().alias("n_measurements"))
            .select(
                *id_fields,
                time_field,
                pl.col("n_measurements").cum_sum().over(id_fields).alias("cumulative_measurements")
            )
        )

    def align_trajectories(self, df: pl.DataFrame) -> pl.DataFrame:
        return (
            self.matching_measurements(df)
            .select(
                subject_id_field,
                prediction_time_field,
                self.binned_time_delta_expr.alias("time_delta_bin"),
                "cumulative_measurements"
            )
            .group_by(subject_id_field, prediction_time_field, "time_delta_bin")
            .agg(pl.col("cumulative_measurements").max())
        )

    def agg_trajectories(self, T: TrajectorySpec) -> pl.DataFrame:
        rel_keys = sorted(k for k in T.trajectories if k.startswith(f"{self.split}/"))

        if not rel_keys:
            raise ValueError(f"No trajectories recorded for {self.split}")
        elif len(rel_keys) == 1:
            return self.align_trajectories(T[rel_keys[0]])
        
        aligned_dfs = [self.align_trajectories(T[k]) for k in tqdm(rel_keys, desc="Aligning Trajectories")]
        id_cols = [subject_id_field, prediction_time_field, "time_delta_bin"]

        meas_dtype = aligned_dfs[0].collect_schema()["cumulative_measurements"]
        cum_meas_list = pl.col("cumulative_measurements").cast(pl.List(meas_dtype)).fill_null([]).alias("cumulative_measurements")

        def check_nulls(df: pl.DataFrame):
            n_nulls = df.select(pl.col("cumulative_measurements").is_null().sum()).item()
            if n_nulls > 0:
                raise ValueError(f"Cumulative measurements should never be null! Got {n_nulls} nulls.")

        df = aligned_dfs[0].with_columns(cum_meas_list)
        check_nulls(df)
        
        for samp_df in tqdm(aligned_dfs[1:], desc="Joining Trajectories"):
            samp_df = samp_df.with_columns(cum_meas_list)
            check_nulls(samp_df)
            
            df = (
                df.join(samp_df, on=id_cols, how="full", coalesce=True)
                .select(
                    *id_cols,
                    pl.concat_list(
                        pl.col("cumulative_measurements").fill_null([]),
                        pl.col("cumulative_measurements_right").fill_null([]),
                    ).alias("cumulative_measurements")
                )
            )
            check_nulls(df)
        
        return self.pivot_trajectories(df)

    def pivot_trajectories(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.pivot(
            on="time_delta_bin",
            index=[subject_id_field, prediction_time_field],
            values="cumulative_measurements",
            aggregate_function=None,
            maintain_order=True,
        ).select(
            subject_id_field, prediction_time_field, *self.time_bins_labels[1:]
        )

def simple_plot(P: PlotSpec, T: TrajectorySpec):
    real = P.pivot_trajectories(P.align_trajectories(T.real_futures(tuning_split)))
    agged = P.agg_trajectories(T)

    def global_agg(df):
        df = df.drop(subject_id_field, prediction_time_field)
        return (df.mean()[0], df.std()[0])

    real_summ = global_agg(real)

    agg_agged = agged.with_columns(
        *[pl.col(c).list.mean() for c in P.time_bins_labels[1:]]
    )
    agg_summ = global_agg(agg_agged)

    # To-do: Use std for fill between

    plt.plot(
        P.time_bins_days,
        real_summ[0].to_numpy()[0],
        label="real"
    )
    plt.plot(
        P.time_bins_days,
        agg_summ[0].to_numpy()[0],
        label="agg"
    )
    plt.legend()
    plt.title("Average measurements post generation over time")
    plt.ylabel("# measurements (cumulative)")
    plt.xlabel("Time (days) since 2016-01-01")
    
def safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    try:
        return roc_auc_score(y_true, y_score)
    except ValueError:
        return np.nan
        
def get_aucs(df: pl.DataFrame) -> dict[str, float]:
    with_df = df.filter(pl.col("h"))
    without_df = df.filter(~pl.col("h"))
    
    return {
        "all": safe_auc(df["y"].to_numpy(), df["p"].to_numpy()),
        "with_hist": safe_auc(with_df["y"].to_numpy(), with_df["p"].to_numpy()),
        "no_hist": safe_auc(without_df["y"].to_numpy(), without_df["p"].to_numpy())
    }

def predictions(
    T: TrajectorySpec, tasks: list[dict], split: str = "tuning"
) -> tuple[pl.DataFrame, dict[str, float], pl.DataFrame]:
    id_fields = [subject_id_field, prediction_time_field]
    agg_exprs = {}
    event_exprs = {}
    task_names = []
    for spec in tasks:
        for timelimit in spec["within"]:
            event_expr = pl.col(code_field).str.contains(spec["code"])
            agg_expr = event_expr & ((pl.col(time_field) - pl.col(prediction_time_field)) < timelimit)
            n = f"task/{spec['name']} within {timelimit}"
            task_names.append(n)
            event_exprs[n] = event_expr.any().alias(n)
            agg_exprs[n] = agg_expr.any().alias(n)
    pfx = f"{split}/"
    rel_dfs = {
        k[len(pfx):]: df.group_by(id_fields, maintain_order=True).agg(**agg_exprs)
        for k, df in T.trajectories.items() if k.startswith(pfx)
    }
    joint_df = None
    for k, df in rel_dfs.items():
        if joint_df is None:
            joint_df = df
            continue
        joint_df = (
            joint_df.join(
                df, on=id_fields, how="inner", maintain_order="left",
                suffix=f"/{k}"
            )
            .select(
                *id_fields,
                *[(pl.col(n)+pl.col(f"{n}/{k}")).alias(n) for n in task_names],
            )
        )

    Y_score = joint_df.select(
        *id_fields,
        *[pl.col(n)/len(rel_dfs) for n in task_names]
    )

    hist = (
        T.real_histories(split)
        .group_by(id_fields, maintain_order=True)
        .agg(**event_exprs)
    )

    Y_true = (
        T
        .real_futures(split)
        .group_by(id_fields, maintain_order=True)
        .agg(**agg_exprs)
    )

    Y_score_and_true = (
        Y_score
        .join(
            Y_true, on=id_fields, how="full", suffix="/true",
            coalesce=True, maintain_order="left_right",
        )
        .join(
            hist, on=id_fields, how="full", suffix="/hist",
            coalesce=True, maintain_order="left_right",
        )
    )

    num_missing = Y_score_and_true.select(
        cs.starts_with("task/").is_null().sum().name.keep()
    )

    aucs = {}
    for task in tqdm(task_names):
        task_df = (
            Y_score_and_true
            .select(
                pl.col(task).alias("p"),
                pl.col(f"{task}/true").alias("y"),
                pl.col(f"{task}/hist").alias("h"),
            )
            .filter(pl.col("p").is_not_null() & pl.col("y").is_not_null())
        )
        aucs[task] = get_aucs(task_df)

    return Y_score_and_true, aucs, num_missing