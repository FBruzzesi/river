from __future__ import annotations

import random
import string
import typing

import narwhals.stable.v2 as nw
import pytest

from river import preprocessing
from river.conftest import FRAME_BACKENDS

if typing.TYPE_CHECKING:
    from typing import Any

    from river.conftest import FrameBackend

# `transform_many` mini-batching is routed through narwhals: pandas keeps the historical sparse
# fast path, while every other backend is encoded densely via `narwhals.Series.to_dummies`. These
# tests pin the cross-backend behaviour (values, native return type, pandas index).


def _categorical_batch(n: int = 40) -> tuple[dict[str, list[str]], list[dict[str, str]]]:
    """Build a reproducible batch of two categorical columns as (columns, row dicts)."""
    rng = random.Random(42)
    alphabet = list(string.ascii_lowercase[:6])
    rows = [{"c1": rng.choice(alphabet), "c2": rng.choice(alphabet)} for _ in range(n)]
    data = {"c1": [r["c1"] for r in rows], "c2": [r["c2"] for r in rows]}
    return data, rows


def _rows(native: Any) -> list[dict[str, int]]:
    """Read a native one-hot frame back into a list of ``{column: int}`` rows via narwhals."""
    frame = nw.from_native(native, eager_only=True)
    return [{col: int(row[col]) for col in frame.columns} for row in frame.iter_rows(named=True)]


CONFIGS: list[dict[str, Any]] = [
    {},
    {"drop_zeros": True},
    {"drop_first": True},
    {"drop_zeros": True, "drop_first": True},
    {"categories": {"c1": {"a", "b"}, "c2": {"c", "d"}}},
    {"categories": {"c1": {"a", "b"}, "c2": {"c", "d"}}, "drop_zeros": True},
]


@pytest.mark.parametrize("config", CONFIGS, ids=lambda c: str(c))
def test_transform_many_is_backend_agnostic(
    frame_backend: FrameBackend, config: dict[str, Any]
) -> None:
    """`transform_many` must yield identical values regardless of the input backend."""
    data, _ = _categorical_batch()

    pandas = FRAME_BACKENDS["pandas"]()
    reference = preprocessing.OneHotEncoder(**config)
    reference.learn_many(pandas.frame(data))
    expected = _rows(reference.transform_many(pandas.frame(data)))

    encoder = preprocessing.OneHotEncoder(**config)
    encoder.learn_many(frame_backend.frame(data))
    got = _rows(encoder.transform_many(frame_backend.frame(data)))

    assert got == expected


@pytest.mark.parametrize("drop_first", [False, True])
def test_transform_many_matches_transform_one(
    frame_backend: FrameBackend, drop_first: bool
) -> None:
    """Each row of `transform_many` must agree with `transform_one` on every backend.

    Uses ``drop_zeros=False`` so the full (sparse-friendly) column set is materialised; missing
    keys are treated as zeros, matching the encoder's own dense/sparse padding.
    """
    data, rows = _categorical_batch()

    encoder = preprocessing.OneHotEncoder(drop_zeros=False, drop_first=drop_first)
    encoder.learn_many(frame_backend.frame(data))

    many = _rows(encoder.transform_many(frame_backend.frame(data)))
    for row, many_row in zip(rows, many):
        one = encoder.transform_one(row)
        keys = set(one) | set(many_row)
        for key in keys:
            assert one.get(key, 0) == many_row.get(key, 0), key


def test_transform_many_returns_native_backend(frame_backend: FrameBackend) -> None:
    """`transform_many` returns the input backend's native frame type."""
    data, _ = _categorical_batch()

    encoder = preprocessing.OneHotEncoder(drop_zeros=True)
    encoder.learn_many(frame_backend.frame(data))
    out = encoder.transform_many(frame_backend.frame(data))

    assert type(out).__module__.split(".")[0] == frame_backend.name


def test_transform_many_pandas_is_sparse_others_dense() -> None:
    """Pandas keeps the sparse fast path; non-pandas backends emit dense integer columns."""
    import pandas as pd

    data, _ = _categorical_batch()

    pandas_out = preprocessing.OneHotEncoder(drop_zeros=True).transform_many(pd.DataFrame(data))
    assert all(isinstance(dtype, pd.SparseDtype) for dtype in pandas_out.dtypes)

    pl = pytest.importorskip("polars")
    polars_out = preprocessing.OneHotEncoder(drop_zeros=True).transform_many(pl.DataFrame(data))
    assert all(dtype.is_integer() for dtype in polars_out.schema.dtypes())


def test_transform_many_preserves_pandas_index() -> None:
    """The pandas fast path keeps the input index on the encoded frame."""
    import pandas as pd

    index = [100, 200, 300]
    X = pd.DataFrame({"c1": ["a", "b", "a"], "c2": ["x", "y", "z"]}, index=index)

    encoder = preprocessing.OneHotEncoder(drop_zeros=False)
    encoder.learn_many(X)
    out = encoder.transform_many(X)

    assert list(out.index) == index


def test_transform_many_explicit_categories_restrict_columns(frame_backend: FrameBackend) -> None:
    """Explicit categories bound the output columns to the provided ones, like scikit-learn."""
    data, _ = _categorical_batch()
    categories = {"c1": {"a", "b"}, "c2": {"c", "d"}}

    encoder = preprocessing.OneHotEncoder(categories=categories)
    encoder.learn_many(frame_backend.frame(data))
    out = nw.from_native(encoder.transform_many(frame_backend.frame(data)), eager_only=True)

    assert set(out.columns) == {"c1_a", "c1_b", "c2_c", "c2_d"}
