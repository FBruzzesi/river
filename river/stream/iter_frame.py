from __future__ import annotations

import narwhals.stable.v1 as nw  # type: ignore[import]

from river import base, stream


def iter_frame(
    X: nw.IntoFrameT, y: nw.Series | nw.IntoFrameT | None = None, **kwargs
) -> base.typing.Stream:
    """Iterates over the rows of a `pandas.DataFrame`.

    Parameters
    ----------
    X
        A dataframe of features. Supports any eager dataframe that is currently supported in Narwhals.
    y
        A series or a dataframe with one column per target.
    kwargs
        Extra keyword arguments are passed to the underlying call to `stream.iter_array`.

    Examples
    --------

    >>> import pandas as pd
    >>> import polars as pl
    >>> import pyarrow as pa
    >>> from river import stream

    >>> X_data = {
    ...     'x1': [1, 2, 3],
    ...     'x2': ['blue', 'yellow', 'yellow'],
    ... }
    >>> y_data = [True, False, False]

    >>> X_pd, y_pd = pd.DataFrame(X_data), pd.Series(y_data)
    >>> X_pl, y_pl = pl.DataFrame(X_data), pl.Series(y_data)
    >>> X_pa, y_pa = pa.table(X_data), pa.chunked_array([y_data])
    
    >>> for xi, yi in stream.iter_frame(X_pd, y_pd):
    ...     print(xi, yi)
    {'x1': 1, 'x2': 'blue'} True
    {'x1': 2, 'x2': 'yellow'} False
    {'x1': 3, 'x2': 'yellow'} False

    >>> for xi, yi in stream.iter_frame(X_pl, y_pl):
    ...     print(xi, yi)
    {'x1': 1, 'x2': 'blue'} True
    {'x1': 2, 'x2': 'yellow'} False
    {'x1': 3, 'x2': 'yellow'} False
    
    >>> for xi, yi in stream.iter_frame(X_pa, y_pa):
    ...     print(xi, yi)
    {'x1': 1, 'x2': 'blue'} True
    {'x1': 2, 'x2': 'yellow'} False
    {'x1': 3, 'x2': 'yellow'} False
    """
    X = nw.from_native(X, eager_only=True, strict=True)
    y = nw.from_native(y, eager_only=True, strict=False, allow_series=True)

    kwargs["feature_names"] = X.columns
    if isinstance(y, nw.DataFrame):
        kwargs["target_names"] = y.columns

    yield from stream.iter_array(X=X.to_numpy(), y=y if y is None else y.to_numpy(), **kwargs)
