# Exercise 3 — Broadcasting (medium)

Implement in `broadcast.py`, with no loops:

```python
normalize_rows(x, eps=1e-5)   # (x - row mean) / sqrt(row variance + eps)
center_columns(x)             # subtract each column's mean
```

`normalize_rows` is the core of layer normalisation. Work out the shapes first. For `x` of shape `(rows, cols)`, `x.mean(axis=1)` has shape `(rows,)`, which would line up against the **columns** (broadcasting aligns from the right). Pass `keepdims=True` to get `(rows, 1)`, which stretches across the columns instead.

**Test:** `uv run pytest course/R-readiness/R.4-python-and-numpy/exercises/03-broadcast`
