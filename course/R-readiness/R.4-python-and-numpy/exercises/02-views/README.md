# Exercise 2 — Views, not copies (easy)

Implement in `views.py`. Four functions must return **views**: `row`, `column`, `transposed`, `every_other_row`. They share the original buffer, so no bytes move and a write through the view changes the original. `column_copy` must do the opposite and return an independent, contiguous copy.

Use `a.strides` and `np.shares_memory(a, b)` in a Python shell to look before you code:

```python
a = np.arange(12, dtype=np.float32).reshape(3, 4)
a.strides        # (16, 4): next row = +16 bytes, next column = +4 bytes
a[:, 2].strides  # (16,)
```

This is the same idea as the strides in P0.1: an array is a buffer plus a shape plus strides.

**Test:** `uv run pytest course/R-readiness/R.4-python-and-numpy/exercises/02-views`
