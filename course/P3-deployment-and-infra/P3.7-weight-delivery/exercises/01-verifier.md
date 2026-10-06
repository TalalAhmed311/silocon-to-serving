# Exercise 1: manifest + checksum verifier (T0)

Implement `verify(root, manifest) -> list[str]` in `verify_impl.py`. It returns one problem string per issue, and an empty list means the copy is good:

- `missing: <path>` when a listed file doesn't exist
- `size: <path> …` when the size differs (check this **before** hashing: it's free)
- `sha256: <path>` when the hash differs
- `unexpected: <path>` for files present but not listed (`manifest.json` itself excepted)

`test_verifier.py` builds a tiny registry with `weights.manifest.build`, then corrupts it four ways. `S2S_SOLUTIONS=1` runs the reference, `platform/weights/manifest.py`.

**Think:** why does the cache verify once, at install time, rather than at every model load? What does that assume about the local disk, and when would you re-verify anyway?
