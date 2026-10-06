# Model presets

Each file is the subset of a model's Hugging Face `config.json` that the calculator reads.

**Status: UNVERIFIED.** The author wrote these from memory of the public configs. Before you cite a number computed from one, replace it with the real `config.json`:

```bash
huggingface-cli download <repo> config.json --local-dir /tmp/cfg
```

Then set `"_verified": "<repo>@<revision> YYYY-MM-DD"` in the preset.
