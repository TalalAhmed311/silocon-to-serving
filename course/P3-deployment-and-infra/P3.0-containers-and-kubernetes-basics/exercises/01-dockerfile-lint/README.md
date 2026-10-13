# Exercise 1 — A beginner Dockerfile linter (easy, T0)

Write `lint(dockerfile_text)` in `dockerlint.py`. It reads a Dockerfile as text and returns findings such as `D3 line 2: ...`. There are four rules: no tag (D1), the `latest` tag (D2), `COPY . .` before the dependency install (D3, the layer-cache mistake from step 3), and no non-root `USER` (D4).

This is the beginner version. [P3.1](../../../P3.1-gpu-containers/README.md) has a second linter, `examples/02_dockerfile_policy.py`, with different rules: digests in strict mode, `ADD` from a URL, secrets in `ENV`, and model weights in the image. Read it after you pass this one; do not copy from it, because the rules differ (it has no cache-order rule and no `latest` rule).

No Docker is needed. You are only parsing text.

```bash
uv run pytest course/P3-deployment-and-infra/P3.0-containers-and-kubernetes-basics/exercises/01-dockerfile-lint
```

Then point your linter at the real files in `env/`:

```bash
uv run python -c "import sys; sys.path.insert(0, 'course/P3-deployment-and-infra/P3.0-containers-and-kubernetes-basics/exercises/solutions/01-dockerfile-lint'); import dockerlint, pathlib; [print(p, dockerlint.lint(p.read_text())) for p in sorted(pathlib.Path('env').glob('Dockerfile.*'))]"
```

(Use your own folder instead of `solutions/` once your version passes.) Which findings are real and which are a limit of a four-rule linter?
