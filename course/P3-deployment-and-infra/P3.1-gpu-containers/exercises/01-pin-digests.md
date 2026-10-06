# Exercise 1 — Pin every base image by digest (T0)

1. For each `FROM` in `env/Dockerfile.*` (and the `COPY --from=ghcr.io/astral-sh/uv:0.5` images), resolve the digest:
   `docker buildx imagetools inspect python:3.12-slim --format '{{json .Manifest.Digest}}'`, or `docker pull` + `docker inspect --format '{{index .RepoDigests 0}}'`.
2. Rewrite each line as `FROM image:tag@sha256:<digest>`. Keep the tag for readability; the digest is what's enforced.
3. `test_policy.py` runs the linter in `--strict` mode over `env/`. It **fails until you pin them**. That is the point.
4. Add a CI step (or a Renovate/Dependabot config) that bumps digests deliberately. Write two sentences on why "deliberately" matters.
