# Exercise 4 — Reproducible build (T0)

Build `env/Dockerfile.platform` twice, with BuildKit, from a clean cache, using `SOURCE_DATE_EPOCH=$(git log -1 --format=%ct)` and `--build-arg SOURCE_DATE_EPOCH`, plus `docker buildx build --output type=image,rewrite-timestamp=true`. Compare the two image digests.

If they differ, find the source of nondeterminism. Usual suspects: timestamps, `apt-get` mirror drift (pin package versions, or snapshot.debian.org), and unpinned base images (exercise 1). Write down what you had to change. A reproducible image is one you can *prove* you rebuilt from source.
