# P3.0 — Containers and Kubernetes from zero

| | |
|---|---|
| **Tier** | ![T0](https://img.shields.io/badge/tier-T0%20laptop-2ea44f) Docker and kind on a laptop, no GPU, no cloud |
| **Time** | ≈30 min reading + ≈5 h hands-on |
| **Prerequisites** | [R.3](../../R-readiness/R.3-terminal-git-build/README.md) (terminal), P2.1 (you have run a server). [R.7](../../R-readiness/R.7-cloud-account-and-cost-safety/README.md) is not needed here: nothing in this module touches a cloud |
| **You will build** | a Dockerfile linter, a Deployment and Service YAML, and a local kind cluster running it |

Already know Docker and the basics of Kubernetes (Pods, Deployments, Services, `kubectl apply`)? Answer the lesson's checkpoints. If you get them all right, skip to [P3.1](../P3.1-gpu-containers/README.md).

## Learning objectives

1. Explain what an image, a layer and a container are, and why a container is not a virtual machine.
2. Read and write a short Dockerfile, and order it so the layer cache helps.
3. Publish ports, mount volumes and pass environment variables with `docker run`.
4. Say why a tag can move and a digest cannot.
5. Explain Kubernetes as a loop that drives actual state toward desired state.
6. Write a Deployment and a Service, and connect them with labels and selectors.
7. Use `kubectl apply/get/describe/logs` on a kind cluster, and read a resource request such as `nvidia.com/gpu: 1`.

## Why this matters

[P3.1](../P3.1-gpu-containers/README.md) puts vLLM in a container and [P3.3](../P3.3-kubernetes-and-gpu-operator/README.md) runs it on Kubernetes with GPUs. Both assume you already know what a layer is and what a Deployment does. This module teaches only that, with no GPU.

---

## 1. Works on my machine

A program needs more than its code: an interpreter, libraries, system packages. Two machines differ in some of these, and the same code behaves differently. The fix is to ship the whole user space together with the program, once, and run it the same way everywhere.

## 2. Images, layers and containers

An **image** is a stack of read-only **layers**, each a set of file changes. A **container** is a running process that sees those layers plus one thin writable layer on top. When the container is removed, its writable layer goes with it. Layers are shared: ten images built from the same base store that base once.

```bash
docker pull python:3.12-slim      # downloads the layers (the tag is an example)
docker history python:3.12-slim   # shows them, with sizes
docker run --rm python:3.12-slim python -c "print('hi')"
```

## 3. A Dockerfile

```dockerfile
FROM python:3.12-slim               # the base layers
WORKDIR /app
COPY requirements.txt .             # copy only the dependency list
RUN pip install -r requirements.txt # one layer: the installed packages
COPY . .                            # copy your code (changes often)
USER 10001                          # do not run as root
CMD ["python", "-m", "app"]         # what runs at start (not a layer of files)
```

Each `RUN`, `COPY` and `ADD` makes a layer. Docker caches a layer when its instruction and inputs are unchanged. **When a layer changes, every layer after it is rebuilt.** So put what changes rarely at the top, and your code at the bottom. `COPY . .` before `pip install` makes every code edit reinstall every package. The lesson's simulator shows it.

## 4. Isolation: a container is not a VM

A virtual machine runs its own kernel on emulated hardware. A container runs on the **host's kernel**. Two Linux features make it look separate:

- **namespaces** limit what a process can see: its own process list, network, hostname and file tree;
- **cgroups** limit what it can use: memory, CPU, and so on.

Fast to start, small, and sharing one kernel. The sharing is also why container isolation is weaker than a VM's, and why P3.8 covers hardening.

## 5. Ports, volumes and environment

```bash
docker run --rm -p 8000:8000 -v $PWD/data:/data -e MODE=dev myimage
#               host:container  host dir:dir      variable
```

- `-p host:container` publishes a container port on the host. Without it, the host cannot reach the server.
- `-v` mounts a host directory or named volume, so data outlives the container.
- `-e` sets an environment variable, the usual way to configure a container. Do not bake secrets into an image.

## 6. Registries, tags and digests

A **registry** stores images. `python:3.12-slim` is a name plus a **tag**. A tag is a label that its owner can move to a new image, so the same Dockerfile can build a different thing next month. A **digest** (`@sha256:...`) names exact content and never changes. The course pins by digest where it matters; see [`env/`](../../../env/README.md) and [P3.1 exercise 1](../P3.1-gpu-containers/exercises/01-pin-digests.md).

## 7. Why one machine is not enough

One server fails, runs out of GPU, and needs updates without downtime. **Kubernetes** runs containers across many machines. Its core idea is a **desired-state loop**. You declare "run 3 copies of this". A **controller** compares the actual state with the desired state, and acts until they match: start a pod if one died, remove one if there are too many. You never say how to restart.

## 8. Pods, Deployments and Services

- A **Pod** is the smallest unit: one or more containers that run together on one node.
- A **Deployment** declares "keep N identical Pods" and handles rolling updates.
- A **Service** gives a set of Pods one stable address, because Pod IPs change.
- **Labels** are key/value tags on objects. A **selector** picks objects by label. The Deployment finds its Pods by selector, and the Service finds the same Pods by selector. A mismatch means no Pods, or a Service with no endpoints.

## 9. kubectl and kind

```bash
kubectl apply -f deployment.yaml     # declare the desired state
kubectl get pods                     # what exists now
kubectl describe pod <name>          # events: why is it Pending?
kubectl logs <name>                  # the container's output
kubectl port-forward service/echo 8080:80
```

**kind** runs a Kubernetes cluster as Docker containers on your laptop, which makes it a free place to learn. The repo's config is [`platform/deploy/kind/kind-config.yaml`](../../../platform/deploy/kind/kind-config.yaml): a control plane and two workers.

## 10. Requests, limits and GPUs

A container asks for resources:

```yaml
resources:
  requests: { cpu: 100m, memory: 128Mi }    # what the scheduler reserves
  limits:   { cpu: 500m, memory: 256Mi }    # the ceiling the container may use
```

The scheduler places a Pod only on a node with enough unreserved resources. Over its memory limit, a container is killed. A GPU is "just another resource": a node advertises `nvidia.com/gpu: 1`, and a Pod asks for it with `limits: { nvidia.com/gpu: 1 }`. Nothing else about Pods changes. How a node learns it has a GPU is [P3.3](../P3.3-kubernetes-and-gpu-operator/README.md).

---

## Exercises

| # | Exercise | Difficulty | Check |
|---|---|---|---|
| 1 | [Dockerfile linter](exercises/01-dockerfile-lint/README.md) | easy | `uv run pytest course/P3-deployment-and-infra/P3.0-containers-and-kubernetes-basics/exercises/01-dockerfile-lint` |
| 2 | [Deployment and Service YAML](exercises/02-deployment-yaml/README.md) | medium | `uv run pytest course/P3-deployment-and-infra/P3.0-containers-and-kubernetes-basics/exercises/02-deployment-yaml` |
| 3 | [kind walkthrough](exercises/03-kind-walkthrough/README.md) | hands-on | `TODO(run)`: needs Docker, kind, kubectl |

Verify the reference solutions: `S2S_SOLUTIONS=1 uv run pytest course/P3-deployment-and-infra/P3.0-containers-and-kubernetes-basics`. Exercises 1 and 2 need no Docker. Exercise 3 and any `docker` command above are `TODO(run)`: this build environment has no Docker.

## Quiz

[quiz.md](quiz.md)

## Go further

All links are UNVERIFIED from the build environment (egress blocked). Check them.

- [Docker: Get started](https://docs.docker.com/get-started/)
- [Kubernetes basics tutorial](https://kubernetes.io/docs/tutorials/kubernetes-basics/)
- [kind quick start](https://kind.sigs.k8s.io/docs/user/quick-start/)

## Next

[P3.1 — GPU containers](../P3.1-gpu-containers/README.md).
