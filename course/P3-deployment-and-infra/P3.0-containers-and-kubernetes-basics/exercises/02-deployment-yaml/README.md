# Exercise 2 — Deployment and Service YAML (medium, T0)

Fill in `deployment.yaml` so it describes a tiny HTTP echo app:

- a **Deployment** named `echo` with 2 replicas, whose selector matches the pod template's labels (`app: echo`);
- the image `hashicorp/http-echo:1.0` (a pinned tag; the exact tag is **UNVERIFIED** from this build environment, so check it exists on the registry before you run it), listening on port 5678;
- a **readiness probe** that does an HTTP GET on that port, so a pod receives traffic only when it answers;
- **requests and limits** for cpu and memory, with requests no larger than limits;
- a **Service** of type `ClusterIP` that selects the same pods and sends port 80 to the container's port 5678.

The test parses the YAML with `pyyaml`. It checks fields and that the three label links (selector, pod template, Service) agree. It never talks to a cluster. Running it on a real cluster is the [kind walkthrough](../03-kind-walkthrough/README.md).

```bash
uv run pytest course/P3-deployment-and-infra/P3.0-containers-and-kubernetes-basics/exercises/02-deployment-yaml
```

A selector that doesn't match the pod labels is the most common first mistake: the Deployment is rejected, or the Service has no endpoints and every request fails. The test names which link is broken.
