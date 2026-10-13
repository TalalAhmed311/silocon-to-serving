# Exercise 3 — Kind walkthrough (T0, no test)

Run your exercise 2 YAML on a real Kubernetes cluster on your laptop. [kind](https://kind.sigs.k8s.io/docs/user/quick-start/) runs a cluster's nodes as Docker containers. You need Docker, `kind` and `kubectl`. No GPU.

Every run here is `TODO(run)`: this build environment has no Docker. Paste what you see into your lab notes.

## 1. Create the cluster

```bash
kind create cluster --name s2s --config platform/deploy/kind/kind-config.yaml
kubectl cluster-info --context kind-s2s
kubectl get nodes
```

Expect three nodes: one control plane and two workers (the config labels one worker as a pretend GPU pool). Status `Ready` can take a minute. `TODO(run)`

## 2. Apply the YAML

```bash
kubectl apply -f course/P3-deployment-and-infra/P3.0-containers-and-kubernetes-basics/exercises/02-deployment-yaml/deployment.yaml
kubectl get deployment echo
kubectl get pods -l app=echo -o wide
```

Expect two pods, `READY 1/1` after the probe passes. Note which node each pod landed on: you did not choose. `TODO(run)`

If a pod stays `ContainerCreating`, `kubectl describe pod <name>` shows the events at the bottom (usually an image pull). If it never becomes ready, `kubectl logs <name>` shows the app's output. Those two commands solve most first problems.

## 3. Reach it

```bash
kubectl get service echo
kubectl get endpoints echo           # two pod IPs; empty means the selector matches nothing
kubectl port-forward service/echo 8080:80
```

In a second terminal:

```bash
curl -s localhost:8080/
```

Expect `hello from the echo app`. `port-forward` tunnels from your laptop to the Service; nothing is exposed to the network. `TODO(run)`

## 4. Watch the loop heal

```bash
kubectl delete pod -l app=echo --wait=false
kubectl get pods -l app=echo -w     # Ctrl-C when two are Running again
kubectl scale deployment echo --replicas=3
kubectl get pods -l app=echo
```

You declared 2 (then 3). The Deployment controller created pods until reality matched. `TODO(run)`

## 5. Break it on purpose

Change the Service selector to `app: ecko`, re-apply, and run `kubectl get endpoints echo`. Explain what you see, then fix it.

## 6. Delete the cluster

```bash
kubectl delete -f course/P3-deployment-and-infra/P3.0-containers-and-kubernetes-basics/exercises/02-deployment-yaml/deployment.yaml
kind delete cluster --name s2s
kind get clusters       # s2s is gone
docker ps               # no kind containers left
```

A kind cluster costs nothing but laptop CPU and RAM while it runs. Delete it anyway, so it does not eat memory.
