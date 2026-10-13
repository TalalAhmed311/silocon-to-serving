# P3.0 quiz

<details><summary><b>1.</b> What is the difference between an image and a container?</summary>

An image is a stack of read-only layers: a recipe's output, stored on disk or in a registry. A container is a running process started from an image, with a thin writable layer on top. You can start many containers from one image.
</details>

<details><summary><b>2.</b> A Dockerfile has <code>COPY . .</code> on line 3 and <code>RUN pip install -r requirements.txt</code> on line 4. You change one line of your code. What rebuilds, and what is a better order?</summary>

Line 3's inputs changed, so its layer and every layer after it rebuild, including the full <code>pip install</code>. Copy only <code>requirements.txt</code> first, run the install, then <code>COPY . .</code>. A code edit then rebuilds only the last layer.
</details>

<details><summary><b>3.</b> Why is a container not a virtual machine?</summary>

A container is a normal process on the host's kernel, limited by namespaces (what it can see) and cgroups (what it can use). A VM runs its own kernel. That makes containers start fast and stay small, and makes their isolation weaker than a VM's.
</details>

<details><summary><b>4.</b> What does <code>-p 8000:8000</code> do, and what happens without it?</summary>

It publishes container port 8000 on host port 8000 (host first, then container). Without it the server runs but the host cannot reach it from outside the container's own network.
</details>

<details><summary><b>5.</b> Why pin an image by digest instead of by tag?</summary>

A tag is a movable label; its owner can point <code>python:3.12-slim</code> at a new image tomorrow. A digest names exact content, so the same Dockerfile builds the same thing. Tags are readable; digests are reproducible.
</details>

<details><summary><b>6.</b> You declared 3 replicas and a node dies, taking one Pod with it. What happens, and who does it?</summary>

The Deployment's controller sees 2 actual against 3 desired and creates a new Pod, which the scheduler places on a healthy node. You did not tell it to restart anything. You declared the end state and the loop did the rest.
</details>

<details><summary><b>7.</b> A Service has no endpoints (<code>kubectl get endpoints</code> is empty). What is the first thing to check?</summary>

Whether the Service's selector labels match the Pod labels exactly, and whether the Pods are Ready (a failing readiness probe removes a Pod from the endpoints).
</details>

<details><summary><b>8.</b> How does a Pod ask for a GPU?</summary>

It lists <code>nvidia.com/gpu: 1</code> under <code>resources.limits</code>. The node must advertise that resource; the scheduler then places the Pod only on a node that has a free one. It is a counted resource like any other, but whole units only.
</details>
