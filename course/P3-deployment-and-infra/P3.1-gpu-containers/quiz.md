# P3.1 quiz

<details><summary><b>1.</b> Where does <code>libcuda.so</code> in a GPU container come from?</summary>

The host. The NVIDIA Container Toolkit injects the host driver's user-space libraries at container start. The image provides the CUDA *runtime* libraries.
</details>

<details><summary><b>2.</b> <code>nvidia-smi</code> inside the container says "CUDA Version 12.4". What does that number describe?</summary>

The highest CUDA version the **host driver** supports, not the toolkit or runtime inside the image.
</details>

<details><summary><b>3.</b> Why not bake model weights into the serving image?</summary>

Huge layers slow every pull (cold starts), force a rebuild for each model update, and duplicate storage per tag. Load the weights at runtime from a registry or cache (P3.7).
</details>

<details><summary><b>4.</b> Why pin by digest and not only by tag?</summary>

Tags are mutable pointers. A digest is content-addressed, so the build is exactly reproducible and auditable.
</details>

<details><summary><b>5.</b> A later layer runs <code>rm /secret</code>. Is the secret gone from the image?</summary>

No. It still exists in the earlier layer and can be extracted. Use build secrets (`RUN --mount=type=secret`) or runtime injection.
</details>
