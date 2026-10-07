# Exercise 3 — Non-root + read-only root FS (T0)

Run the platform image the way a hardened K8s pod would:

```bash
docker run --rm --read-only --tmpfs /tmp --user 10001 --cap-drop ALL --security-opt no-new-privileges -p 8001:8001 s2s-platform:dev
curl -s localhost:8001/health
```

If anything fails (a library writing to `$HOME` or a cache directory), fix it with environment variables that point caches at `/tmp`. Never fix it by dropping `--read-only`. Then write the equivalent K8s `securityContext` (`runAsNonRoot`, `readOnlyRootFilesystem`, `allowPrivilegeEscalation: false`, `capabilities.drop: [ALL]`) for P3.3.
