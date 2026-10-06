# Exercise 3: tamper-evident audit log (T0)

`platform/tenancy/audit.py:AuditLog` writes the hash chain. Implement `verify(path, key, checkpoint=None) -> list[str]` in `audit_impl.py`. `test_audit.py` writes a 20-record log with the reference writer and then:

| tamper | your verify must report |
|---|---|
| none | `[]` |
| edit a field of record 7 | a problem naming line 7 (hash mismatch) |
| delete record 7 | a problem at line 7 (broken link: its `prev` no longer matches) |
| swap records 7 and 8 | a problem at line 7 |
| truncate the last 5 records | nothing from the chain itself. **With** the checkpoint (the hash of record 20, stored elsewhere): a truncation problem |
| rewrite everything with the wrong key | a problem at line 1 when verified with the right key |

`S2S_SOLUTIONS=1` runs the reference `audit.verify`.

**Then:** wire `AuditLog` into the gateway, with one record per request: tenant, key-hash prefix, model, backend, outcome, prompt/completion tokens. **No prompt text.** Where should the checkpoint go on AWS, and which IAM permission must the gateway's role *not* have?
