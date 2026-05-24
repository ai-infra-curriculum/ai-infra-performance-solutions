# Prefix-Aware Routing — Solution

`router.py` consistent-hashes on the system prompt → same replica receives
requests sharing a prefix → maximizes prefix cache hit rate (2-3× speedup
when paired with `--enable-prefix-caching`).
