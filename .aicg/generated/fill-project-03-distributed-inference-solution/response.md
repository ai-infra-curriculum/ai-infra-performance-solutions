Wrote `projects/project-03-distributed-inference/SOLUTION.md` following the project-01/project-02 template (six sections per the output contract: solution overview, worked walkthrough, validation steps, rubric checklist, common mistakes, references), and flipped the project-03 row in `projects/README.md` from `<!-- needs-research: -->` Planned to a published link.

Key shape of the solution:
- Frames project-03 as a **composition exercise** over mod-006 (TP / HPA / prefix routing / cold-start), mod-007 (canary / spot / multi-tier), and mod-008 (CUDA Graphs / NCCL / MIG), with a written SLO contract as the design pillar.
- 13 worked design choices (§ 2.1–2.13) covering SLO-first, TP-within-node + replicas-across, continuous batching, paged-attention KV budget, prefix-routing with failover metrics, queue-depth HPA, surge buffer, readiness vs liveness, graceful spot drain, canary on production deltas, CUDA Graphs on decode, and the quarantine flow inherited from prior projects.
- Anti-patterns and common mistakes explicitly cross-reference mod-006/project-01/project-02 to avoid duplication.
- Spec-specific numbers (PR-N gates, D-numbering, exact thresholds) are marked `<!-- spec-pin: ... -->` per the project-02 convention so they can be reconciled against the learning-repo `requirements.md` / `rubric.md` before publish.
- No invented metrics or incidents; claims tied to module SOLUTION.md docs in this repo or to named primary sources (NCCL / Kubernetes / vLLM / Nsight docs; PagedAttention / Orca / Splitwise / Mooncake papers).

I did **not** touch CURRICULUM.md's "Shipped (autonomous)" table — that header says it's auto-appended by the AICG runner on verify, matching how the project-01/02 rows landed.
