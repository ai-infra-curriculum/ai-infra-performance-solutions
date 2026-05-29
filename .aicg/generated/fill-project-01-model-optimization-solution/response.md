Authored the project-01 solution. Summary of the work packet result:

**Created**
- `projects/project-01-model-optimization/SOLUTION.md` (604 lines) — reviewer's companion to the learning-repo project spec, following the work-packet output contract: solution overview, worked answer / implementation walkthrough, validation steps (`make all` → `make verify` → gate spot-checks), rubric-and-anti-pattern checklist mapped to D1-D8 and PR-1..PR-10, common mistakes catalogued by phase (baseline → FP16 → PTQ → QAT → pruning → distillation → TRT → bench), and references to project artifacts, in-repo module SOLUTIONs, NVIDIA/PyTorch/ONNX official docs, and the foundational papers the spec cites.

**Modified (consistency)**
- `projects/README.md` — replaced the placeholder with a project index, including a `<!-- needs-research: ... -->` block for the two project solutions that don't exist yet (per the source policy, these mark unverified content and block auto-merge).
- `SOLUTION_OVERVIEW.md` and `README.md` — updated the "this repo is module-only by design" / "projects live in the learning repo" statements that were now stale (the audit-report flagged the project gap as an error to fill).

**Scope and source policy**
- The work packet listed no exercises (this is a single-project capstone, not a multi-exercise module), so a single `SOLUTION.md` matched the work-plan actions.
- No metrics or incidents were invented. Numerical targets (3.0x speedup, 2.0 pp accuracy drop, ≤ 90 min runtime, etc.) are quoted from the project's own `requirements.md` / `rubric.md` / `STEP_BY_STEP.md`. Hardware peak numbers (A100 312 TOPS INT8, 2.039 TB/s HBM2e) are quoted from the learning repo's own `STEP_BY_STEP.md` § 6.4.
- The only `<!-- needs-research: -->` blocks are in `projects/README.md` for the two unwritten sibling projects.
