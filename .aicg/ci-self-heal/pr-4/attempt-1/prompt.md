# Address CI failures on PR #4

## Goal

The PR you just opened failed CI. Fix the failures listed
below by editing files on the current branch. Do NOT regenerate
the content from scratch — make the minimal edit needed to
satisfy each failing check.

## Failed checks

### 1. `Markdown lint` (failure)

- Details: <https://github.com/ai-infra-curriculum/ai-infra-performance-solutions/actions/runs/26623477254/job/78454573917>
- Annotations:
  - `.github:2` (warning): Node.js 20 actions are deprecated. The following actions are running on Node.js 20 and may not work as expected: actions/checkout@v4, DavidAnson/markdownlint-cli2-action@v16. Actions will be forced to run with Node.js 24 by default starting June 2nd, 2026. Node.js 20 will be removed from the runner 
  - `projects/project-02-gpu-optimization/SOLUTION.md:531` (failure): projects/project-02-gpu-optimization/SOLUTION.md:531:1 MD004/ul-style Unordered list style [Expected: asterisk; Actual: dash] https://github.com/DavidAnson/markdownlint/blob/v0.34.0/doc/md004.md
  - `projects/project-02-gpu-optimization/SOLUTION.md:529` (failure): projects/project-02-gpu-optimization/SOLUTION.md:529:1 MD004/ul-style Unordered list style [Expected: asterisk; Actual: dash] https://github.com/DavidAnson/markdownlint/blob/v0.34.0/doc/md004.md
  - `projects/project-02-gpu-optimization/SOLUTION.md:527` (failure): projects/project-02-gpu-optimization/SOLUTION.md:527:1 MD004/ul-style Unordered list style [Expected: asterisk; Actual: dash] https://github.com/DavidAnson/markdownlint/blob/v0.34.0/doc/md004.md
  - `projects/project-02-gpu-optimization/SOLUTION.md:525` (failure): projects/project-02-gpu-optimization/SOLUTION.md:525:1 MD004/ul-style Unordered list style [Expected: asterisk; Actual: dash] https://github.com/DavidAnson/markdownlint/blob/v0.34.0/doc/md004.md
  - `projects/project-02-gpu-optimization/SOLUTION.md:522` (failure): projects/project-02-gpu-optimization/SOLUTION.md:522:1 MD004/ul-style Unordered list style [Expected: asterisk; Actual: dash] https://github.com/DavidAnson/markdownlint/blob/v0.34.0/doc/md004.md
  - `projects/project-02-gpu-optimization/SOLUTION.md:332` (failure): projects/project-02-gpu-optimization/SOLUTION.md:332:1 MD004/ul-style Unordered list style [Expected: asterisk; Actual: dash] https://github.com/DavidAnson/markdownlint/blob/v0.34.0/doc/md004.md
  - `projects/project-02-gpu-optimization/SOLUTION.md:328` (failure): projects/project-02-gpu-optimization/SOLUTION.md:328:1 MD004/ul-style Unordered list style [Expected: asterisk; Actual: dash] https://github.com/DavidAnson/markdownlint/blob/v0.34.0/doc/md004.md
  - `projects/project-02-gpu-optimization/SOLUTION.md:243` (failure): projects/project-02-gpu-optimization/SOLUTION.md:243:1 MD004/ul-style Unordered list style [Expected: asterisk; Actual: dash] https://github.com/DavidAnson/markdownlint/blob/v0.34.0/doc/md004.md
  - `projects/project-02-gpu-optimization/SOLUTION.md:239` (failure): projects/project-02-gpu-optimization/SOLUTION.md:239:1 MD004/ul-style Unordered list style [Expected: asterisk; Actual: dash] https://github.com/DavidAnson/markdownlint/blob/v0.34.0/doc/md004.md
  - `projects/project-02-gpu-optimization/SOLUTION.md:216` (failure): projects/project-02-gpu-optimization/SOLUTION.md:216:11 MD038/no-space-in-code Spaces inside code span elements [Context: "`** with `"] https://github.com/DavidAnson/markdownlint/blob/v0.34.0/doc/md038.md

## Output contract

- Edit ONLY files inside this repo on the current branch.
- Preserve the existing structure; do not delete sections.
- Do NOT touch CURRICULUM.md, README.md, or VERSIONS.md.
- One atomic commit covering all fixes is fine.
