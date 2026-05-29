The edit is in place but `git add`/`git commit` aren't authorized in this session. The fix:

**`projects/project-02-gpu-optimization/SOLUTION.md:215-218`** — reflowed the code span so it doesn't wrap across a line that begins with `   * w + b`. That wrap was being parsed as an asterisk-style bullet (forcing MD004 to expect asterisks and failing every later dash bullet) and also tripping MD038 (spaces inside the cross-line code span). With the code span on a single line, the first real bullet (line 239) is a dash, dash becomes the consistent style, and MD038 no longer fires.

The Node.js 20 deprecation warning on `.github:2` is an informational annotation, not a failing check, and per the output contract I haven't touched any workflow files.

Please run `git add projects/project-02-gpu-optimization/SOLUTION.md && git commit` (or approve when re-prompted) — the working-tree diff is the one shown above.
