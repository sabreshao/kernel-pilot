# Passing a Long Prompt to `humanize:humanize-kernel-agent-loop` via File

The slash command's `ARGUMENTS` are just inline text — there is no built-in
`--prompt-file` flag. But Claude Code can read a file and treat its contents as
the task spec. Below are three clean ways to feed a long kernel-task spec to
the loop, from simplest to most pre-baked.

---

## Option 1 — Reference the file inline (recommended)

Point the slash command at an absolute path. Claude will `Read` the file and
treat its contents as the K / R / W spec.

```text
/humanize:humanize-kernel-agent-loop see /home/sashao/tasks/my_kernel_task.md
```

or

```text
/humanize:humanize-kernel-agent-loop read /home/sashao/tasks/my_kernel_task.md and run the loop on it
```

**Pros:** zero setup, works for any file size, file stays on disk.
**Cons:** Claude has to open the file as a first step.

---

## Option 2 — Use Claude Code's `@` file-mention

Type `@` in the prompt box and pick the file. Claude Code inlines the file
content into the message before sending:

```text
/humanize:humanize-kernel-agent-loop run the loop using this spec: @tasks/my_kernel_task.md
```

**Pros:** content is in-message, no extra Read call.
**Cons:** subject to message-size limits; large files may be truncated.

---

## Option 3 — Pre-populate the refined plan and skip plan synthesis

If your spec is already polished (i.e., it already reads like a refined plan
with K/R/W, acceptance checks, baselines), drop it directly where the loop
expects it and tell Claude to skip plan synthesis.

```bash
# inside your chosen workspace root
mkdir -p .humanize/kernel-agent
cp /path/to/my_kernel_task.md .humanize/kernel-agent/refined-plan.md
```

Then invoke:

```text
/humanize:humanize-kernel-agent-loop use the refined plan already at
.humanize/kernel-agent/refined-plan.md — skip plan synthesis, go straight to
scaffold + RLCR setup
```

**Pros:** bypasses the "synthesize a plan from your words" step entirely;
best for very long, structured specs.
**Cons:** you take responsibility for the plan being complete and well-formed.

---

## Recommended File Template

To minimize back-and-forth, structure your prompt file like this:

```markdown
# Task
<kernel name + one-line goal>

## K — Kernel definition
- Inputs (shapes, dtypes, layouts):
- Outputs (shapes, dtypes, layouts):
- Math / fusion boundaries:
- Any algorithmic constraints (causal mask, grouped, paged, etc.):

## R — Correctness reference
- Reference impl path or formula:
- How to invoke it (CLI / Python snippet):
- Tolerance (rtol/atol per dtype):

## W — Workload
- Shapes or distribution (list or generator):
- Batch / seqlen / heads / hidden ranges if applicable:
- Single focused case vs. multi-regime:

## Target
- GPU: MI300X (gfx942) | MI350 (gfx950)
- Baseline to beat: <e.g. hipBLASLt, Triton kernel X, SGLang fused_moe>
- Acceptance: <e.g. ≥1.3× median latency vs baseline, no correctness regression>

## Constraints / Out-of-scope
- Dtypes you don't care about:
- Build / toolchain constraints (ROCm version, Python version, etc.):
- License limits on borrowed code:
- Anything explicitly forbidden (e.g., "no inline asm", "no new deps"):

## Optional: Prior art hints
- Known similar kernels / PRs / wiki pages worth checking first:
- Known failed approaches to avoid:
```

If you provide a file with these sections filled in, the loop can go straight
to scaffold + RLCR without clarification rounds.

---

## Quick Decision Guide

| Situation                                     | Use      |
| --------------------------------------------- | -------- |
| Spec is moderately long, lives in a file      | Option 1 |
| Spec is short-to-medium, want it in-message   | Option 2 |
| Spec is already a complete, polished plan     | Option 3 |

When in doubt: **Option 1**.
