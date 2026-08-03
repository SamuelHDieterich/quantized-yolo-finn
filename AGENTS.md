# Agent Instructions

This file provides instructions and context for AI coding agents working on this project.
It is the single source of truth for project docs; `CLAUDE.md` is a symlink to it.

## Project

A quantized YOLO object detection pipeline: full-precision training (Ultralytics) →
Brevitas QAT → export to FINN-compatible QONNX → FINN dataflow compilation → FPGA
bitstream synthesis and on-board inference.

Five phases, tracked as bd epics (`bd list --tree` for the full breakdown):
1. Full-precision YOLO training
2. Quantization-aware training (Brevitas)
3. Export to FINN-compatible ONNX
4. FINN dataflow compilation
5. Synthesis and on-board inference

## Build & Test

```bash
devenv shell    # enters the Nix dev shell (uv-managed Python venv, see .python-version)
```

No test suite is wired up yet (tracked as `qyf-735.2.8`). Once it lands, this section
should list the pytest invocation.

## Conventions

- Python only, managed with `uv` inside the devenv shell.
- Pipeline config lives in `configs/*.yaml`; a pydantic model layer is planned (`qyf-735.2.4`).
- Issue tracking is bd (beads), not markdown TODOs — see below.

## Non-Interactive Shell Commands

**ALWAYS use non-interactive flags** with file operations to avoid hanging on confirmation prompts.

Shell commands like `cp`, `mv`, and `rm` may be aliased to include `-i` (interactive) mode on some systems, causing the agent to hang indefinitely waiting for y/n input.

**Use these forms instead:**
```bash
# Force overwrite without prompting
cp -f source dest           # NOT: cp source dest
mv -f source dest           # NOT: mv source dest
rm -f file                  # NOT: rm file

# For recursive operations
rm -rf directory            # NOT: rm -r directory
cp -rf source dest          # NOT: cp -r source dest
```

**Other commands that may prompt:**
- `scp` - use `-o BatchMode=yes` for non-interactive
- `ssh` - use `-o BatchMode=yes` to fail instead of prompting
- `apt-get` - use `-y` flag
- `brew` - use `HOMEBREW_NO_AUTO_UPDATE=1` env var

<!-- BEGIN BEADS INTEGRATION v:1 profile:minimal hash:ca08a54f -->
## Beads Issue Tracker

This project uses **bd (beads)** for issue tracking. Run `bd prime` to see full workflow context and commands.

### Quick Reference

```bash
bd ready              # Find available work
bd show <id>          # View issue details
bd update <id> --claim  # Claim work
bd close <id>         # Complete work
```

### Rules

- Use `bd` for ALL task tracking — do NOT use TodoWrite, TaskCreate, or markdown TODO lists
- Run `bd prime` for detailed command reference and session close protocol
- Use `bd remember` for persistent knowledge — do NOT use MEMORY.md files

## Session Completion

**When ending a work session**, you MUST complete ALL steps below. Work is NOT complete until `git push` succeeds.

**MANDATORY WORKFLOW:**

1. **File issues for remaining work** - Create issues for anything that needs follow-up
2. **Run quality gates** (if code changed) - Tests, linters, builds
3. **Update issue status** - Close finished work, update in-progress items
4. **PUSH TO REMOTE** - This is MANDATORY:
   ```bash
   git pull --rebase
   bd dolt push
   git push
   git status  # MUST show "up to date with origin"
   ```
5. **Clean up** - Clear stashes, prune remote branches
6. **Verify** - All changes committed AND pushed
7. **Hand off** - Provide context for next session

**CRITICAL RULES:**
- Work is NOT complete until `git push` succeeds
- NEVER stop before pushing - that leaves work stranded locally
- NEVER say "ready to push when you are" - YOU must push
- If push fails, resolve and retry until it succeeds
<!-- END BEADS INTEGRATION -->
