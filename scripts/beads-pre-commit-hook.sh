#!/bin/sh
# Runs beads' pre-commit checks as a devenv/prek-managed hook step.
#
# beads' own `bd hooks install` writes this same logic straight into .git/hooks/pre-commit,
# but devenv's git-hooks (prek) integration regenerates that file from .pre-commit-config.yaml
# on every `devenv shell` entry, silently dropping anything appended to it.
# Running it as a declared hook here means Nix regenerates the call every time instead of
# fighting over the same file.
set -eu

if ! command -v bd >/dev/null 2>&1; then
    exit 0
fi

export BD_GIT_HOOK=1
bd_timeout="${BEADS_HOOK_TIMEOUT:-300}"

if command -v timeout >/dev/null 2>&1; then
    timeout "$bd_timeout" bd hooks run pre-commit "$@"
    bd_exit=$?
    if [ "$bd_exit" -eq 124 ]; then
        echo >&2 "beads: hook 'pre-commit' timed out after ${bd_timeout}s — continuing without beads"
        bd_exit=0
    fi
else
    bd hooks run pre-commit "$@"
    bd_exit=$?
fi

if [ "$bd_exit" -eq 3 ]; then
    echo >&2 "beads: database not initialized — skipping hook 'pre-commit'"
    bd_exit=0
fi

exit "$bd_exit"
