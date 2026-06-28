# Installing the backlog/ideas hooks

Auto-mode declined to write `.claude/settings.json` directly (sensitive file —
modifies the agent's own hook config). To enable the hooks, **append** the
following to your `.claude/settings.json` `hooks` block. If the file already
has a `"hooks"` key, merge the entries; otherwise add the whole block.

```jsonc
{
  "enabledPlugins": { "context-forge@context-forge": true },
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Bash",
        "hooks": [
          {
            "type": "command",
            "command": "uv run python .claude/hooks/backlog_git_commit_dispatcher.py --mode pre"
          }
        ]
      }
    ],
    "PostToolUse": [
      {
        "matcher": "Bash",
        "hooks": [
          {
            "type": "command",
            "command": "uv run python .claude/hooks/backlog_git_commit_dispatcher.py --mode post"
          }
        ]
      }
    ],
    "Stop": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "uv run python -m backlog.session_audit --max-count 10"
          }
        ]
      }
    ]
  }
}
```

## What each hook does

| Hook | Trigger | Behavior |
|---|---|---|
| PreToolUse / Bash | every Bash invocation | dispatcher inspects the command; if it's `git commit`, runs `backlog.lint` + `ideas.lint`; exits non-zero to block bad-data commits. Otherwise exits 0 silently. |
| PostToolUse / Bash | every Bash invocation | dispatcher inspects the command; if `git commit`, runs `backlog.commit_sync HEAD` to append history events + suggest status transitions. Always exits 0. |
| Stop | session end | runs `backlog.session_audit` to surface forgotten history updates across the session's commits. Advisory only. |

## Opt-out

To bypass the pre-commit lint gate for a specific commit, include
`[skip-backlog-lint]` or `--no-verify` somewhere in the commit message.

## Testing without enabling hooks

You can dry-run the hooks manually at any time:

```bash
uv run python -m backlog.lint
uv run python -m ideas.lint
uv run python -m backlog.commit_sync HEAD
uv run python -m backlog.session_audit
```
