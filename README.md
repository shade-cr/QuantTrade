# QuantTrade

QuantTrade is an equity swing-trading meta-labeling research pipeline, built on the López-de-Prado AFML framework (Advances in Financial Machine Learning, 2018). It was cherry-picked from the QuantHack hackathon project and repurposed for multi-asset equity research.

The pipeline implements two-stage meta-labeling: a rule-based primary (side signal) feeds a calibrated ML meta-learner that filters on confidence, using triple-barrier labels, purged walk-forward validation, and AFML-correct sample weights.

## Quickstart

```powershell
uv sync                           # install deps into .venv
copy .env.example .env            # then set FRED_API_KEY
uv run pytest -q                  # run all tests
```

## Design spec

See [docs/superpowers/specs/2026-06-28-quanttrade-equity-swing-repurpose-design.md](docs/superpowers/specs/2026-06-28-quanttrade-equity-swing-repurpose-design.md) for the full asset-universe, feature-set, and pipeline-repurposing decisions.

For Claude Code guidance (architecture, invariants, workflow), see [CLAUDE.md](CLAUDE.md).
