uv run python -m pipeline.regimes --asset NVDA --frequency D1 --print-sanity
uv run python -m phase5.regime_stats --asset NVDA --frequency D1 --asset-class equity --out signals/regime_stats/
