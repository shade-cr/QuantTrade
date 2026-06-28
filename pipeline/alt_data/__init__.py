"""Alternative-data sources for Phase 5 standalone primaries (B0015 family).

Each module loads a single alt-data series with strict point-in-time
discipline. Loaders are pure functions: take a target_index, return a
DataFrame aligned to that index with .shift(1) publication lag applied.

Used by pipeline/primaries_phase5/phase5_<name>.py modules. NOT by
pipeline/features.py — alt-data accessed by primaries does NOT flow into
build_tier2_features() outputs (and therefore is invisible to the
meta-labeler, preserving the anti-orange-twice commitment).
"""
