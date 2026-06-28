"""Console-encoding crash-proofing for the Jun-15 day-one CLI toolchain.

Found by the 2026-06-12 dress rehearsal: ``run_ll_validation.py --synthetic``
crashed with UnicodeEncodeError on the Windows cp1252 console while printing
the per-trial summary (the sigma character U+03C3 is not encodable in
cp1252) — AFTER writing the report but before the operator-facing trial
table. On Jun-15 that would kill the runner mid-output on the operator's
machine. The fix is ``pipeline.cli_console.make_console_crash_proof()``:
reconfigure stdout/stderr with errors="replace" so unencodable characters
degrade to "?" instead of raising. Every day-one CLI must call it at the
top of ``main()``.
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.cli_console import make_console_crash_proof  # noqa: E402

# DAY_ONE_SCRIPTS list removed — those scripts (run_osr_*, run_ll_*, tick_data_report)
# are hackathon live-ops not seeded into QuantTrade; TestDayOneScriptsAreGuarded dropped.


class TestMakeConsoleCrashProof:
    def _cp1252_stdout(self) -> io.TextIOWrapper:
        """A strict cp1252 stream — the Windows console that crashed the
        rehearsal. Printing U+03C3 to it raises UnicodeEncodeError."""
        return io.TextIOWrapper(io.BytesIO(), encoding="cp1252",
                                errors="strict")

    def test_sigma_print_survives_a_cp1252_console(self, monkeypatch):
        stream = self._cp1252_stdout()
        monkeypatch.setattr(sys, "stdout", stream)
        monkeypatch.setattr(sys, "stderr", self._cp1252_stdout())
        make_console_crash_proof()
        print("trial 7 (BTC event 2.0σ): n_val=0")  # must not raise
        sys.stdout.flush()

    def test_without_the_guard_the_same_print_raises(self, monkeypatch):
        import pytest

        stream = self._cp1252_stdout()
        monkeypatch.setattr(sys, "stdout", stream)
        with pytest.raises(UnicodeEncodeError):
            print("2.0σ")
            sys.stdout.flush()

    def test_tolerates_streams_without_reconfigure(self, monkeypatch):
        """pytest's capsys (and any plain object) lacks .reconfigure —
        the guard must be a no-op there, never an AttributeError."""
        monkeypatch.setattr(sys, "stdout", io.StringIO())
        monkeypatch.setattr(sys, "stderr", io.StringIO())
        make_console_crash_proof()  # must not raise


