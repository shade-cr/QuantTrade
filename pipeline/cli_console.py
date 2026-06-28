"""Crash-proof console output for operator-facing CLI scripts.

Born in the 2026-06-12 Jun-15 dress rehearsal: ``run_ll_validation.py``
died with UnicodeEncodeError on the Windows cp1252 console printing the
per-trial sigma label (U+03C3) — after writing its report, before the
operator-facing summary. Any CLI that prints non-ASCII (σ, §, em-dash)
crashes on consoles whose codepage lacks the character (cp1252, cp850).

The contract: a status print must NEVER kill a runner. Reconfiguring the
standard streams with ``errors="replace"`` keeps the console's native
encoding (no mojibake for what it CAN show) and degrades unencodable
characters to "?" instead of raising. File artifacts are unaffected —
reports are written explicitly with ``encoding="utf-8"``.
"""
from __future__ import annotations

import sys


def make_console_crash_proof() -> None:
    """Reconfigure stdout/stderr to replace unencodable characters.

    Call at the top of every CLI ``main()``. No-op for streams without
    ``reconfigure`` (pytest capture, StringIO) — those accept any str.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(errors="replace")
