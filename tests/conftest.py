"""Gemeinsame Fixtures: Importpfad und das Handbeispiel, an dem jede Einheit von Hand geprüft wird (DEMO-PLAYBOOK §1b)."""

import sys
from fractions import Fraction
from pathlib import Path

import pytest

WURZEL = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WURZEL))
sys.path.insert(0, str(WURZEL / "tools"))


@pytest.fixture
def handbeispiel():
    """3x3-Prior in Hundertsteln, Zellen zeilenweise (Summe 100, kein Gleichstand der Grenzzuwächse); q = 1/2 je Durchgang, also alpha = ln 2."""
    return [Fraction(v, 100) for v in (7, 19, 11, 15, 27, 9, 5, 6, 1)]
