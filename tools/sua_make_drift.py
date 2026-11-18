"""Rechnet die 30 erfundenen Zwei-Hypothesen-Szenarien der Ansicht 4 vor und schreibt data/sua_drift.json.

    python tools/sua_make_drift.py            # 30 Szenarien mit je 40 000 Teilchen
    python tools/sua_make_drift.py --check    # rechnet 5 Szenarien neu und vergleicht mit der Datei (Band 0,02 im Exponenten)

Je Szenario: Fläche des kleinsten Zellgebiets mit 90 % der Teilchen (Zählung auf Zellen, keine Gauß-Annahme) bei t = 1, 4 und 8; Flächenexponent
zwischen t = 1 und 4 (je Zeitverdopplung) und zwischen t = 4 und 8. Zufall nur über den SplitMix64 (Szenario Nr. = Zufalls-Seed).
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

WURZEL = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WURZEL))

import sua_drift as D  # noqa: E402
import sua_messung as M  # noqa: E402

DATEI = WURZEL / "data" / "sua_drift.json"
SZENARIEN = tuple(range(1, 31))
BAND = 0.02


def lauf() -> None:
    t0 = time.time()
    ex14, ex48 = [], []
    for nr in SZENARIEN:
        a, b = D.szenario_exponenten(nr)
        ex14.append(a)
        ex48.append(b)
    out = {"meta": {"szenarien": len(SZENARIEN), "teilchen": 40000, "text": "Eingefroren mit tools/sua_make_drift.py; erfundene Szenarien."},
           "exponent_1_4": {"werte": ex14, **M.stat(np.array(ex14), "drift14"), "min": min(ex14), "max": max(ex14)},
           "exponent_4_8": {"werte": ex48, **M.stat(np.array(ex48), "drift48"), "min": min(ex48), "max": max(ex48)}}
    DATEI.parent.mkdir(exist_ok=True)
    DATEI.write_text(json.dumps(out, separators=(",", ":")), encoding="utf-8")
    print(f"1 bis 4: Mittel {np.mean(ex14):.3f} [{out['exponent_1_4']['lo']:.3f}; {out['exponent_1_4']['hi']:.3f}], Spanne {min(ex14):.2f} bis {max(ex14):.2f}")
    print(f"4 bis 8: Mittel {np.mean(ex48):.3f} [{out['exponent_4_8']['lo']:.3f}; {out['exponent_4_8']['hi']:.3f}], Spanne {min(ex48):.2f} bis {max(ex48):.2f}")
    print(f"fertig nach {time.time() - t0:.0f} s")


def pruefen() -> int:
    daten = json.loads(DATEI.read_text(encoding="utf-8"))
    fehler = 0
    for nr in (1, 8, 15, 22, 30):
        a, b = D.szenario_exponenten(nr)
        for name, neu, alt in (("1 bis 4", a, daten["exponent_1_4"]["werte"][nr - 1]), ("4 bis 8", b, daten["exponent_4_8"]["werte"][nr - 1])):
            ok = abs(neu - alt) <= BAND
            print(("OK    " if ok else "FEHLER"), f"Szenario {nr} {name}: gespeichert {alt:.4f}, neu {neu:.4f}")
            fehler += 0 if ok else 1
    return fehler


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    if a.check:
        sys.exit(1 if pruefen() else 0)
    lauf()
