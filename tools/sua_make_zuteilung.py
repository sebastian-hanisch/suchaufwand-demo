"""Rechnet die Messreihen der Ansichten 1 und 2 vor und schreibt data/sua_messreihe.json (eingefroren: die App rechnet sie nie live).

    python tools/sua_make_zuteilung.py            # alle Messreihen (einige Minuten)
    python tools/sua_make_zuteilung.py --check    # rechnet 6 Zellen neu und vergleicht mit der Datei (Band 0,05 Prozentpunkte)

Alle Zellen: 30 erfundene Instanzen (Zufalls-Seed 1 bis 30), über Entdeckungsstärke und Budget gepaart; Zufall nur über den SplitMix64.
Die Funktionen stehen in `sua_messung.py`, die App liest die Datei über `sua_frozen.py`.
"""

import argparse
import json
import sys
import time
from pathlib import Path

WURZEL = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WURZEL))

import sua_messung as M  # noqa: E402
import sua_search as S  # noqa: E402

DATEI = WURZEL / "data" / "sua_messreihe.json"
BAND_PUNKTE = 0.05


def lauf() -> None:
    t0 = time.time()
    daten: dict = {"meta": {"instanzen": list(M.INSTANZEN), "text": "Eingefroren mit tools/sua_make_zuteilung.py; alle Instanzen erfunden; Lücken in Prozentpunkten."}}
    daten["s1"] = {}
    for n in M.GITTER:
        for art in ("ein_gipfel", "zwei_gipfel", "flach"):
            for alpha in M.ALPHAS:
                for kappa in M.KAPPAS:
                    daten["s1"][f"{n}|{art}|{alpha}|{kappa}"] = M.zelle_s1(n, art, alpha, kappa)
        print(f"s1 {n}x{n}: {time.time() - t0:.0f} s", flush=True)
    daten["tau"] = {}
    for art in ("ein_gipfel", "zwei_gipfel"):
        for alpha in (0.5, 1.0, 2.0):
            for kappa in (0.25, 0.5, 1.0):
                for tau in M.TAUS:
                    daten["tau"][f"{art}|{alpha}|{kappa}|{tau}"] = M.zelle_tau(art, alpha, kappa, tau)
    print(f"tau: {time.time() - t0:.0f} s", flush=True)
    daten["s2"] = {}
    for art in ("ein_gipfel", "zwei_gipfel"):
        for gamma in M.GAMMAS:
            for alpha in M.ALPHAS_S2:
                for kappa in M.KAPPAS:
                    daten["s2"][f"{art}|{gamma}|{alpha}|{kappa}"] = M.zelle_s2(art, gamma, alpha, kappa)
        print(f"s2 {art}: {time.time() - t0:.0f} s", flush=True)
    daten["rucksack_klein"] = M.kleine_rucksack_instanzen()
    daten["nichtfund"] = {f"{art}|{alpha}|{kappa}": M.zelle_nichtfund(art, alpha, kappa)
                          for art in ("ein_gipfel", "zwei_gipfel") for alpha in M.ALPHAS_NF for kappa in M.KAPPAS_NF}
    daten["falsches_alpha"] = {f"{art}|{kappa}": M.zelle_falsches_alpha(art, kappa) for art in ("ein_gipfel", "zwei_gipfel") for kappa in M.KAPPAS_NF}
    print(f"nichtfund: {time.time() - t0:.0f} s", flush=True)
    daten["rechteck_runden"] = {f"{art}|{alpha}|{B}": M.zelle_rechteck_runden(art, alpha, B)
                                for art in ("ein_gipfel", "zwei_gipfel") for alpha in (0.5, 1.0, 2.0) for B in (32, 64)}
    print(f"rechteck_runden: {time.time() - t0:.0f} s", flush=True)
    daten["posterior_band"] = {f"{art}|{alpha}|{kappa}": M.posterior_band(art, alpha, kappa)
                               for art in S.PRIORS for alpha in (0.5, 1.0, 2.0, 3.0) for kappa in (0.25, 0.5, 1.0)}
    daten["aufzaehlung"] = M.aufzaehlung_faelle()
    print(f"fertig nach {time.time() - t0:.0f} s")
    DATEI.parent.mkdir(exist_ok=True)
    DATEI.write_text(json.dumps(daten, separators=(",", ":")), encoding="utf-8")


def pruefen() -> int:
    daten = json.loads(DATEI.read_text(encoding="utf-8"))
    fehler = 0

    def vergleiche(name: str, neu: float, alt: float) -> None:
        nonlocal fehler
        ok = abs(neu - alt) <= BAND_PUNKTE
        print(("OK    " if ok else "FEHLER"), name, f"gespeichert {alt:.3f}, neu {neu:.3f}")
        fehler += 0 if ok else 1

    for n, art, alpha, kappa in ((8, "ein_gipfel", 1.0, 1.0), (8, "zwei_gipfel", 0.5, 0.25), (12, "ein_gipfel", 2.0, 0.5)):
        z = M.zelle_s1(n, art, alpha, kappa)
        alt = daten["s1"][f"{n}|{art}|{alpha}|{kappa}"]
        for k in ("voll", "durchgang", "rechteck"):
            vergleiche(f"s1 {n} {art} {alpha} {kappa} {k}", z[k]["mittel"], alt[k]["mittel"])
    z = M.zelle_s2("ein_gipfel", 1.0, 1.0, 0.5)
    vergleiche("s2 ein_gipfel 1.0 1.0 0.5 voll_diff", z["voll_diff"]["mittel"], daten["s2"]["ein_gipfel|1.0|1.0|0.5"]["voll_diff"]["mittel"])
    z = M.zelle_nichtfund("zwei_gipfel", 1.0, 0.5)
    vergleiche("nichtfund zwei_gipfel 1.0 0.5 alt", z["alt"]["mittel"], daten["nichtfund"]["zwei_gipfel|1.0|0.5"]["alt"]["mittel"])
    z = M.zelle_falsches_alpha("ein_gipfel", 1.0)
    vergleiche("falsches_alpha ein_gipfel 1.0 (10)", z["10.0"]["mittel"], daten["falsches_alpha"]["ein_gipfel|1.0"]["10.0"]["mittel"])
    return fehler


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    if a.check:
        sys.exit(1 if pruefen() else 0)
    lauf()
