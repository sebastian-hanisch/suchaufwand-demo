"""Rechnet die Wege der Ansicht 3 mit CP-SAT vor und schreibt data/sua_weg.json (eingefroren: die App rechnet CP-SAT nie live).

    python tools/sua_make_weg.py --teil 5      # 5x5: Wegoptimum und Orienteering-Fassung je Instanz, dazu die Tabelle der Lücken (dauert rund 15 Minuten)
    python tools/sua_make_weg.py --teil 8      # 8x8: nur die Tabelle (30 Instanzen je Zelle, 10 s je Lösung)
    python tools/sua_make_weg.py --check       # prüft die gespeicherten Wege und löst 10 Instanzen neu (Band)

Instanzen: Zufalls-Seed 1 bis 30 je Prior (`sua_search.make_prior`). Gelöst wird je Instanz mit 6 s (5x5) bzw. 10 s (8x8) Zeitgrenze; ist die Lösung nicht
bewiesen, steht das im Datensatz (`bew` = 0) und die App sagt „beste gefundene Lösung“. Die Lücken beziehen sich auf das beste bekannte Ergebnis
(Weg des Lösers oder ein Verfahren, falls ein Verfahren besser ist). OR-Tools steht nur in requirements-dev.txt.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

WURZEL = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WURZEL))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import sua_path as W  # noqa: E402
import sua_search as S  # noqa: E402
import sua_zufall as Z  # noqa: E402

DATEI = WURZEL / "data" / "sua_weg.json"
INSTANZEN = tuple(range(1, 31))
PRIORS5 = ("ein_gipfel", "zwei_gipfel")
ALPHAS5 = (0.3, 1.0, 3.0, 10.0)
TS5 = (6, 10, 15)
ALPHAS8 = (0.5, 3.0)
TS8 = (16, 32)
ZEIT5, ZEIT8 = 6.0, 10.0
VERFAHREN = ("greedy_kurz", "greedy_ziel", "maehen_ganz", "maehen_rechteck")


def _stat(werte, seed: int) -> dict:
    x = np.asarray(werte, dtype=float)
    lo, hi = Z.bootstrap_intervall(x, seed)
    return {"mittel": float(x.mean()), "lo": lo, "hi": hi, "anteil_positiv": float(np.mean(x > 1e-7))}


def verfahren_werte(p, alpha: float, n: int, T: int) -> dict:
    wege = {"greedy_kurz": W.greedy_kurzsichtig(p, alpha, n, T), "greedy_ziel": W.greedy_mit_ziel(p, alpha, n, T),
            "maehen_ganz": W.maehen_ganz(n, T), "maehen_rechteck": W.maehen_im_rechteck(p, alpha, n, T)}
    return {k: W.weg_wert(p, alpha, w) for k, w in wege.items()}


def zelle_tabelle(zeilen: list[dict], seed: int) -> dict:
    """Aggregiert die Instanzzeilen einer Zelle zu Mittel, Intervall und Anteil je Verfahren (Prozentpunkte Entdeckungswahrscheinlichkeit)."""
    opt_eff = [max([z["opt"]] + [z[v] for v in VERFAHREN] + ([z["o_wert"]] if "o_wert" in z else [])) for z in zeilen]
    out = {"opt_mittel": float(np.mean(opt_eff)) * 100, "unbewiesen": int(sum(1 for z in zeilen if not z["bew"])), "n": len(zeilen)}
    out["wegpreis"] = _stat([100 * (z["schranke"] - o) for z, o in zip(zeilen, opt_eff)], seed)
    for v in VERFAHREN:
        out[v] = _stat([100 * (o - z[v]) for z, o in zip(zeilen, opt_eff)], seed + 1)
    if "o_wert" in zeilen[0]:
        out["orient"] = _stat([100 * (o - z["o_wert"]) for z, o in zip(zeilen, opt_eff)], seed + 2)
        out["ueberschaetzung"] = _stat([100 * (z["o_praemie"] - z["o_wert"]) for z in zeilen], seed + 3)
        out["unbewiesen_orient"] = int(sum(1 for z in zeilen if not z["o_bew"]))
    return out


def lauf(teil: int) -> None:
    from sua_cpsat import exakt_weg, orienteering_weg

    n = 5 if teil == 5 else 8
    priors, alphas, Ts = (PRIORS5, ALPHAS5, TS5) if teil == 5 else (PRIORS5, ALPHAS8, TS8)
    zeit = ZEIT5 if teil == 5 else ZEIT8
    daten = json.loads(DATEI.read_text(encoding="utf-8")) if DATEI.exists() else {}
    daten["meta"] = {"zeit_5x5_s": ZEIT5, "zeit_8x8_s": ZEIT8, "instanzen": list(INSTANZEN), "basis": 0,
                     "text": "Eingefroren mit tools/sua_make_weg.py (CP-SAT, OR-Tools); alle Instanzen erfunden."}
    kopf = f"g{n}"
    daten[kopf] = {"inst": {}, "tabelle": {}}
    seed = 11
    t0 = time.time()
    for prior in priors:
        for alpha in alphas:
            for T in Ts:
                zeilen = []
                for inst in INSTANZEN:
                    p = S.make_prior(prior, n, inst)
                    weg, bew = exakt_weg(p, alpha, n, T, zeit=zeit)
                    z = {"opt": W.weg_wert(p, alpha, weg), "bew": int(bew), "weg": weg, "schranke": S.wert(p, alpha, S.greedy_zuteilung(p, alpha, T))}
                    z.update(verfahren_werte(p, alpha, n, T))
                    if teil == 5:
                        ow, obew = orienteering_weg(p, alpha, n, T, zeit=zeit)
                        z.update({"o_wert": W.weg_wert(p, alpha, ow), "o_praemie": W.praemien_summe(p, ow), "o_bew": int(obew), "o_weg": ow})
                    zeilen.append(z)
                schluessel = f"{prior}|{alpha}|{T}"
                daten[kopf]["tabelle"][schluessel] = zelle_tabelle(zeilen, seed)
                seed += 10
                if teil == 5:
                    daten[kopf]["inst"][schluessel] = [{k: (round(v, 12) if isinstance(v, float) else v) for k, v in z.items()
                                                          if k in ("opt", "bew", "weg", "o_wert", "o_praemie", "o_bew", "o_weg")} for z in zeilen]
                print(f"{schluessel}: {time.time() - t0:.0f} s, unbewiesen {daten[kopf]['tabelle'][schluessel]['unbewiesen']}", flush=True)
                DATEI.parent.mkdir(exist_ok=True)
                DATEI.write_text(json.dumps(daten, separators=(",", ":")), encoding="utf-8")
    print(f"fertig nach {time.time() - t0:.0f} s")


def pruefen() -> int:
    """Gespeicherte Wege gültig und mit gespeichertem Wert; 10 Instanzen neu gelöst (bewiesene Werte gleich auf 1e-6, sonst höchstens schlechter)."""
    from sua_cpsat import exakt_weg

    daten = json.loads(DATEI.read_text(encoding="utf-8"))
    fehler = 0
    gesehen = 0
    for schluessel, liste in daten["g5"]["inst"].items():
        prior, alpha, T = schluessel.split("|")
        alpha, T = float(alpha), int(T)
        assert len(liste) == len(INSTANZEN), schluessel
        for inst, z in zip(INSTANZEN, liste):
            p = S.make_prior(prior, 5, inst)
            for weg, wert in ((z["weg"], z["opt"]), (z["o_weg"], z["o_wert"])):
                if not W.ist_gueltiger_weg(weg, 5, T) or abs(W.weg_wert(p, alpha, weg) - wert) > 1e-9:
                    print("FEHLER gespeicherter Weg:", schluessel, inst)
                    fehler += 1
            gesehen += 1
    print(f"{gesehen} gespeicherte Instanzen geprüft")
    schluessel_liste = list(daten["g5"]["inst"])
    for k in range(10):
        schluessel = schluessel_liste[(k * 7) % len(schluessel_liste)]
        prior, alpha, T = schluessel.split("|")
        alpha, T = float(alpha), int(T)
        inst = INSTANZEN[(k * 11) % len(INSTANZEN)]
        z = daten["g5"]["inst"][schluessel][inst - 1]
        p = S.make_prior(prior, 5, inst)
        weg, bew = exakt_weg(p, alpha, 5, T, zeit=ZEIT5)
        neu = W.weg_wert(p, alpha, weg)
        ok = abs(neu - z["opt"]) < 1e-6 if (bew and z["bew"]) else neu >= z["opt"] - 1e-6 or not bew
        print(("OK    " if ok else "FEHLER"), schluessel, inst, f"gespeichert {z['opt']:.9f}, neu {neu:.9f}, bewiesen {bew}")
        fehler += 0 if ok else 1
    return fehler


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--teil", type=int, choices=(5, 8))
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    if a.check:
        sys.exit(1 if pruefen() else 0)
    if a.teil:
        lauf(a.teil)
    else:
        ap.error("--teil 5, --teil 8 oder --check")
