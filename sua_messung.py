"""Messreihen der Ansichten 1 und 2 (je Zelle 30 erfundene Instanzen mit Zufalls-Seed 1 bis 30, über alle Regler gepaart).

Die Funktionen rechnen **eine Zelle** der Messreihe; `tools/sua_make_zuteilung.py` fährt alle Zellen und friert das Ergebnis in
`data/sua_messreihe.json` ein, `tests/test_claims.py` rechnet einzelne Zellen neu und vergleicht im Band. Lücke = Optimum minus Verfahren in
Prozentpunkten Entdeckungswahrscheinlichkeit; Intervall = 95-%-Bootstrap-Intervall der Mittelwerte (2000 Ziehungen aus dem SplitMix64).
"""

import itertools
from math import exp

import numpy as np

import sua_search as S
import sua_zufall as Z
from sua_rng import SplitMix64

INSTANZEN = tuple(range(1, 31))
GITTER = (8, 12)
ALPHAS = (0.2, 0.5, 1.0, 2.0, 3.0, 10.0)
KAPPAS = (0.125, 0.25, 0.5, 1.0, 2.0)
TAUS = (0.5, 0.8, 0.95, 0.99)
GAMMAS = (0.0, 0.5, 1.0)
ALPHAS_S2 = (0.5, 1.0, 3.0, 10.0)
ALPHAS_NF = (0.5, 1.0, 2.0)
KAPPAS_NF = (0.25, 0.5, 1.0, 2.0)
ALPHAS_ANG = (0.25, 0.5, 2.0, 4.0, 10.0)


def seed_aus(text: str) -> int:
    """Fester Seed aus einem Text (FNV-1a mit 64 Bit; nie Pythons hash(), der je Prozess gesalzen ist)."""
    h = 0xCBF29CE484222325
    for b in text.encode("utf-8"):
        h = ((h ^ b) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h % 1_000_000_007


def stat(werte, schluessel: str) -> dict:
    x = np.asarray(werte, dtype=float)
    lo, hi = Z.bootstrap_intervall(x, seed_aus(schluessel))
    return {"mittel": float(x.mean()), "lo": lo, "hi": hi, "anteil": float(np.mean(x > 1e-7))}


def budget(kappa: float, basis: float) -> int:
    """Budget in ganzen Durchgängen (Kosteneinheiten): kappa mal der Zellenzahl bzw. der Gesamtkosten, gerundet."""
    return int(round(kappa * basis))


# ------------------------------------------------------------------------------------------------ Ansicht 1: gleicher Preis
def zelle_s1(n: int, art: str, alpha: float, kappa: float, tau: float = 0.95, instanzen=INSTANZEN) -> dict:
    N = n * n
    B = budget(kappa, N)
    c = np.ones(N)
    L = {k: [] for k in ("opt", "stetig", "voll", "durchgang", "rechteck")}
    nie = {k: [] for k in L}
    masse = {k: [] for k in L}
    for inst in instanzen:
        p = S.make_prior(art, n, inst)
        z = {"opt": S.greedy_zuteilung(p, alpha, B), "voll": S.voll_strategie(c, alpha, B, p, tau), "durchgang": S.durchgang_strategie(c, B, p),
             "rechteck": S.rechteck_strategie(p, c, alpha, B, n)}
        for k, zz in z.items():
            L[k].append(S.wert(p, alpha, zz))
            a, m = S.nie_abgesucht(p, zz)
            nie[k].append(a)
            masse[k].append(m)
        L["stetig"].append(S.obergrenze_teilbar(p, c, alpha, B))
    opt = np.array(L["opt"])
    kopf = f"s1|{n}|{art}|{alpha}|{kappa}|{tau}"
    zeile = {"B": B, "P_opt": float(opt.mean()), "stetig_ueber_opt": float(np.mean(np.array(L["stetig"]) - opt)) * 100,
             "opt_nie": float(np.mean(nie["opt"])), "opt_masse": float(np.mean(masse["opt"]))}
    for k in ("voll", "durchgang", "rechteck"):
        zeile[k] = stat((opt - np.array(L[k])) * 100, f"{kopf}|{k}")
        zeile[k]["nie"] = float(np.mean(nie[k]))
        zeile[k]["masse"] = float(np.mean(masse[k]))
    return zeile


def zelle_tau(art: str, alpha: float, kappa: float, tau: float, n: int = 8) -> dict:
    """Lücke der Strategie „vollständig“ in Abhängigkeit von der Schwelle tau (eine Zelle der kleinen Messreihe)."""
    z = zelle_s1(n, art, alpha, kappa, tau)
    return {"B": z["B"], "voll": z["voll"]}


# ------------------------------------------------------------------------------------------------ Ansicht 1: Gelände
def zelle_s2(art: str, gamma: float, alpha: float, kappa: float, n: int = 8, instanzen=INSTANZEN) -> dict:
    N = n * n
    L = {k: [] for k in ("dp", "greedy_rat", "voll_p", "voll_rat", "durch_p", "durch_rat", "rechteck")}
    cvs, spannen = [], []
    for inst in instanzen:
        p = S.make_prior(art, n, inst)
        ci = S.make_costs(n, gamma, inst)
        cf = ci.astype(float)
        cvs.append(cf.std() / cf.mean())
        spannen.append(cf.max() / cf.min())
        B = budget(kappa, cf.sum())
        rat = p / cf
        L["dp"].append(S.rucksack_dp(p, ci, alpha, B))
        L["greedy_rat"].append(S.wert(p, alpha, S.greedy_verhaeltnis(p, cf, alpha, B)))
        L["voll_p"].append(S.wert(p, alpha, S.voll_strategie(cf, alpha, B, p)))
        L["voll_rat"].append(S.wert(p, alpha, S.voll_strategie(cf, alpha, B, rat)))
        L["durch_p"].append(S.wert(p, alpha, S.durchgang_strategie(cf, B, p)))
        L["durch_rat"].append(S.wert(p, alpha, S.durchgang_strategie(cf, B, rat)))
        L["rechteck"].append(S.wert(p, alpha, S.rechteck_strategie(p, cf, alpha, B, n)))
    dp = np.array(L["dp"])
    kopf = f"s2|{n}|{art}|{gamma}|{alpha}|{kappa}"
    zeile = {"kosten_cv": float(np.mean(cvs)), "kosten_max_min": float(np.mean(spannen)), "P_opt": float(dp.mean())}
    for k in ("greedy_rat", "voll_p", "voll_rat", "durch_p", "durch_rat", "rechteck"):
        zeile[k] = stat((dp - np.array(L[k])) * 100, f"{kopf}|{k}")
    for name, a, b in (("voll_diff", "voll_rat", "voll_p"), ("durch_diff", "durch_rat", "durch_p")):
        d = (np.array(L[a]) - np.array(L[b])) * 100
        zeile[name] = stat(d, f"{kopf}|{name}")
        zeile[name]["schlechter"] = float(np.mean(d < -1e-7))
        zeile[name]["max_abweichung_null"] = float(np.max(np.abs(d)))
    return zeile


def kleine_rucksack_instanzen() -> dict:
    """30 kleine Instanzen (3x3, Kosten 1 bis 3, Budget 8, Entdeckungsstärke 0,5 / 1 / 2,5 reihum): Greedy nach Zuwachs je Kosten gegen die exakte Rucksack-Rechnung."""
    luecken = []
    for k in range(30):
        rng = SplitMix64(31_000 + k)
        p = S._prior_ein_gipfel(3, rng)
        c = 1 + Z.integers(rng, 9, 3)
        alpha = (0.5, 1.0, 2.5)[k % 3]
        exakt = S.rucksack_dp(p, c, alpha, 8)
        luecken.append(100.0 * (exakt - S.wert(p, alpha, S.greedy_verhaeltnis(p, c.astype(float), alpha, 8))))
    l = np.array(luecken)
    return {"instanzen": 30, "unter_optimum": int(np.sum(l > 1e-7)), "max_luecke": float(l.max())}


# ------------------------------------------------------------------------------------------------ Ansicht 2: Nichtfund
def zelle_nichtfund(art: str, alpha: float, kappa: float, n: int = 8, instanzen=INSTANZEN) -> dict:
    N = n * n
    B = budget(kappa, N)
    B1, B2 = B // 2, B - B // 2
    a, gb, gc, diff = [], [], [], []
    for inst in instanzen:
        p = S.make_prior(art, n, inst)
        r = S.nichtfund_runden(p, alpha, B1, B2)
        a.append(r["gesamt_a"])
        gb.append((r["gesamt_a"] - r["gesamt_b"]) * 100)
        gc.append((r["gesamt_a"] - r["gesamt_c"]) * 100)
        diff.append(abs(r["gesamt_a"] - r["einplan"]))
    kopf = f"nf|{art}|{alpha}|{kappa}"
    return {"B": B, "P_a": float(np.mean(a)) * 100, "alt": stat(gb, kopf + "|alt"), "voll": stat(gc, kopf + "|voll"),
            "einplan_max_abw": float(max(diff)), "einplan_gleich": int(sum(1 for d in diff if d < 1e-9)), "n": len(diff)}


def zelle_falsches_alpha(art: str, kappa: float, n: int = 8, instanzen=INSTANZEN) -> dict:
    N = n * n
    B = budget(kappa, N)
    out = {"B": B}
    for aa in ALPHAS_ANG:
        g = []
        for inst in instanzen:
            p = S.make_prior(art, n, inst)
            _, _, pr, pa = S.falsches_alpha(p, 1.0, aa, B)
            g.append((pr - pa) * 100)
        out[str(aa)] = stat(g, f"fa|{art}|{kappa}|{aa}")
    return out


def zelle_rechteck_runden(art: str, alpha: float, B: int, n: int = 8, instanzen=INSTANZEN) -> dict:
    gb, gc, gd, vorteil = [], [], [], []
    for inst in instanzen:
        p = S.make_prior(art, n, inst)
        r = S.rechteck_runden(p, alpha, n, B)
        gb.append((r["optimum"] - r["kurzsichtig"]) * 100)
        gc.append((r["optimum"] - r["vorausschau"]) * 100)
        gd.append((r["optimum"] - r["ein_rechteck"]) * 100)
        vorteil.append((r["vorausschau"] - r["kurzsichtig"]) * 100)
    kopf = f"rr|{art}|{alpha}|{B}"
    out = {"kurzsichtig": stat(gb, kopf + "|b"), "vorausschau": stat(gc, kopf + "|c"), "ein_rechteck": stat(gd, kopf + "|d")}
    out["vorteil_anteil"] = float(np.mean(np.array(vorteil) > 1e-7))
    out["vorteil_mittel"] = float(np.mean(vorteil))
    return out


# ------------------------------------------------------------------------------------------------ Orakel-Zählung und Posterior-Band
def kompositionen(zellen: int, b: int) -> np.ndarray:
    """Alle Zuteilungen von b Durchgängen auf `zellen` Zellen (Mehrfachmengen), als Zeilen mit der Zahl der Durchgänge je Zelle."""
    zeilen = np.zeros((0, zellen), dtype=np.int8)
    idx = np.array(list(itertools.combinations_with_replacement(range(zellen), b)), dtype=np.int64)
    zeilen = np.zeros((len(idx), zellen), dtype=np.int8)
    for j in range(b):
        np.add.at(zeilen, (np.arange(len(idx)), idx[:, j]), 1)
    return zeilen


def aufzaehlung_abweichung(n: int, b: int, art: str, alpha: float, inst: int) -> float:
    """Abstand zwischen dem Wert der Grenzzuwachs-Zuteilung und dem besten Wert aller Zuteilungen (Aufzählung)."""
    p = S.make_prior(art, n, inst)
    zz = kompositionen(n * n, b).astype(float)
    bester = float((p[None, :] * (1.0 - np.exp(-alpha * zz))).sum(axis=1).max())
    return abs(bester - S.wert(p, alpha, S.greedy_zuteilung(p, alpha, b)))


def aufzaehlung_faelle(vollstaendig: bool = True) -> dict:
    """Die 120 Fälle: 4x4 mit Budget 3, 5 und 8 (2 Priors x 5 Instanzen x 3 Entdeckungsstärken = 90) und 6x6 mit Budget 3 und 4 (5 Instanzen x 3 = 30)."""
    faelle, maxd = 0, 0.0
    for b in (3, 5, 8):
        for art in ("ein_gipfel", "zwei_gipfel"):
            for inst in range(1, 6):
                for alpha in (0.3, 1.0, 2.5):
                    maxd = max(maxd, aufzaehlung_abweichung(4, b, art, alpha, inst))
                    faelle += 1
    for b in (3, 4):
        for inst in range(1, 6):
            for alpha in (0.3, 1.0, 2.5):
                maxd = max(maxd, aufzaehlung_abweichung(6, b, "ein_gipfel", alpha, inst))
                faelle += 1
    return {"faelle": faelle, "max_abweichung": maxd}


def posterior_band(art: str, alpha: float, kappa: float, n: int = 8, instanzen=INSTANZEN) -> dict:
    """Nach der optimalen ganzzahligen Zuteilung: gilt die Schwelle auf dem Grenzzuwachs, wie weit liegen die besuchten Zellen im Posterior auseinander,
    und liegt eine unbesuchte Zelle über einer besuchten? Dazu bei teilbarem Aufwand die Abweichung vom gleichen Niveau."""
    B = budget(kappa, n * n)
    linie_ok, quot_max, unbesucht_ueber, stetig_abw = 0, 1.0, 0, 0.0
    for inst in instanzen:
        p = S.make_prior(art, n, inst)
        z = S.greedy_zuteilung(p, alpha, B)
        zuletzt, naechster = S.linie_grenzzuwachs(p, alpha, z)
        linie_ok += int(zuletzt >= naechster * (1 - 1e-12))
        w = p * np.exp(-alpha * z)
        bes = z > 0
        if bes.any():
            quot_max = max(quot_max, float(w[bes].max() / w[bes].min()))
            if (~bes).any() and w[~bes].max() > w[bes].min() * (1 + 1e-12):
                unbesucht_ueber += 1
        zs, _ = S.wasserstand(p, np.ones(n * n), alpha, float(B))
        bs = zs > 1e-12
        ws = (p * np.exp(-alpha * zs))[bs]
        stetig_abw = max(stetig_abw, float((ws.max() - ws.min()) / ws.max())) if bs.any() else stetig_abw
    return {"n": len(instanzen), "linie_ok": linie_ok, "quotient_max": quot_max, "exp_alpha": exp(alpha), "unbesucht_ueber_besucht": unbesucht_ueber,
            "stetig_abweichung": stetig_abw}
