"""Auswertung: Rechnung je Ansicht (eine Instanz, live), Aggregate der vorgerechneten Messreihen und Verdikte als reine Funktionen.

Jedes Verdikt ist `verdict_*(...) -> (Art, Text)`; die App ruft `getattr(st, Art)(Text)`. Zahlen kommen aus der Rechnung oder aus den
eingefrorenen Messreihen (`data/`), nie aus dem Text. Alle Instanzen sind erfunden; Lücken sind Prozentpunkte Entdeckungswahrscheinlichkeit.
"""

from math import exp

import numpy as np

import sua_constants as C
import sua_drift as D
import sua_frozen as F
import sua_path as W
import sua_search as S

N_INSTANZEN = 30
ALPHA_ANGENOMMEN_TEXT = {"0.25": "0,25", "0.5": "0,5", "2.0": "2", "4.0": "4", "10.0": "10"}


def z(x: float, n: int = 1) -> str:
    """Zahl mit Dezimalkomma."""
    return f"{x:.{n}f}".replace(".", ",")


def ivl(st: dict, n: int = 1) -> str:
    """Mittelwert mit Bootstrap-Intervall: „27,9 [26,3; 29,6]“."""
    return f"{z(st['mittel'], n)} [{z(st['lo'], n)}; {z(st['hi'], n)}]"


def prior_name(art: str) -> str:
    return C.PRIOR_SATZ[art]


# ================================================================================================ Ansicht 1: Zuteilung
def analyse_zuteilung(n: int, art: str, alpha: float, kappa: float, tau: float, seed: int) -> dict:
    """Eine Instanz: Optimum (Grenzzuwachs), „vollständig“, „ein Durchgang je Zelle“, bestes Rechteck; Lücken in Prozentpunkten."""
    N = n * n
    B = int(round(kappa * N))
    p = S.make_prior(art, n, seed)
    c = np.ones(N)
    zs = {"opt": S.greedy_zuteilung(p, alpha, B), "voll": S.voll_strategie(c, alpha, B, p, tau), "durchgang": S.durchgang_strategie(c, B, p),
          "rechteck": S.rechteck_strategie(p, c, alpha, B, n)}
    P = {k: S.wert(p, alpha, zz) for k, zz in zs.items()}
    return {"n": n, "p": p, "B": B, "z": zs, "P": P, "luecke": {k: S.luecke_punkte(P["opt"], P[k]) for k in ("voll", "durchgang", "rechteck")},
            "nie": {k: S.nie_abgesucht(p, zz) for k, zz in zs.items()}, "obergrenze": S.obergrenze_teilbar(p, c, alpha, float(B)),
            "linie": S.linie_grenzzuwachs(p, alpha, zs["opt"])}


def aggregiere_s1(s1: dict, verfahren: str, alpha: float, mit_flach: bool = False) -> dict:
    """Mittlere Lücke eines Verfahrens bei der Entdeckungsstärke alpha über die Zellen von 8x8 und 12x12 mit ein oder zwei Gipfeln (20 Zellen),
    dazu der größte Zellenmittelwert und die Zahl der Zellen, deren Intervall über 0 liegt."""
    zellen = []
    for key, row in s1.items():
        n, art, a, kappa = key.split("|")
        if float(a) == alpha and (mit_flach or art != "flach"):
            zellen.append(row[verfahren])
    if not zellen:
        return {}
    return {"zellen": len(zellen), "mittel": float(np.mean([c["mittel"] for c in zellen])), "maximum": float(max(c["mittel"] for c in zellen)),
            "ueber_null": int(sum(1 for c in zellen if c["lo"] > 0))}


def verdict_zuteilung(n: int, art: str, alpha: float, kappa: float, tau: float, row: dict | None, tau_row: dict | None) -> tuple[str, str]:
    """Verdikt Ansicht 1 (Zuteilung): Zahlen der vorgerechneten Zelle (Mittel über 30 Instanzen, Intervall in eckigen Klammern)."""
    if row is None:
        return "warning", "Für diese Einstellung liegt keine vorgerechnete Messreihe vor."
    voll = row["voll"] if tau == 0.95 else (tau_row["voll"] if tau_row else None)
    voll_text = (f"„vollständig“ (τ = {z(tau, 2)}) {ivl(voll)}" if voll else f"„vollständig“ (τ = {z(tau, 2)}) ist für diese Zelle nicht vorgerechnet (nur τ = 0,95; die Zahl der gewählten Instanz steht oben)")
    text = (f"{n}×{n}, {prior_name(art)}, α = {z(alpha, 1)}, Budget {row['B']} Durchgänge: Das Optimum erreicht im Mittel über {N_INSTANZEN} Instanzen {z(100 * row['P_opt'])} %. "
            f"„Ein Durchgang je Zelle“ liegt {ivl(row['durchgang'])} Prozentpunkte dahinter, {voll_text}, das beste Rechteck {ivl(row['rechteck'])}. "
            f"Das Optimum lässt {z(100 * row['opt_nie'], 0)} % der Zellen unberührt, darin {z(100 * row['opt_masse'])} % der Prior-Masse.")
    if max(row["durchgang"]["mittel"], row["rechteck"]["mittel"], (voll or row["voll"])["mittel"]) < 0.01:
        text += " Bei dieser Entdeckungsstärke und diesem Budget sind alle drei Verfahren gleich (Lücke unter 0,01 Punkten)."
    if art == "flach" and row["durchgang"]["mittel"] < 1e-9:
        text += " Bei flachem Prior ist „ein Durchgang je Zelle“ das Optimum (Lücke 0): es gibt nichts zu priorisieren."
    return "info", text


# ================================================================================================ Ansicht 1: Gelände
def analyse_gelaende(art: str, gamma: float, alpha: float, kappa: float, seed: int) -> dict:
    """Eine Instanz 8x8 mit Gelände: ganzzahlige Kosten je Durchgang, exaktes Rucksack-Optimum, aktualisierender Greedy nach Zuwachs je Kosten,
    „vollständig“ und „ein Durchgang“ nach Prior bzw. nach Prior je Aufwand, bestes Rechteck."""
    n = C.GITTER_GELAENDE
    p = S.make_prior(art, n, seed)
    ci = S.make_costs(n, gamma, seed)
    cf = ci.astype(float)
    B = int(round(kappa * cf.sum()))
    rat = p / cf
    zs = {"greedy_rat": S.greedy_verhaeltnis(p, cf, alpha, B), "voll_p": S.voll_strategie(cf, alpha, B, p), "voll_rat": S.voll_strategie(cf, alpha, B, rat),
          "durch_p": S.durchgang_strategie(cf, B, p), "durch_rat": S.durchgang_strategie(cf, B, rat), "rechteck": S.rechteck_strategie(p, cf, alpha, B, n)}
    P = {k: S.wert(p, alpha, zz) for k, zz in zs.items()}
    P["dp"] = S.rucksack_dp(p, ci, alpha, B)
    return {"n": n, "p": p, "c": ci, "B": B, "z": zs, "P": P, "luecke": {k: S.luecke_punkte(P["dp"], P[k]) for k in zs}}


def aggregiere_s2(s2: dict, gamma: float) -> dict:
    """Über die Zellen eines Geländefaktors (2 Priors x 4 Entdeckungsstärken x 5 Budgets): Mittel der Paardifferenz „nach Prior je Aufwand minus nach
    Prior allein“ für „vollständig“ und „ein Durchgang“, Zellen mit Intervall über 0 und unter 0, Greedy-Lücke gegen das exakte Optimum."""
    zellen = [row for key, row in s2.items() if float(key.split("|")[1]) == gamma]
    if not zellen:
        return {}
    out = {"zellen": len(zellen)}
    for name in ("voll_diff", "durch_diff"):
        out[name + "_mittel"] = float(np.mean([r[name]["mittel"] for r in zellen]))
        out[name + "_ueber"] = int(sum(1 for r in zellen if r[name]["lo"] > 0))
        out[name + "_unter"] = int(sum(1 for r in zellen if r[name]["hi"] < 0))
        out[name + "_gleich"] = int(sum(1 for r in zellen if r[name]["max_abweichung_null"] < 1e-9))
        out[name + "_max"] = float(max(r[name]["mittel"] for r in zellen))
    out["greedy_mittel"] = float(np.mean([r["greedy_rat"]["mittel"] for r in zellen]))
    out["greedy_max"] = float(max(r["greedy_rat"]["mittel"] for r in zellen))
    return out


def verdict_gelaende(gamma: float, agg: dict, klein: dict | None, zelle: dict | None) -> tuple[str, str]:
    """Verdikt Ansicht 1 (Gelände). `agg` aus `aggregiere_s2`, `klein` = Ergebnis der 30 kleinen Rucksack-Instanzen, `zelle` = die gewählte Zelle."""
    if not agg:
        return "warning", "Für diesen Geländefaktor liegt keine vorgerechnete Messreihe vor."
    m = agg["zellen"]
    if gamma == 0:
        text = (f"Bei γ = 0 kostet jeder Durchgang überall gleich viel: Ordnen nach Prior je Aufwand und nach Prior allein sind identisch (Differenz 0 in "
                f"{agg['voll_diff_gleich']} von {m} Zellen bei „vollständig“, {agg['durch_diff_gleich']} von {m} bei „ein Durchgang“).")
    else:
        text = (f"Gelände γ = {z(gamma, 1)}: Ordnen nach Prior je Aufwand statt nach Prior allein gewinnt bei „vollständig“ im Mittel {z(agg['voll_diff_mittel'])} "
                f"Punkte (größter Zellenmittelwert {z(agg['voll_diff_max'])}), bei „ein Durchgang“ {z(agg['durch_diff_mittel'])}; in {agg['voll_diff_ueber']} bzw. "
                f"{agg['durch_diff_ueber']} von {m} Zellen liegt das Intervall über 0, in {agg['voll_diff_unter']} bzw. {agg['durch_diff_unter']} darunter (je {N_INSTANZEN} Instanzen, 8×8).")
    text += (f" Der aktualisierende Greedy nach Zuwachs je Kosten liegt im Mittel {z(agg['greedy_mittel'], 2)} Punkte hinter dem exakten Optimum (größter Zellenmittelwert {z(agg['greedy_max'], 2)}).")
    if klein:
        text += (f" In {klein['unter_optimum']} von {klein['instanzen']} kleinen Instanzen (3×3, Kosten 1 bis 3, Budget 8) liegt er darunter, größte Lücke {z(klein['max_luecke'], 2)} (eine einzelne Stichprobe, keine Obergrenze).")
    if zelle and gamma > 0:
        text += f" Gewählte Zelle: Ordnen nach Prior je Aufwand gewinnt bei „ein Durchgang“ {ivl(zelle['durch_diff'])} Punkte."
    return "info", text


# ================================================================================================ Ansicht 2: Nichtfund und falsches α
def analyse_nichtfund(art: str, alpha: float, kappa: float, anteil: float, seed: int) -> dict:
    n = C.GITTER_NICHTFUND
    B = int(round(kappa * n * n))
    B1 = int(round(anteil * B))
    B2 = B - B1
    p = S.make_prior(art, n, seed)
    r = S.nichtfund_runden(p, alpha, B1, B2)
    r.update({"p": p, "n": n, "B": B, "B1": B1, "B2": B2, "linie1": S.linie_grenzzuwachs(p, alpha, r["z1"])})
    r["luecke_alt"] = S.luecke_punkte(r["gesamt_a"], r["gesamt_b"])
    r["luecke_voll"] = S.luecke_punkte(r["gesamt_a"], r["gesamt_c"])
    r["aktualisiert_minus_einplan"] = r["gesamt_a"] - r["einplan"]
    return r


def aggregiere_nichtfund(nf: dict) -> dict:
    """Über die 24 Zellen (2 Priors x 3 Entdeckungsstärken x 4 Budgets)."""
    if not nf:
        return {}
    alt = [r["alt"] for r in nf.values()]
    voll = [r["voll"] for r in nf.values()]
    return {"zellen": len(nf), "einplan_zellen": int(sum(1 for r in nf.values() if r["einplan_gleich"] == r["n"])),
            "einplan_max_abw": float(max(r["einplan_max_abw"] for r in nf.values())),
            "alt_min": min(a["mittel"] for a in alt), "alt_max": max(a["mittel"] for a in alt), "alt_mittel": float(np.mean([a["mittel"] for a in alt])),
            "alt_alle_schlechter": int(sum(1 for a in alt if a["anteil"] >= 1.0 - 1e-12)),
            "voll_min": min(a["mittel"] for a in voll), "voll_max": max(a["mittel"] for a in voll), "voll_mittel": float(np.mean([a["mittel"] for a in voll]))}


def aggregiere_rechteck_runden(rr: dict) -> dict:
    if not rr:
        return {}
    kurz = [r["kurzsichtig"]["mittel"] for r in rr.values()]
    vor = [r["vorausschau"]["mittel"] for r in rr.values()]
    return {"zellen": len(rr), "kurz_min": min(kurz), "kurz_max": max(kurz), "vor_min": min(vor), "vor_max": max(vor),
            "vorausschau_besser": int(sum(1 for r in rr.values() if r["vorteil_mittel"] > 0))}


def verdict_nichtfund(zelle: dict | None, agg: dict, agg_rr: dict, anteil: float) -> tuple[str, str]:
    if not agg or not agg_rr:
        return "warning", "Für den Nichtfund liegt keine vorgerechnete Messreihe vor."
    m = agg["zellen"]
    text = (f"Aktualisieren in Runden ist eine Zuteilung mit dem ganzen Budget: {agg['einplan_zellen']} von {m} Zellen (8×8, zwei Runden, {N_INSTANZEN} Instanzen je Zelle) ganzzahlig exakt gleich "
            f"(größte Abweichung {agg['einplan_max_abw']:.0e}). Gegenüber „alles auf einmal planen“ bringt das Aktualisieren deshalb 0, **solange der Aufwand frei verteilbar ist**. "
            f"Darf jede Runde nur ein Rechteck absuchen, liegt kurzsichtiges Aktualisieren {z(agg_rr['kurz_min'], 2)} bis {z(agg_rr['kurz_max'], 2)} Punkte hinter dem frei verteilten Optimum "
            f"und ist nicht mehr optimal (mit Vorausschau {z(agg_rr['vor_min'], 2)} bis {z(agg_rr['vor_max'], 2)}, in {agg_rr['vorausschau_besser']} von {agg_rr['zellen']} Zellen im Mittel besser). "
            f"Die alte Karte weiterbenutzen kostet {z(agg['alt_min'])} bis {z(agg['alt_max'])} Punkte (Mittel {z(agg['alt_mittel'])}; in {agg['alt_alle_schlechter']} von {m} Zellen alle {N_INSTANZEN} Instanzen schlechter), "
            f"„dort sofort weiter bis voll“ {z(agg['voll_min'])} bis {z(agg['voll_max'])} (Mittel {z(agg['voll_mittel'])}). "
            "Gilt für 8×8, zwei Runden, die Entdeckungsfunktion 1 − e^(−α·z) und eine Verteilung, die beisammen bleibt.")
    if zelle:
        text += (f" Gewählte Zelle: alte Karte {ivl(zelle['alt'])}, „dort weiter bis voll“ {ivl(zelle['voll'])} Punkte hinter der aktualisierten Zuteilung"
                 f" (Gesamt {z(zelle['P_a'])} %).")
    if anteil != 0.5:
        text += " Die vorgerechneten Zahlen gelten für 50 % des Budgets in Runde 1."
    return "info", text


def analyse_falsches_alpha(art: str, alpha_ang: float, kappa: float, seed: int) -> dict:
    n = C.GITTER_NICHTFUND
    B = int(round(kappa * n * n))
    p = S.make_prior(art, n, seed)
    z_r, z_a, p_r, p_a = S.falsches_alpha(p, C.ALPHA_WAHR, alpha_ang, B)
    return {"n": n, "p": p, "B": B, "z_richtig": z_r, "z_angenommen": z_a, "P_richtig": p_r, "P_angenommen": p_a, "luecke": S.luecke_punkte(p_r, p_a)}


def aggregiere_falsches_alpha(fa: dict) -> dict:
    """Über die 8 Zellen (2 Priors x 4 Budgets): größte Zellenmittel je angenommenem α."""
    if not fa:
        return {}
    out = {"zellen": len(fa)}
    for aa in C.ALPHAS_ANGENOMMEN:
        if aa == C.ALPHA_WAHR:
            continue
        werte = [r[str(aa)]["mittel"] for r in fa.values()]
        out[str(aa)] = {"min": min(werte), "max": max(werte), "mittel": sum(werte) / len(werte)}
    return out


def verdict_falsches_alpha(agg: dict, zelle: dict | None, alpha_ang: float) -> tuple[str, str]:
    if not agg:
        return "warning", "Für das falsche α liegt keine vorgerechnete Messreihe vor."
    hoch, tief = agg["10.0"], agg["0.25"]
    f2 = [agg["0.5"], agg["2.0"]]
    text = (f"Zuteilung mit α = 10 statt 1: bis {z(hoch['max'])} Punkte hinter der Zuteilung mit richtigem α (8×8, {agg['zellen']} Zellen, {N_INSTANZEN} Instanzen je Zelle); mit α = 0,25 bis {z(tief['max'])}; "
            f"um den Faktor 2 daneben (0,5 oder 2) {z(min(f['min'] for f in f2))} bis {z(max(f['max'] for f in f2))}; α = 4 bis {z(agg['4.0']['max'])}. ")
    vier = agg["4.0"]   # gleicher Faktor 4 wie α = 0,25: nur dieser Paarvergleich sagt etwas über „zu hoch gegen zu niedrig“
    mittel = f"im Mittel über die Zellen {z(tief['mittel'])} (α = 0,25) gegen {z(vier['mittel'])} (α = 4)"
    if tief["mittel"] > vier["mittel"]:
        text += f"Beim gleichen Faktor 4 schadet ein zu niedriges α etwas mehr als ein zu hohes: {mittel}. "
    elif tief["mittel"] < vier["mittel"]:
        text += f"Beim gleichen Faktor 4 schadet ein zu hohes α etwas mehr als ein zu niedriges: {mittel}. "
    else:
        text += f"Beim gleichen Faktor 4 schadet ein zu hohes und ein zu niedriges α gleich viel: {mittel}. "
    groesster = max((aa for aa in agg if aa != "zellen"), key=lambda aa: agg[aa]["max"])
    text += f"Der größte Zellenwert ({z(agg[groesster]['max'])}) gehört zu α = {ALPHA_ANGENOMMEN_TEXT[groesster]}. "
    text += "Gemessen nur für den wahren Wert α = 1 und diese Zellen."
    if zelle and str(alpha_ang) in zelle:
        text += f" Gewählte Zelle, α angenommen = {z(alpha_ang, 2)}: {ivl(zelle[str(alpha_ang)], 2)} Punkte."
    return "info", text


# ================================================================================================ Ansicht 3: Weg
def optimum_flach(T: int, alpha: float, N: int = 25) -> float | None:
    """Bei flachem Prior ist jeder Weg über T verschiedene Zellen optimal (T <= N): P = T/N * (1 - exp(-alpha))."""
    return T / N * (1.0 - exp(-alpha)) if T <= N else None


def analyse_weg(art: str, alpha: float, T: int, inst: int) -> dict:
    """Eine 5x5-Instanz: vier Verfahren live, Wegoptimum und Orienteering-Fassung aus den eingefrorenen Lösungen."""
    n = C.GITTER_WEG
    N = n * n
    p = S.make_prior(art, n, inst)
    wege = {"greedy_kurz": W.greedy_kurzsichtig(p, alpha, n, T), "greedy_ziel": W.greedy_mit_ziel(p, alpha, n, T), "maehen_ganz": W.maehen_ganz(n, T),
            "maehen_rechteck": W.maehen_im_rechteck(p, alpha, n, T)}
    bew = {}
    if art == "flach":
        wege["exakt"] = W.maehen_ganz(n, T)
        wege["orient"] = W.maehen_ganz(n, T)
        bew = {"exakt": True, "orient": True}
    else:
        fro = F.weg_instanz(art, alpha, T, inst)
        if fro and W.ist_gueltiger_weg(fro["weg"], n, T) and abs(W.weg_wert(p, alpha, fro["weg"]) - fro["opt"]) < 1e-9:
            wege["exakt"], bew["exakt"] = fro["weg"], bool(fro["bew"])
        if fro and W.ist_gueltiger_weg(fro["o_weg"], n, T) and abs(W.weg_wert(p, alpha, fro["o_weg"]) - fro["o_wert"]) < 1e-9:
            wege["orient"], bew["orient"] = fro["o_weg"], bool(fro["o_bew"])
    P = {k: W.weg_wert(p, alpha, w) for k, w in wege.items()}
    beste = max(P.values())
    schranke = S.wert(p, alpha, S.greedy_zuteilung(p, alpha, T))
    return {"n": n, "p": p, "T": T, "wege": wege, "P": P, "bew": bew, "beste": beste, "schranke": schranke, "wegpreis": S.luecke_punkte(schranke, beste),
            "luecke": {k: S.luecke_punkte(beste, v) for k, v in P.items() if k != "exakt"},
            "ueberschaetzung": (100.0 * (W.praemien_summe(p, wege["orient"]) - P["orient"])) if "orient" in wege else None}


def verdict_weg(art: str, alpha: float, T: int, tab: dict | None, live: dict) -> tuple[str, str]:
    """Verdikt Ansicht 3 aus der vorgerechneten Zelle (Mittel über 30 Instanzen); bei flachem Prior aus der Rechnung."""
    if art == "flach":
        g = live["luecke"]
        return "info", (f"Flacher Prior, T = {T}: jeder Weg über {T} verschiedene Zellen ist optimal (P = {z(100 * live['beste'])} %), die Instanz hat nichts zu priorisieren. "
                        f"Mähen über das ganze Gebiet: Lücke {z(g['maehen_ganz'], 2)}, Mähen im Rechteck {z(g['maehen_rechteck'], 2)}, Greedy kurzsichtig {z(g['greedy_kurz'], 2)}, "
                        f"mit Ziel {z(g['greedy_ziel'], 2)} Punkte; die Orienteering-Fassung ist hier gleich dem Optimum.")
    if tab is None:
        return "warning", "Für diese Einstellung liegt keine vorgerechnete Tabelle vor."
    text = (f"5×5, {prior_name(art)}, α = {z(alpha, 1)}, T = {T}: Das Wegoptimum erreicht im Mittel über {tab['n']} Instanzen {z(tab['opt_mittel'])} % "
            f"({tab['unbewiesen']} Lösungen nicht bewiesen). Wegpreis (Optimum ohne Weg minus Wegoptimum) {ivl(tab['wegpreis'])}; Greedy kurzsichtig {ivl(tab['greedy_kurz'])}, "
            f"mit Ziel {ivl(tab['greedy_ziel'])}; Mähen über das ganze Gebiet {ivl(tab['maehen_ganz'])}, Mähen im besten Rechteck {ivl(tab['maehen_rechteck'])} Punkte hinter dem Optimum. "
            f"Die Orienteering-Fassung (jede Zelle nur beim ersten Besuch gewertet, günstigster Fall) liegt {ivl(tab['orient'])} Punkte dahinter und überschätzt die Entdeckungswahrscheinlichkeit um {ivl(tab['ueberschaetzung'])}.")
    if alpha >= 3.0:
        text += (f" Bei großem α zählt jede Zelle praktisch nur einmal: das ist in der Wertung ein Orienteering-Problem (Prämie je Zelle einmalig; Fahrzeuge, Rückkehr zum Depot und Fahrzeiten kommen dort hinzu), siehe [gewinn-touren-demo]({C.ORIENTEERING_LINK}).")
    return "info", text


# ================================================================================================ Ansicht 4: Drift und Verzögerung
def analyse_drift(s0: float, Dif: float, sU: float, u: float, hyp: int, dv: float, t: float, t0: float, verzoegerung: float, L0: float) -> dict | None:
    """Kennzahlen der Ansicht 4. Ohne jede Streuung (s0 = D = sU = 0) gibt es keine Wolke: None.

    Aufwand E so, dass L zum Startzeitpunkt t0 den Wert L0 hat. Eine Wolke: geschlossene Form; zwei Hypothesen: Raster-Lösung (Wolken gleichen Gewichts,
    Mitten im Abstand 2 dv t). Verlust = 1 - P(t0 (1 + d)) / P(t0) bei festem Aufwand."""
    par = (s0, Dif, sU)
    if min(D.varianz(1.0, *par), D.varianz(t0, *par), D.varianz(t, *par), D.varianz(8.0, *par)) <= 0:
        return None
    E = D.aufwand_fuer_L0(L0, D.varianz(t0, *par))

    def p_bei(d: float) -> float:
        zt = t0 * (1.0 + d)
        return D.p_raster(E, 1.0, D.varianz(zt, *par), 2.0 * dv * zt) if hyp == 2 else D.p_gauss(E, 1.0, D.varianz(zt, *par))

    delays = sorted({0.0, *C.VERZOEGERUNGEN, verzoegerung})
    pd = {d: p_bei(d) for d in delays}
    p0 = pd[0.0]
    return {"par": par, "t": t, "t0": t0, "verzoegerung": verzoegerung, "p0": p0, "p_verzoegert": pd[verzoegerung], "verlust": 1.0 - pd[verzoegerung] / p0,
            "verluste": {d: 1.0 - pd[d] / p0 for d in C.VERZOEGERUNGEN}, "elastizitaet": D.elastizitaet_sekante(p0, pd[verzoegerung], 1.0 + verzoegerung),
            "kurve": [(t0 * (1.0 + d), pd[d]) for d in delays], "verhaeltnis": {tt: D.flaechenverhaeltnis(tt, par) for tt in (2.0, 4.0, 8.0)},
            "exponent_1_8": D.flaechenexponent(par, 1.0, 8.0), "L0": L0, "hyp": hyp, "dv": dv, "u": u}


def verdict_drift(a: dict | None, szenarien: dict | None) -> tuple[str, str]:
    if a is None:
        return "warning", "Ohne jede Streuung (s0 = D = sU = 0) gibt es keine Wolke: mindestens einer der drei Werte muss größer als 0 sein."
    v = a["verluste"]
    text = (f"Mit diesen Werten wächst die Fläche des 90-%-Bereichs zwischen t = 1 und t = 8 mit dem Exponenten {z(a['exponent_1_8'], 2)} (1 = wie die Zeit, 2 = wie das Quadrat der Zeit); "
            f"Fläche bei t = 2 / 4 / 8 gegen t = 1: {z(a['verhaeltnis'][2.0], 2)} / {z(a['verhaeltnis'][4.0], 2)} / {z(a['verhaeltnis'][8.0], 2)}. "
            f"Ein Start um {z(100 * a['verzoegerung'], 0)} % später (t₀ = {z(a['t0'], 1)}) kostet {z(100 * a['verlust'])} % von P ({z(100 * a['p0'])} % bei pünktlichem Start; "
            f"L₀ = {z(a['L0'], 1)}, {'zwei Hypothesen' if a['hyp'] == 2 else 'eine Wolke'}); bei 10 / 25 / 50 / 100 % Verzögerung {z(100 * v[0.10])} / {z(100 * v[0.25])} / {z(100 * v[0.50])} / {z(100 * v[1.0])} %.")
    if szenarien:
        e1, e2 = szenarien["exponent_1_4"], szenarien["exponent_4_8"]
        text += (f" In {szenarien['meta']['szenarien']} erfundenen Zwei-Hypothesen-Szenarien (Zählung auf Zellen, {str(szenarien['meta']['teilchen'])[:-3]} 000 Teilchen) lag der Flächenexponent zwischen t = 1 und 4 bei {ivl(e1, 2)} "
                 f"(Spanne {z(e1['min'], 2)} bis {z(e1['max'], 2)}), zwischen t = 4 und 8 bei {ivl(e2, 2)} (Spanne {z(e2['min'], 2)} bis {z(e2['max'], 2)}).")
    text += (" Gilt für Gauß-Wolken, ein konstantes Strömungsfeld, isotrope Streuung und feste Seiten der Querwind-Hypothesen (kein Seitenwechsel); "
             "der Verlust ist relativ zu P bei pünktlichem Start gemeint.")
    return "info", text


def verdict_elastizitaet(a: dict | None) -> tuple[str, str]:
    """Zweite Kennzahl: Verlust und Verzögerung im Verhältnis (Elastizität), Art nach dem Wert."""
    if a is None:
        return "warning", "Keine Wolke, keine Elastizität."
    el = a["elastizitaet"]
    if el < 1:
        return "success", f"Die Elastizität des Verlusts liegt bei {z(el, 2)}: P fällt prozentual langsamer, als die Startzeit steigt (unter 1 heißt unterproportional)."
    return "warning", f"Die Elastizität des Verlusts liegt bei {z(el, 2)}: P fällt prozentual mindestens so schnell, wie die Startzeit steigt (Randfall: nur unsicherer Strom, knapper Aufwand)."
