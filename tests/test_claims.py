"""Claims: die Ergebniszahlen der Befunde in der README werden hier aus den eingefrorenen Messreihen (`data/`) gelesen oder neu gerechnet und mit dem Wortlaut der README verglichen
(nicht geprüft: Testtoleranzen und die Zahlen der früheren Messung; der Test prüft Zahlen, nicht jede Formulierung darum).

`belegt(fragment, berechnet)`: das Fragment steht in der README und ist gleich der berechneten, formatierten Zahl. Zahlen aus `data/` sind Messungen der Werkzeuge
`tools/sua_make_*.py`; einzelne Zellen werden im Band neu gerechnet (Normalzahlen hängen im letzten Bit von der Plattform ab, deshalb nie bitgenau)."""

import re
from pathlib import Path

import numpy as np
import pytest

import sua_constants as C
import sua_drift as D
import sua_evaluation as E
import sua_frozen as F
import sua_messung as M
import sua_search as S
from sua_evaluation import ivl, z

README = (Path(__file__).resolve().parent.parent / "README.md").read_text(encoding="utf-8")
S1, S2, TAU = F.messreihe("s1"), F.messreihe("s2"), F.messreihe("tau")
W5 = F.laden("sua_weg")["g5"]["tabelle"]
W8 = F.laden("sua_weg")["g8"]["tabelle"]


def belegt(fragment: str, berechnet: str):
    assert fragment == berechnet, f"README {fragment!r}, berechnet {berechnet!r}"
    assert fragment in README, fragment


def zellen(t, f):
    return {k: v for k, v in t.items() if f(*k.split("|"))}


def spanne(t, key, n=1):
    v = [x[key]["mittel"] for x in t.values()]
    return z(min(v), n), z(max(v), n)


# ---------------------------------------------------------------------------------------------- Ansicht 1: Zuteilung
def test_tabelle_mittlere_luecke_je_alpha():
    zeilen = {"voll": "vollständig (τ = 0,95)", "durchgang": "ein Durchgang je Zelle", "rechteck": "bestes Rechteck"}
    for k, name in zeilen.items():
        zeile = next(l for l in README.splitlines() if l.startswith(f"| {name} |"))
        zellen_ = [c.strip() for c in zeile.strip("|").split("|")][1:]
        for al, text in zip(C.ALPHAS, zellen_):
            a = E.aggregiere_s1(S1, k, al)
            erwartet = f"{z(a['mittel'])} ({z(a['maximum'])})" if al < 10 else z(a["mittel"])
            assert text == erwartet, (k, al, text, erwartet)
            assert a["zellen"] == 20


def test_intervall_ueber_null_in_95_93_95_von_100_zellen():
    for k, n in (("voll", 95), ("durchgang", 93), ("rechteck", 95)):
        zs = [r[k] for key, r in S1.items() if key.split("|")[1] != "flach" and float(key.split("|")[2]) <= 3.0]
        assert len(zs) == 100 and sum(1 for c in zs if c["lo"] > 0) == n
    assert "95 von 100 Zellen („vollständig“), 93 von 100 („ein Durchgang“) und 95 von 100 (Rechteck)" in README


def test_alpha_zehn_unter_0_01_und_genug_aufwand_unter_0_25():
    a10 = [r[k]["mittel"] for key, r in S1.items() if float(key.split("|")[2]) == 10.0 for k in ("voll", "durchgang", "rechteck")]
    assert len(a10) == 90 and max(a10) < 0.01 and max(a10) > 0.005                     # die frühere Messung nannte 0,005
    genug = max(r[k]["mittel"] for key, r in S1.items() if float(key.split("|")[3]) == 2.0 and float(key.split("|")[2]) >= 3.0 for k in ("voll", "durchgang", "rechteck"))
    belegt("0,25", z(genug, 2))
    assert "unter 0,01 Punkten" in README and "in allen 30 Zellen" in README


def test_kernzeile_8x8_ein_gipfel_alpha_1():
    zeilen = [S1[f"8|ein_gipfel|1.0|{k}"] for k in C.KAPPAS]
    belegt("39,5 / 57,8 / 76,4 / 90,6 / 97,8 %", " / ".join(z(100 * r["P_opt"]) for r in zeilen) + " %")
    belegt("0,5 / 4,2 / 14,7 / 27,4 [25,3; 29,4] / 11,3", " / ".join(ivl(r["durchgang"]) if r["B"] == 64 else z(r["durchgang"]["mittel"]) for r in zeilen))
    belegt("11,6 / 10,7 / 7,3 / 3,0 / 3,3", " / ".join(z(r["voll"]["mittel"]) for r in zeilen))
    belegt("0,2 / 0,6 / 1,3 / 1,3 / 0,8", " / ".join(z(r["rechteck"]["mittel"]) for r in zeilen))
    assert [r["B"] for r in zeilen] == [8, 16, 32, 64, 128]


def test_nie_abgesuchte_zellen():
    r = S1["8|ein_gipfel|1.0|1.0"]
    belegt("57 % der Zellen unberührt, darin 2,5 %", f"{z(100 * r['opt_nie'], 0)} % der Zellen unberührt, darin {z(100 * r['opt_masse'])} %")
    r = S1["8|ein_gipfel|1.0|0.125"]
    belegt("88 % mit 40 %", f"{z(100 * r['opt_nie'], 0)} % mit {z(100 * r['opt_masse'], 0)} %")
    assert S1["8|ein_gipfel|1.0|1.0"]["durchgang"]["nie"] == 0.0                         # „ein Durchgang“ lässt ab Budget N keine Zelle aus, per Konstruktion
    assert S1["8|ein_gipfel|1.0|2.0"]["durchgang"]["nie"] == 0.0


def test_zwei_gipfel_knappes_budget():
    r = S1["8|zwei_gipfel|1.0|0.125"]
    belegt("9,7 Punkte hinter dem Optimum, „vollständig“ 9,0", f"{z(r['rechteck']['mittel'])} Punkte hinter dem Optimum, „vollständig“ {z(r['voll']['mittel'])}")


def test_vollstaendig_ist_eine_familie_ueber_tau():
    v = [r["voll"]["mittel"] for r in TAU.values()]
    assert len(TAU) == 72
    belegt("zwischen 0,6 und 31,9 Punkten", f"zwischen {z(min(v))} und {z(max(v))} Punkten")
    bestes = {(k.split("|")[0], k.split("|")[1], k.split("|")[2]): min(C.TAUS, key=lambda t: TAU["|".join(k.split("|")[:3]) + f"|{t}"]["voll"]["mittel"]) for k in TAU}
    assert len(bestes) == 18 and len(set(bestes.values())) >= 3                         # je nach Budget und alpha gewinnt ein anderes tau
    assert 0.99 not in bestes.values()                                                  # „0,99 in keiner der 18 Kombinationen“
    nach_budget = lambda kappa: [t for k, t in bestes.items() if k[2] == kappa]
    assert sum(1 for t in nach_budget("0.25") if t == 0.5) == 5 and sum(1 for t in nach_budget("1.0") if t == 0.95) == 5     # „bei 0,25·N meist 0,5, bei N meist 0,95“ (je 5 von 6)
    assert "bei 0,25·N meist 0,5, bei N meist 0,95, 0,99 in keiner der 18 Kombinationen aus Prior, α und Budget" in README


def test_flacher_prior_ein_durchgang_ist_das_optimum():
    flach = {k: r for k, r in S1.items() if "|flach|" in k}
    assert len(flach) == 60 and max(r["durchgang"]["mittel"] for r in flach.values()) == 0.0
    belegt("37,4", z(max(r["voll"]["mittel"] for r in flach.values())))
    assert "(30 Zellen)" in README and sum(1 for k in flach if k.startswith("8|")) == 30


def test_obergrenze_bei_teilbarem_aufwand():
    alle = [r["stetig_ueber_opt"] for r in S1.values()]
    ohne = {k: r["stetig_ueber_opt"] for k, r in S1.items() if "|flach|" not in k}
    assert len(alle) == 180 and len(ohne) == 120
    belegt("über alle 180 Zellen (einschließlich flachem Prior) 5,1 Punkte", f"über alle {len(alle)} Zellen (einschließlich flachem Prior) {z(np.mean(alle))} Punkte")
    belegt("ohne flachen Prior (120 Zellen) 2,2", f"ohne flachen Prior ({len(ohne)} Zellen) {z(np.mean(list(ohne.values())))}")
    belegt("0,10 / 0,53 / 1,65 / 4,41 / 7,15 / 16,62", " / ".join(z(np.mean([r["stetig_ueber_opt"] for k, r in S1.items() if float(k.split("|")[2]) == a]), 2) for a in C.ALPHAS))
    belegt("0,05 / 0,25 / 0,68 / 1,72 / 2,87 / 7,44", " / ".join(z(np.mean([v for k, v in ohne.items() if float(k.split("|")[2]) == a]), 2) for a in C.ALPHAS))
    belegt("66,8", z(max(alle)))
    groesster = max(S1, key=lambda k: S1[k]["stetig_ueber_opt"])
    assert groesster == "8|flach|10.0|0.25"                                             # „gehört zum flachen Prior bei α = 10 und Budget 0,25·N“
    belegt("ohne flachen Prior ist er 35,1", f"ohne flachen Prior ist er {z(max(ohne.values()))}")
    assert min(alle) >= -1e-9                                                           # ganze Durchgänge nie über der Obergrenze


def test_aufzaehlung_120_faelle_und_band():
    a = F.messreihe("aufzaehlung")
    assert a["faelle"] == 120 and a["max_abweichung"] < 1e-12
    assert "in 120 Fällen (4×4 Budget 3, 5, 8: 90; 6×6 Budget 3, 4: 30)" in README
    assert M.aufzaehlung_abweichung(4, 5, "zwei_gipfel", 1.0, 3) < 1e-12                # eine Zelle im Test neu gerechnet


def test_schwelle_auf_dem_grenzzuwachs_und_posterior_band():
    pb = F.messreihe("posterior_band")
    assert len(pb) == 36 and sum(r["n"] for r in pb.values()) == 1080
    assert sum(r["linie_ok"] for r in pb.values()) == 1080
    assert all(r["quotient_max"] <= r["exp_alpha"] * (1 + 1e-9) for r in pb.values())
    assert max(r["quotient_max"] / r["exp_alpha"] for r in pb.values()) > 1 - 1e-9      # der Quotient erreicht e^alpha
    erreicht = sorted(k for k, r in pb.items() if r["quotient_max"] / r["exp_alpha"] > 1 - 1e-9)
    assert erreicht == ["zwei_gipfel|1.0|1.0", "zwei_gipfel|2.0|1.0", "zwei_gipfel|3.0|1.0"]    # nur 3 von 36 Zellen (zwei Gipfel, Budget N), dort Gleichstände der Zählwerte
    assert "nur bei Gleichstand der Prior-Werte erreicht, nie darüber" in README
    ueber = sum(r["unbesucht_ueber_besucht"] for r in pb.values())
    ohne = sum(r["unbesucht_ueber_besucht"] for k, r in pb.items() if not k.startswith("flach"))
    belegt("911 von 1080 Instanzen (671 von 720 ohne flachen Prior)", f"{ueber} von 1080 Instanzen ({ohne} von 720 ohne flachen Prior)")
    belegt("4,5·10⁻¹⁵", f"{max(r['stetig_abweichung'] for r in pb.values()) * 1e15:.1f}".replace(".", ",") + "·10⁻¹⁵")
    assert "1080 von 1080 Instanzen" in README


def test_eine_zelle_der_ansicht_1_im_band_neu_gerechnet():
    neu = M.zelle_s1(8, "ein_gipfel", 1.0, 1.0)
    alt = S1["8|ein_gipfel|1.0|1.0"]
    for k in ("voll", "durchgang", "rechteck"):
        assert neu[k]["mittel"] == pytest.approx(alt[k]["mittel"], abs=0.05)


# ---------------------------------------------------------------------------------------------- Ansicht 1: Gelände
def test_gelaende_aggregate():
    a5, a10, a0 = E.aggregiere_s2(S2, 0.5), E.aggregiere_s2(S2, 1.0), E.aggregiere_s2(S2, 0.0)
    assert a5["zellen"] == a10["zellen"] == a0["zellen"] == 40
    belegt("im Mittel 1,1 Punkte (γ = 0,5) bzw. 2,0 (γ = 1,0), größter Zellenmittelwert 5,1",
           f"im Mittel {z(a5['voll_diff_mittel'])} Punkte (γ = 0,5) bzw. {z(a10['voll_diff_mittel'])} (γ = 1,0), größter Zellenmittelwert {z(max(a5['voll_diff_max'], a10['voll_diff_max']))}")
    assert (a5["voll_diff_ueber"], a10["voll_diff_ueber"], a5["voll_diff_unter"], a10["voll_diff_unter"]) == (30, 30, 0, 0)
    assert "über 0 in 30 von 40 Zellen, in keiner darunter" in README.replace("das Intervall liegt in 30 von 40 Zellen über 0, in keiner darunter", "über 0 in 30 von 40 Zellen, in keiner darunter")
    belegt("0,4 bzw. 0,7 Punkte, über 0 in 24 bzw. 19 von 40 Zellen",
           f"{z(a5['durch_diff_mittel'])} bzw. {z(a10['durch_diff_mittel'])} Punkte, über 0 in {a5['durch_diff_ueber']} bzw. {a10['durch_diff_ueber']} von 40 Zellen".replace("0,4 bzw.", "0,4 bzw."))
    assert a5["durch_diff_unter"] == a10["durch_diff_unter"] == 0
    assert a0["voll_diff_gleich"] == a0["durch_diff_gleich"] == 40 and "Differenz 0 in 40 von 40 Zellen" in README
    belegt("0,02 (γ = 0,5) bzw. 0,05 (γ = 1,0)", f"{z(a5['greedy_mittel'], 2)} (γ = 0,5) bzw. {z(a10['greedy_mittel'], 2)} (γ = 1,0)")
    belegt("größter Zellenmittelwert 0,29", f"größter Zellenmittelwert {z(max(a5['greedy_max'], a10['greedy_max']), 2)}")


def test_gelaende_kostenstreuung_und_gegenfall():
    cv = [np.mean([r["kosten_cv"] for k, r in S2.items() if float(k.split("|")[1]) == g]) for g in (0.5, 1.0)]
    belegt("0,53 bei γ = 0,5 und 1,03 bei γ = 1,0", f"{z(cv[0], 2)} bei γ = 0,5 und {z(cv[1], 2)} bei γ = 1,0")
    belegt("Budget 0,125 der Gesamtkosten: 2,8 Punkte", "Budget 0,125 der Gesamtkosten: " + z(S2["ein_gipfel|1.0|10.0|0.125"]["voll_diff"]["mittel"]) + " Punkte")   # im Gelände ist das Budget ein Anteil der Gesamtkosten
    assert S2["ein_gipfel|1.0|10.0|0.125"]["durch_diff"]["mittel"] > 2.0
    for art in ("ein_gipfel", "zwei_gipfel"):                                           # „bei Budget gleich den Gesamtkosten und darüber verschwunden (Differenz 0,0)“
        assert all(z(S2[f"{art}|1.0|10.0|{k}"][f]["mittel"]) == "0,0" for k in ("1.0", "2.0") for f in ("voll_diff", "durch_diff"))
    assert "bei Budget gleich den Gesamtkosten und darüber ist er verschwunden (Differenz 0,0)" in README


def test_rucksack_luecke_kleiner_instanzen_live_gleich_eingefroren():
    k = F.messreihe("rucksack_klein")
    live = M.kleine_rucksack_instanzen()
    assert live["unter_optimum"] == k["unter_optimum"] == 1 and live["max_luecke"] == pytest.approx(k["max_luecke"], abs=0.01)
    belegt("in 1 von 30 kleinen Instanzen (3×3, Kosten 1 bis 3, Budget 8) liegt der Greedy unter dem Optimum, größte Lücke 0,47 Punkte",
           f"in {k['unter_optimum']} von {k['instanzen']} kleinen Instanzen (3×3, Kosten 1 bis 3, Budget 8) liegt der Greedy unter dem Optimum, größte Lücke {z(k['max_luecke'], 2)} Punkte")


def test_eine_gelaendezelle_im_band_neu_gerechnet():
    neu = M.zelle_s2("ein_gipfel", 1.0, 1.0, 0.5)
    alt = S2["ein_gipfel|1.0|1.0|0.5"]
    assert neu["voll_diff"]["mittel"] == pytest.approx(alt["voll_diff"]["mittel"], abs=0.05) and neu["greedy_rat"]["mittel"] == pytest.approx(alt["greedy_rat"]["mittel"], abs=0.05)


# ---------------------------------------------------------------------------------------------- Ansicht 2
def test_nichtfund_aggregate():
    nf = F.messreihe("nichtfund")
    agg = E.aggregiere_nichtfund(nf)
    assert agg["zellen"] == 24 and agg["einplan_zellen"] == 24 and agg["einplan_max_abw"] < 1e-12
    belegt("0,2 bis 13,2 Punkte hinter dem aktualisierten Vorgehen (Mittel der Zellenmittel 3,5; in 23 von 24 Zellen alle 30 Instanzen schlechter)",
           f"{z(agg['alt_min'])} bis {z(agg['alt_max'])} Punkte hinter dem aktualisierten Vorgehen (Mittel der Zellenmittel {z(agg['alt_mittel'])}; in {agg['alt_alle_schlechter']} von 24 Zellen alle 30 Instanzen schlechter)")
    belegt("0,06 bis 8,1 (Mittel 4,2)", f"{z(agg['voll_min'], 2)} bis {z(agg['voll_max'])} (Mittel {z(agg['voll_mittel'])})")
    assert "ganzzahlig exakt gleich in 24 von 24 Zellen" in README


def test_rechteck_muster_ueber_runden():
    rr = E.aggregiere_rechteck_runden(F.messreihe("rechteck_runden"))
    belegt("0,53 bis 6,25 Punkte", f"{z(rr['kurz_min'], 2)} bis {z(rr['kurz_max'], 2)} Punkte")
    belegt("0,20 bis 2,67, im Mittel besser als kurzsichtig in 12 von 12 Zellen", f"{z(rr['vor_min'], 2)} bis {z(rr['vor_max'], 2)}, im Mittel besser als kurzsichtig in {rr['vorausschau_besser']} von {rr['zellen']} Zellen")


def test_falsches_alpha_aggregate():
    a = E.aggregiere_falsches_alpha(F.messreihe("falsches_alpha"))
    assert a["zellen"] == 8
    belegt("α = 10 bis 16,4 Punkte", f"α = 10 bis {z(a['10.0']['max'])} Punkte")
    belegt("mit 0,25 bis 9,7, mit 4 bis 9,0", f"mit 0,25 bis {z(a['0.25']['max'])}, mit 4 bis {z(a['4.0']['max'])}")
    belegt("(0,5 oder 2) 0,4 bis 2,4", f"(0,5 oder 2) {z(min(a['0.5']['min'], a['2.0']['min']))} bis {z(max(a['0.5']['max'], a['2.0']['max']))}")
    assert a["10.0"]["max"] > a["0.25"]["max"]                                          # der größte Schaden liegt bei α = 10 (Faktor 10 daneben)
    belegt("Faktor 4: im Mittel über die 8 Zellen 7,0 gegen 5,4", f"Faktor 4: im Mittel über die {a['zellen']} Zellen {z(a['0.25']['mittel'])} gegen {z(a['4.0']['mittel'])}")
    fa = F.messreihe("falsches_alpha")
    assert sum(1 for r in fa.values() if r["0.25"]["mittel"] > r["4.0"]["mittel"]) == 7  # zu niedrig schadet beim gleichen Faktor 4 in 7 von 8 Zellen mehr
    assert a["0.25"]["mittel"] > a["4.0"]["mittel"] and "Beim gleichen Faktor schadet ein zu niedriges α etwas mehr als ein zu hohes" in README
    belegt("Faktor 2: 1,77 gegen 1,67, nur in 5 von 8 Zellen", f"Faktor 2: {z(a['0.5']['mittel'], 2)} gegen {z(a['2.0']['mittel'], 2)}, nur in {sum(1 for r in fa.values() if r['0.5']['mittel'] > r['2.0']['mittel'])} von {a['zellen']} Zellen")
    assert "in 7 von 8 Zellen" in README


def test_eine_nichtfund_zelle_im_band_neu_gerechnet():
    neu = M.zelle_nichtfund("zwei_gipfel", 1.0, 0.5)
    alt = F.zelle("nichtfund", "zwei_gipfel", 1.0, 0.5)
    assert neu["alt"]["mittel"] == pytest.approx(alt["alt"]["mittel"], abs=0.05) and neu["einplan_max_abw"] < 1e-12


# ---------------------------------------------------------------------------------------------- Ansicht 3
def test_weg_tabelle_5x5_und_8x8():
    ein = zellen(W5, lambda a, al, T: a == "ein_gipfel")
    zwei = zellen(W5, lambda a, al, T: a == "zwei_gipfel")
    assert len(W5) == 24 and len(W8) == 8 and sum(t["unbewiesen"] for t in W5.values()) == 0 and sum(t["unbewiesen"] for t in W8.values()) == 5
    zeilen = {"wegpreis": "Wegpreis (Optimum ohne Weg minus Wegoptimum)", "greedy_kurz": "Greedy kurzsichtig", "greedy_ziel": "Greedy mit Ziel", "maehen_ganz": "Mähen über das ganze Gebiet",
              "maehen_rechteck": "Mähen im besten Rechteck"}
    for key, name in zeilen.items():
        zeile = next(l for l in README.splitlines() if l.startswith(f"| {name} |"))
        zl = [c.strip() for c in zeile.strip("|").split("|")][1:]
        nd = 2 if key in ("wegpreis", "greedy_kurz", "greedy_ziel", "maehen_rechteck") else 1
        for tab, text in zip((ein, zwei, W8), zl):
            lo, hi = spanne(tab, key, 1)
            vals = [x[key]["mittel"] for x in tab.values()]
            erwartet = f"{z(min(vals), 2 if min(vals) < 1 else 1)} bis {z(max(vals), 1)}"
            assert text == erwartet, (key, text, erwartet)


def test_weg_mähen_ganz_intervall_ueber_null_und_rechteck_aussage():
    assert all(t["maehen_ganz"]["lo"] > 0 for t in W5.values()) and all(t["maehen_ganz"]["lo"] > 0 for t in W8.values())
    assert "in allen 24 (5×5) und 8 (8×8) Zellen mit dem Intervall über 0" in README
    ein = [t["maehen_rechteck"]["mittel"] for k, t in W5.items() if k.startswith("ein_gipfel")]
    ein8 = [t["maehen_rechteck"]["mittel"] for k, t in W8.items() if k.startswith("ein_gipfel")]
    belegt("höchstens 1,1 bei 5×5, 1,3 bei 8×8", f"höchstens {z(max(ein))} bei 5×5, {z(max(ein8))} bei 8×8")


def test_orienteering_fassung():
    a = zellen(W5, lambda ar, al, T: float(al) <= 1.0 and int(T) >= 10)
    assert len(a) == 8
    v = [t["orient"]["mittel"] for t in a.values()]
    belegt("4,4 bis 21,3 Punkte hinter dem Wegoptimum (8 Zellen)", f"{z(min(v))} bis {z(max(v))} Punkte hinter dem Wegoptimum ({len(a)} Zellen)")
    t6 = zellen(W5, lambda ar, al, T: float(al) <= 1.0 and int(T) == 6)
    belegt("T = 6 höchstens 3,6", f"T = 6 höchstens {z(max(t['orient']['mittel'] for t in t6.values()))}")
    a10 = zellen(W5, lambda ar, al, T: float(al) == 10.0)
    assert len(a10) == 6 and all(t["orient"]["mittel"] < 0.005 and t["ueberschaetzung"]["mittel"] < 0.005 for t in a10.values())
    assert "bei α = 10 ist die Lücke in allen 6 Zellen gleich 0,00" in README
    u = [t["ueberschaetzung"]["mittel"] for t in zellen(W5, lambda ar, al, T: float(al) <= 1.0).values()]
    belegt("um 20,2 bis 73,0 Punkte", f"um {z(min(u))} bis {z(max(u))} Punkte")
    assert "alle 720 CP-SAT-Lösungen bewiesen optimal" in README


def test_flacher_prior_weg_optimum_und_maehen():
    for T in (6, 10, 15):
        a = E.analyse_weg("flach", 1.0, T, 1)
        assert a["beste"] == pytest.approx(E.optimum_flach(T, 1.0)) and a["luecke"]["maehen_ganz"] == pytest.approx(0.0, abs=1e-9) and a["luecke"]["orient"] == pytest.approx(0.0, abs=1e-9)


def test_live_weg_ansicht_nutzt_die_eingefrorenen_loesungen():
    a = E.analyse_weg("ein_gipfel", 1.0, 15, 1)
    fro = F.weg_instanz("ein_gipfel", 1.0, 15, 1)
    assert a["P"]["exakt"] == pytest.approx(fro["opt"], abs=1e-9) and a["P"]["orient"] == pytest.approx(fro["o_wert"], abs=1e-9) and a["bew"]["exakt"] is True
    assert a["P"]["exakt"] >= max(a["P"][k] for k in ("greedy_kurz", "greedy_ziel", "maehen_ganz", "maehen_rechteck")) - 1e-9


# ---------------------------------------------------------------------------------------------- Ansicht 4
def test_flaeche_und_exponenten_formeln():
    belegt("2 / 4 / 8 (Exponent 1,00)", " / ".join(str(round(D.flaechenverhaeltnis(t, (0.0, 0.5, 0.0)))) for t in (2, 4, 8)) + f" (Exponent {z(D.flaechenexponent((0.0, 0.5, 0.0), 1, 8), 2)})")
    belegt("4 / 16 / 64 (Exponent 2,00)", " / ".join(str(round(D.flaechenverhaeltnis(t, (0.0, 0.0, 0.3)))) for t in (2, 4, 8)) + f" (Exponent {z(D.flaechenexponent((0.0, 0.0, 0.3), 1, 8), 2)})")
    belegt("1,61 / 3,08 / 7,06 (Exponent 0,94", " / ".join(z(D.flaechenverhaeltnis(t, D.PARAMETER_GEMISCHT), 2) for t in (2, 4, 8)) + f" (Exponent {z(D.flaechenexponent(D.PARAMETER_GEMISCHT, 1, 8), 2)}")


def test_dreissig_szenarien_eingefroren():
    d = F.drift_szenarien()
    assert d["meta"]["szenarien"] == 30 and len(d["exponent_1_4"]["werte"]) == 30
    e1, e2 = d["exponent_1_4"], d["exponent_4_8"]
    belegt("1,07 [1,01; 1,14] (Spanne 0,74 bis 1,47)", f"{ivl(e1, 2)} (Spanne {z(e1['min'], 2)} bis {z(e1['max'], 2)})")
    belegt("1,41 [1,36; 1,47] (Spanne 1,13 bis 1,75)", f"{ivl(e2, 2)} (Spanne {z(e2['min'], 2)} bis {z(e2['max'], 2)})")
    nr = 5
    a, b = D.szenario_exponenten(nr)
    assert a == pytest.approx(e1["werte"][nr - 1], abs=0.02) and b == pytest.approx(e2["werte"][nr - 1], abs=0.02)


def test_verlust_zeilen():
    par = D.PARAMETER_GEMISCHT
    prozent = lambda L0, dv=None: " / ".join(z(100 * (D.verlust_eine_wolke(L0, par, 1.0, d) if dv is None else D.verlust_zwei_hypothesen(L0, par, 1.0, d, dv))) for d in D.FAKTOREN_VERZOEGERUNG)
    belegt("bei L₀ = 0,5 4,6 / 10,7 / 19,6 / 33,4 % von P", f"bei L₀ = 0,5 {prozent(0.5)} % von P")
    belegt("bei L₀ = 1 3,8 / 9,0 / 16,7 / 29,2 %", f"bei L₀ = 1 {prozent(1.0)} %")
    belegt("(dv = 0,8, L₀ = 1) 5,3 / 12,5 / 23,1 / 39,5 %", f"(dv = 0,8, L₀ = 1) {prozent(1.0, 0.8)} %")
    belegt("mit dv = 0,3 4,0 / 9,6 / 17,8 / 30,9 %", f"mit dv = 0,3 {prozent(1.0, 0.3)} %")


def test_elastizitaet_unter_eins_in_neun_zeilen_und_randfall():
    par = D.PARAMETER_GEMISCHT
    zeilen = []
    for L0 in (0.5, 1.0, 2.0):
        E1 = D.aufwand_fuer_L0(L0, D.varianz(1.0, *par))
        for dv in (None, 0.3, 0.8):
            f = (lambda t: D.p_gauss(E1, 1.0, D.varianz(t, *par))) if dv is None else (lambda t, dv=dv: D.p_raster(E1, 1.0, D.varianz(t, *par), 2 * dv * t))
            zeilen.append(D.elastizitaet_sekante(f(1.0), f(1.25), 1.25))
    assert len(zeilen) == 9 and max(zeilen) < 1.0
    assert "in allen 9 gemessenen Zeilen" in README
    belegt("bei 0,28 bis 0,72", f"bei {z(min(zeilen), 2)} bis {z(max(zeilen), 2)}")
    assert "ist in diesen Zeilen **unterproportional**, nicht überproportional" in README
    nur = (0.2, 0.0, 0.3)
    E1 = D.aufwand_fuer_L0(0.5, D.varianz(1.0, *nur))
    belegt("Elastizität bei 1,26", f"Elastizität bei {z(D.elastizitaet_sekante(D.p_gauss(E1, 1.0, D.varianz(1.0, *nur)), D.p_gauss(E1, 1.0, D.varianz(1.25, *nur)), 1.25), 2)}")


# ---------------------------------------------------------------------------------------------- Texte
def test_presets_zeigen_die_erwartete_verdikt_zeile():
    """Je Preset die erwartete Aussage: Gegenfälle melden „gleich“, kein Preset ohne Wirkung meldet eine Wirkung."""
    row = lambda n, art, al, k: F.zelle("s1", n, art, al, k)
    assert "alle drei Verfahren gleich" in E.verdict_zuteilung(8, "ein_gipfel", 10.0, 1.0, 0.95, row(8, "ein_gipfel", 10.0, 1.0), None)[1]
    assert "Bei flachem Prior ist „ein Durchgang je Zelle“ das Optimum" in E.verdict_zuteilung(8, "flach", 1.0, 1.0, 0.95, row(8, "flach", 1.0, 1.0), None)[1]
    assert "alle drei Verfahren gleich" not in E.verdict_zuteilung(8, "ein_gipfel", 1.0, 1.0, 0.95, row(8, "ein_gipfel", 1.0, 1.0), None)[1]
    assert "identisch" in E.verdict_gelaende(0.0, E.aggregiere_s2(S2, 0.0), None, None)[1] and "gewinnt bei „vollständig“" in E.verdict_gelaende(1.0, E.aggregiere_s2(S2, 1.0), None, None)[1]
    assert "Orienteering-Problem" in E.verdict_weg("ein_gipfel", 10.0, 10, W5["ein_gipfel|10.0|10"], {})[1]
    assert "Orienteering-Problem" not in E.verdict_weg("ein_gipfel", 1.0, 15, W5["ein_gipfel|1.0|15"], {})[1]


def test_testzahl_der_readme_steht_in_readme_test_counts():
    assert re.search(r"\d+ Tests insgesamt", README)
