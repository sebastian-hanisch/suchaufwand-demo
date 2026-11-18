"""Verdikte an von Hand gesetzten Zeilen (DEMO-PLAYBOOK §12): jede Zahl im Text und jede Schwelle der Art steht hier.

Ein Verdikt ist eine reine Funktion (Art, Text) und bekommt Zeilen der vorgerechneten Messreihen, nicht Seeds. Zahlen mit Dezimalkomma, Intervall in eckigen Klammern."""

import sua_evaluation as E


def st(m, lo=None, hi=None, anteil=1.0, **extra):
    return {"mittel": m, "lo": m - 1 if lo is None else lo, "hi": m + 1 if hi is None else hi, "anteil": anteil, **extra}


ROW = {"B": 64, "P_opt": 0.9058, "opt_nie": 0.572, "opt_masse": 0.0247, "voll": st(3.0, 2.5, 3.5), "durchgang": st(27.4, 25.3, 29.4), "rechteck": st(1.3, 1.1, 1.5)}


def test_verdikt_zuteilung_nennt_optimum_luecken_und_unberuehrte_zellen():
    kind, text = E.verdict_zuteilung(8, "ein_gipfel", 1.0, 1.0, 0.95, ROW, None)
    assert kind == "info"
    assert "8×8, ein Gipfel, α = 1,0, Budget 64 Durchgänge" in text and "im Mittel über 30 Instanzen 90,6 %" in text
    assert "„Ein Durchgang je Zelle“ liegt 27,4 [25,3; 29,4] Prozentpunkte dahinter" in text and "„vollständig“ (τ = 0,95) 3,0 [2,5; 3,5]" in text
    assert "das beste Rechteck 1,3 [1,1; 1,5]" in text and "lässt 57 % der Zellen unberührt, darin 2,5 % der Prior-Masse" in text
    assert "gleich" not in text and "Optimum (Lücke 0)" not in text


def test_verdikt_zuteilung_tau_ungleich_standard_nimmt_die_tau_zeile_oder_sagt_nicht_vorgerechnet():
    kind, text = E.verdict_zuteilung(8, "ein_gipfel", 1.0, 1.0, 0.5, ROW, {"voll": st(27.4, 26.0, 28.0)})
    assert "„vollständig“ (τ = 0,50) 27,4 [26,0; 28,0]" in text
    kind, text = E.verdict_zuteilung(12, "ein_gipfel", 1.0, 1.0, 0.99, ROW, None)
    assert "nicht vorgerechnet" in text and "3,0 [2,5; 3,5]" not in text


def test_verdikt_zuteilung_gegenfaelle_alpha_zehn_und_flach():
    gleich = dict(ROW, voll=st(0.002), durchgang=st(0.0), rechteck=st(0.004))
    assert "alle drei Verfahren gleich (Lücke unter 0,01 Punkten)" in E.verdict_zuteilung(8, "ein_gipfel", 10.0, 1.0, 0.95, gleich, None)[1]
    kopf = dict(ROW, durchgang=st(0.009), rechteck=st(0.009), voll=st(0.009))
    assert "alle drei Verfahren gleich" in E.verdict_zuteilung(8, "ein_gipfel", 10.0, 1.0, 0.95, kopf, None)[1]            # Schwelle: unter 0,01
    knapp = dict(ROW, durchgang=st(0.011), rechteck=st(0.0), voll=st(0.0))
    assert "alle drei Verfahren gleich" not in E.verdict_zuteilung(8, "ein_gipfel", 10.0, 1.0, 0.95, knapp, None)[1]
    flach = dict(ROW, durchgang=st(0.0), voll=st(20.0), rechteck=st(5.0))
    assert "Bei flachem Prior ist „ein Durchgang je Zelle“ das Optimum (Lücke 0)" in E.verdict_zuteilung(8, "flach", 1.0, 1.0, 0.95, flach, None)[1]
    assert "flachem Prior" not in E.verdict_zuteilung(8, "ein_gipfel", 1.0, 1.0, 0.95, flach, None)[1]


def test_aggregat_s1_mittelt_zwanzig_zellen_ohne_flach():
    s1 = {}
    for n in (8, 12):
        for art in ("ein_gipfel", "zwei_gipfel", "flach"):
            for kappa in (0.125, 0.25, 0.5, 1.0, 2.0):
                s1[f"{n}|{art}|1.0|{kappa}"] = {"voll": st(10.0 if art != "flach" else 99.0, 5.0, 11.0)}
    s1["12|ein_gipfel|1.0|2.0"] = {"voll": st(20.0, -1.0, 21.0)}
    a = E.aggregiere_s1(s1, "voll", 1.0)
    assert a["zellen"] == 20 and abs(a["mittel"] - (19 * 10.0 + 20.0) / 20) < 1e-12 and a["maximum"] == 20.0 and a["ueber_null"] == 19
    assert E.aggregiere_s1(s1, "voll", 1.0, mit_flach=True)["zellen"] == 30 and E.aggregiere_s1(s1, "voll", 0.5) == {}


def _s2_zelle(voll, durch, greedy=0.02, null=1.0):
    return {"voll_diff": dict(st(voll, voll - 0.5, voll + 0.5), schlechter=0.0, max_abweichung_null=null), "durch_diff": dict(st(durch, durch - 0.5, durch + 0.5), schlechter=0.0, max_abweichung_null=null),
            "greedy_rat": st(greedy)}


def test_verdikt_gelaende_gamma_positiv_zaehlt_zellen_mit_intervall_ueber_und_unter_null():
    s2 = {f"ein_gipfel|1.0|1.0|{k}": _s2_zelle(2.0, 0.6) for k in range(3)}
    s2["ein_gipfel|1.0|1.0|9"] = _s2_zelle(-0.9, 0.0)                              # Intervall [-1,4; -0,4] liegt unter 0
    s2["ein_gipfel|0.5|1.0|1"] = _s2_zelle(9.0, 9.0)                                # anderer Geländefaktor, zählt nicht
    agg = E.aggregiere_s2(s2, 1.0)
    assert agg["zellen"] == 4 and agg["voll_diff_ueber"] == 3 and agg["voll_diff_unter"] == 1 and abs(agg["voll_diff_mittel"] - 1.275) < 1e-12
    kind, text = E.verdict_gelaende(1.0, agg, {"instanzen": 30, "unter_optimum": 5, "max_luecke": 4.96}, s2["ein_gipfel|1.0|1.0|0"])
    assert kind == "info" and "Gelände γ = 1,0" in text and "bei „vollständig“ im Mittel 1,3 Punkte" in text and "bei „ein Durchgang“ 0,4; in" in text
    assert "in 3 bzw. 3 von 4 Zellen liegt das Intervall über 0, in 1 bzw. 0 darunter" in text
    assert "In 5 von 30 kleinen Instanzen" in text and "größte Lücke 4,96" in text and "0,02 Punkte hinter dem exakten Optimum" in text
    assert "Gewählte Zelle" in text


def test_verdikt_gelaende_gamma_null_nennt_gleichheit_in_allen_zellen():
    s2 = {f"ein_gipfel|0.0|1.0|{k}": _s2_zelle(0.0, 0.0, 0.0, 0.0) for k in range(5)}
    agg = E.aggregiere_s2(s2, 0.0)
    assert agg["voll_diff_gleich"] == 5
    text = E.verdict_gelaende(0.0, agg, None, None)[1]
    assert "Differenz 0 in 5 von 5 Zellen bei „vollständig“, 5 von 5 bei „ein Durchgang“" in text and "kleinen Instanzen" not in text


def _nf(alt, voll, gleich=30, anteil=1.0):
    return {"B": 64, "P_a": 90.0, "alt": st(alt, anteil=anteil), "voll": st(voll), "einplan_max_abw": 0.0, "einplan_gleich": gleich, "n": 30}


def test_verdikt_nichtfund_aktualisiert_ist_einplan_und_negativbefund():
    nf = {f"a|{k}": _nf(1.0 + k, 2.0 + k) for k in range(4)}
    nf["a|4"] = _nf(0.5, 1.0, anteil=0.97)
    agg = E.aggregiere_nichtfund(nf)
    assert agg["zellen"] == 5 and agg["einplan_zellen"] == 5 and agg["alt_alle_schlechter"] == 4 and agg["alt_min"] == 0.5 and agg["alt_max"] == 4.0
    rr = {"k": {"kurzsichtig": st(0.5), "vorausschau": st(0.2), "vorteil_mittel": 0.3}, "m": {"kurzsichtig": st(6.0), "vorausschau": st(2.5), "vorteil_mittel": 1.0}}
    kind, text = E.verdict_nichtfund(nf["a|1"], agg, E.aggregiere_rechteck_runden(rr), 0.5)
    assert kind == "info"
    assert "5 von 5 Zellen" in text and "bringt das Aktualisieren deshalb 0" in text and "solange der Aufwand frei verteilbar ist" in text
    assert "0,50 bis 6,00 Punkte hinter dem frei verteilten Optimum" in text and "in 2 von 2 Zellen im Mittel besser" in text
    assert "kostet 0,5 bis 4,0 Punkte" in text and "in 4 von 5 Zellen alle 30 Instanzen schlechter" in text and "Gewählte Zelle" in text
    assert "50 % des Budgets in Runde 1" not in text and "50 % des Budgets in Runde 1" in E.verdict_nichtfund(None, agg, E.aggregiere_rechteck_runden(rr), 0.25)[1]


def test_verdikt_falsches_alpha_vergleicht_gleiche_faktoren_und_nennt_den_groessten_wert():
    fa = {"x": {"0.25": st(3.0), "0.5": st(1.0), "2.0": st(0.8), "4.0": st(6.0), "10.0": st(12.0)}, "y": {"0.25": st(9.0), "0.5": st(2.0), "2.0": st(2.5), "4.0": st(8.0), "10.0": st(10.0)}}
    agg = E.aggregiere_falsches_alpha(fa)
    assert agg["zellen"] == 2 and agg["10.0"] == {"min": 10.0, "max": 12.0, "mittel": 11.0} and agg["0.25"]["max"] == 9.0 and agg["0.25"]["mittel"] == 6.0 and agg["4.0"]["mittel"] == 7.0
    kind, text = E.verdict_falsches_alpha(agg, fa["x"], 4.0)
    assert "α = 10 statt 1: bis 12,0 Punkte" in text and "α = 0,25 bis 9,0" in text and "(0,5 oder 2) 0,8 bis 2,5" in text and "α = 4 bis 8,0" in text
    # das größte Maximum (12,0 bei α = 10) gehört zum Faktor 10 und entscheidet nicht über „zu hoch gegen zu niedrig“; verglichen wird der gleiche Faktor 4 (Mittel 6,0 gegen 7,0)
    assert "Beim gleichen Faktor 4 schadet ein zu hohes α etwas mehr als ein zu niedriges: im Mittel über die Zellen 6,0 (α = 0,25) gegen 7,0 (α = 4)" in text
    assert "Der größte Zellenwert (12,0) gehört zu α = 10" in text and "α angenommen = 4,00: 6,00 [5,00; 7,00]" in text
    assert "Zu hoch geschätzt schadet in diesen Zellen mehr" not in text and "Zu niedrig geschätzt schadet" not in text
    fa["x"]["0.25"] = st(9.0)
    fa["y"]["0.25"] = st(8.0)
    text2 = E.verdict_falsches_alpha(E.aggregiere_falsches_alpha(fa), None, 4.0)[1]
    assert "Beim gleichen Faktor 4 schadet ein zu niedriges α etwas mehr als ein zu hohes: im Mittel über die Zellen 8,5 (α = 0,25) gegen 7,0 (α = 4)" in text2
    fa["x"]["0.25"] = st(6.0)
    fa["y"]["0.25"] = st(8.0)
    assert "schadet ein zu hohes und ein zu niedriges α gleich viel" in E.verdict_falsches_alpha(E.aggregiere_falsches_alpha(fa), None, 4.0)[1]


TAB = {"n": 30, "unbewiesen": 0, "opt_mittel": 77.1, "wegpreis": st(3.36), "greedy_kurz": st(1.12), "greedy_ziel": st(0.81), "maehen_ganz": st(36.9), "maehen_rechteck": st(1.14),
       "orient": st(14.38), "ueberschaetzung": st(36.12)}


def test_verdikt_weg_nennt_alle_verfahren_und_den_hinweis_nur_bei_grossem_alpha():
    kind, text = E.verdict_weg("ein_gipfel", 1.0, 15, TAB, {})
    assert kind == "info"
    assert "5×5, ein Gipfel, α = 1,0, T = 15" in text and "77,1 %" in text and "Wegpreis (Optimum ohne Weg minus Wegoptimum) 3,4 [2,4; 4,4]" in text
    assert "Greedy kurzsichtig 1,1 [0,1; 2,1], mit Ziel 0,8 [-0,2; 1,8]" in text and "ganze Gebiet 36,9" in text and "im besten Rechteck 1,1" in text
    assert "Orienteering-Fassung" in text and "14,4 [13,4; 15,4]" in text and "um 36,1 [35,1; 37,1]" in text
    assert "gewinn-touren-demo" not in text
    hoch = E.verdict_weg("ein_gipfel", 3.0, 10, TAB, {})[1]
    assert "Orienteering-Problem" in hoch and "(https://github.com/sebastian-hanisch/gewinn-touren-demo)" in hoch
    assert "gewinn-touren-demo" not in E.verdict_weg("ein_gipfel", 1.0, 10, TAB, {})[1]


def test_verdikt_weg_flacher_prior_aus_der_rechnung():
    live = {"beste": 0.3792, "luecke": {"maehen_ganz": 0.0, "maehen_rechteck": 0.0, "greedy_kurz": 1.5, "greedy_ziel": 0.0}}
    text = E.verdict_weg("flach", 1.0, 15, None, live)[1]
    assert "Flacher Prior, T = 15: jeder Weg über 15 verschiedene Zellen ist optimal (P = 37,9 %)" in text and "Mähen über das ganze Gebiet: Lücke 0,00" in text and "Greedy kurzsichtig 1,50" in text


def _drift(el=0.5):
    return {"verhaeltnis": {2.0: 1.61, 4.0: 3.08, 8.0: 7.06}, "exponent_1_8": 0.94, "verzoegerung": 0.25, "verlust": 0.107, "p0": 0.2, "t0": 1.0, "L0": 0.5, "hyp": 1,
            "verluste": {0.10: 0.046, 0.25: 0.107, 0.50: 0.196, 1.0: 0.334}, "elastizitaet": el}


def test_verdikt_drift_nennt_exponent_verlust_und_szenarien():
    szen = {"meta": {"szenarien": 30, "teilchen": 40000}, "exponent_1_4": dict(st(1.07, 1.01, 1.14), min=0.74, max=1.47), "exponent_4_8": dict(st(1.41, 1.36, 1.47), min=1.13, max=1.75)}
    kind, text = E.verdict_drift(_drift(), szen)
    assert kind == "info"
    assert "Exponenten 0,94" in text and "1,61 / 3,08 / 7,06" in text and "Start um 25 % später (t₀ = 1,0) kostet 10,7 % von P (20,0 % bei pünktlichem Start; L₀ = 0,5, eine Wolke)" in text
    assert "4,6 / 10,7 / 19,6 / 33,4 %" in text and "1,07 [1,01; 1,14]" in text and "Spanne 0,74 bis 1,47" in text and "1,41 [1,36; 1,47]" in text
    assert "feste Seiten der Querwind-Hypothesen" in text
    assert "zwei Hypothesen" in E.verdict_drift(dict(_drift(), hyp=2), None)[1] and "Szenarien" not in E.verdict_drift(_drift(), None)[1]


def test_verdikt_elastizitaet_schwelle_bei_eins():
    assert E.verdict_elastizitaet(_drift(0.99))[0] == "success" and "0,99" in E.verdict_elastizitaet(_drift(0.99))[1]
    assert E.verdict_elastizitaet(_drift(1.0))[0] == "warning" and E.verdict_elastizitaet(_drift(1.26))[0] == "warning"
    assert E.verdict_elastizitaet(None)[0] == "warning"
