"""Orakel-Tests Ansicht 3 (Weg): CP-SAT gegen Aufzählung aller Wege, lexikographische Orienteering-Fassung gegen Aufzählung, jedes Verfahren liefert gültige Wege und nie mehr
als das Optimum, und die eingefrorenen Wege in `data/sua_weg.json` sind gültig und haben ihren gespeicherten Wert (kein CP-SAT-Neurechnen im Test: Gleichstände, Version)."""

import numpy as np
import pytest

import sua_frozen as F
import sua_path as W
import sua_search as S
from sua_evaluation import optimum_flach


def cpsat():
    pytest.importorskip("ortools")
    import sua_cpsat

    return sua_cpsat


# ---------------------------------------------------------------------------------------------- Aufzählung als Wahrheit
def test_cpsat_wegoptimum_gleich_aufzaehlung_aller_wege():
    C = cpsat()
    k = 0
    for n, T in ((3, 7), (3, 9), (4, 6)):
        for inst in range(1, 6):
            for alpha in (0.5, 2.0):
                p = S.make_prior(("ein_gipfel", "zwei_gipfel")[inst % 2], n, inst)
                weg, bewiesen = C.exakt_weg(p, alpha, n, T, zeit=20.0, worker=2)
                assert W.ist_gueltiger_weg(weg, n, T) and bewiesen
                assert W.weg_wert(p, alpha, weg) == pytest.approx(C.beste_weg_aufzaehlung(p, alpha, n, T)[0], abs=1e-8)
                k += 1
    assert k == 30


def test_lexikographische_orienteering_fassung_gleich_aufzaehlung():
    C = cpsat()
    for T in (5, 7):
        for alpha in (0.5, 2.0):
            for inst in range(1, 6):
                p = S.make_prior(("ein_gipfel", "zwei_gipfel")[inst % 2], 3, inst)
                weg, bewiesen = C.orienteering_weg(p, alpha, 3, T, zeit=20.0, worker=2)
                _, pm_best, lex_best = C.beste_weg_aufzaehlung(p, alpha, 3, T)
                assert bewiesen
                assert W.praemien_summe(p, weg) == pytest.approx(pm_best, abs=1e-8) and W.weg_wert(p, alpha, weg) == pytest.approx(lex_best, abs=1e-8)


def test_jedes_verfahren_liefert_gueltige_wege_und_nie_mehr_als_das_optimum_3x3():
    C = cpsat()
    for inst in range(1, 9):
        p = S.make_prior(("ein_gipfel", "zwei_gipfel")[inst % 2], 3, inst)
        for alpha, T in ((0.5, 5), (2.0, 7)):
            opt = C.beste_weg_aufzaehlung(p, alpha, 3, T)[0]
            wege = {"kurz": W.greedy_kurzsichtig(p, alpha, 3, T), "ziel": W.greedy_mit_ziel(p, alpha, 3, T), "ganz": W.maehen_ganz(3, T), "rechteck": W.maehen_im_rechteck(p, alpha, 3, T)}
            for name, w in wege.items():
                assert W.ist_gueltiger_weg(w, 3, T), name
                assert W.weg_wert(p, alpha, w) <= opt + 1e-12, name


def test_aufzaehlung_zaehlt_die_wege_wie_die_matrixpotenz():
    C = cpsat()
    A = np.zeros((9, 9), dtype=np.int64)                      # Bleiben oder Nachbarzelle auf dem 3x3-Gitter
    for i, nb in enumerate(W.nachbarn(3)):
        A[i, nb] = 1
    for T in (1, 2, 3, 5):
        assert sum(1 for _ in C.wege_aufzaehlen(3, T)) == np.linalg.matrix_power(A, T - 1)[0].sum()
    assert all(W.ist_gueltiger_weg(w, 3, 3) for w in C.wege_aufzaehlen(3, 3))


def test_flacher_prior_optimum_ist_t_durch_n_mal_q_und_gleich_aufzaehlung():
    C = cpsat()
    p = S.make_prior("flach", 3, 1)
    for T in (4, 6, 9):
        assert C.beste_weg_aufzaehlung(p, 1.0, 3, T)[0] == pytest.approx(optimum_flach(T, 1.0, 9), abs=1e-12)
    assert optimum_flach(26, 1.0) is None


# ---------------------------------------------------------------------------------------------- eingefrorene Wege
def test_eingefrorene_wege_sind_gueltig_und_haben_ihren_wert():
    d = F.laden("sua_weg")
    assert d is not None
    gesehen = 0
    for schluessel, liste in d["g5"]["inst"].items():
        art, alpha, T = schluessel.split("|")
        alpha, T = float(alpha), int(T)
        assert len(liste) == 30
        for inst, z in enumerate(liste, start=1):
            p = S.make_prior(art, 5, inst)
            assert W.ist_gueltiger_weg(z["weg"], 5, T) and W.weg_wert(p, alpha, z["weg"]) == pytest.approx(z["opt"], abs=1e-9)
            assert W.ist_gueltiger_weg(z["o_weg"], 5, T) and W.weg_wert(p, alpha, z["o_weg"]) == pytest.approx(z["o_wert"], abs=1e-9)
            assert W.praemien_summe(p, z["o_weg"]) == pytest.approx(z["o_praemie"], abs=1e-9)
            assert z["o_wert"] <= z["opt"] + 1e-9 or not z["bew"]                  # die Orienteering-Fassung liegt nicht über dem bewiesenen Optimum
            gesehen += 1
    assert gesehen == 720


def test_eingefrorenes_optimum_nie_unter_den_vier_verfahren_wenn_bewiesen():
    d = F.laden("sua_weg")
    for schluessel, liste in list(d["g5"]["inst"].items())[:12]:
        art, alpha, T = schluessel.split("|")
        alpha, T = float(alpha), int(T)
        for inst, z in enumerate(liste, start=1):
            if not z["bew"]:
                continue
            p = S.make_prior(art, 5, inst)
            for w in (W.greedy_kurzsichtig(p, alpha, 5, T), W.greedy_mit_ziel(p, alpha, 5, T), W.maehen_ganz(5, T)):
                assert W.weg_wert(p, alpha, w) <= z["opt"] + 1e-9


def test_eingefrorene_tabelle_stimmt_mit_den_instanzzeilen_ueberein():
    d = F.laden("sua_weg")
    for schluessel, liste in d["g5"]["inst"].items():
        art, alpha, T = schluessel.split("|")
        tab = d["g5"]["tabelle"][schluessel]
        assert tab["n"] == 30 and tab["unbewiesen"] == sum(1 for z in liste if not z["bew"]) and tab["unbewiesen_orient"] == sum(1 for z in liste if not z["o_bew"])
        assert tab["orient"]["mittel"] >= -1e-9 and tab["wegpreis"]["mittel"] >= -1e-9 and tab["ueberschaetzung"]["mittel"] >= -1e-9
        for k in ("greedy_kurz", "greedy_ziel", "maehen_ganz", "maehen_rechteck"):
            assert tab[k]["lo"] <= tab[k]["mittel"] <= tab[k]["hi"] and tab[k]["mittel"] >= -1e-9
    assert len(d["g5"]["tabelle"]) == 24 and len(d["g8"]["tabelle"]) == 8


def test_nicht_vorgerechnete_instanz_gibt_none():
    assert F.weg_instanz("ein_gipfel", 1.0, 15, 31) is None and F.weg_instanz("ein_gipfel", 1.0, 25, 1) is None and F.weg_instanz("flach", 1.0, 15, 1) is None
    assert F.weg_tabelle(5, "ein_gipfel", 2.0, 15) is None and np.isfinite(F.weg_tabelle(5, "ein_gipfel", 1.0, 15)["opt_mittel"])
