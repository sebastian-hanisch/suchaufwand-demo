"""Randfälle (DEMO-PLAYBOOK §12): Budget 0 und 1, alpha sehr klein und sehr groß, Zellen mit Prior 0, ein einziger Gipfel in einer Zelle, flacher Prior, Kosten 1,
Budget unter den Kosten jeder Zelle, T = 1, 1x1-Gitter, Verzögerung 0, Streuung 0, L gegen 0."""

import numpy as np
import pytest

import sua_drift as D
import sua_evaluation as E
import sua_path as W
import sua_search as S


def test_budget_null_und_eins():
    p = S.make_prior("ein_gipfel", 4, 1)
    assert S.greedy_zuteilung(p, 1.0, 0).sum() == 0 and S.wert(p, 1.0, S.greedy_zuteilung(p, 1.0, 0)) == 0.0
    z = S.greedy_zuteilung(p, 1.0, 1)
    assert z.sum() == 1 and int(np.argmax(z)) == int(np.argmax(p))
    assert S.durchgang_strategie(np.ones(16), 0, p).sum() == 0 and S.voll_strategie(np.ones(16), 1.0, 0, p).sum() == 0
    assert S.rucksack_dp(p, np.ones(16, dtype=int), 1.0, 0) == 0.0


def test_alpha_sehr_klein_und_sehr_gross():
    p = S.make_prior("ein_gipfel", 4, 2)
    klein = S.wert(p, 1e-9, S.greedy_zuteilung(p, 1e-9, 10))
    assert 0 <= klein < 1e-7                                                  # Aufwand lohnt nicht, keine Division durch 0
    assert S.wert(p, 1e3, S.greedy_zuteilung(p, 1e3, 16)) == pytest.approx(1.0, abs=1e-12)
    assert S.wert(p, 1e-9, S.voll_strategie(np.ones(16), 1e-9, 10, p)) >= 0


def test_zellen_mit_prior_null_und_ein_einziger_gipfel():
    p = np.zeros(16)
    p[5] = 1.0
    z = S.greedy_zuteilung(p, 1.0, 6)
    assert z[5] == 6 and z.sum() == 6                                         # alles in die einzige Zelle mit Prior
    assert S.posterior(p, 1.0, z)[5] == pytest.approx(1.0)
    assert S.nie_abgesucht(p, z) == (15 / 16, 0.0)
    assert S.wasserstand(p, np.ones(16), 1.0, 3.0)[0][5] == pytest.approx(3.0)


def test_flacher_prior_gleichstand_ueberall():
    p = S.make_prior("flach", 4, 1)
    assert list(S.greedy_zuteilung(p, 1.0, 16)) == [1.0] * 16
    assert S.luecke_punkte(S.wert(p, 1.0, S.greedy_zuteilung(p, 1.0, 16)), S.wert(p, 1.0, S.durchgang_strategie(np.ones(16), 16, p))) == 0.0


def test_budget_kleiner_als_jede_zellenkosten():
    p = S.make_prior("ein_gipfel", 4, 3)
    c = np.full(16, 5.0)
    assert S.greedy_verhaeltnis(p, c, 1.0, 4.0).sum() == 0 and S.voll_strategie(c, 1.0, 4.0, p).sum() == 0 and S.durchgang_strategie(c, 4.0, p).sum() == 0
    assert S.rucksack_dp(p, np.full(16, 5), 1.0, 4) == 0.0


def test_kosten_eins_ueberall_ist_der_gleiche_preis_fall():
    p = S.make_prior("ein_gipfel", 4, 3)
    assert S.wert(p, 1.0, S.greedy_verhaeltnis(p, np.ones(16), 1.0, 9.0)) == pytest.approx(S.wert(p, 1.0, S.greedy_zuteilung(p, 1.0, 9)), abs=1e-12)


def test_geländefaktor_null_gibt_ueberall_kosten_zwei_und_gamma_gross_streut():
    assert (S.make_costs(8, 0.0, 5) == 2).all()
    c = S.make_costs(8, 1.0, 5)
    assert c.min() >= 1 and c.max() > c.min() and c.dtype.kind == "i"


def test_weg_ein_schritt_und_eingesperrt():
    p = S.make_prior("ein_gipfel", 3, 1)
    assert W.greedy_kurzsichtig(p, 1.0, 3, 1) == [0] and W.greedy_mit_ziel(p, 1.0, 3, 1) == [0] and W.maehen_ganz(3, 1) == [0]
    assert W.maehen_im_rechteck(p, 1.0, 3, 1) == [0]
    eins = np.array([1.0])
    assert W.greedy_kurzsichtig(eins, 1.0, 1, 4) == [0, 0, 0, 0] and W.maehen_ganz(1, 4) == [0, 0, 0, 0]
    assert W.greedy_mit_ziel(eins, 1.0, 1, 4) == [0, 0, 0, 0] and W.maehen_im_rechteck(eins, 1.0, 1, 4) == [0, 0, 0, 0]
    assert not W.ist_gueltiger_weg([0, 4], 3, 2) and not W.ist_gueltiger_weg([1, 1], 3, 2) and not W.ist_gueltiger_weg([0], 3, 2)


def test_weg_ueber_die_zahl_der_zellen_hinaus_wiederholt_besuche():
    p = S.make_prior("ein_gipfel", 3, 1)
    w = W.maehen_ganz(3, 20)
    assert len(w) == 20 and W.ist_gueltiger_weg(w, 3, 20) and len(set(w)) == 9


def test_drift_verzoegerung_null_ohne_streuung_und_L_gegen_null():
    par = D.PARAMETER_GEMISCHT
    assert D.verlust_eine_wolke(1.0, par, 1.0, 0.0) == pytest.approx(0.0, abs=1e-15)
    assert E.analyse_drift(0.0, 0.0, 0.0, 0.5, 1, 0.8, 4.0, 1.0, 0.25, 0.5) is None                 # S = 0: keine Wolke
    assert E.analyse_drift(0.0, 0.0, 0.3, 0.5, 1, 0.8, 4.0, 1.0, 0.25, 0.5) is not None
    L = 1e-3
    assert D.p_gauss(L ** 2 * np.pi * 2.0, 1.0, 2.0) == pytest.approx(L ** 2 / 2, rel=2e-3)      # P ~ L^2/2
    assert E.verdict_drift(None, None)[0] == "warning"


def test_verdikte_ohne_vorgerechnete_zellen_geben_eine_warnung_statt_einer_zahl():
    assert E.verdict_zuteilung(8, "ein_gipfel", 1.0, 1.0, 0.95, None, None)[0] == "warning"
    assert E.verdict_gelaende(1.0, {}, None, None)[0] == "warning"
    assert E.verdict_nichtfund(None, {}, {}, 0.5)[0] == "warning"
    assert E.verdict_falsches_alpha({}, None, 2.0)[0] == "warning"
    assert E.verdict_weg("ein_gipfel", 1.0, 15, None, {})[0] == "warning"
    assert E.optimum_flach(5, 1.0, 25) == pytest.approx(5 / 25 * (1 - np.exp(-1.0)))
