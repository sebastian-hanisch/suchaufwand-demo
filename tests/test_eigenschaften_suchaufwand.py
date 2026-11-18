"""Eigenschaftstests (ORACLE-PLAYBOOK §5.2): Aussagen, die für das Verfahren gelten müssen, ohne dass man das richtige Ergebnis kennt; dazu Zweig-Tests (jede Variante
weicht auf mindestens einer Instanz von der Nachbarvariante ab). Je Eigenschaft 20 bis 50 Instanzen mit festem Zufalls-Seed."""

import numpy as np
import pytest

import sua_drift as D
import sua_path as W
import sua_search as S
from sua_rng import SplitMix64


def instanzen(n=6, k=30):
    return [(S.make_prior(S.PRIORS[i % 3], n, i + 1), (0.3, 1.0, 3.0)[i % 3]) for i in range(k)]


def test_entdeckung_waechst_mit_budget_und_alpha():
    for p, alpha in instanzen():
        werte = [S.wert(p, alpha, S.greedy_zuteilung(p, alpha, B)) for B in (0, 1, 3, 9, 27, 81)]
        assert werte[0] == 0.0 and all(a <= b + 1e-12 for a, b in zip(werte, werte[1:]))
        assert S.wert(p, alpha, S.greedy_zuteilung(p, alpha, 20)) <= S.wert(p, 2 * alpha, S.greedy_zuteilung(p, 2 * alpha, 20)) + 1e-12


def test_grenzzuwaechse_je_zelle_fallen_mit_der_zahl_der_durchgaenge():
    for p, alpha in instanzen():
        g = [S.grenzzuwachs_naechster(p, alpha, np.full(len(p), k)) for k in range(5)]
        for a, b in zip(g, g[1:]):
            assert (b <= a + 1e-15).all() and (b[p > 0] < a[p > 0]).all()


def test_zuteilung_summiert_zum_budget_und_ist_ganzzahlig():
    for p, alpha in instanzen():
        z = S.greedy_zuteilung(p, alpha, 25)
        assert z.sum() == 25 and (z == np.round(z)).all() and (z >= 0).all()


def test_vertauschen_von_zellen_aendert_die_entdeckung_nicht():
    rng = SplitMix64(1)
    for p, alpha in instanzen(5, 20):
        perm = list(range(len(p)))
        rng.shuffle(perm)
        assert S.wert(p[perm], alpha, S.greedy_zuteilung(p[perm], alpha, 12)) == pytest.approx(S.wert(p, alpha, S.greedy_zuteilung(p, alpha, 12)), abs=1e-12)


def test_kosten_und_budget_mit_c_malnehmen_aendert_das_optimum_nicht():
    for k in range(20):
        p = S.make_prior("ein_gipfel", 4, k + 1)
        c = S.make_costs(4, 1.0, k + 1)
        alpha, B = (0.5, 1.0, 2.5)[k % 3], 12 + k
        assert S.rucksack_dp(p, c * 3, alpha, 3 * B) == pytest.approx(S.rucksack_dp(p, c, alpha, B), abs=1e-12)


def test_gleichstaende_werden_nach_festem_tie_break_aufgeloest():
    p = np.full(16, 1 / 16)
    assert list(S.greedy_folge(p, 1.0, 20)) == [i % 16 for i in range(20)]                         # flach: reihum, kleinster Index zuerst
    assert list(S.durchgang_strategie(np.ones(16), 5, p)) == [1] * 5 + [0] * 11                    # Gleichstand: stabile Sortierung
    assert list(S.greedy_zuteilung(np.array([0.5, 0.5, 0.0]), 1.0, 1)) == [1, 0, 0]


def test_posterior_ist_nach_mehr_durchgaengen_in_der_zelle_kleiner():
    for p, alpha in instanzen():
        z0 = np.zeros(len(p))
        z1 = z0.copy()
        z1[int(np.argmax(p))] += 1
        assert S.posterior(p, alpha, z1)[int(np.argmax(p))] < p[int(np.argmax(p))]


def test_entdeckungsfunktion_nie_ueber_eins_und_prior_summe_eins():
    for p, alpha in instanzen():
        assert p.sum() == pytest.approx(1.0) and 0 <= S.wert(p, alpha, S.greedy_zuteilung(p, alpha, 500)) <= 1 + 1e-12


# ---------------------------------------------------------------------------------------------- Zweig-Tests
def test_zweig_voll_weicht_von_ein_durchgang_ab_und_beide_vom_optimum():
    c = np.ones(64)
    abw_vd = abw_ov = abw_od = 0
    for p, alpha in instanzen(8, 30):
        B = 64
        zo, zv, zd = S.greedy_zuteilung(p, alpha, B), S.voll_strategie(c, alpha, B, p), S.durchgang_strategie(c, B, p)
        abw_vd += int(abs(S.wert(p, alpha, zv) - S.wert(p, alpha, zd)) > 1e-9)
        abw_ov += int(S.wert(p, alpha, zo) - S.wert(p, alpha, zv) > 1e-9)
        abw_od += int(S.wert(p, alpha, zo) - S.wert(p, alpha, zd) > 1e-9)
    assert abw_vd >= 1 and abw_ov >= 1 and abw_od >= 1


def test_zweig_prior_gegen_prior_je_aufwand_weicht_bei_gamma_ab_und_nicht_bei_gamma_null():
    unterschiede = {0.0: 0, 1.0: 0}
    for gamma in (0.0, 1.0):
        for inst in range(1, 21):
            p = S.make_prior("ein_gipfel", 8, inst)
            c = S.make_costs(8, gamma, inst).astype(float)
            B = int(round(0.5 * c.sum()))
            a, b = S.voll_strategie(c, 1.0, B, p), S.voll_strategie(c, 1.0, B, p / c)
            unterschiede[gamma] += int(abs(S.wert(p, 1.0, a) - S.wert(p, 1.0, b)) > 1e-9)
    assert unterschiede[0.0] == 0 and unterschiede[1.0] >= 1


def test_zweig_aktualisiert_gegen_alte_karte_und_dort_weiter():
    a = b = 0
    for inst in range(1, 21):
        r = S.nichtfund_runden(S.make_prior("ein_gipfel", 8, inst), 1.0, 16, 16)
        a += int(r["gesamt_a"] - r["gesamt_b"] > 1e-9)
        b += int(r["gesamt_a"] - r["gesamt_c"] > 1e-9)
    assert a >= 1 and b >= 1


def test_zweig_wege_weichen_voneinander_ab():
    kurz = ziel = rechteck = 0
    for inst in range(1, 16):
        p = S.make_prior("zwei_gipfel", 5, inst)
        T = 10
        wk, wz, wm, wr = W.greedy_kurzsichtig(p, 1.0, 5, T), W.greedy_mit_ziel(p, 1.0, 5, T), W.maehen_ganz(5, T), W.maehen_im_rechteck(p, 1.0, 5, T)
        kurz += int(wk != wz)
        ziel += int(wz != wm)
        rechteck += int(wm != wr)
    assert kurz >= 1 and ziel >= 1 and rechteck >= 1


def test_zweig_falsches_alpha_weicht_ab_wenn_das_alpha_nicht_stimmt():
    p = S.make_prior("ein_gipfel", 8, 3)
    z_r, z_a, pr, pa = S.falsches_alpha(p, 1.0, 10.0, 64)
    assert (z_r != z_a).any() and pa < pr
    z_r2, z_a2, pr2, pa2 = S.falsches_alpha(p, 1.0, 1.0, 64)
    assert (z_r2 == z_a2).all() and pa2 == pr2


def test_drift_verlust_waechst_mit_dem_aufwand_nicht_ueberproportional_bei_gemischter_drift():
    par = D.PARAMETER_GEMISCHT
    for L0 in (0.5, 1.0, 2.0, 4.0):
        v10, v25 = D.verlust_eine_wolke(L0, par, 1.0, 0.1), D.verlust_eine_wolke(L0, par, 1.0, 0.25)
        assert 0 < v10 < v25 and v25 / v10 < 3.0
