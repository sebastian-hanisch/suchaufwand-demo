"""Handbeispiele von Hand (exakte Brüche, q = 1/2 je Durchgang, also alpha = ln 2) gegen den Kern in `sua_search.py`, `sua_path.py` und `sua_drift.py`.

Jede Zahl steht hier als Rechnung oder als Handwert; die Zuteilungen und Wege sind unabhängig mit `fractions` und Aufzählung nachgerechnet."""

import itertools
import math
from fractions import Fraction as F

import numpy as np
import pytest

import sua_drift as D
import sua_path as W
import sua_search as S

ALPHA = math.log(2)
Q = F(1, 2)


def wert_f(p, z, q=Q):
    return sum(pi * (1 - (1 - q) ** zi) for pi, zi in zip(p, z))


def zuteilungen(zellen, b):
    """Alle Zuteilungen von b Durchgängen auf `zellen` Zellen (Sterne und Striche)."""
    for c in itertools.combinations(range(b + zellen - 1), zellen - 1):
        z, vor = [], -1
        for k in c:
            z.append(k - vor - 1)
            vor = k
        z.append(b + zellen - 2 - vor)
        yield z


def pf(p):
    return np.array([float(v) for v in p])


# ---------------------------------------------------------------------------------------------- A: Zuteilung
def test_zuteilung_9_durchgaenge_ist_237_400(handbeispiel):
    z = S.greedy_zuteilung(pf(handbeispiel), ALPHA, 9)
    assert list(z) == [1, 2, 1, 2, 2, 1, 0, 0, 0]
    assert wert_f(handbeispiel, [int(v) for v in z]) == F(237, 400)          # 59,25 %
    assert S.wert(pf(handbeispiel), ALPHA, z) == pytest.approx(0.5925, rel=1e-12)


def test_aufzaehlung_aller_24310_zuteilungen_hat_ein_eindeutiges_optimum(handbeispiel):
    alle = [(wert_f(handbeispiel, z), z) for z in zuteilungen(9, 9)]
    assert len(alle) == math.comb(17, 8) == 24310
    best = max(v for v, _ in alle)
    assert best == F(237, 400) and sum(1 for v, _ in alle if v == best) == 1


def test_die_neun_groessten_grenzzuwaechse_in_hundertsteln(handbeispiel):
    gewinne = sorted((float(handbeispiel[i] * 100) * 0.5 * 0.5 ** k, i, k + 1) for i in range(9) for k in range(4))[::-1]
    assert [g for g, _, _ in gewinne[:9]] == [13.5, 9.5, 7.5, 6.75, 5.5, 4.75, 4.5, 3.75, 3.5]
    assert gewinne[9][0] == 3.375 and gewinne[10][0] == 3.0


def test_ein_durchgang_je_zelle_50_prozent_luecke_9_25(handbeispiel):
    p = pf(handbeispiel)
    z1 = S.durchgang_strategie(np.ones(9), 9, p)
    assert list(z1) == [1] * 9 and wert_f(handbeispiel, [1] * 9) == F(1, 2)
    assert S.luecke_punkte(S.wert(p, ALPHA, S.greedy_zuteilung(p, ALPHA, 9)), S.wert(p, ALPHA, z1)) == pytest.approx(9.25, abs=1e-9)


def test_vollstaendig_mit_zwei_durchgaengen_je_zelle_58_5_prozent(handbeispiel):
    p = pf(handbeispiel)
    # tau = 0,75 bei q = 1/2: ceil(-ln(0,25)/ln 2) = 2 Durchgänge je Zelle
    z = S.voll_strategie(np.ones(9), ALPHA, 9, p, tau=0.75)
    assert list(z) == [0, 2, 2, 2, 2, 1, 0, 0, 0]
    assert wert_f(handbeispiel, [int(v) for v in z]) == F(117, 200)           # 58,5 %
    assert S.luecke_punkte(0.5925, S.wert(p, ALPHA, z)) == pytest.approx(0.75, abs=1e-9)


def test_nie_abgesuchte_zellen_6_7_8_mit_12_prozent_der_masse(handbeispiel):
    p = pf(handbeispiel)
    z = S.greedy_zuteilung(p, ALPHA, 9)
    anteil, masse = S.nie_abgesucht(p, z)
    assert anteil == pytest.approx(3 / 9) and masse == pytest.approx(0.12)


def test_posterior_argmax_nach_jedem_durchgang_ist_die_greedy_folge_und_grenzzuwachs_formel(handbeispiel):
    p = handbeispiel
    z = [0] * 9
    pges = F(0)
    folge = []
    for _ in range(9):
        nicht = 1 - pges
        post = [p[i] * (1 - Q) ** z[i] / nicht for i in range(9)]
        assert sum(post) == 1
        j = post.index(max(post))
        gain = p[j] * (1 - Q) ** z[j] * Q
        assert gain == Q * nicht * post[j]                                     # Grenzzuwachs = q (1 - P) Posterior
        z[j] += 1
        folge.append(j)
        pges += gain
    assert folge == S.greedy_folge(pf(p), ALPHA, 9) == [4, 1, 3, 4, 2, 1, 5, 3, 0]
    assert pges == F(237, 400)


def test_schwelle_auf_dem_grenzzuwachs_und_posterior_nur_im_band(handbeispiel):
    p = handbeispiel
    z = [1, 2, 1, 2, 2, 1, 0, 0, 0]
    genommen = [p[i] * Q * (1 - Q) ** (z[i] - 1) for i in range(9) if z[i] > 0]
    naechste = [p[i] * Q * (1 - Q) ** z[i] for i in range(9)]
    assert min(genommen) == F(35, 1000) and max(naechste) == F(3375, 100000)
    assert min(genommen) >= max(naechste)
    w = [p[i] * (1 - Q) ** z[i] for i in range(9)]
    besucht = [w[i] for i in range(9) if z[i] > 0]
    assert max(besucht) / min(besucht) == F(27, 14) <= 1 / (1 - Q)           # Band e^alpha = 2, nicht gleich hoch
    assert w[7] == F(6, 100) > w[0] == F(35, 1000)                            # unbesuchte Zelle 7 über besuchter Zelle 0
    kleinste, groesste = S.linie_grenzzuwachs(pf(p), ALPHA, np.array(z, dtype=float))
    assert kleinste == pytest.approx(0.035) and groesste == pytest.approx(0.03375)


def test_stetiger_grenzzuwachs_ist_alpha_mal_1_minus_p_mal_posterior(handbeispiel):
    p = pf(handbeispiel)
    z = np.array([1, 2, 1, 2, 2, 1, 0, 0, 0], dtype=float)
    P = S.wert(p, ALPHA, z)
    post = S.posterior(p, ALPHA, z)
    h = 1e-6
    f = lambda zz: p[4] * (1 - math.exp(-ALPHA * zz))
    ableitung = (f(z[4] + h) - f(z[4] - h)) / (2 * h)
    assert ableitung == pytest.approx(ALPHA * (1 - P) * post[4], abs=1e-9)


def test_gegenfall_alpha_gross_alle_verfahren_gleich(handbeispiel):
    p = pf(handbeispiel)
    alpha = 10 * math.log(2)
    zg = S.greedy_zuteilung(p, alpha, 9)
    assert list(zg) == [1] * 9
    q = F(1023, 1024)
    assert wert_f(handbeispiel, [1] * 9, q) == q
    assert S.wert(p, alpha, zg) == pytest.approx(1023 / 1024, rel=1e-12)
    assert S.wert(p, alpha, S.voll_strategie(np.ones(9), alpha, 9, p)) == pytest.approx(1023 / 1024, rel=1e-12)


def test_gegenfall_flacher_prior_ein_durchgang_ist_das_optimum():
    pflach = [F(1, 9)] * 9
    assert wert_f(pflach, [1] * 9) == F(1, 2) == max(wert_f(pflach, z) for z in zuteilungen(9, 9))
    p = np.full(9, 1 / 9)
    assert wert_f(pflach, [int(v) for v in S.voll_strategie(np.ones(9), ALPHA, 9, p, tau=0.75)]) == F(7, 18)   # 38,9 %, Lücke 11,1 Punkte


# ---------------------------------------------------------------------------------------------- B: Nichtfund
def test_nichtfund_runde_1_und_posterior(handbeispiel):
    p = pf(handbeispiel)
    r = S.nichtfund_runden(p, ALPHA, 4, 5)
    assert list(r["z1"]) == [0, 1, 0, 1, 2, 0, 0, 0, 0]
    assert r["d1"] == pytest.approx(149 / 400, rel=1e-12)
    rest = [handbeispiel[i] * (1 - Q) ** int(r["z1"][i]) for i in range(9)]
    assert sum(rest) == F(251, 400) and rest[2] == F(11, 100) > rest[4] == F(27, 400)    # unbesuchte Zelle 2 über besuchter Zelle 4
    assert r["posterior"] == pytest.approx([float(v / sum(rest)) for v in rest], rel=1e-12)


def test_aktualisiert_ist_der_einplan_exakt_und_alte_karte_kostet(handbeispiel):
    p = pf(handbeispiel)
    r = S.nichtfund_runden(p, ALPHA, 4, 5)
    assert r["gesamt_a"] == pytest.approx(0.5925, rel=1e-12) and r["einplan"] == pytest.approx(0.5925, rel=1e-12)
    assert list(r["z1"] + r["z2a"]) == [1, 2, 1, 2, 2, 1, 0, 0, 0]
    assert r["gesamt_b"] == pytest.approx(901 / 1600, rel=1e-12)                              # 56,3125 %
    assert S.luecke_punkte(r["gesamt_a"], r["gesamt_b"]) == pytest.approx(2.9375, abs=1e-9)
    assert S.luecke_punkte(r["gesamt_a"], r["gesamt_c"]) > 0


def test_falsches_alpha_zu_hoch_belegt_andere_zellen(handbeispiel):
    p = pf(handbeispiel)
    _, z_ang, p_r, p_a = S.falsches_alpha(p, ALPHA, 2 * ALPHA, 9)                              # q angenommen 3/4 statt 1/2
    assert list(z_ang) == [1, 1, 1, 1, 2, 1, 1, 1, 0]
    assert p_r == pytest.approx(0.5925, rel=1e-12) and p_a == pytest.approx(0.5625, rel=1e-12)


# ---------------------------------------------------------------------------------------------- C: Weg
def nb3(i):
    y, x = divmod(i, 3)
    return [i] + ([i - 3] if y > 0 else []) + ([i + 3] if y < 2 else []) + ([i - 1] if x > 0 else []) + ([i + 1] if x < 2 else [])


def alle_wege(T, start=0):
    def rec(pos, rest):
        if rest == 0:
            yield [pos]
            return
        for v in nb3(pos):
            for w in rec(v, rest - 1):
                yield [pos] + w
    return rec(start, T - 1)


def test_weg_3x3_T5_wegoptimum_40_75_wegpreis_2_punkte(handbeispiel):
    p = pf(handbeispiel)
    werte = [(wert_f(handbeispiel, [w.count(i) for i in range(9)]), w) for w in alle_wege(5)]
    wopt = max(v for v, _ in werte)
    assert wopt == F(163, 400)                                                                 # 40,75 %
    assert sum(1 for v, _ in werte if v == wopt) == 4
    assert [0, 1, 4, 3, 4] in [w for v, w in werte if v == wopt]
    schranke = wert_f(handbeispiel, [int(v) for v in S.greedy_zuteilung(p, ALPHA, 5)])
    assert schranke == F(171, 400) and schranke - wopt == F(8, 400)                            # Wegpreis 2,0 Punkte
    assert W.weg_wert(p, ALPHA, [0, 1, 4, 3, 4]) == pytest.approx(float(wopt), rel=1e-12)


def test_weg_verfahren_3x3(handbeispiel):
    p = pf(handbeispiel)
    assert W.greedy_kurzsichtig(p, ALPHA, 3, 5) == [0, 1, 4, 3, 4]
    assert W.weg_wert(p, ALPHA, W.greedy_kurzsichtig(p, ALPHA, 3, 5)) == pytest.approx(0.4075, rel=1e-12)
    assert W.maehen_ganz(3, 5) == [0, 1, 2, 5, 4]
    assert W.weg_wert(p, ALPHA, W.maehen_ganz(3, 5)) == pytest.approx(0.365, rel=1e-12)       # 36,5 %
    assert W.praemien_summe(p, [0, 1, 4, 3, 4]) == pytest.approx(0.07 + 0.19 + 0.27 + 0.15)


def test_orienteering_fassung_3x3_hat_praemien_79_prozent_und_wahren_wert_39_5(handbeispiel):
    p = pf(handbeispiel)
    wege = list(alle_wege(5))
    pm = [sum(handbeispiel[i] for i in set(w)) for w in wege]
    beste = max(pm)
    assert beste == F(79, 100)
    lex = max(wert_f(handbeispiel, [w.count(i) for i in range(9)]) for w, m in zip(wege, pm) if m == beste)
    assert lex == F(79, 200)                                                                   # 39,5 %
    assert 100 * (float(F(163, 400)) - float(lex)) == pytest.approx(1.25)                      # 1,25 Punkte hinter dem Wegoptimum


def test_flacher_prior_3x3_jeder_weg_ueber_5_zellen_ist_optimal():
    pflach = [F(1, 9)] * 9
    werte = [wert_f(pflach, [w.count(i) for i in range(9)]) for w in alle_wege(5)]
    assert max(werte) == F(5, 18) == wert_f(pflach, [w.count(i) for w in [[0, 1, 2, 5, 4]] for i in range(9)])


# ---------------------------------------------------------------------------------------------- D: Drift
def test_varianz_und_flaechenverhaeltnisse_in_brüchen():
    par = (1.0, 0.5, 0.5)                                      # s0^2 = 1, 2 D = 1, sU^2 = 1/4
    assert [D.varianz(t, *par) for t in (1, 2, 4, 8)] == [2.25, 4.0, 9.0, 25.0]
    assert [D.flaechenverhaeltnis(t, par) for t in (2, 4, 8)] == pytest.approx([16 / 9, 4.0, 100 / 9])
    assert [D.flaechenverhaeltnis(t, (0.0, 0.5, 0.0)) for t in (2, 4, 8)] == pytest.approx([2, 4, 8])      # nur Diffusion
    assert [D.flaechenverhaeltnis(t, (0.0, 0.0, 0.5)) for t in (2, 4, 8)] == pytest.approx([4, 16, 64])    # nur konstante Geschwindigkeitsunsicherheit


def test_geschlossene_form_beispiel_mit_E_gleich_pi():
    par = (1.0, 0.5, 0.5)
    E = math.pi
    assert D.p_gauss(E, 1.0, D.varianz(1.0, *par)) == pytest.approx(1 - F(5, 3) * math.exp(-2 / 3), rel=1e-12)   # 14,43 %
    verl = [1 - D.p_gauss(E, 1.0, D.varianz(1 + d, *par)) / D.p_gauss(E, 1.0, D.varianz(1.0, *par)) for d in (0.1, 0.25, 0.5, 1.0)]
    assert [round(100 * v, 2) for v in verl] == [5.07, 12.00, 21.98, 37.49]
    assert verl[1] / verl[0] < 2.5
    assert D.laenge_L(E, 1.0, 2.25) == pytest.approx(2 / 3)


def test_kleines_L_P_ungefaehr_L_quadrat_halbe():
    L = 1e-3
    assert (1 - (1 + L) * math.exp(-L)) / (L ** 2 / 2) == pytest.approx(1.0, abs=2e-3)
