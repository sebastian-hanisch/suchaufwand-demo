"""Orakel-Tests Zuteilung, Nichtfund und Rechteck (ORACLE-PLAYBOOK §2, §5, §6): jeder Kern gegen einen anderen Rechenweg.

(a) Grenzzuwachs-Greedy gegen Aufzählung aller ganzzahligen Zuteilungen (4x4 Budget 3 und 5, 6x6 Budget 3); (b) Wasserstand gegen die Bedingungen von Kuhn und Tucker,
SLSQP und einen feinen Greedy; (c) Rucksack-Rechnung gegen rekursive Aufzählung; (d) Posterior gegen die direkte Summe über alle gemeinsamen Ergebnisse und gegen Monte-Carlo;
(e) additives Prinzip bei ganzen und bei teilbaren Durchgängen; (f) Rechteck-Baustein und Vorausschau gegen Aufzählung aller Rechteckpaare; (g) Schwelle auf dem Grenzzuwachs und
Posterior-Band; (i) bei sehr großem alpha ist das Optimum die Reihenfolge der Prior-Werte. Die volle Aufzählung (120 Fälle) steht in `data/sua_messreihe.json`."""

import itertools
from math import exp, log

import numpy as np
import pytest

import sua_messung as M
import sua_search as S
import sua_zufall as Z
from sua_rng import SplitMix64


def aufzaehlen(p, c, alpha, B):
    """Rekursive Aufzählung aller Zuteilungen mit Kosten höchstens B (unabhängig von der Rucksack-Rechnung)."""
    best = 0.0

    def rec(i, rest, acc):
        nonlocal best
        if i == len(p):
            best = max(best, acc)
            return
        k = 0
        while c[i] * k <= rest:
            rec(i + 1, rest - c[i] * k, acc + p[i] * (1 - exp(-alpha * k)))
            k += 1

    rec(0, B, 0.0)
    return best


# ---------------------------------------------------------------------------------------------- (a) Aufzählung
@pytest.mark.parametrize("n,b,faelle", [(4, 3, 6), (4, 5, 6), (6, 3, 3)])
def test_grenzzuwachs_greedy_gleich_aufzaehlung_aller_zuteilungen(n, b, faelle):
    maxd, k = 0.0, 0
    for inst in range(1, faelle + 1):
        for art in ("ein_gipfel", "zwei_gipfel"):
            for alpha in (0.3, 1.0, 2.5):
                maxd = max(maxd, M.aufzaehlung_abweichung(n, b, art, alpha, inst))
                k += 1
    assert k == faelle * 6 and maxd < 1e-12


def test_kompositionen_zaehlen_stimmt():
    assert len(M.kompositionen(16, 5)) == 15504 and len(M.kompositionen(9, 3)) == 165
    assert (M.kompositionen(7, 4).sum(axis=1) == 4).all()


# ---------------------------------------------------------------------------------------------- (b) Wasserstand
def test_wasserstand_erfuellt_kuhn_tucker_und_summiert_zum_budget():
    for inst in range(12):
        p = S.make_prior(("ein_gipfel", "zwei_gipfel")[inst % 2], 4, inst + 1)
        c = S.make_costs(4, (0.0, 0.5, 1.0)[inst % 3], inst + 1).astype(float)
        alpha, B = (0.4, 1.0, 2.5)[inst % 3], (4.0, 8.0, 16.0)[(inst // 3) % 3]
        z, lam = S.wasserstand(p, c, alpha, B)
        assert (c * z).sum() == pytest.approx(B, abs=1e-9)
        marg = p * alpha * np.exp(-alpha * z) / c
        aktiv = z > 1e-12
        assert np.abs(marg[aktiv] - lam).max() < 1e-9                          # besuchte Zellen: alle auf dem Niveau
        assert (p[~aktiv] * alpha / c[~aktiv] - lam).max() <= 1e-9              # unbesuchte Zellen liegen darunter


def test_wasserstand_gleich_slsqp_und_feiner_greedy():
    optimize = pytest.importorskip("scipy.optimize")
    for inst in range(6):
        p = S.make_prior("ein_gipfel" if inst % 2 == 0 else "zwei_gipfel", 4, inst + 1)
        c = np.ones(16)
        alpha, B = (0.5, 1.0, 2.0)[inst % 3], 6.0
        z, _ = S.wasserstand(p, c, alpha, B)
        v = S.wert(p, alpha, z)
        res = optimize.minimize(lambda x: -np.sum(p * (1 - np.exp(-alpha * x))), x0=np.full(16, B / 16), jac=lambda x: -p * alpha * np.exp(-alpha * x),
                                bounds=[(0, None)] * 16, constraints=[{"type": "eq", "fun": lambda x: x.sum() - B, "jac": lambda x: np.ones(16)}], method="SLSQP",
                                options={"ftol": 1e-14, "maxiter": 500})
        assert -res.fun == pytest.approx(v, abs=1e-8)
        zz, schritt = np.zeros(16), B / 3000
        for _ in range(3000):
            zz[int(np.argmax(p * np.exp(-alpha * zz)))] += schritt
        assert S.wert(p, alpha, zz) == pytest.approx(v, abs=2e-4)                # der feine Greedy nähert nur an


def test_ganze_durchgaenge_nie_ueber_der_teilbaren_obergrenze():
    for inst in range(1, 21):
        p = S.make_prior("zwei_gipfel" if inst % 2 else "ein_gipfel", 6, inst)
        alpha = (0.3, 1.0, 3.0, 10.0)[inst % 4]
        for B in (6, 18, 36):
            assert S.wert(p, alpha, S.greedy_zuteilung(p, alpha, B)) <= S.obergrenze_teilbar(p, np.ones(36), alpha, B) + 1e-12


# ---------------------------------------------------------------------------------------------- (c) Rucksack
def test_rucksack_rechnung_gleich_rekursiver_aufzaehlung_3x3():
    for k in range(30):
        rng = SplitMix64(500 + k)
        p = S._prior_ein_gipfel(3, rng)
        c = 1 + Z.integers(rng, 9, 3)
        alpha, B = (0.5, 1.0, 2.5)[k % 3], 4 + k % 8
        assert S.rucksack_dp(p, c, alpha, B) == pytest.approx(aufzaehlen(p, [int(v) for v in c], alpha, B), abs=1e-12)


def test_rucksack_mit_gleichen_kosten_ist_der_grenzzuwachs_greedy():
    for inst in range(1, 11):
        p = S.make_prior("ein_gipfel", 5, inst)
        assert S.rucksack_dp(p, np.ones(25, dtype=int), 1.3, 11) == pytest.approx(S.wert(p, 1.3, S.greedy_zuteilung(p, 1.3, 11)), abs=1e-12)


def test_greedy_nach_zuwachs_je_kosten_nie_ueber_dem_optimum():
    for k in range(40):
        p = S.make_prior("ein_gipfel", 4, k + 1)
        c = S.make_costs(4, 1.0, k + 1)
        alpha, B = (0.5, 1.0, 2.5)[k % 3], 10 + k % 15
        assert S.wert(p, alpha, S.greedy_verhaeltnis(p, c.astype(float), alpha, B)) <= S.rucksack_dp(p, c, alpha, B) + 1e-12


# ---------------------------------------------------------------------------------------------- (d) Posterior
def test_posterior_gleich_direkte_summe_ueber_alle_gemeinsamen_ergebnisse():
    p = np.array([0.4, 0.3, 0.2, 0.1])
    m = np.array([2, 1, 0, 3])
    q = 0.35
    alpha = -log(1 - q)                                                       # 1 - exp(-alpha) = q
    formel = S.posterior(p, alpha, m.astype(float))
    gemeinsam = np.zeros(4)
    for i in range(4):
        for ausgang in itertools.product([0, 1], repeat=int(m[i])):
            if sum(ausgang) == 0:
                gemeinsam[i] += p[i] * np.prod([q if o else 1 - q for o in ausgang])
    assert np.abs(formel - gemeinsam / gemeinsam.sum()).max() < 1e-14


def test_posterior_gleich_monte_carlo_6x6():
    p = S.make_prior("ein_gipfel", 6, 3)
    rng = SplitMix64(41)
    zz = Z.integers(rng, 36, 4).astype(float)
    alpha, N = 0.8, 400_000
    ort = np.minimum(np.searchsorted(np.cumsum(p), Z.uniform(rng, N)), 35)
    nicht = Z.uniform(rng, N) >= 1 - np.exp(-alpha * zz[ort])
    emp = np.bincount(ort[nicht], minlength=36) / nicht.sum()
    post = S.posterior(p, alpha, zz)
    se = np.sqrt(post * (1 - post) / nicht.sum())
    assert np.all(np.abs(emp - post) < 5 * se + 1e-9)


def test_posterior_summiert_zu_eins_und_bleibt_null_wo_der_prior_null_ist():
    p = S.make_prior("zwei_gipfel", 8, 2)
    post = S.posterior(p, 1.0, S.greedy_zuteilung(p, 1.0, 40))
    assert post.sum() == pytest.approx(1.0) and (post[p == 0] == 0).all() and (p == 0).any()


# ---------------------------------------------------------------------------------------------- (e) additives Prinzip
def test_aktualisieren_in_runden_ist_eine_zuteilung_mit_dem_ganzen_budget_ganzzahlig_exakt():
    k = 0
    for inst in range(1, 31):
        art = ("ein_gipfel", "zwei_gipfel")[inst % 2]
        for alpha, B in ((0.5, 16), (1.0, 32), (2.0, 64)):
            p = S.make_prior(art, 8, inst)
            r = S.nichtfund_runden(p, alpha, B // 2, B - B // 2)
            assert r["gesamt_a"] == pytest.approx(r["einplan"], abs=1e-12)
            assert list(r["z1"] + r["z2a"]) == list(S.greedy_zuteilung(p, alpha, B))
            k += 1
    assert k == 90


def test_runden_mit_posterior_sind_eine_zuteilung_bei_teilbarem_aufwand():
    for inst in range(1, 11):
        p = S.make_prior(("ein_gipfel", "zwei_gipfel")[inst % 2], 8, inst)
        alpha, c = (0.5, 1.0, 3.0)[inst % 3], np.ones(64)
        z1, _ = S.wasserstand(p, c, alpha, 32.0)
        q, rest, ptot, zsum = p.copy(), 1.0, 0.0, np.zeros(64)
        for _ in range(4):
            zr, _ = S.wasserstand(q, c, alpha, 8.0)
            d = S.wert(q, alpha, zr)
            ptot += rest * d
            rest *= 1 - d
            q = S.posterior(q, alpha, zr)
            zsum += zr
        assert ptot == pytest.approx(S.wert(p, alpha, z1), abs=1e-9) and np.abs(zsum - z1).max() < 1e-8


def test_alte_karte_und_dort_weiter_sind_nie_besser_als_aktualisiert():
    for inst in range(1, 21):
        p = S.make_prior(("ein_gipfel", "zwei_gipfel")[inst % 2], 8, inst)
        r = S.nichtfund_runden(p, 1.0, 16, 16)
        assert r["gesamt_b"] <= r["gesamt_a"] + 1e-12 and r["gesamt_c"] <= r["gesamt_a"] + 1e-12


# ---------------------------------------------------------------------------------------------- (f) Rechteck
def rechteck_brute(p, alpha, B, n):
    best = -1.0
    for y0 in range(n):
        for y1 in range(y0 + 1, n + 1):
            for x0 in range(n):
                for x1 in range(x0 + 1, n + 1):
                    idx = [y * n + x for y in range(y0, y1) for x in range(x0, x1)]
                    m = B // len(idx)
                    z = np.zeros(n * n)
                    z[idx] = m
                    for i in sorted(idx, key=lambda i: -p[i])[: B - m * len(idx)]:
                        z[i] += 1
                    best = max(best, S.wert(p, alpha, z))
    return best


def test_rechteck_strategie_mit_allen_kandidaten_gleich_aufzaehlung_4x4():
    for inst in range(1, 13):
        p = S.make_prior(("ein_gipfel", "zwei_gipfel")[inst % 2], 4, inst)
        alpha, B = (0.3, 1.0, 2.5)[inst % 3], (3, 6, 10)[inst % 3]
        z = S.rechteck_strategie(p, np.ones(16), alpha, B, 4, vorauswahl=200)
        assert S.wert(p, alpha, z) == pytest.approx(rechteck_brute(p, alpha, B, 4), abs=1e-12)


def test_rechteck_baustein_und_vorausschau_gleich_aufzaehlung_aller_rechteckpaare_4x4():
    R = S.rechtecke(4)

    def flaeche(r):
        return (r[1] - r[0]) * (r[3] - r[2])

    for inst in range(1, 5):
        p = S.make_prior(("ein_gipfel", "zwei_gipfel")[inst % 2], 4, inst)
        for alpha, B in ((0.5, 8), (1.0, 6)):
            B1, B2 = B // 2, B - B // 2
            kand1 = [tuple(r) for r in R if flaeche(r) <= B1]
            kand2 = [tuple(r) for r in R if flaeche(r) <= B2]
            alle = max(S.wert(p, alpha, S.plan_rechteck(p, 4, r, B1)) for r in kand1)                     # Baustein: Vorauswahl von 6 gegen alle Rechtecke
            assert S.bestes_rechteck_muster(p, alpha, 4, B1)[0] == pytest.approx(alle, abs=1e-12)
            paare = -1.0                                                                                    # Paar-Aufzählung
            for r1, r2 in itertools.product(kand1, kand2):
                z1 = S.plan_rechteck(p, 4, r1, B1)
                paare = max(paare, S.wert(p, alpha, z1 + S.plan_rechteck(p * np.exp(-alpha * z1), 4, r2, B2)))
            voraus = -1.0                                                                                   # Vorausschau mit allen Rechtecken als Kandidaten
            for r1 in kand1:
                z1 = S.plan_rechteck(p, 4, r1, B1)
                z2 = S.bestes_rechteck_muster(p * np.exp(-alpha * z1), alpha, 4, B2, 200)[1]
                voraus = max(voraus, S.wert(p, alpha, z1 + z2))
            assert voraus == pytest.approx(paare, abs=1e-12)


def test_kurzsichtiges_rechteck_ist_nicht_optimal_und_die_vorausschau_hilft():
    r = [S.rechteck_runden(S.make_prior("zwei_gipfel", 8, i), 1.0, 8, 32) for i in range(1, 9)]
    assert all(x["kurzsichtig"] <= x["optimum"] + 1e-12 and x["vorausschau"] >= x["kurzsichtig"] - 1e-12 for x in r)
    assert sum(x["optimum"] - x["kurzsichtig"] > 1e-3 for x in r) >= 6                 # nicht mehr optimal
    assert sum(x["vorausschau"] > x["kurzsichtig"] + 1e-9 for x in r) >= 1             # Vorausschau besser in mindestens einer Instanz


# ---------------------------------------------------------------------------------------------- (g) Schwelle und Band, (i) alpha groß
def test_schwelle_auf_dem_grenzzuwachs_und_posterior_band_bei_ganzen_durchgaengen():
    k = 0
    for inst in range(1, 16):
        for art in S.PRIORS:
            for alpha in (0.5, 1.0, 3.0):
                p = S.make_prior(art, 8, inst)
                z = S.greedy_zuteilung(p, alpha, 32)
                zuletzt, naechster = S.linie_grenzzuwachs(p, alpha, z)
                assert zuletzt >= naechster * (1 - 1e-12)
                w = p * np.exp(-alpha * z)
                bes = z > 0
                assert w[bes].max() / w[bes].min() <= exp(alpha) * (1 + 1e-9)
                k += 1
    assert k == 135


def test_stetig_stehen_die_besuchten_zellen_im_posterior_auf_gleicher_hoehe():
    for inst in range(1, 11):
        p = S.make_prior("zwei_gipfel", 8, inst)
        z, _ = S.wasserstand(p, np.ones(64), 1.0, 20.0)
        w = (p * np.exp(-z))[z > 1e-12]
        assert (w.max() - w.min()) / w.max() < 1e-12


def test_sehr_grosses_alpha_optimum_ist_die_reihenfolge_der_prior_werte():
    p = S.make_prior("ein_gipfel", 8, 5)
    c = np.ones(64)
    for B in (8, 16, 32, 64):
        po = S.wert(p, 50.0, S.greedy_zuteilung(p, 50.0, B))
        assert S.wert(p, 50.0, S.voll_strategie(c, 50.0, B, p)) == pytest.approx(po, abs=1e-9)
        assert S.wert(p, 50.0, S.durchgang_strategie(c, B, p)) == pytest.approx(po, abs=1e-9)
