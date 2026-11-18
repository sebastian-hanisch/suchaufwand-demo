"""Orakel-Tests Ansicht 4 (Drift): Teilchen gegen die Varianzformel, geschlossene Form gegen Raster, Radialintegral und den Literaturwert für alpha = 1, Elastizität gegen den Differenzenquotienten,
zwei weit getrennte Wolken gegen die Formel mit halbem Aufwand."""

from math import exp, log, pi, sqrt

import numpy as np
import pytest

import sua_drift as D
import sua_zufall as Z
from sua_rng import SplitMix64


def test_teilchen_gegen_varianzformel_radius_und_mittelpunkt():
    par = D.PARAMETER_GEMISCHT
    s0, Dif, sU = par
    n, dt = 100_000, 0.25                                       # Euler-Maruyama ist für Drift und Diffusion mit konstanten Koeffizienten exakt
    rng = SplitMix64(2026)
    x = s0 * Z.normals(rng, 2 * n).reshape(n, 2)
    u = np.array([0.5, 0.0]) + sU * Z.normals(rng, 2 * n).reshape(n, 2)
    t = 0.0
    for ziel in (1.0, 2.0, 4.0):
        while t < ziel - 1e-9:
            x += u * dt + sqrt(2 * Dif * dt) * Z.normals(rng, 2 * n).reshape(n, 2)         # Euler-Maruyama, unabhängig von der Formel
            t += dt
        S = D.varianz(ziel, *par)
        assert x.var(axis=0).mean() == pytest.approx(S, rel=0.03)
        r = np.linalg.norm(x - np.array([0.5 * ziel, 0.0]), axis=1)
        assert np.quantile(r, 0.9) == pytest.approx(sqrt(-2 * log(0.1) * S), rel=0.03)
        assert np.abs(x.mean(axis=0) - np.array([0.5 * ziel, 0.0])).max() < 0.05


def test_wolke_aus_einheitsnormalen_hat_die_varianz_S_von_t():
    par = (1.0, 0.5, 0.3)
    einheit = D.einheitsnormale(60_000, 17)
    for t in (1.0, 4.0, 8.0):
        pos = D.wolke(par, t, 0.5, 1, 0.8, einheit)
        assert pos.var(axis=0).mean() == pytest.approx(D.varianz(t, *par), rel=0.03)
    pos2 = D.wolke(par, 4.0, 0.5, 2, 0.8, einheit)
    assert abs(pos2[:, 1][einheit["seite"]].mean() - 0.8 * 4.0) < 0.1 and abs(pos2[:, 1][~einheit["seite"]].mean() + 0.8 * 4.0) < 0.1


@pytest.mark.parametrize("S,E,alpha", [(1.0, 0.5, 1.0), (1.0, 4.0, 2.0), (4.0, 20.0, 0.5), (4.0, 100.0, 1.0)])
def test_geschlossene_form_gleich_raster(S, E, alpha):
    assert D.p_raster(E, alpha, S, 0.0, 24) == pytest.approx(D.p_gauss(E, alpha, S), abs=2e-3)


def test_geschlossene_form_gleich_radialintegral_und_literaturwert():
    integrate = pytest.importorskip("scipy.integrate")
    optimize = pytest.importorskip("scipy.optimize")
    for E, S in ((0.5, 1.0), (3.0, 2.25), (10.0, 4.0)):
        dichte = lambda r: exp(-r * r / (2 * S)) / (2 * pi * S)
        # Wasserstand-Lösung über Radien: abgesucht wird dort, wo die Dichte über dem Niveau lam liegt; der Aufwand ist das Integral von ln(dichte/lam)
        def aufwand(lam):
            r0 = sqrt(2 * S * log(1 / (2 * pi * S * lam)))
            return integrate.quad(lambda r: 2 * pi * r * log(dichte(r) / lam), 0, r0, epsabs=1e-12, epsrel=1e-12)[0], r0

        lam = optimize.brentq(lambda l: aufwand(l)[0] - E, 1e-6, 1 / (2 * pi * S) * (1 - 1e-9), xtol=1e-12)
        r0 = aufwand(lam)[1]
        radial = integrate.quad(lambda r: 2 * pi * r * (dichte(r) - lam), 0, r0, epsabs=1e-12, epsrel=1e-12)[0]
        assert radial == pytest.approx(D.p_gauss(E, 1.0, S), abs=1e-8)
        # Literaturwert alpha = 1 (Kreis-Normalverteilung, exponentielle Entdeckung): P = 1 - (1 + H sqrt(T)) exp(-H sqrt(T)) mit H sqrt(T) = sqrt(E / (pi sigma^2))
        h = sqrt(E / (pi * S))
        assert 1 - (1 + h) * exp(-h) == pytest.approx(D.p_gauss(E, 1.0, S), rel=1e-12)
        assert r0 ** 2 == pytest.approx(2 * S * h, rel=1e-6)                               # Rand des abgesuchten Kreises: r0^2 = 2 S L


def test_elastizitaet_analytisch_gleich_differenzenquotient():
    for par in ((1.0, 0.5, 0.0), (0.2, 0.0, 0.3), (1.0, 0.5, 0.3)):
        for E in (1.0, 10.0, 60.0):
            for t0 in (1.0, 3.0):
                e = 1e-5
                f = lambda t: log(D.p_gauss(E, 1.0, D.varianz(t, *par)))
                num = -(f(t0 * (1 + e)) - f(t0 * (1 - e))) / (log(1 + e) - log(1 - e))
                L0 = D.laenge_L(E, 1.0, D.varianz(t0, *par))
                assert D.elastizitaet_analytisch(L0, par, t0) == pytest.approx(num, abs=1e-6)


def test_zwei_weit_getrennte_wolken_gleich_formel_mit_halbem_aufwand():
    for S in (1.0, 4.0):
        for E in (1.0, 8.0, 40.0):
            assert D.p_raster(E, 1.0, S, 12 * sqrt(S), 24) == pytest.approx(D.p_gauss(E / 2, 1.0, S), abs=2e-3)


def test_verlust_waechst_mit_der_verzoegerung_und_ist_null_ohne_verzoegerung():
    par = D.PARAMETER_GEMISCHT
    v = [D.verlust_eine_wolke(1.0, par, 1.0, d) for d in (0.0, 0.1, 0.25, 0.5, 1.0)]
    assert v[0] == pytest.approx(0.0, abs=1e-15) and all(a < b for a, b in zip(v, v[1:])) and v[-1] < 1
    w = [D.verlust_zwei_hypothesen(1.0, par, 1.0, d, 0.8) for d in (0.0, 0.1, 0.25)]
    assert w[0] == pytest.approx(0.0, abs=1e-12) and w[1] < w[2]


def test_flaeche_90_der_teilchen_waechst_wie_S_bei_einer_wolke():
    par = (1.0, 0.5, 0.3)
    einheit = D.einheitsnormale(40_000, 23)
    fl = {t: D.flaeche_90(D.wolke(par, t, 0.5, 1, 0.8, einheit), D.varianz(t, *par), 0.0) for t in (1.0, 4.0)}
    assert fl[4.0] / fl[1.0] == pytest.approx(D.flaechenverhaeltnis(4.0, par), rel=0.15)     # Zählung auf Zellen, kein Gauß-Maß: Band 15 %
