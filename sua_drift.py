"""Ansicht 4: Drift und Verzögerung (Modell erfunden, einheitenlos).

x(t) = x0 + U t + sqrt(2 D) W(t) je Achse mit x0 ~ N(0, s0^2), U ~ N(Ū, sU^2) je Teilchen konstant und W Brownsche Bewegung. Dann ist x(t)
normalverteilt mit der Varianz je Achse S(t) = s0^2 + 2 D t + sU^2 t^2. Die Fläche eines Gebiets mit festem Anteil der Wahrscheinlichkeit ist
proportional zu S(t). Bei optimal verteiltem, **teilbarem** Aufwand E, Entdeckung 1 - exp(-alpha * Aufwandsdichte) und einer Gauß-Wolke gilt
P = 1 - (1 + L) exp(-L) mit L = sqrt(alpha E / (pi S)).

Herkunft: neu geschrieben für diese Demo; die geschlossene Form ist hergeleitet und im Test gegen ein Raster und ein Radialintegral geprüft.
"""

from math import log, pi, sqrt

import numpy as np

import sua_zufall as Z
from sua_rng import SplitMix64
from sua_search import wasserstand

PARAMETER_GEMISCHT = (1.0, 0.5, 0.3)  # s0, D, sU des Standardbeispiels
FAKTOREN_VERZOEGERUNG = (0.10, 0.25, 0.50, 1.00)


def varianz(t: float, s0: float, D: float, sU: float) -> float:
    """Varianz je Achse S(t) = s0^2 + 2 D t + sU^2 t^2."""
    return s0 ** 2 + 2.0 * D * t + (sU * t) ** 2


def flaechenverhaeltnis(t: float, par: tuple, t_bezug: float = 1.0) -> float:
    """Fläche des Gebiets mit festem Wahrscheinlichkeitsanteil zur Zeit t gegen die zur Zeit t_bezug (= S(t) / S(t_bezug))."""
    return varianz(t, *par) / varianz(t_bezug, *par)


def flaechenexponent(par: tuple, t1: float, t2: float) -> float:
    """Exponent e mit Fläche ~ t^e zwischen t1 und t2: log(S(t2)/S(t1)) / log(t2/t1)."""
    return log(varianz(t2, *par) / varianz(t1, *par)) / log(t2 / t1)


def laenge_L(E: float, alpha: float, S: float) -> float:
    return sqrt(alpha * E / (pi * S))


def p_gauss(E: float, alpha: float, S: float) -> float:
    """Optimale Entdeckungswahrscheinlichkeit bei Gauß-Wolke (Varianz S je Achse) und teilbarem Aufwand E: 1 - (1 + L) exp(-L)."""
    L = laenge_L(E, alpha, S)
    return 1.0 - (1.0 + L) * float(np.exp(-L))


def aufwand_fuer_L0(L0: float, S0: float, alpha: float = 1.0) -> float:
    """Aufwand E, bei dem L zum Bezugszeitpunkt (Varianz S0) den Wert L0 hat: E = L0^2 pi S0 / alpha."""
    return L0 ** 2 * pi * S0 / alpha


def verlust_eine_wolke(L0: float, par: tuple, t0: float, verzoegerung: float, alpha: float = 1.0) -> float:
    """Relativer Verlust von P, wenn die Suche um verzoegerung * t0 später beginnt: 1 - P(t0 + d) / P(t0) bei festem Aufwand E."""
    E = aufwand_fuer_L0(L0, varianz(t0, *par), alpha)
    p0 = p_gauss(E, alpha, varianz(t0, *par))
    return 1.0 - p_gauss(E, alpha, varianz(t0 * (1.0 + verzoegerung), *par)) / p0


def elastizitaet_sekante(p0: float, p1: float, faktor: float) -> float:
    """Elastizität des Verlusts über die Verzögerung: -ln(P1/P0) / ln(faktor) mit faktor = (t0 + d)/t0."""
    return -log(p1 / p0) / log(faktor)


def elastizitaet_analytisch(L0: float, par: tuple, t0: float, alpha: float = 1.0) -> float:
    """Elastizität -d ln P / d ln t im Punkt t0 (Ableitung): L^2 exp(-L) / P * 1/2 * d ln S / d ln t."""
    s0_, D, sU = par
    S = varianz(t0, *par)
    L = L0
    p = 1.0 - (1.0 + L) * float(np.exp(-L))
    dlns = (2.0 * D * t0 + 2.0 * (sU * t0) ** 2) / S
    return L * L * float(np.exp(-L)) / p * 0.5 * dlns


def p_raster(E: float, alpha: float, S: float, abstand: float, feinheit: int = 24) -> float:
    """Optimale Entdeckungswahrscheinlichkeit für zwei gleich schwere Gauß-Wolken (Varianz S je Achse, Mitten im Abstand `abstand` auf der x-Achse)
    bei teilbarem Aufwand E, numerisch auf einem Raster der Zellbreite sigma/feinheit (Wasserstand-Lösung). abstand = 0 ist eine Wolke."""
    sig = sqrt(S)
    h = sig / feinheit
    rx = abstand / 2.0 + 7.0 * sig
    ry = 7.0 * sig
    gx = (np.arange(int(2 * rx / h)) + 0.5) * h - rx
    gy = (np.arange(int(2 * ry / h)) + 0.5) * h - ry
    X, Y = np.meshgrid(gx, gy)

    def dichte(cx: float) -> np.ndarray:
        return np.exp(-((X - cx) ** 2 + Y ** 2) / (2.0 * S)) / (2.0 * pi * S)

    dens = 0.5 * dichte(-abstand / 2.0) + 0.5 * dichte(abstand / 2.0)
    pc = (dens * h * h).ravel()
    pc = pc / pc.sum()
    z, _ = wasserstand(pc, np.ones_like(pc), alpha / (h * h), E)
    return float(np.sum(pc * (1.0 - np.exp(-alpha / (h * h) * z))))


def verlust_zwei_hypothesen(L0: float, par: tuple, t0: float, verzoegerung: float, dv: float, alpha: float = 1.0, feinheit: int = 24) -> float:
    """Wie `verlust_eine_wolke`, aber mit zwei Hypothesen (Wolken gleichen Gewichts, Mitten im Abstand 2 dv t) und Raster-Lösung.
    Der Aufwand E ist derselbe wie bei einer Wolke mit dem Wert L0 zur Zeit t0."""
    E = aufwand_fuer_L0(L0, varianz(t0, *par), alpha)
    p0 = p_raster(E, alpha, varianz(t0, *par), 2.0 * dv * t0, feinheit)
    t1 = t0 * (1.0 + verzoegerung)
    return 1.0 - p_raster(E, alpha, varianz(t1, *par), 2.0 * dv * t1, feinheit) / p0


# ------------------------------------------------------------------------------------------------ Wolke aus Teilchen
def einheitsnormale(n_teilchen: int, seed: int) -> dict:
    """Feste Standardnormalzahlen je Teilchen (Start, Geschwindigkeit, Diffusion, Seite), damit sich die Wolke mit der Zeit stetig bewegt."""
    rng = SplitMix64(int(seed))
    return {
        "x0": Z.normals(rng, 2 * n_teilchen).reshape(n_teilchen, 2),
        "u": Z.normals(rng, 2 * n_teilchen).reshape(n_teilchen, 2),
        "w": Z.normals(rng, 2 * n_teilchen).reshape(n_teilchen, 2),
        "seite": Z.uniform(rng, n_teilchen) < 0.5,
    }


def wolke(par: tuple, t: float, u_mittel: float, hypothesen: int, dv: float, einheit: dict) -> np.ndarray:
    """Teilchenpositionen zur Zeit t (Feld n x 2). Zwei Hypothesen: Geschwindigkeit quer zur Windrichtung (y) um +dv bzw. -dv verschoben, je die Hälfte."""
    s0, D, sU = par
    n = len(einheit["x0"])
    mittel = np.zeros((n, 2))
    mittel[:, 0] = u_mittel
    if hypothesen == 2:
        mittel[:, 1] = np.where(einheit["seite"], dv, -dv)
    return s0 * einheit["x0"] + (mittel + sU * einheit["u"]) * t + sqrt(2.0 * D * t) * einheit["w"]


def flaeche_90(pos: np.ndarray, S_ref: float, abstand: float) -> float:
    """Fläche des kleinsten Zellgebiets mit 90 % der Teilchen. Zellbreite 0,15 * sqrt(S + (Abstand/2)^2): gleiche relative Auflösung zu jeder Zeit
    (Zählung auf Zellen, keine Gauß-Annahme)."""
    h = 0.15 * sqrt(S_ref + (abstand / 2.0) ** 2)
    ix = np.floor((pos[:, 0] - pos[:, 0].min()) / h).astype(int)
    iy = np.floor((pos[:, 1] - pos[:, 1].min()) / h).astype(int)
    zaehl = np.bincount(ix * (iy.max() + 1) + iy)
    kum = np.sort(zaehl)[::-1].cumsum()
    k = int(np.searchsorted(kum, 0.9 * len(pos))) + 1
    return k * h * h


def szenario_exponenten(nr: int, n_teilchen: int = 40000) -> tuple[float, float]:
    """Ein erfundenes Zwei-Hypothesen-Szenario Nr. nr (Zufalls-Seed nr): Flächenexponent zwischen t = 1 und 4 (je Zeitverdopplung) und zwischen 4 und 8.
    Parameter zufällig: s0 in [0,2; 1], D in [0; 0,5], sU in [0,1; 0,35], Windgeschwindigkeit in [0,5; 1], Hypothesenabstand dv in [0,3; 0,8]."""
    rng = SplitMix64(7000 + nr)
    s0 = 0.2 + 0.8 * rng.random()
    D = 0.5 * rng.random()
    sU = 0.1 + 0.25 * rng.random()
    u_mittel = 0.5 + 0.5 * rng.random()
    dv = 0.3 + 0.5 * rng.random()
    par = (s0, D, sU)
    einheit = einheitsnormale(n_teilchen, 9000 + nr)
    fl = {}
    for t in (1.0, 4.0, 8.0):
        pos = wolke(par, t, u_mittel, 2, dv, einheit)
        fl[t] = flaeche_90(pos, varianz(t, *par), dv * t * 2.0)
    return float(np.log2(fl[4.0] / fl[1.0]) / 2.0), float(np.log2(fl[8.0] / fl[4.0]))
