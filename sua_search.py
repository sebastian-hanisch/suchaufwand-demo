"""Kern der Ansichten 1 und 2: Prior, Gelände, Zuteilung ganzer Durchgänge, Posterior, Nichtfund, Rechteck-Muster.

Modell (alle Instanzen erfunden): ruhendes Ziel in einer von N Zellen eines n x n-Gitters, Prior p (Summe 1). Ein Durchgang in Zelle i
kostet c_i Aufwandseinheiten (ohne Gelände 1) und entdeckt das Ziel, falls es dort ist, mit q = 1 - exp(-alpha). Bei z_i Durchgängen
entdeckt die Suche mit 1 - exp(-alpha * z_i) (Zufallssuche). P = sum p_i (1 - exp(-alpha z_i)) ist die Entdeckungswahrscheinlichkeit.

Herkunft: neu geschrieben für diese Demo nach den Formeln der Aufgabenbeschreibung; keine Zeile aus einer anderen Demo.
"""

from math import ceil, cos, exp, log, pi, sin

import numpy as np

import sua_zufall as Z
from sua_rng import SplitMix64

PRIORS = ("ein_gipfel", "zwei_gipfel", "flach")
_PRIOR_ID = {"ein_gipfel": 1, "zwei_gipfel": 2, "flach": 3}
TEILCHEN_PRIOR = 4000  # Driftteilchen, aus denen der Prior „zwei Gipfel“ gezählt wird


def instanz_rng(art: str, n: int, seed: int) -> SplitMix64:
    """Zufallsstrom einer Instanz: der Zufalls-Seed der Demo, die Art des Priors und die Gittergröße bestimmen ihn eindeutig."""
    return SplitMix64(int(seed) * 1_000_003 + 97 * _PRIOR_ID[art] + n)


# ------------------------------------------------------------------------------------------------ Prior und Gelände
def _prior_ein_gipfel(n: int, rng: SplitMix64) -> np.ndarray:
    """Gauß-Kuppe: Mitte und Breiten aus dem Zufalls-Seed."""
    y, x = np.mgrid[0:n, 0:n]
    cy, cx = (n - 1) * rng.random(), (n - 1) * rng.random()
    sy, sx = (0.10 + 0.15 * rng.random()) * n, (0.10 + 0.15 * rng.random()) * n
    g = np.exp(-0.5 * (((y - cy) / sy) ** 2 + ((x - cx) / sx) ** 2))
    return (g / g.sum()).ravel()


def _prior_zwei_gipfel(n: int, rng: SplitMix64) -> np.ndarray:
    """Zwei Gipfel aus Driftteilchen: zwei Driftrichtungen (links und rechts vom Wind), die Teilchen werden auf das Gitter gezählt."""
    ziel = np.array([0.15 + 0.7 * rng.random(), 0.15 + 0.7 * rng.random()]) * n
    ang = 2 * pi * rng.random()
    d = (0.25 + 0.15 * rng.random()) * n
    w = 0.35 + 0.30 * rng.random()
    mitten = np.array([[ziel[0] + d * sin(ang), ziel[1] + d * cos(ang)], [ziel[0] - d * sin(ang), ziel[1] - d * cos(ang)]])
    s = (0.06 + 0.06 * rng.random()) * n
    seite = Z.uniform(rng, TEILCHEN_PRIOR) < w
    pos = np.where(seite[:, None], mitten[0], mitten[1]) + s * Z.normals(rng, 2 * TEILCHEN_PRIOR).reshape(TEILCHEN_PRIOR, 2)
    iy = np.clip(np.floor(pos[:, 0]).astype(int), 0, n - 1)
    ix = np.clip(np.floor(pos[:, 1]).astype(int), 0, n - 1)
    h = np.bincount(iy * n + ix, minlength=n * n).astype(float)
    return h / h.sum()


def make_prior(art: str, n: int, seed: int) -> np.ndarray:
    """Prior als Feld der Länge n*n (Summe 1), Zellen zeilenweise."""
    if art == "flach":
        return np.full(n * n, 1.0 / (n * n))
    rng = instanz_rng(art, n, seed)
    return _prior_ein_gipfel(n, rng) if art == "ein_gipfel" else _prior_zwei_gipfel(n, rng)


def _glatt(a: np.ndarray, sigma: float) -> np.ndarray:
    """Gauß-Glättung, getrennt nach Achsen, gespiegelter Rand."""
    radius = int(4.0 * sigma + 0.5)
    t = np.arange(-radius, radius + 1)
    k = np.exp(-0.5 * (t / sigma) ** 2)
    k /= k.sum()
    for achse in (0, 1):
        pad = [(0, 0), (0, 0)]
        pad[achse] = (radius, radius)
        b = np.pad(a, pad, mode="symmetric")
        a = np.apply_along_axis(lambda v: np.convolve(v, k, mode="valid"), achse, b)
    return a


def make_costs(n: int, gamma: float, seed: int) -> np.ndarray:
    """Ganzzahlige Aufwandskosten je Durchgang (Gelände): glattes Zufallsfeld, Kosten etwa 2 im Mittel, mindestens 1; gamma = 0 gibt überall 2."""
    if gamma == 0:
        return np.full(n * n, 2, dtype=int)
    rng = SplitMix64(int(seed) * 1_000_003 + 7919 + n)
    g = _glatt(Z.normals(rng, n * n).reshape(n, n), n / 8.0)
    g = (g - g.mean()) / g.std()
    c = np.exp(gamma * g)
    c = c / c.mean()
    return np.maximum(1, np.round(2 * c)).astype(int).ravel()


# ------------------------------------------------------------------------------------------------ Bewertung
def wert(p, alpha: float, z) -> float:
    """Entdeckungswahrscheinlichkeit P bei z_i Durchgängen je Zelle."""
    return float(np.sum(np.asarray(p) * (1.0 - np.exp(-alpha * np.asarray(z, dtype=float)))))


def luecke_punkte(optimum: float, verfahren: float) -> float:
    """Abstand in Prozentpunkten Entdeckungswahrscheinlichkeit (nie Prozent des Optimums)."""
    return 100.0 * (optimum - verfahren)


def nie_abgesucht(p, z) -> tuple[float, float]:
    """(Anteil der Zellen ohne Durchgang, Prior-Masse in diesen Zellen)."""
    z = np.asarray(z)
    leer = z == 0
    return float(leer.mean()), float(np.asarray(p)[leer].sum())


def grenzzuwachs_naechster(p, alpha: float, z) -> np.ndarray:
    """Zuwachs von P durch den nächsten Durchgang in jeder Zelle: p * exp(-alpha z) * (1 - exp(-alpha))."""
    return np.asarray(p) * np.exp(-alpha * np.asarray(z, dtype=float)) * (1.0 - exp(-alpha))


def posterior(p, alpha: float, z) -> np.ndarray:
    """Aufenthaltswahrscheinlichkeit nach Nichtfund bei z_i Durchgängen (Bayes-Regel); ohne Rest-Masse bleibt der Prior."""
    w = np.asarray(p) * np.exp(-alpha * np.asarray(z, dtype=float))
    s = float(w.sum())
    return w / s if s > 0 else np.asarray(p).copy()


# ------------------------------------------------------------------------------------------------ Zuteilung ganzer Durchgänge
def greedy_folge(p, alpha: float, B: int) -> list[int]:
    """Der nächste Durchgang geht in die Zelle mit dem größten Grenzzuwachs (gleicher Preis je Zelle), B Durchgänge in Reihenfolge.

    Bei gleichem Preis ist der größte Grenzzuwachs dasselbe wie der größte aktuelle Posterior. Gleichstand: kleinster Index
    (Grenzzuwachs relativ zum größten auf 12 Stellen gerundet, damit die letzte Binärstelle nichts entscheidet)."""
    g = np.asarray(p, dtype=float) * (1.0 - exp(-alpha))
    folge: list[int] = []
    f = exp(-alpha)
    for _ in range(int(B)):
        gmax = float(g.max())
        if gmax <= 0:
            break
        i = int(np.argmax(np.round(g / gmax, 12)))
        folge.append(i)
        g[i] *= f
    return folge


def greedy_zuteilung(p, alpha: float, B: int) -> np.ndarray:
    """Optimale ganzzahlige Zuteilung bei gleichem Preis (durch Aufzählung belegt, siehe tests/test_oracle_suchaufwand.py)."""
    z = np.zeros(len(p))
    for i in greedy_folge(p, alpha, B):
        z[i] += 1.0
    return z


def greedy_verhaeltnis(p, c, alpha: float, B: float) -> np.ndarray:
    """Aktualisierender Greedy nach Zuwachs je Kosten, solange ein Durchgang ins Budget passt (nur eine Näherung bei ungleichen Kosten)."""
    p, c = np.asarray(p, dtype=float), np.asarray(c, dtype=float)
    z = np.zeros(len(p))
    rest = float(B)
    while True:
        m = p * np.exp(-alpha * z) * (1.0 - exp(-alpha)) / c
        m[c > rest + 1e-12] = -1.0
        mx = float(m.max())
        if mx <= 0:
            break
        j = int(np.argmax(np.round(m / mx, 12)))
        z[j] += 1.0
        rest -= c[j]
    return z


def voll_strategie(c, alpha: float, B: float, key, tau: float = 0.95) -> np.ndarray:
    """Zellen in der Reihenfolge von `key` je „vollständig“ absuchen (so viele Durchgänge, dass die Entdeckung in der Zelle mindestens tau erreicht),
    dann die nächste Zelle; ist die Liste durch, geht es von vorn los, solange ein Durchgang ins Budget passt."""
    c = np.asarray(c, dtype=float)
    mv = max(1, int(ceil(-log(1.0 - tau) / alpha - 1e-12)))
    order = np.argsort(-np.asarray(key), kind="stable")
    z = np.zeros(len(c))
    rest = float(B)
    while True:
        gekauft = False
        for i in order:
            for _ in range(mv):
                if c[i] <= rest + 1e-12:
                    z[i] += 1.0
                    rest -= c[i]
                    gekauft = True
                else:
                    break
        if not gekauft:
            return z


def durchgang_strategie(c, B: float, key) -> np.ndarray:
    """Ein Durchgang je Zelle in der Reihenfolge von `key`, reihum, solange ein Durchgang ins Budget passt."""
    c = np.asarray(c, dtype=float)
    order = np.argsort(-np.asarray(key), kind="stable")
    z = np.zeros(len(c))
    rest = float(B)
    while True:
        gekauft = False
        for i in order:
            if c[i] <= rest + 1e-12:
                z[i] += 1.0
                rest -= c[i]
                gekauft = True
        if not gekauft:
            return z


_RECHTECKE: dict[int, np.ndarray] = {}


def rechtecke(n: int) -> np.ndarray:
    """Alle Rechtecke eines n x n-Gitters als Zeilen (y0, y1, x0, x1), Ende ausgeschlossen."""
    if n not in _RECHTECKE:
        _RECHTECKE[n] = np.array([(y0, y1, x0, x1) for y0 in range(n) for y1 in range(y0 + 1, n + 1) for x0 in range(n) for x1 in range(x0 + 1, n + 1)])
    return _RECHTECKE[n]


def _summen(feld: np.ndarray, n: int) -> np.ndarray:
    s = np.zeros((n + 1, n + 1))
    s[1:, 1:] = feld.reshape(n, n).cumsum(0).cumsum(1)
    return s


def _rechteck_summe(s: np.ndarray, r: np.ndarray) -> np.ndarray:
    y0, y1, x0, x1 = r.T
    return s[y1, x1] - s[y0, x1] - s[y1, x0] + s[y0, x0]


def rechteck_strategie(p, c, alpha: float, B: float, n: int, vorauswahl: int = 6) -> np.ndarray:
    """Gleichmäßig viele Durchgänge in einem Rechteck (Art des Planers in SAROPS): das beste Rechteck, Rest auf die Zellen mit größtem Prior.

    Vorauswahl der `vorauswahl` besten Rechtecke nach dem Wert bei teilbarem Aufwand im Rechteck, danach der wahre Wert ganzer Durchgänge."""
    p, c = np.asarray(p, dtype=float), np.asarray(c, dtype=float)
    r = rechtecke(n)
    pr = _rechteck_summe(_summen(p, n), r)
    cr = _rechteck_summe(_summen(c, n), r)
    v = pr * (1.0 - np.exp(-alpha * B / cr))
    beste_wert, beste_z = -1.0, np.zeros(n * n)
    P, C = p.reshape(n, n), c.reshape(n, n)
    for k in np.argsort(-v, kind="stable")[:vorauswahl]:
        y0, y1, x0, x1 = r[k]
        sub_c = C[y0:y1, x0:x1].ravel()
        sub_p = P[y0:y1, x0:x1].ravel()
        m = int(np.floor(B / sub_c.sum() + 1e-12))
        sz = np.full(len(sub_c), float(m))
        rest = B - m * sub_c.sum()
        for i in np.argsort(-sub_p, kind="stable"):
            if sub_c[i] <= rest + 1e-12:
                sz[i] += 1.0
                rest -= sub_c[i]
        zz = np.zeros((n, n))
        zz[y0:y1, x0:x1] = sz.reshape(y1 - y0, x1 - x0)
        zz = zz.ravel()
        wv = wert(p, alpha, zz)
        if wv > beste_wert:
            beste_wert, beste_z = wv, zz
    return beste_z


def obergrenze_teilbar(p, c, alpha: float, B: float) -> float:
    """Entdeckungswahrscheinlichkeit des Optimums bei **teilbarem** Aufwand (Wasserstand-Lösung): nur eine Obergrenze für ganze Durchgänge."""
    z, _ = wasserstand(p, c, alpha, B)
    return wert(p, alpha, z)


def wasserstand(p, c, alpha: float, B: float):
    """Optimum von sum p (1 - exp(-alpha z)) mit sum c z = B und z >= 0 bei teilbarem Aufwand (Bedingungen von Lagrange und Kuhn-Tucker, geschlossen).

    Zurück: (z, Niveau). Alle besuchten Zellen haben denselben Grenzwert p*alpha*exp(-alpha z)/c, das Niveau."""
    p, c = np.asarray(p, dtype=float), np.asarray(c, dtype=float)
    z = np.zeros_like(p)
    idx = np.where(p > 0)[0]
    if len(idx) == 0 or B <= 0:
        return z, float("inf")
    lr = np.log(p[idx] * alpha / c[idx])
    o = np.argsort(-lr, kind="stable")
    lr_s, c_s = lr[o], c[idx][o]
    cs = np.cumsum(c_s)
    csl = np.cumsum(c_s * lr_s)
    ln_lam = (csl - alpha * B) / cs
    gueltig = np.where(lr_s >= ln_lam - 1e-12)[0]
    ll = ln_lam[gueltig.max()]
    z[idx] = np.maximum(0.0, (lr - ll) / alpha)
    return z, float(np.exp(ll))


def rucksack_dp(p, c_int, alpha: float, B: int) -> float:
    """Exaktes Optimum bei ganzzahligen Kosten (Rucksack-Rechnung über Zellen und Durchgänge), nur der Wert.

    Je Zelle bis zu so vielen Durchgängen, dass der Grenzzuwachs unter 1e-12 fällt (bei alpha = 0,5 sind das 56)."""
    p = np.asarray(p, dtype=float)
    B = int(B)
    best = np.zeros(B + 1)
    kmax_alpha = int(ceil(27.7 / alpha)) + 1
    for i in range(len(p)):
        ci = int(c_int[i])
        if p[i] <= 0 or ci > B:
            continue
        neu = best.copy()
        for k in range(1, min(kmax_alpha, B // ci) + 1):
            sh = ci * k
            neu[sh:] = np.maximum(neu[sh:], best[: B + 1 - sh] + p[i] * (1.0 - exp(-alpha * k)))
        best = neu
    return float(best[B])


# ------------------------------------------------------------------------------------------------ Nichtfund (Ansicht 2)
def nichtfund_runden(p, alpha: float, B1: int, B2: int) -> dict:
    """Zwei Runden: Runde 1 optimal auf dem Prior, Nichtfund, Runde 2 a) auf der aktualisierten Karte, b) auf der alten Karte (die Zuteilung der Runde 1
    wird mit dem Budget der Runde 2 auf dem Prior wiederholt), c) „dort weiter bis voll“: die Zellen mit dem größten Posterior je bis 95 % absuchen.
    Alle Gesamtwerte mit der wahren Entdeckungswahrscheinlichkeit des Gesamtplans (Zuteilung ganzer Durchgänge, gleicher Preis)."""
    p = np.asarray(p, dtype=float)
    z1 = greedy_zuteilung(p, alpha, B1)
    d1 = wert(p, alpha, z1)
    post = posterior(p, alpha, z1)
    z2a = greedy_zuteilung(post, alpha, B2)
    gesamt_a = d1 + (1.0 - d1) * wert(post, alpha, z2a)
    einplan = wert(p, alpha, greedy_zuteilung(p, alpha, B1 + B2))
    z2b = greedy_zuteilung(p, alpha, B2)
    gesamt_b = wert(p, alpha, z1 + z2b)
    mv = int(ceil(-log(0.05) / alpha))
    z2c = np.zeros(len(p))
    rest = B2
    for i in np.argsort(-post, kind="stable"):
        k = min(mv, rest)
        z2c[i] = k
        rest -= k
        if rest <= 0:
            break
    gesamt_c = wert(p, alpha, z1 + z2c)
    return {"z1": z1, "z2a": z2a, "z2b": z2b, "z2c": z2c, "d1": d1, "posterior": post, "gesamt_a": gesamt_a, "gesamt_b": gesamt_b,
            "gesamt_c": gesamt_c, "einplan": einplan}


def falsches_alpha(p, alpha_wahr: float, alpha_ang: float, B: int) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Zuteilung mit dem angenommenen alpha gegen die mit dem richtigen; beide mit dem wahren alpha bewertet.
    Zurück: (z richtig, z angenommen, P richtig, P angenommen)."""
    z_r = greedy_zuteilung(p, alpha_wahr, B)
    z_a = greedy_zuteilung(p, alpha_ang, B)
    return z_r, z_a, wert(p, alpha_wahr, z_r), wert(p, alpha_wahr, z_a)


def linie_grenzzuwachs(p, alpha: float, z) -> tuple[float, float]:
    """(kleinster Grenzzuwachs eines genommenen Durchgangs, größter Grenzzuwachs des nächsten Durchgangs) nach der Zuteilung z.

    Bei der optimalen ganzzahligen Zuteilung ist der erste Wert nicht kleiner als der zweite (exakt); der Posterior der besuchten Zellen
    liegt dagegen nur in einem Band vom Faktor exp(alpha). Ohne Durchgang gibt der erste Wert nan zurück."""
    z = np.asarray(z, dtype=float)
    p = np.asarray(p, dtype=float)
    q = 1.0 - exp(-alpha)
    naechster = grenzzuwachs_naechster(p, alpha, z)
    besucht = z > 0
    zuletzt = p[besucht] * np.exp(-alpha * (z[besucht] - 1.0)) * q
    return (float(zuletzt.min()) if besucht.any() else float("nan")), float(naechster.max())


# ------------------------------------------------------------------------------------------------ Rechteck-Muster über Runden
def plan_rechteck(w, n: int, r, B: int) -> np.ndarray:
    """B // Fläche Durchgänge je Zelle des Rechtecks r, der Rest einzeln auf die Zellen mit dem größten Gewicht."""
    y0, y1, x0, x1 = (int(v) for v in r)
    zz = np.zeros((n, n))
    flaeche = (y1 - y0) * (x1 - x0)
    m = B // flaeche
    rest = B - m * flaeche
    sub = np.full(flaeche, float(m))
    gew = np.asarray(w).reshape(n, n)[y0:y1, x0:x1].ravel()
    for i in np.argsort(-gew, kind="stable")[:rest]:
        sub[i] += 1.0
    zz[y0:y1, x0:x1] = sub.reshape(y1 - y0, x1 - x0)
    return zz.ravel()


def rechteck_kandidaten(w, alpha: float, n: int, B: int, top: int) -> list[tuple]:
    """Rechtecke mit Fläche höchstens B (das Rechteck wird ganz abgedeckt), die besten nach dem Wert bei teilbarem Aufwand im Rechteck."""
    r = rechtecke(n)
    pr = _rechteck_summe(_summen(np.asarray(w, dtype=float), n), r)
    flaeche = (r[:, 1] - r[:, 0]) * (r[:, 3] - r[:, 2])
    v = pr * (1.0 - np.exp(-alpha * B / flaeche))
    v[flaeche > B] = -1.0
    return [tuple(r[i]) for i in np.argsort(-v, kind="stable")[:top] if v[i] >= 0]


def bestes_rechteck_muster(w, alpha: float, n: int, B: int, vorauswahl: int = 6) -> tuple[float, np.ndarray]:
    """Bestes Rechteck-Muster (ganz abgedeckt) für die Gewichte w: unter den `vorauswahl` Kandidaten der mit dem höchsten Wert."""
    beste = (-1.0, np.zeros(n * n))
    for r in rechteck_kandidaten(w, alpha, n, B, vorauswahl):
        z = plan_rechteck(w, n, r, B)
        v = wert(w, alpha, z)
        if v > beste[0]:
            beste = (v, z)
    return beste


def rechteck_runden(p, alpha: float, n: int, B: int, vorausschau: int = 40) -> dict:
    """Zwei Runden zu je B/2 Durchgängen, in jeder Runde ein ganz abgedecktes Rechteck.
    b) kurzsichtig: je Runde das beste Rechteck auf der aktualisierten Karte; c) mit Vorausschau: unter den `vorausschau` besten Rechtecken der Runde 1
    jeweils das beste der Runde 2, bestes Paar (untere Schranke für das beste Rechteckpaar); d) ein Rechteck mit dem ganzen Budget.
    Alle Werte: wahre Entdeckungswahrscheinlichkeit, dazu das frei verteilte Optimum (Zuteilung ganzer Durchgänge mit B)."""
    p = np.asarray(p, dtype=float)
    B1, B2 = B // 2, B - B // 2
    _, z1 = bestes_rechteck_muster(p, alpha, n, B1)
    _, z2 = bestes_rechteck_muster(p * np.exp(-alpha * z1), alpha, n, B2)
    pb = wert(p, alpha, z1 + z2)
    pc = pb
    for r1 in rechteck_kandidaten(p, alpha, n, B1, vorausschau):
        zz1 = plan_rechteck(p, n, r1, B1)
        _, zz2 = bestes_rechteck_muster(p * np.exp(-alpha * zz1), alpha, n, B2)
        pc = max(pc, wert(p, alpha, zz1 + zz2))
    _, zd = bestes_rechteck_muster(p, alpha, n, B)
    return {"optimum": wert(p, alpha, greedy_zuteilung(p, alpha, B)), "kurzsichtig": pb, "vorausschau": pc, "ein_rechteck": wert(p, alpha, zd)}
