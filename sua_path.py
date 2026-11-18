"""Ansicht 3: Suche mit Weg. Ein Team läuft auf dem n x n-Gitter (Schritt zu einer Nachbarzelle oder Bleiben) und sucht in **jedem** Zeitschritt
einen Durchgang in seiner aktuellen Zelle; die Anfahrt zählt also als Suchen. Zelle 0 (oben links) ist die Basis. Der Wert eines Weges ist
sum p_i (1 - exp(-alpha z_i)), z_i = Zahl der Zeitschritte in Zelle i.

Die exakten Wege (Wegoptimum und Orienteering-Fassung) rechnet `tools/sua_make_weg.py` mit CP-SAT vor; die App liest sie aus `data/sua_weg.json`.
Herkunft: neu geschrieben für diese Demo, keine Zeile aus einer anderen Demo.
"""

from math import exp

import numpy as np


def nachbarn(n: int) -> list[list[int]]:
    """Je Zelle die erreichbaren Zellen des nächsten Schritts in fester Reihenfolge: Bleiben, oben, unten, links, rechts."""
    nb = []
    for i in range(n * n):
        y, x = divmod(i, n)
        liste = [i]
        if y > 0:
            liste.append(i - n)
        if y < n - 1:
            liste.append(i + n)
        if x > 0:
            liste.append(i - 1)
        if x < n - 1:
            liste.append(i + 1)
        nb.append(liste)
    return nb


def ist_gueltiger_weg(weg, n: int, T: int, start: int = 0) -> bool:
    """T Zellen, beginnend an der Basis, jeder Schritt Bleiben oder zur Nachbarzelle."""
    weg = [int(v) for v in weg]
    if len(weg) != T or not weg or weg[0] != start:
        return False
    nb = nachbarn(n)
    return all(0 <= weg[t] < n * n and weg[t + 1] in nb[weg[t]] for t in range(T - 1))


def durchgaenge(weg, N: int) -> np.ndarray:
    return np.bincount(np.asarray(weg, dtype=int), minlength=N).astype(float)


def weg_wert(p, alpha: float, weg) -> float:
    """Entdeckungswahrscheinlichkeit eines Weges."""
    z = durchgaenge(weg, len(p))
    return float(np.sum(np.asarray(p) * (1.0 - np.exp(-alpha * z))))


def praemien_summe(p, weg) -> float:
    """Orienteering-Wert eines Weges: Prior-Masse der besuchten Zellen, jede nur einmal gezählt."""
    return float(sum(p[i] for i in set(int(v) for v in weg)))


def greedy_kurzsichtig(p, alpha: float, n: int, T: int, start: int = 0) -> list[int]:
    """In jedem Schritt unter Bleiben und den Nachbarzellen die mit dem größten Grenzzuwachs (= aktueller Posterior bis auf einen gemeinsamen Faktor).
    Gleichstand (relativ auf 12 Stellen): die erste in der Reihenfolge Bleiben, oben, unten, links, rechts."""
    nb = nachbarn(n)
    z = np.zeros(n * n)
    weg = [start]
    z[start] += 1
    for _ in range(T - 1):
        kand = nb[weg[-1]]
        g = np.array([p[v] * exp(-alpha * z[v]) for v in kand])
        v = kand[int(np.argmax(np.round(g / max(float(g.max()), 1e-300), 12)))]
        weg.append(v)
        z[v] += 1
    return weg


def greedy_mit_ziel(p, alpha: float, n: int, T: int, start: int = 0) -> list[int]:
    """Greedy mit Ziel, in jedem Schritt neu bestimmt: Für jedes noch erreichbare Ziel die Summe der Grenzzuwächse auf dem besten kürzesten Weg
    (ohne die aktuelle Zelle, mit dem Ziel) durch die Zahl der Schritte teilen; das beste Verhältnis gewinnt (Bleiben zählt mit seinem eigenen
    Grenzzuwachs), der erste Schritt dorthin wird gegangen."""
    N = n * n
    q = 1.0 - exp(-alpha)
    z = np.zeros(N)
    weg = [start]
    z[start] += 1
    for _ in range(T - 1):
        c = weg[-1]
        uebrig = T - len(weg)
        g = np.asarray(p) * np.exp(-alpha * z) * q
        cy, cx = divmod(c, n)
        beste_rate, naechste = float(g[c]), c
        for j in range(N):
            if j == c:
                continue
            jy, jx = divmod(j, n)
            d = abs(jy - cy) + abs(jx - cx)
            if d > uebrig:
                continue
            sy = 1 if jy >= cy else -1
            sx = 1 if jx >= cx else -1
            V: dict[int, float] = {}
            for yy in range(jy, cy - sy, -sy):
                for xx in range(jx, cx - sx, -sx):
                    u = yy * n + xx
                    if u == j:
                        V[u] = float(g[u])
                        continue
                    opt = []
                    if yy != jy:
                        opt.append(V[(yy + sy) * n + xx])
                    if xx != jx:
                        opt.append(V[yy * n + xx + sx])
                    V[u] = float(g[u]) + max(opt)
            kand = []
            if cy != jy:
                kand.append((V[(cy + sy) * n + cx], (cy + sy) * n + cx))
            if cx != jx:
                kand.append((V[cy * n + cx + sx], cy * n + cx + sx))
            wert_pfad, nxt = max(kand, key=lambda a: a[0])
            rate = wert_pfad / d
            if rate > beste_rate + 1e-15:
                beste_rate, naechste = rate, nxt
        weg.append(naechste)
        z[naechste] += 1
    return weg


def _schlange(y0: int, y1: int, x0: int, x1: int, ecke: int, zeilenweise: bool) -> list[tuple[int, int]]:
    ys, xs = list(range(y0, y1)), list(range(x0, x1))
    if ecke in (2, 3):
        ys = ys[::-1]
    if ecke in (1, 3):
        xs = xs[::-1]
    pfad = []
    if zeilenweise:
        for r, yy in enumerate(ys):
            pfad += [(yy, xx) for xx in (xs if r % 2 == 0 else xs[::-1])]
    else:
        for r, xx in enumerate(xs):
            pfad += [(yy, xx) for yy in (ys if r % 2 == 0 else ys[::-1])]
    return pfad


def _hin_und_her(pfad: list, T: int) -> list:
    """Den Pfad hin, dann rückwärts, dann wieder vor (ohne die Endzelle doppelt), bis T Zellen."""
    if len(pfad) == 1:
        return (pfad * T)[:T]
    seq = list(pfad)
    rueck, vor, hin = pfad[-2::-1], pfad[1:], True
    while len(seq) < T:
        seq += rueck if hin else vor
        hin = not hin
    return seq[:T]


def maehen_ganz(n: int, T: int) -> list[int]:
    """Schlangenlinie über das ganze Gebiet ab der Ecke (zeilenweise), nach dem Ende hin und her."""
    return [y * n + x for y, x in _hin_und_her(_schlange(0, n, 0, n, 0, True), T)]


def maehen_im_rechteck(p, alpha: float, n: int, T: int) -> list[int]:
    """Das beste Rechteck, gemäht in Schlangenlinie (vier Startecken, zeilen- oder spaltenweise), mit Anfahrt von der Basis auf zwei Wegen;
    alle Schritte zählen, auch die Anfahrt. Bestes aller Rechtecke nach dem wahren Wert; Gleichstand: das zuerst gefundene."""
    p = np.asarray(p)
    beste, bester = -1.0, [0]
    for y0 in range(n):
        for y1 in range(y0 + 1, n + 1):
            for x0 in range(n):
                for x1 in range(x0 + 1, n + 1):
                    for ecke in range(4):
                        ey = y0 if ecke in (0, 1) else y1 - 1
                        ex = x0 if ecke in (0, 2) else x1 - 1
                        for zeilenweise in (True, False):
                            pfad = _schlange(y0, y1, x0, x1, ecke, zeilenweise)
                            for anfahrt in (0, 1):
                                if anfahrt == 0:
                                    a = [(yy, 0) for yy in range(ey + 1)] + [(ey, xx) for xx in range(1, ex + 1)]
                                else:
                                    a = [(0, xx) for xx in range(ex + 1)] + [(yy, ex) for yy in range(1, ey + 1)]
                                a = a[:-1]
                                seq = (a + pfad)[:T] if len(a) + len(pfad) >= T else a + _hin_und_her(pfad, T - len(a))
                                idx = [yy * n + xx for yy, xx in seq]
                                v = weg_wert(p, alpha, idx)
                                if v > beste:
                                    beste, bester = v, idx
    return bester
