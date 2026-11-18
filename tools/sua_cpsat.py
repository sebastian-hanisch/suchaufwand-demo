"""CP-SAT-Modelle für die Suche mit Weg (nur Werkzeug und Tests; OR-Tools steht deshalb nur in requirements-dev.txt).

Das Team steht in jedem der T Zeitschritte in genau einer Zelle (Schritt 0 an der Basis), zwischen zwei Schritten bleibt es oder geht in eine
Nachbarzelle. Der Wert einer Zelle mit z Besuchen ist p (1 - exp(-alpha z)); CP-SAT rechnet mit ganzzahlig skalierten Grenzzuwächsen
p (1 - exp(-alpha)) exp(-alpha k) für den k-ten Besuch (Skala 1e9), die Zahlen werden danach aus dem Weg neu berechnet.
Zwei Fassungen: `exakt_weg` maximiert den wahren Wert; `orienteering_weg` löst erst das Orienteering-Problem (Prior-Masse der besuchten Zellen,
jede nur einmal gezählt) und wählt unter allen Wegen mit diesem Optimum den mit dem höchsten wahren Wert (günstigster Fall für die Orienteering-Fassung).
"""

import sys
from math import exp
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ortools.sat.python import cp_model  # noqa: E402

from sua_path import nachbarn, praemien_summe, weg_wert  # noqa: E402

SKALA = 10 ** 9


def _grundmodell(n: int, T: int, start: int = 0):
    N = n * n
    nb = nachbarn(n)
    m = cp_model.CpModel()
    x = [[m.NewBoolVar(f"x{t}_{i}") for i in range(N)] for t in range(T)]
    for t in range(T):
        m.AddExactlyOne(x[t])
    m.Add(x[0][start] == 1)
    for t in range(T - 1):
        for j in range(N):
            m.Add(x[t + 1][j] <= sum(x[t][i] for i in nb[j]))
    return m, x


def _wahrer_wert_ziel(m, x, p, alpha: float, T: int) -> list:
    """Zielterme: je Zelle und Besuch k eine Boolesche Variable mit dem Grenzzuwachs als Gewicht (fallend in k, deshalb nutzt der Löser sie in Reihenfolge)."""
    N = len(p)
    ziel = []
    for i in range(N):
        zi = sum(x[t][i] for t in range(T))
        ks = [k for k in range(T) if int(round(p[i] * exp(-alpha * k) * (1 - exp(-alpha)) * SKALA)) > 0]
        if not ks:
            continue
        ys = [m.NewBoolVar(f"y{i}_{k}") for k in ks]
        for a in range(len(ks) - 1):
            m.Add(ys[a] >= ys[a + 1])
        m.Add(sum(ys) <= zi)
        for a, k in enumerate(ks):
            ziel.append(int(round(p[i] * exp(-alpha * k) * (1 - exp(-alpha)) * SKALA)) * ys[a])
    return ziel


def _loesen(m, zeit: float, worker: int):
    s = cp_model.CpSolver()
    s.parameters.max_time_in_seconds = zeit
    s.parameters.num_workers = worker
    s.parameters.random_seed = 1
    status = s.Solve(m)
    return s, status


def _weg(s, x, N: int, T: int) -> list[int]:
    return [next(i for i in range(N) if s.Value(x[t][i])) for t in range(T)]


def exakt_weg(p, alpha: float, n: int, T: int, start: int = 0, zeit: float = 6.0, worker: int = 4) -> tuple[list[int], bool]:
    """(Weg, bewiesen optimal?) für den wahren Wert; bei Zeitablauf der beste gefundene Weg."""
    N = n * n
    m, x = _grundmodell(n, T, start)
    m.Maximize(sum(_wahrer_wert_ziel(m, x, p, alpha, T)))
    s, status = _loesen(m, zeit, worker)
    return _weg(s, x, N, T), status == cp_model.OPTIMAL


def orienteering_weg(p, alpha: float, n: int, T: int, start: int = 0, zeit: float = 6.0, worker: int = 4) -> tuple[list[int], bool]:
    """(Weg, bewiesen?) der lexikographischen Orienteering-Fassung (siehe Modulkopf)."""
    N = n * n
    gew = [int(round(p[i] * SKALA)) for i in range(N)]
    m, x = _grundmodell(n, T, start)
    v = [m.NewBoolVar(f"v{i}") for i in range(N)]
    for i in range(N):
        m.Add(v[i] <= sum(x[t][i] for t in range(T)))
    m.Maximize(sum(gew[i] * v[i] for i in range(N)))
    s1, st1 = _loesen(m, zeit, worker)
    stufe1 = int(round(s1.ObjectiveValue()))
    m2, x2 = _grundmodell(n, T, start)
    v2 = [m2.NewBoolVar(f"v{i}") for i in range(N)]
    for i in range(N):
        m2.Add(v2[i] <= sum(x2[t][i] for t in range(T)))
    m2.Add(sum(gew[i] * v2[i] for i in range(N)) >= stufe1)
    m2.Maximize(sum(_wahrer_wert_ziel(m2, x2, p, alpha, T)))
    s2, st2 = _loesen(m2, zeit, worker)
    return _weg(s2, x2, N, T), (st1 == cp_model.OPTIMAL and st2 == cp_model.OPTIMAL)


def wege_aufzaehlen(n: int, T: int, start: int = 0):
    """Alle Wege (Orakel für kleine Gitter und kurze Wege)."""
    nb = nachbarn(n)

    def rec(pos, rest):
        if rest == 0:
            yield [pos]
            return
        for v in nb[pos]:
            for w in rec(v, rest - 1):
                yield [pos] + w

    return rec(start, T - 1)


def beste_weg_aufzaehlung(p, alpha: float, n: int, T: int, start: int = 0) -> tuple[float, float, float]:
    """(Wegoptimum, Prämien-Summe des Orienteering-Optimums, wahrer Wert des besten Orienteering-Weges) durch Aufzählung aller Wege."""
    beste, bestes_pm, bester_lex = -1.0, -1.0, -1.0
    for w in wege_aufzaehlen(n, T, start):
        wv = weg_wert(p, alpha, w)
        pm = praemien_summe(p, w)
        beste = max(beste, wv)
        if pm > bestes_pm + 1e-12:
            bestes_pm, bester_lex = pm, wv
        elif abs(pm - bestes_pm) <= 1e-12:
            bester_lex = max(bester_lex, wv)
    return beste, bestes_pm, bester_lex


__all__ = ["exakt_weg", "orienteering_weg", "wege_aufzaehlen", "beste_weg_aufzaehlung"]
