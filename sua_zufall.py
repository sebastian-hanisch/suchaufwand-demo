"""Zufallszahlen in Blöcken auf dem ganzzahligen SplitMix64 aus `sua_rng.py`.

Der Ganzzahlstrom ist derselbe wie bei `SplitMix64.next_u64()` (Test: `tests/test_zufall.py`), nur als Block mit numpy gerechnet
(Ganzzahlarithmetik mod 2^64, auf jeder Plattform gleich). Gleichverteilte Zahlen und Ganzzahlen sind deshalb bitgleich. Die
Normalzahlen (Box-Muller) nehmen `np.log`, `np.cos` und `np.sin`: deren letztes Bit hängt von der Plattform ab. Sie dienen nur
Zählungen in Zellen und Flächenmessungen; jede Aussage darüber ist als Band geprüft, nie bitgenau.
"""

import numpy as np

from sua_rng import MASK, SplitMix64

_GOLD = 0x9E3779B97F4A7C15
_M1 = np.uint64(0xBF58476D1CE4E5B9)
_M2 = np.uint64(0x94D049BB133111EB)


def u64_block(rng: SplitMix64, n: int) -> np.ndarray:
    """Die nächsten n Ausgaben von `rng.next_u64()` als uint64-Feld; der Zustand von `rng` rückt um n Schritte vor."""
    if n <= 0:
        return np.zeros(0, dtype=np.uint64)
    k = np.arange(1, n + 1, dtype=np.uint64)
    z = np.uint64(rng.state) + k * np.uint64(_GOLD)
    z = (z ^ (z >> np.uint64(30))) * _M1
    z = (z ^ (z >> np.uint64(27))) * _M2
    z = z ^ (z >> np.uint64(31))
    rng.state = (rng.state + n * _GOLD) & MASK
    return z


def uniform(rng: SplitMix64, n: int) -> np.ndarray:
    """Gleichverteilt in [0, 1) mit 53 Bit, wie `SplitMix64.random()`."""
    return (u64_block(rng, n) >> np.uint64(11)).astype(np.float64) / float(1 << 53)


def integers(rng: SplitMix64, n: int, high: int) -> np.ndarray:
    """Ganzzahlen 0..high-1 (Rest der Division wie `SplitMix64.below()`)."""
    return (u64_block(rng, n) % np.uint64(high)).astype(np.int64)


def normals(rng: SplitMix64, n: int) -> np.ndarray:
    """Standardnormalverteilte Zahlen (Box-Muller, beide Werte eines Paares)."""
    m = (n + 1) // 2
    u = uniform(rng, 2 * m)
    u1 = np.maximum(u[:m], 1e-12)
    radius = np.sqrt(-2.0 * np.log(u1))
    winkel = 2.0 * np.pi * u[m:]
    return np.concatenate([radius * np.cos(winkel), radius * np.sin(winkel)])[:n]


def bootstrap_intervall(werte, seed: int, ziehungen: int = 2000, anteil: float = 95.0) -> tuple[float, float]:
    """Intervall der Mittelwerte (Perzentil-Bootstrap) mit Ziehungen aus dem SplitMix64; fester Zufalls-Seed, deshalb reproduzierbar."""
    x = np.asarray(werte, dtype=float)
    if len(x) == 0:
        return float("nan"), float("nan")
    rng = SplitMix64(seed)
    idx = integers(rng, ziehungen * len(x), len(x)).reshape(ziehungen, len(x))
    mittel = x[idx].mean(axis=1)
    rand = (100.0 - anteil) / 2.0
    return float(np.percentile(mittel, rand)), float(np.percentile(mittel, 100.0 - rand))
