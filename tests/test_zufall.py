"""Zufall: der Blockgenerator liefert denselben Ganzzahlstrom wie der skalare SplitMix64 (Plattformfestigkeit der Messreihen)."""

import numpy as np

import sua_zufall as Z
from sua_rng import SplitMix64


def test_block_ist_derselbe_strom_wie_der_skalare_generator():
    a, b = SplitMix64(12345), SplitMix64(12345)
    skalar = [a.next_u64() for _ in range(1000)]
    block = [int(v) for v in Z.u64_block(b, 1000)]
    assert skalar == block and a.state == b.state


def test_zwei_bloecke_hintereinander_sind_ein_grosser_block():
    a, b = SplitMix64(7), SplitMix64(7)
    ganz = Z.u64_block(a, 50)
    teile = np.concatenate([Z.u64_block(b, 20), Z.u64_block(b, 30)])
    assert np.array_equal(ganz, teile)


def test_gleichverteilt_wie_random_und_ganzzahlen_wie_below():
    a, b, c, d = SplitMix64(3), SplitMix64(3), SplitMix64(4), SplitMix64(4)
    assert list(Z.uniform(a, 10)) == [b.random() for _ in range(10)]
    assert list(Z.integers(c, 10, 17)) == [d.below(17) for _ in range(10)]


def test_normalzahlen_haben_mittel_null_und_varianz_eins():
    x = Z.normals(SplitMix64(99), 200_000)
    assert abs(x.mean()) < 0.01 and abs(x.var() - 1.0) < 0.02
    assert len(Z.normals(SplitMix64(1), 7)) == 7  # ungerade Zahl


def test_gleicher_seed_gleiche_folge():
    assert np.array_equal(Z.normals(SplitMix64(5), 50), Z.normals(SplitMix64(5), 50))
    assert not np.array_equal(Z.normals(SplitMix64(5), 50), Z.normals(SplitMix64(6), 50))


def test_bootstrap_intervall_ist_reproduzierbar_und_umschliesst_das_mittel():
    x = np.arange(30, dtype=float)
    lo, hi = Z.bootstrap_intervall(x, 11)
    assert (lo, hi) == Z.bootstrap_intervall(x, 11)
    assert lo < x.mean() < hi
    assert Z.bootstrap_intervall(np.full(30, 2.0), 11) == (2.0, 2.0)
