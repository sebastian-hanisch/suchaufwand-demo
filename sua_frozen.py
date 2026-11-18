"""Vorgerechnete Messreihen und Wege (Ordner data/): Lesezugriff. Erzeugt von `tools/sua_make_zuteilung.py`, `tools/sua_make_weg.py` und
`tools/sua_make_drift.py`; die App rechnet diese Dinge (30 Instanzen je Zelle, CP-SAT, 40 000 Teilchen) nie live."""

import json
from functools import lru_cache
from pathlib import Path

DATA = Path(__file__).resolve().parent / "data"


@lru_cache(maxsize=None)
def laden(name: str) -> dict | None:
    """Inhalt von data/<name>.json oder None, wenn die Datei fehlt."""
    pfad = DATA / f"{name}.json"
    if not pfad.exists():
        return None
    return json.loads(pfad.read_text(encoding="utf-8"))


def schluessel(*teile) -> str:
    return "|".join(str(t) for t in teile)


def messreihe(reihe: str) -> dict:
    d = laden("sua_messreihe")
    return d[reihe] if d and reihe in d else {}


def zelle(reihe: str, *teile) -> dict | None:
    return messreihe(reihe).get(schluessel(*teile))


def weg_tabelle(gitter: int, art: str, alpha: float, T: int) -> dict | None:
    d = laden("sua_weg")
    if not d or f"g{gitter}" not in d:
        return None
    return d[f"g{gitter}"]["tabelle"].get(schluessel(art, alpha, T))


def weg_instanz(art: str, alpha: float, T: int, inst: int) -> dict | None:
    """Gespeicherte Lösungen (5x5) einer Instanz: Wegoptimum und Orienteering-Fassung, je mit Weg, Wert und Beweisstatus."""
    d = laden("sua_weg")
    if not d or "g5" not in d:
        return None
    liste = d["g5"]["inst"].get(schluessel(art, alpha, T))
    if not liste or not 1 <= inst <= len(liste):
        return None
    return liste[inst - 1]


def drift_szenarien() -> dict | None:
    return laden("sua_drift")
