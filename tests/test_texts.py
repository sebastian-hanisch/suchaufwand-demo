"""Texte: keine Planungswörter, App- und README-Fassung derselben Texte stimmen überein, Hinweis „Anwendungsbeispiel, keine Beratung“ steht in beiden."""

import re
from pathlib import Path

import sua_constants as C
import sua_texts as T

ROOT = Path(__file__).resolve().parent.parent
README = (ROOT / "README.md").read_text(encoding="utf-8")
# Wörter aus der Planung, zusammengesetzt, damit diese Datei sie nicht selbst enthält
VERBOTEN = [r"Einzel" + r"plan", r"Vor" + r"messung", r"Phase " + r"0", r"ZUSCH" + r"NITT", r"St" + r"ück\b", r"\bPl" + r"an\b", r"\bLin" + r"ie\b", r"Saat" + r"wert", r"läuft lo" + r"kal",
            r"\btr" + r"ägt\b", r"\btr" + r"agen\b", r"gepl" + r"ant"]


def _dateien():
    return [ROOT / "README.md", ROOT / "app.py", *sorted(ROOT.glob("sua_*.py")), *sorted((ROOT / "tools").glob("*.py")), *sorted((ROOT / "tests").glob("*.py"))]


def test_keine_planungswoerter_in_readme_app_texten_docstrings_und_tests():
    treffer = []
    for pfad in _dateien():
        for nr, zeile in enumerate(pfad.read_text(encoding="utf-8").splitlines(), start=1):
            if "Diese Demo ist Teil des Portfolios" in zeile or "VERBOTEN" in zeile or "Wörter aus der Planung" in zeile:
                continue
            for muster in VERBOTEN:
                if re.search(muster, zeile):
                    treffer.append(f"{pfad.name}:{nr}: {muster}")
    assert not treffer, treffer


def test_hinweis_steht_in_app_und_readme_und_nennt_keine_beratung():
    assert C.HINWEIS in T.EINLEITUNG
    assert "**Anwendungsbeispiel, keine Beratung.**" in README and "keine Einsatzempfehlung" in README and "keine Einsatzempfehlung" in C.HINWEIS


def test_die_annahmen_der_app_stehen_wortgleich_in_der_readme():
    for zeile in T.GRENZEN_ALLGEMEIN.strip().splitlines():
        assert zeile.strip() in README, zeile


def test_jede_ansicht_hat_alle_texte_und_die_formeln_haben_je_einen_block():
    for a in C.ANSICHTEN:
        for quelle in (T.KERNFRAGEN, T.ANSICHT_EINLEITUNG, T.METHODEN, T.GRENZEN, T.MATHE_LESEHILFE, T.MATHE_GLEICHUNGEN, T.MATHE_SATZ):
            assert quelle[a]
        assert all("$$" not in g and "%" not in g.replace(r"\%", "") for g in T.MATHE_GLEICHUNGEN[a])          # ein Block je Gleichung, Prozentzeichen geschützt
    assert "Skizze" in T.MATHE_SATZ["drift"] and "vollständige Modell" in T.MATHE_SATZ["zuteilung"]


def test_orienteering_link_und_lawine_ohne_zahl():
    assert C.ORIENTEERING_LINK == "https://github.com/sebastian-hanisch/gewinn-touren-demo" and C.ORIENTEERING_LINK in README
    lawine = next(z for z in T.GRENZEN["weg"].splitlines() if "früher Fund" in z) + " " + T.GRENZEN["weg"].split("früher Fund")[1]
    assert not re.search(r"\d", lawine.split("Eine Zahl dazu")[0].replace("Zustand", ""))


def test_nachbar_demos_sind_github_links_ohne_zusatz():
    for name in ("gewinn-touren-demo", "lagrange-kkt-demo", "aufloesung-messorte-demo", "particle-filter-demo", "bayes-opt-demo"):
        assert f"(https://github.com/sebastian-hanisch/{name})" in README
