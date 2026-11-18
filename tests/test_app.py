"""End-to-end Smoke-Test mit Streamlits AppTest: lädt app.py, schaltet jede Ansicht, klickt jedes Preset und fährt jeden Regler an seine Grenzen; kein Python-Fehler,
insbesondere kein `StreamlitDuplicateElementId` (gleiche Diagramme brauchen ein key=) und keine Ausnahme bei Reglern am Rand (Streuung 0, Zeit am Maximum)."""

import os

import pytest
from streamlit.testing.v1 import AppTest

import sua_constants as C
import sua_presets as P

APP_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
TIMEOUT = int(os.environ.get("APPTEST_TIMEOUT", "300"))  # unter Last (parallele Läufe) reichen 180 s nicht immer; Umgebungsvariable zum Anpassen
ARTEN = ("slider", "number_input", "select_slider", "radio")


def _app(ansicht: str | None = None) -> AppTest:
    at = AppTest.from_file(APP_PATH, default_timeout=TIMEOUT)
    at.run()
    if ansicht:
        at.radio(key="ansicht").set_value(ansicht).run()
    assert not at.exception, [str(e) for e in at.exception]
    return at


def _regler(at, key):
    for art in ARTEN:
        try:
            return getattr(at.main, art)(key=key)
        except KeyError:
            continue
    raise AssertionError(key)


def test_app_laedt_ohne_ausnahme_und_zeigt_hinweis_und_kernfrage():
    at = _app()
    texte = " ".join(m.value for m in at.markdown)
    assert "Anwendungsbeispiel, keine Beratung" in texte and "Kernfrage" in texte
    assert [e.label for e in at.expander if "Wie wir das erreichen" in e.label] and [e.label for e in at.expander if "Wo die Annahmen enden" in e.label]
    assert [e.label for e in at.expander if "Mathematische Formulierung" in e.label]


@pytest.mark.parametrize("ansicht", C.ANSICHTEN)
def test_jede_ansicht_zeigt_kernfrage_verdikt_pdf_und_drei_expander(ansicht):
    at = _app(ansicht)
    assert any(m.value.startswith("## ❓ Kernfrage") for m in at.markdown)
    assert at.info or at.success
    labels = [e.label for e in at.expander]
    for teil in ("Wie wir das erreichen", "Wo die Annahmen enden", "Mathematische Formulierung"):
        assert any(teil in label for label in labels), (ansicht, teil)
    assert len(at.get("download_button")) >= 1


def test_jedes_preset_laeuft_ohne_ausnahme():
    for name in C.PRESETS:
        at = _app()
        at.sidebar.button(key=f"preset_{name}").click().run(timeout=TIMEOUT)
        assert not at.exception, (name, [str(e) for e in at.exception])


@pytest.mark.parametrize("key", [k for k in P.SETTING_SPECS if k != "ansicht"])
def test_jeder_regler_an_seinen_grenzen_ohne_ausnahme(key):
    s = P.SETTING_SPECS[key]
    for rand in (0, -1):
        at = _app(s.ansicht)
        if key == "d_dv":
            _regler(at, "d_hyp").set_value(2).run(timeout=TIMEOUT)            # der Regler wirkt (und ist bedienbar) nur bei zwei Hypothesen
        wert = s.optionen[rand] if s.optionen is not None else (s.lo if rand == 0 else s.hi)
        _regler(at, key).set_value(wert).run(timeout=TIMEOUT)
        assert not at.exception, (key, wert, [str(e) for e in at.exception])


def test_ohne_jede_streuung_gibt_die_drift_ansicht_eine_warnung_statt_eines_absturzes():
    at = _app("drift")
    for key in ("d_s0", "d_D", "d_sU"):
        _regler(at, key).set_value(0.0)
    at.run(timeout=TIMEOUT)
    assert not at.exception and any("keine Wolke" in w.value for w in at.warning)


def test_diagramme_haben_eindeutige_schluessel_bei_gleichem_inhalt():
    at = _app("zuteilung")
    at.sidebar.button(key="preset_Gelände γ = 1").click().run(timeout=TIMEOUT)
    assert not at.exception
