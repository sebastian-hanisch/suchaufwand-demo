"""Layout (DEMO-PLAYBOOK §2): fester Kopf, Schnellstart oben in der Seitenleiste, aktives Preset hervorgehoben; Regler stehen im Hauptbereich beim Experiment.

Die Hervorhebung ist zustandslos (`aktives_preset` vergleicht jeden Regler aus SETTING_SPECS mit dem Preset). Der Test unten ändert JEDEN Regler jeder Ansicht einzeln:
sonst bliebe das Preset trotz Änderung markiert."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import sua_constants as C
import sua_presets as P

APP = str(Path(__file__).resolve().parent.parent / "app.py")
REGLER_ARTEN = ("slider", "number_input", "select_slider", "selectbox", "radio", "toggle", "checkbox", "text_input")


def _app(ansicht: str | None = None) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=240)
    at.run()
    if ansicht:
        at.radio(key="ansicht").set_value(ansicht).run()
    assert not at.exception, [str(e) for e in at.exception]
    return at


def _typen(at) -> dict:
    return {b.label: b.proto.type for b in at.sidebar.button if b.label in C.PRESETS}


def _regler(at, key):
    for art in REGLER_ARTEN:
        try:
            return getattr(at.main, art)(key=key)
        except KeyError:
            continue
    raise AssertionError(f"Regler {key} steht nicht im Hauptbereich")


def test_presets_stehen_in_der_seitenleiste_und_nicht_im_hauptbereich():
    at = _app()
    assert [b.label for b in at.sidebar.button][: len(C.PRESETS)] == list(C.PRESETS)
    assert not [b for b in at.main.button if b.label in C.PRESETS]
    assert not [w for art in REGLER_ARTEN for w in getattr(at.sidebar, art) if w.key]          # keine Regler in der Seitenleiste außer den Presets


def test_titel_steht_im_festen_kopf():
    at = _app()
    assert "st-key-kopf" in " ".join(h.proto.body for h in at.get("html"))
    assert at.title[0].value


def test_beim_start_ist_genau_das_standard_preset_hervorgehoben():
    assert [n for n, t in _typen(_app()).items() if t == "primary"] == [next(iter(C.PRESETS))]


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_angeklicktes_preset_schaltet_die_ansicht_ist_hervorgehoben_und_bleibt_es(name):
    at = _app()
    at.sidebar.button(key=f"preset_{name}").click().run()
    assert not at.exception
    assert at.radio(key="ansicht").value == C.PRESETS[name]["ansicht"]
    assert [n for n, t in _typen(at).items() if t == "primary"] == [name]
    at.run()
    assert [n for n, t in _typen(at).items() if t == "primary"] == [name]


def test_regler_der_gewaehlten_ansicht_stehen_in_setting_specs_und_umgekehrt():
    for ansicht in C.ANSICHTEN:
        at = _app(ansicht)
        schluessel = {w.key for art in REGLER_ARTEN for w in getattr(at.main, art) if w.key} - {"ansicht"}
        assert schluessel == set(P.view_keys(ansicht)), ansicht


@pytest.mark.parametrize("state_key", [k for k in P.SETTING_SPECS if k != "ansicht"])
def test_jede_aenderung_eines_reglers_hebt_die_hervorhebung_auf(state_key):
    s = P.SETTING_SPECS[state_key]
    at = _app(s.ansicht)
    name = next(n for n, p in C.PRESETS.items() if p["ansicht"] == s.ansicht)
    at.sidebar.button(key=f"preset_{name}").click().run()
    assert _typen(at)[name] == "primary"
    if state_key == "d_dv":
        _regler(at, "d_hyp").set_value(2).run()                  # der Abstand ist nur bei zwei Hypothesen bedienbar
    regler = _regler(at, state_key)
    if s.optionen is not None:
        neu = next(o for o in s.optionen if not P._gleich(o, regler.value))
    else:
        neu = s.hi if not P._gleich(regler.value, s.hi) else s.lo
    regler.set_value(neu).run()
    assert not at.exception, [str(e) for e in at.exception]
    assert _typen(at)[name] != "primary"                       # ein anderes Preset darf zufällig passen (Weg: ein Gipfel gegen zwei Gipfel)
    at.sidebar.button(key=f"preset_{name}").click().run()
    assert _typen(at)[name] == "primary"


def test_aenderung_des_umschalters_hebt_die_hervorhebung_auf():
    at = _app()
    at.radio(key="ansicht").set_value("weg").run()
    assert _typen(at)[next(iter(C.PRESETS))] != "primary"        # das Standard-Preset gilt nicht mehr; ein Weg-Preset darf passen
