"""Reglerangaben, Permalink (Begrenzen), Presets und Zufalls-Seed-Knopf.

Jede Ansicht hat eigene Regler im Hauptbereich. Ein Regler, der in einem Lauf nicht gezeichnet wird, verliert seinen Zustand (Streamlit räumt ihn
weg); deshalb schreiben Standardwerte, Permalink und Presets nur Regler der **gezeichneten** Ansicht (DEMO-PLAYBOOK §3, ausblendbare Regler).
"""

import time
from dataclasses import dataclass
from math import isclose

import streamlit as st

import sua_constants as C
from sua_rng import SplitMix64


@dataclass(frozen=True)
class SettingSpec:
    url_param: str
    caster: type
    default: object
    ansicht: str
    optionen: tuple | None = None      # feste Auswahl (Regler mit Auswahlwerten)
    lo: float | None = None            # Grenzen (Schieberegler, Ganzzahl)
    hi: float | None = None
    step: float | None = None


def _s(url, caster, default, ansicht, optionen=None, lo=None, hi=None, step=None):
    return SettingSpec(url, caster, default, ansicht, optionen, lo, hi, step)


SETTING_SPECS = {
    "ansicht": _s("ansicht", str, "zuteilung", "*", optionen=C.ANSICHTEN),
    # Ansicht 1, Experiment „Zuteilung“
    "z_prior": _s("zp", str, "ein_gipfel", "zuteilung", C.PRIORS_DREI),
    "z_gitter": _s("zn", int, 8, "zuteilung", C.GITTER),
    "z_alpha": _s("za", float, 1.0, "zuteilung", C.ALPHAS),
    "z_kappa": _s("zk", float, 1.0, "zuteilung", C.KAPPAS),
    "z_tau": _s("zt", float, 0.95, "zuteilung", C.TAUS),
    "z_seed": _s("zs", int, 1, "zuteilung", lo=0, hi=C.SEED_MAX, step=1),
    # Ansicht 1, Experiment „Gelände“
    "g_prior": _s("gp", str, "ein_gipfel", "zuteilung", C.PRIORS_ZWEI),
    "g_gamma": _s("gg", float, 0.0, "zuteilung", C.GAMMAS),
    "g_alpha": _s("ga", float, 1.0, "zuteilung", C.ALPHAS_GELAENDE),
    "g_kappa": _s("gk", float, 0.5, "zuteilung", C.KAPPAS),
    "g_seed": _s("gs", int, 1, "zuteilung", lo=0, hi=C.SEED_MAX, step=1),
    # Ansicht 2, Experiment „Nichtfund“
    "n_prior": _s("np", str, "ein_gipfel", "nichtfund", C.PRIORS_ZWEI),
    "n_alpha": _s("na", float, 1.0, "nichtfund", C.ALPHAS_NICHTFUND),
    "n_kappa": _s("nk", float, 1.0, "nichtfund", C.KAPPAS_NICHTFUND),
    "n_anteil": _s("nr", float, 0.5, "nichtfund", C.ANTEILE_RUNDE1),
    "n_verfahren": _s("nv", str, "aktualisiert", "nichtfund", C.VERFAHREN_RUNDE2),
    "n_seed": _s("ns", int, 1, "nichtfund", lo=0, hi=C.SEED_MAX, step=1),
    # Ansicht 2, Experiment „Falsches α“
    "f_prior": _s("fp", str, "ein_gipfel", "nichtfund", C.PRIORS_ZWEI),
    "f_alpha": _s("fa", float, 2.0, "nichtfund", C.ALPHAS_ANGENOMMEN),
    "f_kappa": _s("fk", float, 1.0, "nichtfund", C.KAPPAS_NICHTFUND),
    "f_seed": _s("fs", int, 1, "nichtfund", lo=0, hi=C.SEED_MAX, step=1),
    # Ansicht 3
    "w_prior": _s("wp", str, "ein_gipfel", "weg", C.PRIORS_DREI),
    "w_alpha": _s("wa", float, 1.0, "weg", C.ALPHAS_WEG),
    "w_T": _s("wt", int, 15, "weg", C.TS_WEG),
    "w_verfahren": _s("wv", str, "exakt", "weg", C.VERFAHREN_WEG),
    "w_inst": _s("wi", int, 1, "weg", lo=1, hi=30, step=1),
    # Ansicht 4
    "d_s0": _s("s0", float, 1.0, "drift", lo=0.0, hi=2.0, step=0.1),
    "d_D": _s("dd", float, 0.5, "drift", lo=0.0, hi=1.0, step=0.05),
    "d_sU": _s("su", float, 0.3, "drift", lo=0.0, hi=0.6, step=0.05),
    "d_u": _s("du", float, 0.5, "drift", lo=0.0, hi=1.5, step=0.1),
    "d_hyp": _s("dh", int, 1, "drift", C.HYPOTHESEN),
    "d_dv": _s("dv", float, 0.8, "drift", lo=0.1, hi=1.0, step=0.1),
    "d_t": _s("dt", float, 4.0, "drift", lo=1.0, hi=8.0, step=0.5),
    "d_t0": _s("d0", float, 1.0, "drift", lo=0.5, hi=4.0, step=0.5),
    "d_delay": _s("dl", float, 0.25, "drift", C.VERZOEGERUNGEN),
    "d_L0": _s("dL", float, 0.5, "drift", C.L0_WERTE),
}


def spec(key: str) -> SettingSpec:
    return SETTING_SPECS[key]


def view_keys(ansicht: str) -> list[str]:
    """Regler einer Ansicht (ohne den Umschalter selbst)."""
    return [k for k, s in SETTING_SPECS.items() if s.ansicht == ansicht]


def bounds(key: str):
    s = SETTING_SPECS[key]
    return s.lo, s.hi


def _gleich(a, b) -> bool:
    if isinstance(a, float) or isinstance(b, float):
        try:
            return isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-9)
        except (TypeError, ValueError):
            return False
    return a == b


def _begrenzen(s: SettingSpec, wert):
    """Wert aus der Adresszeile auf einen erlaubten Wert zurückführen; ungültig gibt den Standardwert."""
    if s.optionen is not None:
        for o in s.optionen:
            if _gleich(o, wert):
                return o
        return s.default
    if s.lo is not None:
        wert = max(s.lo, wert)
    if s.hi is not None:
        wert = min(s.hi, wert)
    return wert


def init_session_state_defaults():
    """Nur der Umschalter steht immer auf der Seite und bekommt seinen Standardwert; die Regler der Ansichten folgen mit `init_view_defaults`."""
    if "ansicht" not in st.session_state:
        st.session_state["ansicht"] = SETTING_SPECS["ansicht"].default


def init_view_defaults(ansicht: str):
    """Standardwerte der Regler der gezeichneten Ansicht, soweit noch nichts im Zustand steht (unmittelbar vor dem Zeichnen)."""
    for key in view_keys(ansicht):
        if key not in st.session_state:
            st.session_state[key] = SETTING_SPECS[key].default


def load_permalink_settings():
    """Einmal je Sitzung: Werte aus der Adresszeile übernehmen (auf erlaubte Werte begrenzt), nur für die Ansicht der Adresszeile."""
    if "permalink_loaded" in st.session_state:
        return
    qp = st.query_params
    s = SETTING_SPECS["ansicht"]
    if s.url_param in qp and qp[s.url_param] in C.ANSICHTEN:
        st.session_state["ansicht"] = qp[s.url_param]
    ansicht = st.session_state.get("ansicht", s.default)
    for key in view_keys(ansicht):
        sp = SETTING_SPECS[key]
        if sp.url_param in qp:
            try:
                st.session_state[key] = _begrenzen(sp, sp.caster(qp[sp.url_param]))
            except (ValueError, TypeError):
                continue
    st.session_state["permalink_loaded"] = True


def sync_query_params(ansicht: str):
    """Die Adresszeile spiegelt Ansicht und Regler der gezeichneten Ansicht."""
    try:
        st.query_params.clear()
        st.query_params["ansicht"] = ansicht
        for key in view_keys(ansicht):
            st.query_params[SETTING_SPECS[key].url_param] = str(st.session_state.get(key, SETTING_SPECS[key].default))
    except Exception:
        pass


def apply_preset(name: str):
    """Schaltet auf die Ansicht des Presets und setzt deren Regler: die genannten auf die Werte des Presets, alle übrigen auf ihren Standard."""
    preset = C.PRESETS[name]
    ansicht = preset["ansicht"]
    st.session_state["ansicht"] = ansicht
    for key in view_keys(ansicht):
        st.session_state[key] = preset.get(key, SETTING_SPECS[key].default)


def aktives_preset():
    """Name des Presets, dessen Werte GENAU den aktuellen Einstellungen entsprechen, sonst None.

    Zustandslos: verglichen wird jeder Regler aus SETTING_SPECS (ein Preset ohne Eintrag meint den Standardwert) mit dem Session State; Regler
    nicht gezeichneter Ansichten haben keinen Zustand und gelten als Standard. Ein später ergänzter Regler braucht nur seinen Eintrag in
    SETTING_SPECS (tests/test_layout.py prüft für jeden Regler, dass seine Änderung die Markierung aufhebt)."""
    for name, preset in C.PRESETS.items():
        if all(_gleich(st.session_state.get(key, s.default), preset.get(key, s.default)) for key, s in SETTING_SPECS.items()):
            return name
    return None


def sidebar_presets():
    """Schnellstart oben in der Seitenleiste; das aktive Preset ist hervorgehoben (`type="primary"`)."""
    st.header("🎯 Schnellstart")
    st.caption("Ein Beispielszenario laden (schaltet auch die Ansicht um):")
    aktiv = aktives_preset()
    for name in C.PRESETS:
        st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP.get(name, ""), key=f"preset_{name}",
                  type="primary" if aktiv == name else "secondary")


def randomize_seed(key: str):
    st.session_state[key] = SplitMix64(time.time_ns()).below(C.SEED_MAX + 1)  # nur der Knopf nimmt die Uhr; alle Rechnungen nehmen den Seed
