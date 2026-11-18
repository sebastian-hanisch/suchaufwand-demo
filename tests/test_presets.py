"""Presets: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen, Permalink-Begrenzung; jedes Preset ruft seine Ansicht auf."""

import sua_constants as C
import sua_presets as P


def test_every_preset_has_help_and_valid_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP) and len(C.PRESETS) == 12
    for name, preset in C.PRESETS.items():
        assert C.PRESET_HELP[name]
        assert preset["ansicht"] in C.ANSICHTEN
        for key in preset:
            assert key == "ansicht" or (key in P.SETTING_SPECS and P.SETTING_SPECS[key].ansicht == preset["ansicht"]), (name, key)


def test_preset_values_are_inside_the_allowed_values():
    for name, preset in C.PRESETS.items():
        for key, value in preset.items():
            if key == "ansicht":
                continue
            s = P.SETTING_SPECS[key]
            if s.optionen is not None:
                assert any(P._gleich(value, o) for o in s.optionen), (name, key, value)
            else:
                assert s.lo <= value <= s.hi, (name, key, value)


def test_every_default_is_allowed():
    for key, s in P.SETTING_SPECS.items():
        assert (s.optionen is not None and any(P._gleich(s.default, o) for o in s.optionen)) or s.lo <= s.default <= s.hi, key


def test_default_preset_equals_the_default_settings():
    first = next(iter(C.PRESETS.values()))
    assert first["ansicht"] == P.SETTING_SPECS["ansicht"].default
    for key, value in first.items():
        assert P._gleich(P.SETTING_SPECS[key].default, value), key


def test_every_view_has_a_preset_and_the_view_keys_cover_all_settings():
    assert {p["ansicht"] for p in C.PRESETS.values()} == set(C.ANSICHTEN)
    assert sorted(k for a in C.ANSICHTEN for k in P.view_keys(a)) == sorted(k for k in P.SETTING_SPECS if k != "ansicht")


def test_permalink_values_are_limited_to_allowed_values():
    s = P.SETTING_SPECS
    assert P._begrenzen(s["z_alpha"], 99.0) == s["z_alpha"].default and P._begrenzen(s["z_alpha"], 0.5) == 0.5
    assert P._begrenzen(s["z_seed"], -5) == 0 and P._begrenzen(s["z_seed"], 10 ** 9) == C.SEED_MAX
    assert P._begrenzen(s["d_s0"], 7.0) == 2.0 and P._begrenzen(s["w_inst"], 0) == 1
    assert P._begrenzen(s["w_verfahren"], "unsinn") == "exakt"


def test_all_presets_differ_in_at_least_one_setting():
    seen = set()
    for preset in C.PRESETS.values():
        full = tuple(sorted((k, repr(preset.get(k, P.SETTING_SPECS[k].default))) for k in P.SETTING_SPECS))
        assert full not in seen
        seen.add(full)
