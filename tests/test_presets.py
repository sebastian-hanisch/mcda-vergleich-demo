"""Presets: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen; jedes Preset zeigt seinen Effekt."""

import vgl_constants as C
import vgl_evaluation as ev
import vgl_presets as P

STANDARD = "Standardfall (exakte Gewichte, additiver Entscheider)"


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_inside_the_bounds_and_choices():
    for name, p in C.PRESETS.items():
        assert C.N_MIN <= p["n"] <= C.N_MAX and C.M_MIN <= p["m"] <= C.M_MAX and 0 <= p["seed"] <= C.SEED_MAX, name
        assert p["rho"] in C.RHO_CHOICES and p["sigma"] in C.SIGMA_CHOICES and p["decider"] in C.DECIDERS and p["perturb"] in C.PERTURB_CHOICES, name


def test_default_preset_equals_the_default_settings():
    assert ev.Settings(**C.PRESETS[STANDARD]) == ev.Settings()


def test_setting_specs_cover_every_preset_key_and_url_params_are_unique():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS)
    params = [s.url_param for s in P.SETTING_SPECS.values()]
    assert len(params) == len(set(params))


def test_labels_cover_every_choice():
    assert set(C.METHOD_LABELS) == set(C.METHODS) == set(C.METHOD_SHORT) == set(C.KNOWS_WEIGHTS)
    assert set(C.DECIDER_LABELS) == set(C.DECIDERS) == set(C.DECIDER_SHORT)
    assert set(C.WEIGHTED) <= set(C.METHODS) and len(C.WEIGHTED) == 7
    assert [m for m in C.METHODS if not C.KNOWS_WEIGHTS[m]] == ["equal", "dea_ce", "smaa"]


def test_sweep_seeds_are_the_documented_fixed_set():
    assert len(C.SWEEP_SEEDS) == 200 and len(set(C.SWEEP_SEEDS)) == 200 and C.SWEEP_SEEDS[0] == 200000


def _analysis(name, **override):
    p = dict(C.PRESETS[name])
    p.update(override)
    return ev.analyse(ev.Settings(**p))


def test_every_preset_shows_the_effect_it_is_named_for():
    """Die Presets sind auf einen Seed abgestimmt, bei dem der Effekt sichtbar ist; ändert sich Generator oder Auswertung, soll das hier auffallen."""
    a = _analysis(STANDARD)
    assert a.picks["sum_nom"] == a.best and 5 <= a.hits <= 8 and a.distinct >= 3 and not a.all_agree

    noisy = _analysis("Verrauschte Gewichte")
    assert noisy.settings.seed == a.settings.seed and noisy.picks["sum_nom"] != noisy.best and noisy.hits < a.hits

    floor = _analysis("Nicht additiv")
    assert sum(r == 1.0 for r in floor.regrets.values()) >= 5 and floor.u[floor.best] > C.EXCLUDED_BELOW

    corr = _analysis("Korrelierte Kriterien")
    assert corr.all_agree and corr.agreed_right and corr.hits == 10

    wrong = _analysis("Einig und falsch")
    assert wrong.weighted_agree and wrong.agreed_right is False and not wrong.all_agree and wrong.hits >= 1
