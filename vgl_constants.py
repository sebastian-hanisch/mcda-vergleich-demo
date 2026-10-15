"""Konstanten: Regler-Grenzen, Verfahren, Entscheider, feste Annahmen der Experimente und Presets."""

N_MIN, N_MAX, DEFAULT_N = 6, 20, 12
M_MIN, M_MAX, DEFAULT_M = 3, 8, 5
SEED_MAX, DEFAULT_SEED = 999999, 24           # je Preset geprüft (tests/test_presets.py)

RHO_CHOICES = (0.0, 0.5)                      # Gleichkorrelation der Kriterien (Gauß-Copula des Generators)
DEFAULT_RHO = 0.0
SIGMA_CHOICES = (0.0, 0.3, 0.6, 1.0)          # Erhebungsrauschen der Gewichte: lognormal, w_hat ~ w * exp(sigma * Z), danach auf Summe 1
DEFAULT_SIGMA = 0.0
PERTURB_CHOICES = (0.0, 0.05, 0.10)          # Störung der Rohwerte: lognormal, x ~ x * exp(s * Z)
DEFAULT_PERTURB = 0.0

DECIDERS = ("guete_linear", "guete_konkav", "guete_cobb", "roh_linear", "untergrenze")
DECIDER_LABELS = {
    "guete_linear": "Linear in der Güte (Quantil innerhalb des Kriteriums)",
    "guete_konkav": "Konkav in der Güte (Wurzel, gewichtet)",
    "guete_cobb": "Cobb-Douglas in der Güte (Logarithmus, gewichtet)",
    "roh_linear": "Linear in den Rohwerten (feste Nenn-Spannweite)",
    "untergrenze": "Mit Untergrenze (unter Güte 0,1 in einem Kriterium: ausgeschlossen)",
}
DECIDER_SHORT = {"guete_linear": "Güte linear", "guete_konkav": "Güte konkav", "guete_cobb": "Güte Cobb-Douglas", "roh_linear": "Rohwerte linear", "untergrenze": "Untergrenze"}
DEFAULT_DECIDER = "guete_linear"
THETA_FLOOR = 0.1                             # Güte, unter der der Entscheider mit Untergrenze eine Alternative ausschließt
UNDER_FLOOR = -1e9                            # Nutzen einer ausgeschlossenen Alternative
EXCLUDED_BELOW = -1e8                         # Nutzen darunter heißt ausgeschlossen

METHODS = ("equal", "sum_nom", "sum_minmax", "topsis_vec", "topsis_nom", "prom_v", "electre", "ahp_dist", "dea_ce", "smaa")
WEIGHTED = ("sum_nom", "sum_minmax", "topsis_vec", "topsis_nom", "prom_v", "electre", "ahp_dist")      # bekommen die (verrauschten) Gewichte des Entscheiders
METHOD_LABELS = {
    "equal": "Gleiche Gewichte (Summe, Nenn-Spannweite)",
    "sum_nom": "Gewichtete Summe, Nenn-Spannweite",
    "sum_minmax": "Gewichtete Summe, Min-Max der Stichprobe",
    "topsis_vec": "TOPSIS, Vektor-Normierung",
    "topsis_nom": "TOPSIS, Nenn-Spannweite",
    "prom_v": "PROMETHEE II (V-Form, p = 0,5)",
    "electre": "ELECTRE I (c* = 0,6, Copeland-Zahl)",
    "ahp_dist": "AHP (verteilend)",
    "dea_ce": "DEA-Kreuzeffizienz (CCR)",
    "smaa": "SMAA ohne Gewichtswissen (erwarteter Rang)",
}
METHOD_SHORT = {"equal": "gleiche Gewichte", "sum_nom": "Summe (Nenn)", "sum_minmax": "Summe (Min-Max)", "topsis_vec": "TOPSIS (Vektor)", "topsis_nom": "TOPSIS (Nenn)",
                "prom_v": "PROMETHEE II", "electre": "ELECTRE I", "ahp_dist": "AHP", "dea_ce": "DEA-Kreuzeffizienz", "smaa": "SMAA"}
KNOWS_WEIGHTS = {m: m in WEIGHTED for m in METHODS}

NS_DRAWS = 1500                               # Ziehungen des Gewichtsraums für SMAA
ELECTRE_CSTAR = 0.6                           # Konkordanzschwelle
PROMETHEE_P = 0.5                             # Präferenzschwelle der V-Form (Anteil der Spannweite)
TOL_TIE = 1e-9                                # LP-Rauschen: Wertungen werden vor dem Vergleich auf neun Stellen gerundet
TIE_DECIMALS = 9

# Feste Instanzen für alle Experimente: Mittel über SWEEP_SEEDS, nie ein einzelner Seed als Befund.
SWEEP_SEEDS = tuple(range(200000, 200200))
EXP_N = 12
EXP_M = 5
OFFSET_NOISE = 104729                         # Strom des Erhebungsrauschens der Gewichte: SplitMix64(seed + OFFSET_NOISE)
OFFSET_SMAA = 5                               # Strom der SMAA-Ziehungen
OFFSET_PERTURB = 7919                         # Strom der Störung der Rohwerte

COLOR_BEST = "#D68A2E"
COLOR_PICK = "#2F6B65"
COLOR_BAR = "#8A96A6"
COLOR_LINE = "rgba(80,80,80,0.7)"
COLOR_METHOD = ("#4c78a8", "#2F6B65", "#59a14f", "#7a5195", "#c0392b", "#D68A2E", "#76b7b2", "#9c755f", "#b07aa1", "#8A96A6")

PRESETS = {
    "Standardfall (exakte Gewichte, additiver Entscheider)": {"n": DEFAULT_N, "m": DEFAULT_M, "rho": 0.0, "sigma": 0.0, "decider": "guete_linear", "perturb": 0.0,
                                                              "seed": DEFAULT_SEED},
    "Verrauschte Gewichte": {"n": 12, "m": 5, "rho": 0.0, "sigma": 0.6, "decider": "guete_linear", "perturb": 0.0, "seed": 24},
    "Nicht additiv": {"n": 12, "m": 5, "rho": 0.0, "sigma": 0.0, "decider": "untergrenze", "perturb": 0.0, "seed": 14},
    "Korrelierte Kriterien": {"n": 12, "m": 5, "rho": 0.5, "sigma": 0.0, "decider": "guete_linear", "perturb": 0.0, "seed": 22},
    "Einig und falsch": {"n": 12, "m": 5, "rho": 0.0, "sigma": 0.6, "decider": "guete_linear", "perturb": 0.0, "seed": 23},
}
PRESET_HELP = {
    "Standardfall (exakte Gewichte, additiver Entscheider)": "Die Verfahren kennen die Gewichte des Entscheiders und er ist additiv: Die Summe und PROMETHEE treffen am häufigsten.",
    "Verrauschte Gewichte": "Dieselbe Matrix wie im Standardfall, aber die Gewichte sind verrauscht (σ = 0,6): Die Verfahren, die Gewichte brauchen, verlieren im Mittel Treffer.",
    "Nicht additiv": "Der Entscheider schließt Alternativen mit einem sehr schlechten Kriterium aus: Keines der gewichteten Verfahren schützt davor.",
    "Korrelierte Kriterien": "Kriterien mit Korrelation 0,5: Die Verfahren liegen dichter beieinander, weil weniger Raum für Zielkonflikte bleibt.",
    "Einig und falsch": "Verrauschte Gewichte, und alle gewichteten Verfahren wählen dasselbe, aber nicht den Besten: Sie teilen die Eingabe und damit den Fehler.",
}
