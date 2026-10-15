# Welches Verfahren, gemessen woran? – Streamlit-Demo

<!-- Nach dem Deploy hierher den Live-Link setzen (Leerzeile davor und danach):
**[→ Demo live ausprobieren](https://sebastianhanisch-mcda-vergleich-demo.streamlit.app/)** -->

Fünfzehntes und letztes Stück der Linie **Mehrkriterien-Ranking** in der „Konzepte“-Reihe für die Website „Sebastian Hanisch – Operations Research und Machine Learning“, der
Zusammenfluss der Linie: Kind von [SMAA](https://github.com/sebastian-hanisch/smaa-demo), [ordinaler Regression](https://github.com/sebastian-hanisch/ordinale-regression-demo),
[Kriterien-Wechselwirkung](https://github.com/sebastian-hanisch/kriterien-wechselwirkung-demo), [Zuordnung gegen Profile](https://github.com/sebastian-hanisch/zuordnung-profile-demo),
[DEA](https://github.com/sebastian-hanisch/dea-demo) und [Gruppenentscheidung](https://github.com/sebastian-hanisch/gruppen-entscheidung-demo). **Zehn Verfahren der Linie** bekommen
dieselbe Entscheidungsmatrix und werden gegen den **verborgenen Nutzen** eines simulierten Entscheiders gemessen: Platz 1 getroffen, Platz 1 unter den ersten drei, **Kendalls τ**,
**Nutzenverlust**, **Stabilität** bei gestörten Rohwerten und **Übereinstimmung** der Verfahren untereinander.

Vehikel ist dieselbe **neutrale, simulierte Entscheidungsmatrix** wie in den Vorgängern (Generator `entscheidungsmatrix_generator.py`, wortgleich). Der Entscheider ist ein Orakel,
kein Mensch. Die Zahlen gelten **in dieser simulierten Welt**, für diese Entscheider und diese Gewichte; es ist keine Verfahrensempfehlung.

## Warum dieses Problem

**Welches Verfahren trifft, hängt weniger am Verfahren als an dem, was es weiß und was es annimmt.** Die Kernfrage der Demo: Was kostet es, wenn die Gewichte ungenau sind, wenn der
Entscheider nicht additiv ist, und was zeigt es, dass mehrere Verfahren denselben Sieger wählen?

## Befunde und Korrekturen gegenüber dem Plan

- **Bei exakt bekannten Gewichten und additivem Entscheider treffen drei Verfahren im Rauschen beieinander:** gewichtete Summe auf der Nenn-Spannweite 77,5 %, auf Min-Max 79,0 %,
  PROMETHEE II 78,0 % (Entscheider mit linearem Güte-Nutzen). TOPSIS 70,0 / 71,0 %, ELECTRE I 70,5 %, AHP 68,0 %; gleiche Gewichte 54,5 %, SMAA ohne Gewichtswissen 54,0 %, DEA-
  Kreuzeffizienz 47,5 %. Der Standardfehler ist 3 Punkte; die drei vorderen unterscheiden sich nicht belastbar.
- **Die Verfahren mit eigener Aggregationsregel verlieren 6,5 bis 9,5 Punkte an die Summe,** weil der Entscheider additiv ist (ELECTRE, TOPSIS, AHP). Das ist ein Befund über diese
  Welt: Ein nicht additiver Entscheider würde die Reihenfolge verschieben.
- **Mit Untergrenze trifft das beste Verfahren 71,5 %** (Summe, beide Skalen); der Nutzenverlust der gewichteten Verfahren liegt zwischen 14,3 und 28,2 %, bei gleichen Gewichten
  12,1 % und bei SMAA 11,1 %: Die gewichteten Verfahren treffen häufiger, verlieren im Mittel aber mehr Nutzen (eine ausgeschlossene Wahl zählt 100 %).
- **Gewichte kosten Treffer:** Mit Erhebungsrauschen 0,6 fällt die Summe von 77,5 auf 63,5 %, ELECTRE von 70,5 auf 53,5 %; bei Rauschen 1,0 sind es 54,0 % und damit nicht mehr als
  die gleichen Gewichte (54,5 %).
- **Übereinstimmung ist kein Beleg:** Die sieben Verfahren, die dieselben Gewichte bekommen, wählen in 33,0 bis 39,0 % der Matrizen denselben Sieger. Der ist bei exakten Gewichten
  immer der Beste (100,0 %), bei Rauschen 0,6 nur in 87,7 %, bei 1,0 in 75,8 %: Sie teilen die Eingabe und damit den Fehler.
- **Korrelierte Kriterien rücken die Verfahren zusammen:** Bei ρ = 0,5 schrumpft die Spanne zwischen bestem und schlechtestem Verfahren von 31,5 auf 11,0 Punkte.
- **Stabilität ist keine Richtigkeit:** Bei 5 % Rauschen auf den Rohwerten bleibt der Sieger bei TOPSIS (Vektor) in 94,5 % der Matrizen, bei ELECTRE I in 82,0 %.
- **Jedes Experiment startet auf eigenen Klick** (einige Sekunden bis rund 30 Sekunden).

## Modell

- **Matrix:** n Alternativen (6 bis 20, Standard 12), m Kriterien (3 bis 8, Standard 5), Gleichkorrelation ρ der Kriterien 0 oder 0,5.
- **Entscheider:** verborgene Gewichte (gleichverteilt über dem Simplex) und eine von fünf Nutzenformen: linear, konkav, Cobb-Douglas (je in der Güte), linear in den Rohwerten auf der
  Nenn-Spannweite, linear mit Untergrenze (unter Güte 0,1 in einem Kriterium ausgeschlossen).
- **Verfahren (je eine Spielart):** gleiche Gewichte; gewichtete Summe auf der Nenn-Spannweite und auf Min-Max der Stichprobe; TOPSIS mit Vektor-Normierung und auf der Nenn-Spannweite;
  PROMETHEE II (V-Form, p = 0,5); ELECTRE I (c* = 0,6, ohne Veto, Copeland-Zahl); AHP verteilend (aus den Rohwerten, ↓ als Kehrwert); DEA-Kreuzeffizienz (CCR, ↓ sind Eingaben);
  SMAA ohne Gewichtswissen (1 500 Ziehungen, erwarteter Rang).
- **Eingabe der Verfahren:** Die sieben gewichteten bekommen die Gewichte des Entscheiders mit lognormalem Erhebungsrauschen σ (0 / 0,3 / 0,6 / 1,0); gleiche Gewichte, DEA und SMAA
  kennen sie nicht.
- **Maße:** Platz 1 getroffen (Gleichstand: kleinere Nummer, Wertungen auf neun Stellen gerundet), Platz 1 unter den ersten drei, τ-b zwischen Wertung und Nutzen, Nutzenverlust in Anteilen
  der Nutzenspannweite (eine ausgeschlossene Wahl zählt 100 %), Stabilität (der Sieger bleibt bei 5 % oder 10 % lognormalem Rauschen auf den Rohwerten), Übereinstimmung (τ zwischen den
  Rangfolgen, alle einig, davon richtig).
- **Fester Rahmen der Experimente:** 200 feste Matrizen (`SWEEP_SEEDS`), 12 Alternativen, 5 Kriterien, unabhängig von den Reglern.

Formeln im 📐-Expander der App.

## Methodik

1. **Dieselbe Matrix, dieselben Gewichte, alle zehn Verfahren:** Vergleiche sind gepaart.
2. **Maßstab ist ein Orakel,** nicht ein Konsens der Verfahren: der verborgene Nutzen.
3. **Gegenproben:** Rauschen der Gewichte, andere Nutzenform, korrelierte Kriterien, gestörte Rohwerte.
4. **Keine Verfahrensempfehlung:** Jede Aussage nennt Welt, Entscheider und Gewichtswissen.

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen: `tests/test_claims.py`. 200 feste Matrizen, 12 Alternativen, 5 Kriterien. **Der Standardfehler eines Anteils ist 1,5 bis 3,5 Punkte; Unterschiede unter etwa 5 Punkten sind
nicht belastbar und stehen nicht als Aussage.**

| Frage | Befund | Test |
|---|---|---|
| Platz 1 je Verfahren (Güte linear, exakte Gewichte, ρ = 0): gleiche Gewichte, Summe Nenn, Summe Min-Max, TOPSIS Vektor / Nenn, PROMETHEE II, ELECTRE I, AHP, DEA, SMAA | **54,5 / 77,5 / 79,0 / 70,0 / 71,0 / 78,0 / 70,5 / 68,0 / 47,5 / 54,0 %** | `test_the_sum_and_promethee_lead_and_the_weight_free_methods_trail` |
| Platz 1 unter den ersten drei; τ zum Nutzen (Güte linear) | Summe Nenn 98,5 %, τ 0,89; PROMETHEE II 99,0 %, 0,89; TOPSIS 94,5 / 93,5 %, 0,81 / 0,82; ELECTRE I 96,0 %, 0,80; AHP 97,5 %, 0,83; gleiche Gewichte 83,0 %, 0,55; DEA 76,5 %, 0,49; SMAA 83,5 %, 0,56. | dasselbe |
| Platz 1 je Entscheider (Summe Nenn / PROMETHEE II / AHP / DEA): konkav, Cobb-Douglas, Rohwerte, Untergrenze | Summe Nenn **81,0 / 76,5 / 100,0 / 71,5 %**; PROMETHEE II 83,0 / 78,0 / 89,5 / 70,5 %; AHP 63,0 / 57,0 / 53,5 / 58,0 %; DEA 50,5 / 53,5 / 45,5 / 51,5 %. Der Rohwert-Entscheider ist die Identität der Summe auf der Nenn-Spannweite (100 %). | `test_the_hit_rate_depends_on_the_decider` |
| Nutzenverlust (Güte linear / Untergrenze) | Summe Nenn 1,6 / 14,3 %; PROMETHEE II 1,5 / 15,8 %; TOPSIS (Vektor) 2,9 / 15,6 %; ELECTRE I 3,1 / 24,0 %; AHP 5,5 / 28,2 %; gleiche Gewichte 10,1 / 12,1 %; DEA 12,8 / 19,1 %; SMAA 10,0 / 11,1 %. | `test_the_floor_decider_costs_the_weighted_methods_most` |
| Erhebungsrauschen σ = 0 / 0,3 / 0,6 / 1,0: Platz 1 der Summe (Nenn), von ELECTRE I, von PROMETHEE II | Summe **77,5 / 75,0 / 63,5 / 54,0 %**; ELECTRE I 70,5 / 67,0 / 53,5 / 44,5 %; PROMETHEE II 78,0 / 72,0 / 65,5 / 58,0 %; gleiche Gewichte, DEA, SMAA unverändert 54,5 / 47,5 / 54,0 %. | `test_noisy_weights_cost_the_weighted_methods_hits` |
| Korrelierte Kriterien (ρ = 0,5, exakte Gewichte): Platz 1 und Spanne | Summe 84,0 %, PROMETHEE II 84,5 %, ELECTRE I 75,0 %, gleiche Gewichte 75,5 %, DEA 73,5 %, SMAA 76,0 %; Spanne **31,5 → 11,0 Punkte**. | `test_correlated_criteria_pull_the_methods_together` |
| Die sieben gewichteten wählen denselben Sieger: Anteil / davon der Beste (σ = 0 / 0,3 / 0,6 / 1,0) | **34,5 / 39,0 / 36,5 / 33,0 %**; **100,0 / 96,2 / 87,7 / 75,8 %**. | `test_agreement_among_weighted_methods_is_not_evidence` |
| Alle zehn einig: Anteil / davon der Beste; mittleres τ zwischen den Rangfolgen | ρ = 0: **18,0 %**, 100,0 %, τ 0,66; ρ = 0,5: **46,5 %**, 100,0 %, τ 0,82. | dasselbe |
| Stabilität: Sieger bleibt bei 5 % / 10 % Rauschen auf den Rohwerten | TOPSIS (Vektor) **94,5 / 85,0 %**; AHP 94,0 / 87,5 %; PROMETHEE II 94,0 / 85,5 %; Summe Nenn 88,5 / 78,5 %; gleiche Gewichte 85,5 / 76,5 %; **ELECTRE I 82,0 / 70,5 %**. | `test_stability_differs_but_is_not_correctness` |

## Ehrliche Grenzen

- **Der Entscheider ist in vier von fünf Formen additiv.** Die Summe und PROMETHEE gewinnen gerade deshalb; die Aussagen gelten für diese Welt, nicht für Entscheidungen im Allgemeinen.
- **Jedes Verfahren in einer Spielart,** die in der Literatur uneinheitlich sind: ELECTRE ohne Veto, PROMETHEE nur V-Form, AHP aus Rohwerten statt aus Paarvergleichen, TOPSIS in
  zwei Normierungen, DEA radial, SMAA ohne Vorwissen. Fuzzy-Varianten und neuere Einzelverfahren sind nicht geprüft.
- **Gewichtsrauschen ist lognormal und unabhängig;** echte Erhebungsfehler sind strukturiert.
- **Ein Generator,** n = 12, m = 5, unabhängige oder gleich stark korrelierte Kriterien; keine Cluster, keine Ausreißer.
- **Kleine Unterschiede sind Rauschen:** Summe gegen PROMETHEE II, die beiden TOPSIS, die beiden Summen.
- **Die Vergleichsstudien zu MCDA-Verfahren und zur Verfahrensvermehrung sind hier nicht gelesen;** weder sie noch ihre Maße (zwölf Ähnlichkeitsmaße) stehen im Text.
- **Nicht dabei:** ordinale Regression, Choquet-Integral, Zuordnung gegen Profile und Gruppenaggregation brauchen andere Eingaben und stehen in ihren Stücken.

## Verifikation

- **Kern:** die sieben gewichteten Verfahren und gleiche Gewichte gegen die Handrechnung der Mini-Instanz (`tests/conftest.py`: drei Alternativen, zwei Kriterien; alle Verfahren
  setzen A vorn und sind uneinig über B gegen C) und gegen unabhängige Schleifen auf Zufallsmatrizen (`tests/test_methods.py`); DEA und SMAA von Hand (`tests/test_dea_smaa.py`).
- **Orakel** (`tests/test_oracle_vergleich.py`): Summe auf der Nenn-Spannweite trifft den Rohwert-Entscheider immer; gleiche Gewichte, DEA und SMAA ignorieren die gemeldeten
  Gewichte; DEA ändert sich nicht bei Umskalierung; Dominanz bei allen zehn Verfahren; Min-Max-Summe, PROMETHEE und ELECTRE invariant gegen affine Umskalierung, die Nenn-Summe nicht;
  Rauschen bleibt eine Verteilung; eine überall beste Alternative wählen alle zehn.
- **Randfälle** (`tests/test_edge_cases.py`): zwei und drei Alternativen, kleinste und größte Instanz, gleiche Alternativen, ein Kriterium ohne Streuung, sehr großes Rauschen, keine Störung.
- **Generator** (`tests/test_generator.py`): SplitMix64 gegen unabhängige Python-Fassung, Randverteilung, Spearman-Korrelation der Copula, Richtungen und Bereiche.
- **Aussagen:** jede Zahl dieser README in `tests/test_claims.py`; Verdikte an von Hand gesetzten Zeilen in `test_verdicts.py`; Presets in `test_presets.py` (jedes zeigt seinen Effekt).
- **Mutationstest** vor dem ersten Commit (`tools/mutate_demo.py` der Website, mit `--budget` und `--timeout`); Stand in `tests/mutation_state.json`.
- **Oberfläche:** AppTest-Rauchtest (jeder Button, jeder Regler am Minimum und Maximum, jede Auswahl), alle Presets, Permalink, PDF und jedes Experiment erst nach eigenem Klick, Footer,
  Laufzeit-Importe.

## Dateistruktur

| Datei | Zweck |
|---|---|
| `.github/workflows/tests.yml` | CI: Push, Pull Request, wöchentlicher Canary |
| `.gitignore` | venv, Caches, Streamlit-Secrets |
| `.streamlit/config.toml` | Streamlit-Einstellungen (keine Nutzungsstatistik) |
| `app.py` | Streamlit-App |
| `entscheidungsmatrix_generator.py` | Generator der Linie (wortgleich in allen Repos der Linie, kein Präfix): SplitMix64, Copula, Rohwerte, Entscheider |
| `vgl_constants.py` | Regler-Grenzen, Verfahren, Entscheider, feste Annahmen der Experimente, Presets |
| `vgl_score.py` | Nenn-Spannweite, Normierungen, Nutzen der Entscheider, Erhebungsrauschen, Störung, Kendalls τ-b, Platz des Besten, Nutzenverlust |
| `vgl_methods.py` | sieben gewichtete Verfahren und gleiche Gewichte als reine Funktionen (eigene Kopien) |
| `vgl_dea.py` | DEA (CCR, Multiplikatorform) und Kreuzeffizienz |
| `vgl_smaa.py` | Gewichtsziehung, Rangmatrix, erwarteter Rang |
| `vgl_evaluation.py` | Fall der App, vier Experimente, Verdikte (reine Funktionen) |
| `vgl_presets.py` | Permalink (Begrenzen), Presets, Zufalls-Seed-Button |
| `vgl_visualization.py` | Verfahrenstabelle, Nutzenbalken, τ-Matrix, Tabellen und Abbildungen der Experimente (Achsen gesperrt) |
| `vgl_pdf_export.py` | PDF-Export des aktuellen Falls (fpdf2, erst nach Klick) |
| `requirements.txt` | Laufzeit-Bibliotheken (ungepinnt) |
| `requirements-dev.txt` | Laufzeit plus pytest |
| `tests/` | Mini-Instanz, Verfahren, Hilfsfunktionen, DEA und SMAA, Orakel, Randfälle, Generator, Auswertung, Verdikte, Aussagen, Presets, AppTest, Footer, Laufzeit-Importe |

## Bewusst nicht umgesetzt

- **Eine Verfahrensempfehlung oder Rangliste der Verfahren:** Die Daten tragen sie nicht über den additiven Entscheider hinaus.
- **Die zwölf Ähnlichkeitsmaße der Originalstudie:** nicht gelesen; hier gelten Platz 1, τ, Nutzenverlust und Stabilität.
- **Weitere Verfahren (ordinale Regression, Choquet, Zuordnung, Gruppe) als Mitspieler:** sie brauchen Beispielurteile, Wechselwirkungen, Profile oder mehrere Entscheider.
- **Fuzzy- und neuere Einzelverfahren:** nicht geprüft.
- **Strukturierte Erhebungsfehler der Gewichte:** das Rauschen ist lognormal und unabhängig.

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Gebaut mit Streamlit, Plotly, NumPy, SciPy, pandas und fpdf2.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html).
