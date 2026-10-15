"""Welches Verfahren, gemessen woran? - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Fünfzehntes und letztes Stück der Linie Mehrkriterien-Ranking (Zusammenfluss). Dieselbe neutrale Entscheidungsmatrix; zehn Verfahren der Linie gegen den verborgenen Nutzen
eines simulierten Entscheiders: Platz 1, Platz 1 unter den ersten drei, Kendalls τ, Nutzenverlust, Stabilität bei Störung der Rohwerte, Übereinstimmung zwischen den Verfahren.
Kern: vgl_methods.py (sieben gewichtete Verfahren), vgl_dea.py, vgl_smaa.py, vgl_score.py; Generator: entscheidungsmatrix_generator.py (wortgleich in allen Repos der Linie).

Lauffähig mit: streamlit run app.py
"""

import streamlit as st

import vgl_constants as C
from vgl_evaluation import (
    Settings,
    analyse,
    de,
    experiment_agreement,
    experiment_methods,
    experiment_stability,
    experiment_weights,
    mean_off_diagonal,
    pct,
    verdict_agreement,
    verdict_case,
    verdict_methods,
    verdict_stability,
    verdict_weights,
)
from vgl_pdf_export import build_pdf
from vgl_presets import (
    apply_preset,
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sync_query_params,
)
from vgl_visualization import (
    agreement_table,
    alt_names,
    build_methods,
    build_tau_matrix,
    build_utility,
    build_weights,
    methods_detail_table,
    methods_hit_table,
    methods_table,
    stability_table,
    style_methods,
    weights_table,
)

st.set_page_config(page_title="Welches Verfahren, gemessen woran? – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner=False)
def _analysis(settings):
    return analyse(settings)


@st.cache_data(show_spinner="Experiment 1 rechnet …")
def _exp_methods():
    return experiment_methods()


@st.cache_data(show_spinner="Experiment 2 rechnet …")
def _exp_weights():
    return experiment_weights()


@st.cache_data(show_spinner="Experiment 3 rechnet …")
def _exp_agreement():
    return experiment_agreement()


@st.cache_data(show_spinner="Experiment 4 rechnet …")
def _exp_stability():
    return experiment_stability()


def verdict(kind_text):
    kind, text = kind_text
    getattr(st, kind)(text)


st.title("⚖️ Welches Verfahren, gemessen woran?")
st.markdown(
    """
**Welches Verfahren trifft, hängt weniger am Verfahren als an dem, was es weiß und was es annimmt.** Dies ist das fünfzehnte und letzte Stück der Linie
**Mehrkriterien-Ranking**: Zehn Verfahren der Linie bekommen dieselbe Entscheidungsmatrix und werden gegen den **verborgenen Nutzen** eines simulierten Entscheiders gemessen
(Platz 1 getroffen, Kendalls τ, Nutzenverlust, Stabilität). Die Zahlen gelten **in dieser simulierten Welt**, für diesen Entscheider und diese Gewichte; es ist keine
Verfahrensempfehlung.

Weiter unten: vier Experimente (**🔬**), **🚧 Wo die Annahmen enden** und der Expander **📐 Mathematische Formulierung**.
"""
)

with st.expander("So wird verglichen", expanded=True):
    st.markdown(
        """
1. **Welt:** Eine neutrale Matrix aus Alternativen und Kriterien (Punkte ↑, Euro und Minuten ↓), ein simulierter Entscheider mit verborgenen Gewichten und einer von fünf Nutzenformen.
2. **Verfahren:** gleiche Gewichte, gewichtete Summe (Nenn-Spannweite, Min-Max), TOPSIS (Vektor-, Nenn-Normierung), PROMETHEE II, ELECTRE I, AHP, DEA-Kreuzeffizienz, SMAA. Die
   sieben gewichteten (Summe, TOPSIS, PROMETHEE II, ELECTRE I, AHP) bekommen die Gewichte des Entscheiders **mit Erhebungsrauschen σ**; gleiche Gewichte, DEA und SMAA kennen sie
   nicht. Je Verfahren eine Spielart.
3. **Maße:** Platz 1 getroffen, Platz 1 unter den ersten drei, τ zum Nutzen, Nutzenverlust der Wahl in Anteilen der Nutzenspannweite, Stabilität des Siegers bei gestörten
   Rohwerten, Übereinstimmung der Verfahren untereinander.
4. **Was ein Vergleich in dieser Welt zeigt und was nicht:** Er zeigt, wie sich die Verfahren unter diesen Annahmen zueinander verhalten, nicht, welches in einer echten
   Entscheidung richtig ist.
        """
    )

st.caption("🎯 Schnellstart – ein Beispielszenario laden:")
cols = st.columns(len(C.PRESETS))
for col, name in zip(cols, C.PRESETS):
    with col:
        st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP.get(name, ""), key=f"preset_{name}")

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    st.markdown("**Matrix**")
    n = st.slider("Alternativen n", *bounds("n_slider"), key="n_slider", help="Wie viele Alternativen verglichen werden.")
    m = st.slider("Kriterien m", *bounds("m_slider"), key="m_slider", help="Nach wie vielen Kriterien jede Alternative gemessen ist.")
    rho = st.selectbox("Korrelation ρ der Kriterien", C.RHO_CHOICES, key="rho_select", format_func=lambda v: de(v, 1),
                       help="0: unabhängige Kriterien (viele Zielkonflikte). 0,5: positiv korrelierte Kriterien (weniger Zielkonflikte).")
    st.markdown("**Entscheider und Gewichte**")
    decider = st.selectbox("Entscheider", C.DECIDERS, key="decider_select", format_func=lambda x: C.DECIDER_LABELS[x],
                           help="Der Nutzen, gegen den gemessen wird. Vier Formen sind additiv; die mit Untergrenze nicht.")
    sigma = st.selectbox("Erhebungsrauschen σ der Gewichte", C.SIGMA_CHOICES, key="sigma_select", format_func=lambda v: de(v, 1),
                         help="Wie ungenau die Gewichte sind, die die Gewichtsverfahren bekommen: 0 = exakt, 1,0 = jedes Gewicht typisch um den Faktor 2,7 nach oben oder unten verschoben.")
    st.markdown("**Störung**")
    perturb = st.selectbox("Störung der Rohwerte", C.PERTURB_CHOICES, key="perturb_select", format_func=lambda v: f"{pct(v, 0)} %",
                           help="Multiplikatives lognormales Rauschen auf allen Rohwerten; die Tabelle zeigt, welcher Sieger bleibt.")
    seed = st.number_input("Zufalls-Seed der Instanz", *bounds("seed_input"), key="seed_input", step=1)
    st.button("🎲 Neue Matrix würfeln", width="stretch", on_click=randomize_seed)

sync_query_params({"n_slider": int(n), "m_slider": int(m), "rho_select": float(rho), "sigma_select": float(sigma), "decider_select": decider, "perturb_select": float(perturb),
                   "seed_input": int(seed)})

settings = Settings(int(n), int(m), float(rho), float(sigma), decider, float(perturb), int(seed))
a = _analysis(settings)
names = alt_names(settings.n)

# --- Ergebnis ------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Ergebnis")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Bester des Entscheiders", names[a.best], delta=C.DECIDER_SHORT[settings.decider], delta_color="off", help="Die Alternative mit dem größten verborgenen Nutzen.")
c2.metric("Verfahren, die ihn wählen", f"{a.hits} von {len(C.METHODS)}", delta=f"{a.distinct} verschiedene Sieger", delta_color="off",
          help="Wie viele der zehn Verfahren den Besten des Entscheiders auf Platz 1 setzen.")
c3.metric("Einig", "alle zehn" if a.all_agree else ("7 gewichtete" if a.weighted_agree else "nein"),
          delta="" if not a.weighted_agree else ("und der Beste" if a.agreed_right else "aber nicht der Beste"), delta_color="off", help="Ob die Verfahren denselben Sieger wählen, und ob das der Beste des Entscheiders ist.")
c4.metric("Mittleres τ zwischen den Verfahren", de(mean_off_diagonal(a.tau_matrix), 2), delta=f"Nutzenverlust Summe (Nenn) {pct(a.regrets['sum_nom'])} %", delta_color="off",
          help="Wie ähnlich sich die Rangfolgen der zehn Verfahren sind; darunter, was die Wahl der Summe auf der Nenn-Spannweite den Entscheider kostet.")

st.dataframe(style_methods(methods_table(a)), hide_index=True, width="stretch")
st.caption("Platz des Besten: der Platz, auf dem das Verfahren die beste Alternative des Entscheiders setzt (1 = getroffen). Nutzenverlust: in Anteilen der Nutzenspannweite der "
           "erlaubten Alternativen; eine ausgeschlossene Wahl zählt 100 %.")

tab_u, tab_t = st.tabs(["Nutzen des Entscheiders je Alternative", "τ zwischen den Rangfolgen der Verfahren"])
with tab_u:
    st.plotly_chart(build_utility(a), width="stretch", key="utility_chart")
with tab_t:
    st.plotly_chart(build_tau_matrix(a), width="stretch", key="tau_chart")
    st.caption("Kendalls τ-b zwischen den Rangfolgen je zweier Verfahren: 1 = gleiche Ordnung, 0 = unabhängig, −1 = umgekehrt.")

if st.button("📄 PDF des aktuellen Falls erstellen", key="pdf_make"):
    st.session_state["pdf_made"] = (settings, build_pdf(a))
made = st.session_state.get("pdf_made")
if made and made[0] == settings:
    st.download_button("⬇️ PDF herunterladen", data=made[1], file_name="verfahrensvergleich.pdf", mime="application/pdf", key="pdf_download")

# --- Kernfrage -----------------------------------------------------------------------------------------------------------------------------

st.markdown("## ❓ Kernfrage: Was zeigt die Übereinstimmung der Verfahren?")
verdict(verdict_case(a))
if settings.decider == "untergrenze":
    excluded = [C.METHOD_SHORT[m_] for m_ in C.METHODS if a.regrets[m_] >= 1.0]
    if excluded:
        st.caption("Der Entscheider schließt Alternativen mit einem sehr schlechten Kriterium aus. Eine ausgeschlossene Alternative wählen: " + ", ".join(excluded) + ".")

st.markdown("---")

# --- Experimente ---------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Die vier Experimente")
st.caption(f"{len(C.SWEEP_SEEDS)} feste Matrizen (SWEEP_SEEDS), {C.EXP_N} Alternativen, {C.EXP_M} Kriterien, unabhängig von den Reglern; jedes Experiment rechnet einige "
           "Sekunden bis rund 30 Sekunden und startet erst auf Klick. Standardfehler der Anteile 1,5 bis 3,5 Punkte: Unterschiede unter etwa 5 Punkten sind nicht belastbar.")

st.markdown("### Experiment 1: Welche Verfahren treffen?")
st.caption("Zehn Verfahren gegen fünf Entscheider, exakte Gewichte, unabhängige Kriterien: Platz 1, Platz 1 unter den ersten drei, τ und Nutzenverlust.")
if st.button("▶ Experiment 1 rechnen", key="exp_methods_start"):
    st.session_state["exp_methods"] = _exp_methods()
if st.session_state.get("exp_methods"):
    res = st.session_state["exp_methods"]
    st.plotly_chart(build_methods(res), width="stretch", key="methods_chart")
    verdict(verdict_methods(res))
    st.dataframe(methods_hit_table(res), hide_index=True, width="stretch")
    st.caption("Entscheider mit linearem Güte-Nutzen, alle Maße:")
    st.dataframe(methods_detail_table(res), hide_index=True, width="stretch")

st.markdown("### Experiment 2: Was kosten die Gewichte?")
st.caption("Erhebungsrauschen σ der Gewichte (0 bis 1,0) und Korrelation ρ der Kriterien (0 und 0,5): Platz 1 je Verfahren, Entscheider mit linearem Güte-Nutzen.")
if st.button("▶ Experiment 2 rechnen", key="exp_weights_start"):
    st.session_state["exp_weights"] = _exp_weights()
if st.session_state.get("exp_weights"):
    res = st.session_state["exp_weights"]
    st.plotly_chart(build_weights(res), width="stretch", key="weights_chart")
    verdict(verdict_weights(res))
    st.dataframe(weights_table(res), hide_index=True, width="stretch")

st.markdown("### Experiment 3: Übereinstimmung ist kein Beleg")
st.caption("Wie oft wählen die sieben gewichteten Verfahren denselben Sieger, und wie oft ist das der Beste des Entscheiders?")
if st.button("▶ Experiment 3 rechnen", key="exp_agreement_start"):
    st.session_state["exp_agreement"] = _exp_agreement()
if st.session_state.get("exp_agreement"):
    res = st.session_state["exp_agreement"]
    verdict(verdict_agreement(res))
    st.dataframe(agreement_table(res), hide_index=True, width="stretch")

st.markdown("### Experiment 4: Wie stabil?")
st.caption("Die Rohwerte werden mit 5 % und 10 % lognormalem Rauschen gestört, die Gewichte bleiben: Bleibt der Sieger, wie ähnlich bleibt die Rangfolge?")
if st.button("▶ Experiment 4 rechnen", key="exp_stability_start"):
    st.session_state["exp_stability"] = _exp_stability()
if st.session_state.get("exp_stability"):
    res = st.session_state["exp_stability"]
    verdict(verdict_stability(res))
    st.dataframe(stability_table(res), hide_index=True, width="stretch")

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Der Entscheider ist in vier von fünf Formen additiv** | Die Summe und PROMETHEE gewinnen gerade deshalb; mit Untergrenze verlieren auch die besten gewichteten Verfahren deutlich (Experiment 1). Die Aussagen gelten für diese Welt, nicht für Entscheidungen im Allgemeinen. | – |
| **Je Verfahren eine Spielart** | ELECTRE ohne Veto, PROMETHEE nur mit V-Form, AHP aus Rohwerten statt aus Paarvergleichen, TOPSIS in zwei Normierungen, DEA radial. Andere Spielarten können anders abschneiden. | – |
| **Gewichtsrauschen ist lognormal und unabhängig** | Echte Erhebungsfehler sind strukturiert (Anker, Gruppierung, Reihenfolge). | – |
| **Ein Generator** | Unabhängige oder gleich stark korrelierte Kriterien, n = 12, m = 5; keine Cluster, keine Ausreißer. | – |
| **Standardfehler 1,5 bis 3,5 Punkte** | Unterschiede unter etwa 5 Punkten (Summe gegen PROMETHEE, die beiden TOPSIS) sind nicht belastbar. | – |
| **Nicht dabei** | Ordinale Regression, Choquet-Integral, Zuordnung gegen Profile und Gruppenaggregation brauchen andere Eingaben und stehen in ihren Stücken; Fuzzy-Varianten und neuere Einzelverfahren sind nicht geprüft. | die Stücke der Linie |
| **Die Originalstudien sind nicht geprüft** | Vergleichsstudien zu MCDA-Verfahren und zur Vermehrung von Verfahren sind hier nicht gelesen und nicht zitiert. | – |
| **Keine Verfahrensempfehlung** | Die Aussage lautet „in dieser Welt, für diesen Entscheider, mit diesen Gewichten“. | – |
"""
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Welt.** $n$ Alternativen mit Güten $g_{ij} \in (0,1)$, $m$ Kriterien (Gauß-Copula mit Gleichkorrelation $\rho$), Rohwerte $x_{ij}$ über je eine streng monotone Abbildung. Der
Entscheider hat verborgene Gewichte $w$ (gleichverteilt über dem Simplex) und einen Nutzen $u$: linear $\sum_j w_j g_{ij}$, konkav $\sum_j w_j \sqrt{g_{ij}}$, Cobb-Douglas
$\sum_j w_j \ln g_{ij}$, linear in den auf die Nenn-Spannweite normierten Rohwerten, oder linear mit Untergrenze ($u = -\infty$, wenn ein $g_{ij} < 0{,}1$).

**Verfahren** mit den Gewichten $\hat w_j = w_j e^{\sigma Z_j} / \sum_k w_k e^{\sigma Z_k}$ ($Z$ standardnormal): gewichtete Summe $\sum_j \hat w_j\, \tilde x_{ij}$ auf der
Nenn-Spannweite oder Min-Max; TOPSIS mit Abstand zu Ideal $d^+$ und Antiideal $d^-$, Wertung $d^- / (d^+ + d^-)$; PROMETHEE II mit Nettofluss
$\phi(a) = \frac{1}{n-1} \sum_b [\pi(a,b) - \pi(b,a)]$ und $\pi(a,b) = \sum_j \hat w_j \min(1, \max(0, (\tilde x_{aj} - \tilde x_{bj})/p))$; ELECTRE I mit Konkordanz
$c(a,b) = \sum_{j:\, x_{aj} \ge x_{bj}} \hat w_j \ge c^\ast$ und Copeland-Zahl; AHP verteilend (Spalte durch ihre Summe); DEA-Kreuzeffizienz (CCR, Mittel über die Optimalgewichte
aller Bewerter); SMAA mit erwartetem Rang unter Gewichten gleichverteilt über dem Simplex. Gleiche Gewichte: $\hat w_j = 1/m$.

**Maße.** Platz 1 getroffen: $\arg\max_a s(a) = \arg\max_a u(a)$. Kendalls $\tau_b$ zwischen Wertung $s$ und Nutzen $u$ (über die erlaubten Alternativen). Nutzenverlust
$(\max u - u(a^\ast)) / (\max u - \min u)$ über die erlaubten Alternativen; eine ausgeschlossene Wahl zählt 1. Stabilität: Anteil der Matrizen, in denen der Sieger nach der
Störung $x \mapsto x\, e^{s Z}$ derselbe bleibt.

**Literatur:** Vergleichsstudien zu MCDA-Verfahren (etwa Simulationsstudien zu Rangübereinstimmung) sind hier nicht im Original geprüft und deshalb nicht mit Autoren angegeben.
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zur Reihe: [Mehrkriterien-Ranking: von Pareto zum Vergleich](https://sebastianhanisch.net/konzepte-mehrkriterien-ranking.html)."
)
