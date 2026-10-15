"""PDF-Export des aktuellen Falls (fpdf2). Kernschriften sind latin-1: Umlaute gehen, Pfeile und griechische Buchstaben werden ersetzt."""

from __future__ import annotations

from fpdf import FPDF

import vgl_constants as C
from vgl_evaluation import Analysis, de, mean_off_diagonal, pct

_REPLACE = {"↑": "+", "↓": "-", "→": "->", "↔": "<->", "–": "-", "—": "-", "τ": "tau", "σ": "sigma", "ρ": "rho", "×": "x", "≥": ">=", "≤": "<=", "≠": "!=", "−": "-", "€": "EUR",
            "·": "*", "★": "*", "…": "..."}


def _s(text) -> str:
    out = str(text)
    for old, new in _REPLACE.items():
        out = out.replace(old, new)
    return out.encode("latin-1", "replace").decode("latin-1")


def build_pdf(a: Analysis) -> bytes:
    s = a.settings
    names = [f"A{i + 1}" for i in range(s.n)]
    pdf = FPDF(orientation="L", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 15)
    pdf.cell(0, 9, _s("Welches Verfahren, gemessen woran?"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 5, _s(f"{s.n} Alternativen, {s.m} Kriterien, Korrelation rho = {de(s.rho, 1)}, Erhebungsrauschen der Gewichte sigma = {de(s.sigma, 1)}, Entscheider: "
                            f"{C.DECIDER_LABELS[s.decider]}, Störung der Rohwerte {pct(s.perturb, 0)} %. Seed {s.seed}. Alles simuliert."), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    pdf.multi_cell(0, 5, _s(f"Der Beste des Entscheiders ist {names[a.best]}; {a.hits} von 10 Verfahren wählen ihn, es gibt {a.distinct} verschiedene Sieger. "
                            f"Mittleres tau zwischen den Rangfolgen der Verfahren: {de(mean_off_diagonal(a.tau_matrix), 2)}."), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)
    pdf.set_font("Helvetica", "B", 9)
    pdf.cell(0, 6, _s("Verfahren (Sieger, Platz des Besten, tau zum Nutzen, Nutzenverlust)"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 8)
    for m in C.METHODS:
        after = "" if a.picks_perturbed is None else f", nach Störung {names[a.picks_perturbed[m]]}"
        pdf.cell(0, 5, _s(f"{C.METHOD_LABELS[m]}: Sieger {names[a.picks[m]]}, Platz des Besten {a.ranks[m]}, tau {de(a.taus[m], 2)}, Nutzenverlust {pct(a.regrets[m])} %{after}"),
                 new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)
    pdf.set_font("Helvetica", "I", 8)
    pdf.multi_cell(0, 4, _s("Die Aussagen gelten in dieser simulierten Welt für diesen Entscheider mit diesen Gewichten und sind keine Verfahrensempfehlung. Der Entscheider ist ein "
                            "simuliertes Orakel, kein Mensch. Quelle: mcda-vergleich-demo, Sebastian Hanisch."), new_x="LMARGIN", new_y="NEXT")
    return bytes(pdf.output())
