"""Gemeinsame Fixtures: die Mini-Instanz, an der jede Einheit von Hand geprüft wird (DEMO-PLAYBOOK §1b).

Drei Alternativen A, B, C und zwei Kriterien (K1 ↑ Punkte, K2 ↓ Euro; Nenn-Spannweiten [0; 100] und [800; 5000]), Gewichte (0,6; 0,4):
A (80; 1000 Euro), B (50; 3000 Euro), C (20; 800 Euro); vorzeichenbehaftet (80; -1000), (50; -3000), (20; -800).
Nenn-Spannweite normiert: A (0,8; 0,9524), B (0,5; 0,4762), C (0,2; 1,0): Summe 0,861 / 0,4905 / 0,52.
Min-Max: A (1; 0,9091), B (0,5; 0), C (0; 1): Summe 0,9636 / 0,3 / 0,4; PROMETHEE II (V-Form, p = 0,5): Nettofluss 0,7636 / -0,4 / -0,3636.
ELECTRE I (c* = 0,6): A übertrifft B und C, B übertrifft C: Copeland 2 / 0 / -2. TOPSIS (Nenn) 0,8345 / 0,4927 / 0,4652; AHP verteilend 0,4748 / 0,2516 / 0,2735.
DEA (CCR, Eingabe = Euro, Ausgabe = Punkte): Verhältnisse 0,08 / 0,0167 / 0,025: theta 1 / 0,2083 / 0,3125, Kreuzeffizienz gleich theta (ein Eingang, ein Ausgang).
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

MINI_S = np.array([[80.0, -1000.0], [50.0, -3000.0], [20.0, -800.0]])
MINI_RAW = np.abs(MINI_S)
MINI_D = np.array([1.0, -1.0])
MINI_W = np.array([0.6, 0.4])
MINI_NOM = np.array([[0.8, 4000.0 / 4200.0], [0.5, 2000.0 / 4200.0], [0.2, 1.0]])


@pytest.fixture
def mini():
    return MINI_S.copy(), MINI_W.copy()
