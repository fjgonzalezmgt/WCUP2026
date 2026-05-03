"""Constantes de configuracion globales del paquete wcup2026.

Define rutas de datos, titulos de la app, URLs de referencia, nombres de grupos,
columnas del modelo y paleta de colores para los graficos.
"""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "teams_seed.csv"
RESULTS_DIR = PROJECT_ROOT / "resultados"
MONTECARLO_RESULTS_PATH = RESULTS_DIR / "resultado_montecarlo.xlsx"
LLM_ANALYSIS_PATH = RESULTS_DIR / "resultado_llm.md"

APP_TITLE = "WCUP 2026 Predictor"
APP_CAPTION = "Simulador Monte Carlo para explorar quien podria ganar el Mundial 2026."

FIFA_GROUPS_URL = "https://www.fifa.com/en/tournaments/mens/worldcup/canadamexicousa2026/articles/final-draw-results"
FIFA_SCHEDULE_URL = "https://www.fifa.com/en/tournaments/mens/worldcup/canadamexicousa2026/articles/match-schedule-fixtures-results-teams-stadiums"
OPENAI_RESPONSES_URL = "https://platform.openai.com/docs/api-reference/responses"

GROUPS = list("ABCDEFGHIJKL")
FEATURE_COLUMNS = ["attack", "defense", "squad", "form"]
STAGE_COLUMNS = [
    "group_winner",
    "group_runner_up",
    "best_third",
    "round_of_32",
    "round_of_16",
    "quarterfinal",
    "semifinal",
    "final",
    "champion",
]

CHART_COLORS = [
    "#206a5d",
    "#c94c4c",
    "#f2a154",
    "#4b7bec",
    "#6c5ce7",
    "#7f8c8d",
]
