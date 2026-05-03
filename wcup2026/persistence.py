"""Persistencia de resultados generados por la aplicacion.

Guarda y recupera la ultima simulacion Monte Carlo y el ultimo analisis
LLM desde la carpeta ``resultados`` para que la app pueda restaurarlos al
abrir una nueva sesion.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from wcup2026.config import LLM_ANALYSIS_PATH, MONTECARLO_RESULTS_PATH, RESULTS_DIR
from wcup2026.parameters import SimParams


def ensure_results_dir() -> None:
    """Crear la carpeta de resultados si no existe."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def save_montecarlo_results(
    results: pd.DataFrame,
    teams: pd.DataFrame,
    params: SimParams,
    path: Path = MONTECARLO_RESULTS_PATH,
) -> None:
    """Sobrescribir el archivo XLSX con la ultima simulacion Monte Carlo."""
    ensure_results_dir()
    params_df = pd.DataFrame(
        [{"parameter": key, "value": value} for key, value in vars(params).items()]
    )
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        results.to_excel(writer, sheet_name="resultados", index=False)
        teams.to_excel(writer, sheet_name="equipos", index=False)
        params_df.to_excel(writer, sheet_name="parametros", index=False)


def load_montecarlo_results(
    path: Path = MONTECARLO_RESULTS_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """Leer la ultima simulacion guardada, si existe y es valida."""
    if not path.exists():
        return None
    results = pd.read_excel(path, sheet_name="resultados")
    teams = pd.read_excel(path, sheet_name="equipos")
    if results.empty or teams.empty:
        return None
    return results, teams


def save_llm_analysis(text: str, path: Path = LLM_ANALYSIS_PATH) -> None:
    """Sobrescribir el archivo markdown con la ultima salida del LLM."""
    ensure_results_dir()
    path.write_text(text, encoding="utf-8")


def load_llm_analysis(path: Path = LLM_ANALYSIS_PATH) -> str | None:
    """Leer el ultimo analisis LLM guardado, si existe."""
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8").strip()
    return text or None
