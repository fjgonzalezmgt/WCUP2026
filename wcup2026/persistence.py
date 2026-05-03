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
    """Guardar la simulacion Monte Carlo en un archivo XLSX con tres hojas.

    Crea o sobrescribe el archivo con las hojas ``resultados``, ``equipos``
    y ``parametros``.

    Parameters
    ----------
    results : pd.DataFrame
        DataFrame de resultados devuelto por ``simulate_many``.
    teams : pd.DataFrame
        DataFrame original de equipos con ratings.
    params : SimParams
        Hiperparametros usados en la simulacion.
    path : Path, optional
        Ruta del archivo XLSX de salida.  Por defecto
        ``MONTECARLO_RESULTS_PATH``.
    """
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
    """Leer la ultima simulacion Monte Carlo guardada en disco.

    Parameters
    ----------
    path : Path, optional
        Ruta del archivo XLSX.  Por defecto ``MONTECARLO_RESULTS_PATH``.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame] or None
        ``(results, teams)`` si el archivo existe y ambas hojas tienen
        datos; ``None`` en caso contrario.
    """
    if not path.exists():
        return None
    results = pd.read_excel(path, sheet_name="resultados")
    teams = pd.read_excel(path, sheet_name="equipos")
    if results.empty or teams.empty:
        return None
    return results, teams


def save_llm_analysis(text: str, path: Path = LLM_ANALYSIS_PATH) -> None:
    """Guardar el texto del analisis LLM en un archivo Markdown.

    Parameters
    ----------
    text : str
        Contenido del analisis generado por el LLM.
    path : Path, optional
        Ruta del archivo de salida.  Por defecto ``LLM_ANALYSIS_PATH``.
    """
    ensure_results_dir()
    path.write_text(text, encoding="utf-8")


def load_llm_analysis(path: Path = LLM_ANALYSIS_PATH) -> str | None:
    """Leer el ultimo analisis LLM guardado desde disco.

    Parameters
    ----------
    path : Path, optional
        Ruta del archivo Markdown.  Por defecto ``LLM_ANALYSIS_PATH``.

    Returns
    -------
    str or None
        Contenido del archivo si existe y no esta vacio; ``None`` en caso
        contrario.
    """
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8").strip()
    return text or None
