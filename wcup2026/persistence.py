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


POST_GROUP_STATE_PATH = RESULTS_DIR / "post_group_state.xlsx"


def ensure_results_dir() -> None:
    """Crear la carpeta de resultados si no existe.

    Returns
    -------
    None
        La funcion solo garantiza la existencia de ``RESULTS_DIR``.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def save_montecarlo_results(
    results: pd.DataFrame,
    teams: pd.DataFrame,
    params: SimParams,
    bracket: pd.DataFrame | None = None,
    bracket_probable: pd.DataFrame | None = None,
    path: Path = MONTECARLO_RESULTS_PATH,
) -> None:
    """Guardar la simulacion Monte Carlo en un archivo XLSX con tres o cinco hojas.

    Crea o sobrescribe el archivo con las hojas ``resultados``, ``equipos``,
    ``parametros`` y opcionalmente ``bracket`` y ``bracket_probable``.

    Parameters
    ----------
    results : pd.DataFrame
        DataFrame de resultados devuelto por ``simulate_many``.
    teams : pd.DataFrame
        DataFrame original de equipos con ratings.
    params : SimParams
        Hiperparametros usados en la simulacion.
    bracket : pd.DataFrame or None, optional
        Partidos de eliminatoria de una simulacion representativa.
    bracket_probable : pd.DataFrame or None, optional
        Cuadro mas probable calculado sobre N simulaciones.
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
        if bracket is not None and not bracket.empty:
            bracket.to_excel(writer, sheet_name="bracket", index=False)
        if bracket_probable is not None and not bracket_probable.empty:
            bracket_probable.to_excel(writer, sheet_name="bracket_probable", index=False)


def load_bracket_probable(
    path: Path = MONTECARLO_RESULTS_PATH,
) -> pd.DataFrame | None:
    """Leer el cuadro mas probable guardado en la hoja ``bracket_probable`` del XLSX.

    Parameters
    ----------
    path : Path, optional
        Ruta del archivo XLSX.  Por defecto ``MONTECARLO_RESULTS_PATH``.

    Returns
    -------
    pd.DataFrame or None
        DataFrame con el bracket mas probable si la hoja existe; ``None``
        en caso contrario.
    """
    if not path.exists():
        return None
    try:
        xl = pd.ExcelFile(path)
        if "bracket_probable" not in xl.sheet_names:
            return None
        bp = pd.read_excel(xl, sheet_name="bracket_probable")
        return bp if not bp.empty else None
    except Exception:
        return None


def load_bracket(
    path: Path = MONTECARLO_RESULTS_PATH,
) -> pd.DataFrame | None:
    """Leer el cuadro de eliminacion guardado en la hoja ``bracket`` del XLSX.

    Parameters
    ----------
    path : Path, optional
        Ruta del archivo XLSX.  Por defecto ``MONTECARLO_RESULTS_PATH``.

    Returns
    -------
    pd.DataFrame or None
        DataFrame con partidos de eliminatoria si la hoja existe y tiene
        datos; ``None`` en caso contrario.
    """
    if not path.exists():
        return None
    try:
        xl = pd.ExcelFile(path)
        if "bracket" not in xl.sheet_names:
            return None
        bracket = pd.read_excel(xl, sheet_name="bracket")
        return bracket if not bracket.empty else None
    except Exception:
        return None


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


def save_post_group_state(
    group_results_input: pd.DataFrame,
    knockout_input: pd.DataFrame,
    knockout_results: pd.DataFrame,
    projection: pd.DataFrame | None = None,
    bracket: pd.DataFrame | None = None,
    bracket_probable: pd.DataFrame | None = None,
    path: Path = POST_GROUP_STATE_PATH,
) -> None:
    """Guardar el estado post-grupos para restaurarlo en nuevas sesiones.

    Persiste la tabla editable de grupos, la tabla editable de eliminatorias,
    los resultados KO normalizados y, opcionalmente, la ultima proyeccion y
    llaves generadas.
    """
    ensure_results_dir()
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        group_results_input.to_excel(writer, sheet_name="group_results_input", index=False)
        knockout_input.to_excel(writer, sheet_name="knockout_input", index=False)
        knockout_results.to_excel(writer, sheet_name="knockout_results", index=False)
        if projection is not None and not projection.empty:
            projection.to_excel(writer, sheet_name="projection", index=False)
        if bracket is not None and not bracket.empty:
            bracket.to_excel(writer, sheet_name="bracket", index=False)
        if bracket_probable is not None and not bracket_probable.empty:
            bracket_probable.to_excel(writer, sheet_name="bracket_probable", index=False)


def load_post_group_state(
    path: Path = POST_GROUP_STATE_PATH,
) -> dict[str, pd.DataFrame] | None:
    """Cargar el estado post-grupos previamente guardado."""
    if not path.exists():
        return None
    try:
        xl = pd.ExcelFile(path)
        state: dict[str, pd.DataFrame] = {}
        expected = [
            "group_results_input",
            "knockout_input",
            "knockout_results",
            "projection",
            "bracket",
            "bracket_probable",
        ]
        for sheet in expected:
            if sheet in xl.sheet_names:
                state[sheet] = pd.read_excel(xl, sheet_name=sheet)
        return state if state else None
    except Exception:
        return None
