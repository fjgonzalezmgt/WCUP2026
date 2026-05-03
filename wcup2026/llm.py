"""Integracion con la API de OpenAI para analisis cualitativo del torneo.

Expone funciones auxiliares para verificar la disponibilidad de la clave
de API, construir el payload de analisis y ejecutar la llamada al modelo
de lenguaje.
"""

from __future__ import annotations

import json
import os
from datetime import date
from typing import Any

import pandas as pd


LLM_INSTRUCTIONS = (
    "Eres un analista cuantitativo de futbol. Usa solo el contexto entregado. "
    "Separa claramente lo que sale del modelo de lo que son riesgos cualitativos. "
    "No inventes lesiones, convocatorias ni noticias. Responde en espanol claro, "
    "con bullets cortos y recomendaciones accionables."
)


def api_key_available() -> bool:
    """Comprobar si la variable de entorno ``OPENAI_API_KEY`` esta definida.

    Returns
    -------
    bool
        ``True`` si la clave existe y no es vacia; ``False`` en caso contrario.
    """
    return bool(os.getenv("OPENAI_API_KEY"))


def default_model() -> str:
    """Obtener el nombre del modelo OpenAI configurado.

    Lee la variable de entorno ``OPENAI_MODEL``; si no esta definida
    devuelve ``"gpt-5"`` como valor predeterminado.

    Returns
    -------
    str
        Nombre del modelo OpenAI a usar en las llamadas a la API.
    """
    return os.getenv("OPENAI_MODEL", "gpt-5.5")


def build_analysis_payload(results: pd.DataFrame, teams: pd.DataFrame, notes: str) -> dict[str, Any]:
    """Construir el payload JSON que se envia al LLM para el analisis.

    Selecciona los 12 equipos favoritos del modelo junto con los ratings
    base de todos los equipos y el escenario cualitativo del usuario.

    Parameters
    ----------
    results : pd.DataFrame
        DataFrame de resultados de la simulacion ordenado por probabilidad
        de campeon (salida de ``simulate_many``).
    teams : pd.DataFrame
        DataFrame original con los ratings de los equipos.
    notes : str
        Texto libre del usuario con informacion cualitativa (lesiones,
        contexto, escenarios hipoteticos).

    Returns
    -------
    dict[str, Any]
        Diccionario serializable a JSON con las claves
        ``model_results_top_12``, ``teams``, ``user_scenario`` y
        ``request``.
    """
    return {
        "model_results_top_12": results.head(12)[
            ["team", "group", "overall", "round_of_32_pct", "semifinal_pct", "final_pct", "champion_pct"]
        ].round(2).to_dict(orient="records"),
        "teams": teams[
            ["team", "group", "confederation", "attack", "defense", "squad", "form"]
        ].to_dict(orient="records"),
        "user_scenario": notes,
        "request": (
            "Resume favoritos, riesgos del modelo y ajustes cualitativos sugeridos. "
            "Si propones cambiar variables, usa rangos pequenos en puntos: -5 a +5."
        ),
    }


def call_llm_news_search(model: str, teams: pd.DataFrame) -> str:
    """Usar el LLM para generar un resumen de noticias de lesiones y contexto relevante.

    Parameters
    ----------
    model : str
        Nombre del modelo OpenAI.
    teams : pd.DataFrame
        DataFrame con los equipos participantes.

    Returns
    -------
    str
        Resumen de noticias relevantes listo para pegar en el campo cualitativo.
    """
    from openai import OpenAI

    client = OpenAI()
    if "overall" in teams.columns:
        focus_teams = teams.sort_values("overall", ascending=False).head(18)
    else:
        focus_teams = teams.head(18)

    team_list = ", ".join(focus_teams["team"].tolist())
    all_teams = ", ".join(teams["team"].tolist())
    prompt = (
        f"Fecha de consulta: {date.today().isoformat()}.\n"
        f"Selecciones participantes en el Mundial 2026: {all_teams}.\n"
        f"Prioriza estas selecciones por rating/favoritismo del modelo: {team_list}.\n\n"
        "Busca en internet noticias recientes y relevantes para el Mundial 2026 sobre "
        "lesiones, bajas, suspensiones, convocatorias, minutos recientes, racha de forma "
        "y contexto competitivo de los principales favoritos al titulo. "
        "Usa informacion verificable y reciente; ignora rumores debiles o contenido sin "
        "fuente clara. Organiza por seleccion. "
        "Cada bullet debe incluir fecha aproximada y fuente o URL. "
        "Si no encuentras informacion reciente y confiable de una seleccion, dilo en una "
        "linea breve. Devuelve texto listo para pegar en el campo de analisis cualitativo."
    )
    response = client.responses.create(
        model=model,
        tools=[
            {
                "type": "web_search",
                "search_context_size": "medium",
                "user_location": {
                    "type": "approximate",
                    "country": "US",
                    "timezone": "America/Guatemala",
                },
            }
        ],
        tool_choice="required",
        include=["web_search_call.action.sources"],
        instructions=(
            "Eres un analista de futbol actualizado. Responde en espanol. "
            "Debes usar busqueda web antes de responder. No inventes datos especificos. "
            "No presentes una noticia, lesion o suspension como confirmada si la fuente "
            "no lo respalda. Incluye fuentes visibles para que el usuario pueda revisar."
        ),
        input=prompt,
    )
    return response.output_text


def call_llm_analysis(model: str, payload: dict[str, Any]) -> str:
    """Llamar a la API de OpenAI Responses y devolver el texto generado.

    Parameters
    ----------
    model : str
        Nombre del modelo OpenAI (p.ej. ``"gpt-5"`` o ``"gpt-4o"``).
    payload : dict[str, Any]
        Payload construido con ``build_analysis_payload``.

    Returns
    -------
    str
        Texto de respuesta del modelo.

    Raises
    ------
    openai.OpenAIError
        Si la llamada a la API falla (red, credenciales, cuota, etc.).
    """
    from openai import OpenAI

    client = OpenAI()
    response = client.responses.create(
        model=model,
        instructions=LLM_INSTRUCTIONS,
        input=json.dumps(payload, ensure_ascii=True),
    )
    return response.output_text
