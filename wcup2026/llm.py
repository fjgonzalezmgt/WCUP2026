"""Integracion con la API de OpenAI para analisis cualitativo del torneo.

Expone funciones auxiliares para verificar la disponibilidad de la clave
de API, construir el payload de analisis y ejecutar la llamada al modelo
de lenguaje.
"""

from __future__ import annotations

import json
import os
import re
from datetime import date
from typing import Any

import pandas as pd


LLM_INSTRUCTIONS = (
    "Eres un analista cuantitativo de futbol. Usa solo el contexto entregado. "
    "Separa claramente lo que sale del modelo de lo que son riesgos cualitativos. "
    "No inventes lesiones, convocatorias ni noticias. Responde en espanol claro, "
    "con bullets cortos y conclusiones firmes. Abre siempre con una seccion titulada "
    "'## Veredicto final', sin numeracion ni prefijos. En esa seccion entrega un bloque "
    "conclusivo de 8 a 10 bullets con etiquetas de este estilo: Favorito de mayor techo, "
    "Favoritos mas confiables hoy, Mejor perseguidor, Grande mas vulnerable, Outsider mas "
    "peligroso del modelo, Infravalorado mas solido, Equipo trampa, Candidato de sorpresa "
    "fuerte, Marca grande con mas volatilidad y Outsider que pierde fuerza por contexto "
    "fisico. Ese veredicto final debe mezclar desde el inicio lo que dice el modelo con lo "
    "que cambia por el contexto cualitativo; no presentes primero una lectura puramente "
    "estadistica y luego otra separada. Despues agrega un parrafo breve que empiece "
    "exactamente con 'Conclusion firme:' y sintetice la tesis central integrada. Luego "
    "explica como llegaste ahi en secciones separadas: primero '## Lo que dice el modelo', "
    "despues '## Ajustes cualitativos' y al final '## Escenarios'. "
    "No recomiendes ajustes de pesos, ratings ni variables internas del modelo; "
    "traduce cualquier senal cualitativa a implicaciones deportivas concretas y "
    "escenarios cerrados."
)


def _extract_response_text(response: Any) -> str:
    """Obtener texto legible de una respuesta de OpenAI Responses API.

    Algunas versiones del SDK exponen ``output_text`` vacio aunque el
    contenido exista dentro de ``response.output``. Esta funcion intenta
    ambas rutas para evitar respuestas invisibles en la UI.

    Parameters
    ----------
    response : Any
        Objeto de respuesta devuelto por ``client.responses.create``.

    Returns
    -------
    str
        Texto extraido y concatenado de la respuesta, o cadena vacia si
        no se encontro ningun fragmento de texto.
    """
    output_text = getattr(response, "output_text", None)
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    fragments: list[str] = []
    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            text_value = getattr(content, "text", None)
            if isinstance(text_value, str) and text_value.strip():
                fragments.append(text_value.strip())
                continue

            for part in getattr(content, "annotations", []) or []:
                annotation_text = getattr(part, "text", None)
                if isinstance(annotation_text, str) and annotation_text.strip():
                    fragments.append(annotation_text.strip())

    return "\n\n".join(fragments).strip()


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


def build_analysis_payload(
    results: pd.DataFrame,
    teams: pd.DataFrame,
    notes: str,
    bracket_probable: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Construir el payload JSON que se envia al LLM para el analisis.

    Selecciona los 12 equipos favoritos del modelo junto con los ratings
    base de todos los equipos, el escenario cualitativo del usuario y,
    opcionalmente, el cuadro de eliminacion mas probable.

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
    bracket_probable : pd.DataFrame or None, optional
        Cuadro de eliminacion mas probable calculado sobre N simulaciones.
        Si se proporciona, se incluye en el payload como ``bracket_probable``
        filtrando solo cuartos, semis y final para mantenerlo compacto.

    Returns
    -------
    dict[str, Any]
        Diccionario serializable a JSON con las claves
        ``model_results_top_12``, ``teams``, ``user_scenario``,
        ``bracket_probable`` (si aplica) y ``request``.
    """
    payload: dict[str, Any] = {
        "model_results_top_12": results.head(12)[
            ["team", "group", "overall", "round_of_32_pct", "semifinal_pct", "final_pct", "champion_pct"]
        ].round(2).to_dict(orient="records"),
        "teams": teams[
            ["team", "group", "confederation", "attack", "defense", "squad", "form"]
        ].to_dict(orient="records"),
        "user_scenario": notes,
    }

    if bracket_probable is not None and not bracket_probable.empty:
        late_rounds = {"quarterfinal", "semifinal", "final"}
        bp = bracket_probable[bracket_probable["round"].isin(late_rounds)].copy()
        bp["winner_pct"] = bp["winner_pct"].where(bp["winner_pct"].notna(), other=None)
        payload["bracket_probable"] = bp[
            ["round", "match_id", "team_a", "team_b", "winner", "winner_pct"]
        ].to_dict(orient="records")

    payload["request"] = (
            "Entrega un analisis mas concluyente y menos tecnico. "
            "Empieza obligatoriamente con una seccion markdown titulada '## Veredicto final', "
            "sin numeracion. En esa seccion entrega 8 a 10 bullets conclusivos con etiquetas "
            "claras del tipo del ejemplo: favorito de mayor techo, favoritos mas confiables hoy, "
            "mejor perseguidor, grande mas vulnerable, outsider mas peligroso del modelo, "
            "infravalorado mas solido, equipo trampa, candidato de sorpresa fuerte, marca grande "
            "con mas volatilidad y outsider que pierde fuerza por contexto fisico. "
            "Esos bullets deben incorporar ya la lectura combinada entre resultados del modelo y "
            "analisis cualitativo. "
            "Despues escribe un parrafo corto que empiece exactamente con 'Conclusion firme:' "
            "y cierre la tesis principal. Luego explica paso a paso como llegas a ese veredicto "
            "en tres secciones: '## Lo que dice el modelo', '## Ajustes cualitativos' y "
            "'## Escenarios'. En la ultima seccion cierra con los escenarios base, conservador "
            "y optimista. "
            "Si se incluye 'bracket_probable', incorpora sus caminos al titulo en el analisis: "
            "menciona que equipos llegan a cuartos, semis y la final segun el modelo, cuales son "
            "los duelos clave del cuadro y que tan probable es cada uno segun el porcentaje. "
            "No recomiendes ajustes de pesos, ratings, variables ni escalas numericas del modelo."
    )
    return payload


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
    return _extract_response_text(response)


def _extract_json(text: str) -> Any:
    """Extraer el primer objeto o array JSON valido de un texto.

    Intenta parseo directo, luego extrae de bloque de codigo markdown
    y finalmente busca el primer array u objeto JSON en el texto.

    Parameters
    ----------
    text : str
        Texto que contiene JSON (posiblemente con markdown o texto adicional).

    Returns
    -------
    Any
        Estructura Python resultante del parseo JSON.

    Raises
    ------
    ValueError
        Si no se encontro ningun JSON valido en el texto.
    """
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"```(?:json)?\s*([\s\S]+?)```", text)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass
    match = re.search(r"(\[[\s\S]+?\]|\{[\s\S]+?\})", text)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass
    raise ValueError("No se encontro JSON valido en la respuesta del LLM.")


def call_llm_ratings_update(model: str, teams: pd.DataFrame) -> pd.DataFrame:
    """Usar busqueda web para obtener ratings actualizados de todas las selecciones.

    Construye un prompt estructurado solicitando al LLM que busque en internet
    los datos mas recientes de Elo, ranking FIFA, ataque, defensa, plantilla
    y forma para cada seleccion participante.  Devuelve un DataFrame con los
    valores actualizados listo para fusionarse con el dataset base mediante
    ``apply_ratings_update``.

    Parameters
    ----------
    model : str
        Nombre del modelo OpenAI.
    teams : pd.DataFrame
        DataFrame con los equipos participantes (columna ``team`` requerida).

    Returns
    -------
    pd.DataFrame
        DataFrame con columnas ``team``, ``elo``, ``fifa_rank_proxy``,
        ``attack``, ``defense``, ``squad`` y ``form`` para cada equipo
        devuelto por el LLM.

    Raises
    ------
    ValueError
        Si la respuesta del LLM no contiene JSON parseable.
    openai.OpenAIError
        Si la llamada a la API falla.
    """
    from openai import OpenAI

    client = OpenAI()
    team_list = ", ".join(teams["team"].tolist())
    schema_example = json.dumps(
        {
            "ratings": [
                {
                    "team": "Argentina",
                    "elo": 2080,
                    "fifa_rank_proxy": 95,
                    "attack": 88,
                    "defense": 82,
                    "squad": 87,
                    "form": 80,
                }
            ]
        },
        ensure_ascii=False,
    )

    prompt = (
        f"Fecha: {date.today().isoformat()}. Copa del Mundo 2026.\n"
        f"Selecciones participantes: {team_list}.\n\n"
        "Busca en internet los datos mas recientes de cada seleccion y devuelve "
        "un JSON con este esquema exacto (sin texto adicional):\n"
        f"{schema_example}\n\n"
        "Para cada equipo incluye:\n"
        "- team: nombre exacto tal como aparece en la lista\n"
        "- elo: rating ELO actual (rango tipico 1400-2200)\n"
        "- fifa_rank_proxy: puntuacion de ranking FIFA en escala 0-100 (100 = mejor)\n"
        "- attack: potencial ofensivo 0-100 basado en estadisticas de goles y xG recientes\n"
        "- defense: solidez defensiva 0-100 basada en goles encajados y xGA recientes\n"
        "- squad: calidad del plantel 0-100 segun valor de mercado y profundidad de banquillo\n"
        "- form: forma reciente 0-100 basada en resultados de los ultimos 6 meses\n\n"
        "Devuelve SOLO el JSON valido, sin explicaciones, sin markdown, sin texto extra."
    )

    response = client.responses.create(
        model=model,
        tools=[
            {
                "type": "web_search",
                "search_context_size": "high",
                "user_location": {
                    "type": "approximate",
                    "country": "US",
                    "timezone": "America/Guatemala",
                },
            }
        ],
        tool_choice="required",
        instructions=(
            "Eres un analista cuantitativo de futbol. Usa busqueda web para obtener datos "
            "reales y actualizados. Devuelve UNICAMENTE un JSON valido con el esquema "
            "solicitado. No incluyas markdown, backticks ni texto adicional."
        ),
        input=prompt,
    )

    data = _extract_json(_extract_response_text(response))
    ratings_list = data.get("ratings", data) if isinstance(data, dict) else data
    return pd.DataFrame(ratings_list)


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
    return _extract_response_text(response)
