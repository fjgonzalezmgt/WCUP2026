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

OPENAI_MODEL = "gpt-5.6-luna"


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
    """Obtener el modelo OpenAI unico de la aplicacion.

    El modelo se fija en codigo para que todas las acciones de IA usen la
    misma version y no puedan divergir por configuracion local o de sesion.

    Returns
    -------
    str
        Nombre del modelo OpenAI a usar en las llamadas a la API.
    """
    return OPENAI_MODEL


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
        ordenado desde ronda de 32 hasta la final.

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
        round_order = {
            "round_of_32": 0,
            "round_of_16": 1,
            "quarterfinal": 2,
            "semifinal": 3,
            "final": 4,
        }
        bp = bracket_probable.copy()
        bp["_round_order"] = bp["round"].map(round_order).fillna(99)
        bp = bp.sort_values(["_round_order", "match_id"]).drop(columns="_round_order")
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
            "menciona las llaves probables desde ronda de 32, octavos, cuartos, semis y final, "
            "cuales son los duelos clave del cuadro y que tan probable es cada uno segun el porcentaje. "
            "No recomiendes ajustes de pesos, ratings, variables ni escalas numericas del modelo."
    )
    return payload


def _surviving_teams_from_knockout_state(
    teams: pd.DataFrame,
    knockout_state: pd.DataFrame | None,
) -> pd.DataFrame:
    """Excluir equipos eliminados segun los resultados KO ya confirmados."""
    if knockout_state is None or knockout_state.empty:
        return teams.copy()

    required = ["team_a", "team_b", "winner"]
    if not set(required).issubset(knockout_state.columns):
        return teams.copy()

    eliminated: set[str] = set()
    for row in knockout_state[required].itertuples(index=False):
        team_a = str(row.team_a).strip()
        team_b = str(row.team_b).strip()
        winner = str(row.winner).strip()
        if winner and winner.lower() != "nan" and winner in {team_a, team_b}:
            eliminated.add(team_b if winner == team_a else team_a)

    return teams.loc[~teams["team"].isin(eliminated)].copy()


def call_llm_news_search(
    model: str,
    teams: pd.DataFrame,
    knockout_state: pd.DataFrame | None = None,
) -> str:
    """Usar el LLM para generar un resumen de noticias de lesiones y contexto relevante.

    Parameters
    ----------
    model : str
        Nombre del modelo OpenAI.
    teams : pd.DataFrame
        DataFrame con los equipos participantes.
    knockout_state : pd.DataFrame, optional
        Estado real de las eliminatorias. Sus perdedores confirmados se
        excluyen antes de construir la consulta.

    Returns
    -------
    str
        Resumen de noticias relevantes listo para pegar en el campo cualitativo.
    """
    from openai import OpenAI

    client = OpenAI()
    surviving_teams = _surviving_teams_from_knockout_state(teams, knockout_state)
    if "overall" in surviving_teams.columns:
        focus_teams = surviving_teams.sort_values("overall", ascending=False).head(18)
    else:
        focus_teams = surviving_teams.head(18)

    team_list = ", ".join(focus_teams["team"].tolist())
    all_teams = ", ".join(teams["team"].tolist())
    known_survivors = ", ".join(surviving_teams["team"].tolist())
    prompt = (
        f"Fecha de consulta: {date.today().isoformat()}.\n"
        f"Selecciones participantes en el Mundial 2026: {all_teams}.\n"
        f"Equipos que siguen vivos segun los resultados cargados: {known_survivors}.\n"
        f"Entre ellos, prioriza por rating/favoritismo del modelo: {team_list}.\n\n"
        "Primero verifica con fuentes fiables y resultados actualizados cuales de esas "
        "selecciones siguen vivas en el torneo en la fecha de consulta. Considera vivo a "
        "un equipo mientras conserve posibilidades de avanzar o no haya sido eliminado. "
        "No incluyas noticias de selecciones ya eliminadas. "
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


def call_llm_group_results_update(model: str, teams: pd.DataFrame) -> pd.DataFrame:
    """Usar busqueda web para obtener la tabla actual/final de fase de grupos.

    Devuelve una fila por equipo con grupo, posicion, puntos, goles a favor
    y goles en contra. Si la fase de grupos aun no ha terminado, usa la tabla
    oficial mas reciente disponible.
    """
    from openai import OpenAI

    client = OpenAI()
    clean = teams[["team", "group"]].copy()
    clean["group"] = clean["group"].astype(str).str.upper().str.strip()
    team_records = clean.sort_values(["group", "team"]).to_dict(orient="records")
    schema_example = json.dumps(
        {
            "group_results": [
                {
                    "group": "A",
                    "position": 1,
                    "team": "Mexico",
                    "points": 7,
                    "gf": 5,
                    "ga": 2,
                }
            ]
        },
        ensure_ascii=False,
    )

    prompt = (
        f"Fecha de consulta: {date.today().isoformat()}.\n"
        "Necesito actualizar la tabla de fase de grupos del Mundial FIFA 2026 "
        "para alimentar un simulador de eliminatorias.\n\n"
        "Equipos y grupos esperados, con nombres exactos que debes conservar:\n"
        f"{json.dumps(team_records, ensure_ascii=False)}\n\n"
        "Busca en la web standings/resultados oficiales o fuentes confiables "
        "del Mundial 2026. Si la fase de grupos ya termino, devuelve la tabla final. "
        "Si aun esta en curso, devuelve la tabla actual mas reciente disponible. "
        "Ordena cada grupo por la posicion oficial publicada; si una fuente no da "
        "desempates completos, usa puntos, diferencia de goles, goles a favor y, "
        "solo al final, el orden de la fuente.\n\n"
        "Devuelve SOLO un JSON valido con este esquema exacto, sin markdown ni texto extra:\n"
        f"{schema_example}\n\n"
        "Reglas estrictas:\n"
        "- group_results debe tener 48 filas, una por cada equipo de la lista.\n"
        "- group debe ser una letra A-L.\n"
        "- position debe ser 1, 2, 3 o 4 dentro de cada grupo.\n"
        "- team debe coincidir exactamente con uno de los nombres entregados.\n"
        "- points, gf y ga deben ser enteros no negativos.\n"
        "- No inventes equipos ni cambies nombres."
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
        include=["web_search_call.action.sources"],
        instructions=(
            "Eres un analista de resultados FIFA. Debes usar busqueda web antes de "
            "responder. Prioriza FIFA.com y fuentes deportivas confiables. Devuelve "
            "UNICAMENTE JSON valido con el esquema solicitado. No incluyas explicaciones, "
            "citas, markdown ni texto adicional."
        ),
        input=prompt,
    )

    data = _extract_json(_extract_response_text(response))
    rows = data.get("group_results", data) if isinstance(data, dict) else data
    result = pd.DataFrame(rows)
    expected_columns = ["group", "position", "team", "points", "gf", "ga"]
    missing = set(expected_columns).difference(result.columns)
    if missing:
        raise ValueError(
            "La respuesta de grupos no incluyo columnas requeridas: "
            + ", ".join(sorted(missing))
        )
    result = result[expected_columns].copy()
    for column in ["position", "points", "gf", "ga"]:
        result[column] = pd.to_numeric(result[column], errors="coerce").fillna(0).astype(int)
    result["group"] = result["group"].astype(str).str.upper().str.strip()
    result["team"] = result["team"].astype(str).str.strip()
    return result.sort_values(["group", "position"]).reset_index(drop=True)


def call_llm_knockout_results_update(model: str, fixtures: pd.DataFrame) -> pd.DataFrame:
    """Usar busqueda web para obtener ganadores confirmados de eliminatorias.

    Parameters
    ----------
    model : str
        Nombre del modelo OpenAI.
    fixtures : pd.DataFrame
        Tabla de llaves con columnas ``match_id``, ``round``, ``team_a`` y
        ``team_b``. Se usa para restringir las respuestas a partidos y equipos
        validos del simulador.

    Returns
    -------
    pd.DataFrame
        DataFrame con columnas ``match_id`` y ``winner`` para partidos ya
        jugados y confirmados.
    """
    from openai import OpenAI

    client = OpenAI()
    required = {"match_id", "round", "team_a", "team_b"}
    missing = required.difference(fixtures.columns)
    if missing:
        raise ValueError(
            "La tabla de fixtures no incluye columnas requeridas: "
            + ", ".join(sorted(missing))
        )

    fixture_columns = ["match_id", "round", "team_a", "team_b"]
    if "winner" in fixtures.columns:
        fixture_columns.append("winner")
    valid_rows = fixtures[fixture_columns].copy()
    valid_rows["match_id"] = pd.to_numeric(valid_rows["match_id"], errors="coerce").fillna(0).astype(int)
    valid_rows["round"] = valid_rows["round"].astype(str).str.strip()
    valid_rows["team_a"] = valid_rows["team_a"].astype(str).str.strip()
    valid_rows["team_b"] = valid_rows["team_b"].astype(str).str.strip()
    if "winner" in valid_rows.columns:
        valid_rows["winner"] = valid_rows["winner"].where(valid_rows["winner"].notna(), "")
        valid_rows["winner"] = valid_rows["winner"].astype(str).str.strip()
    valid_rows = valid_rows.sort_values("match_id")

    schema_example = json.dumps(
        {
            "knockout_results": [
                {
                    "match_id": 73,
                    "winner": "South Africa",
                }
            ]
        },
        ensure_ascii=False,
    )

    prompt = (
        f"Fecha de consulta: {date.today().isoformat()} en America/Guatemala.\n"
        "Busca resultados oficiales/confirmados de las rondas eliminatorias del Mundial FIFA 2026.\n"
        "Prioriza resultados publicados o actualizados hoy y en las ultimas 24 horas; "
        "no te quedes con previas, calendarios ni marcadores antiguos si ya hay un resultado final.\n"
        "Debes usar UNICAMENTE los partidos/equipos de esta tabla de llaves. "
        "Si la tabla ya trae ganadores en la columna winner, usalos para interpretar "
        "los cruces de la siguiente fase y busca tambien esos partidos derivados:\n"
        f"{json.dumps(valid_rows.to_dict(orient='records'), ensure_ascii=False)}\n\n"
        "Devuelve SOLO JSON valido con este esquema exacto:\n"
        f"{schema_example}\n\n"
        "Reglas estrictas:\n"
        "- Revisa FIFA.com primero y luego fuentes deportivas confiables con hora/fecha reciente.\n"
        "- Incluye solo partidos ya jugados y con ganador confirmado, incluyendo partidos jugados hoy.\n"
        "- winner debe ser exactamente team_a o team_b del match_id correspondiente.\n"
        "- Si team_a/team_b ya son equipos concretos, busca ese cruce especifico por ambos nombres.\n"
        "- Si un partido no esta jugado o no hay confirmacion fiable, NO lo incluyas.\n"
        "- No inventes partidos, no inventes equipos, no cambies nombres.\n"
        "- No incluyas texto fuera del JSON."
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
        include=["web_search_call.action.sources"],
        instructions=(
            "Eres un analista de resultados FIFA. Debes usar busqueda web antes de responder. "
            "Prioriza FIFA.com y fuentes deportivas confiables. Devuelve UNICAMENTE JSON valido "
            "con el esquema solicitado, sin markdown ni explicaciones."
        ),
        input=prompt,
    )

    data = _extract_json(_extract_response_text(response))
    rows = data.get("knockout_results", data) if isinstance(data, dict) else data
    result = pd.DataFrame(rows)
    if result.empty:
        return pd.DataFrame(columns=["match_id", "winner"])

    expected = {"match_id", "winner"}
    missing_cols = expected.difference(result.columns)
    if missing_cols:
        raise ValueError(
            "La respuesta de eliminatorias no incluyo columnas requeridas: "
            + ", ".join(sorted(missing_cols))
        )

    result = result[["match_id", "winner"]].copy()
    result["match_id"] = pd.to_numeric(result["match_id"], errors="coerce").fillna(0).astype(int)
    result["winner"] = result["winner"].astype(str).str.strip()
    result = result.loc[(result["match_id"] > 0) & (result["winner"] != "")]
    return result.sort_values("match_id").reset_index(drop=True)


def call_llm_analysis(model: str, payload: dict[str, Any]) -> str:
    """Llamar a la API de OpenAI Responses y devolver el texto generado.

    Parameters
    ----------
    model : str
        Nombre del modelo OpenAI (``"gpt-5.6-luna"`` en esta aplicacion).
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
