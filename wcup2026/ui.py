"""Componentes de la interfaz de usuario Streamlit.

Contiene todas las funciones de renderizado de la aplicacion web:
configuracion de pagina, barra lateral de parametros, pestanas de
prediccion, grupos, descripcion del modelo, integracion LLM y editor
de datos.  El punto de entrada principal es ``main()`` llamado desde
``app.py``.
"""

from __future__ import annotations

import html
import importlib
import json
from io import BytesIO
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv

from wcup2026.config import (
    APP_CAPTION,
    APP_TITLE,
    CHART_COLORS,
    DATA_PATH,
    FIFA_GROUPS_URL,
    FIFA_SCHEDULE_URL,
    OPENAI_RESPONSES_URL,
)
from wcup2026.data import (
    apply_ratings_update,
    dataframe_from_csv_text,
    dataframe_to_csv_text,
    load_team_data,
    validate_team_data,
)
from wcup2026.llm import (
    api_key_available,
    build_analysis_payload,
    call_llm_analysis,
    call_llm_news_search,
    call_llm_ratings_update,
    default_model,
)
from wcup2026.parameters import SimParams
from wcup2026.persistence import (
    load_bracket,
    load_bracket_probable,
    load_llm_analysis,
    load_montecarlo_results,
    save_llm_analysis,
    save_montecarlo_results,
)
import wcup2026.report as report_module
from wcup2026.simulator import describe_matchup, simulate_bracket_most_probable, simulate_bracket_sample, simulate_many


def configure_page() -> None:
    """Configurar la pagina Streamlit y cargar variables de entorno.

    Llama a ``st.set_page_config`` con titulo e icono de la app, carga
    el archivo ``.env`` via dotenv e inyecta los estilos CSS personalizados.
    Debe invocarse como primera instruccion Streamlit del script.

    Returns
    -------
    None
        Configura Streamlit y no devuelve ningun valor.
    """
    st.set_page_config(page_title=APP_TITLE, page_icon="WC26", layout="wide")
    load_dotenv()
    inject_style()


def inject_style() -> None:
    """Inyectar CSS personalizado en la pagina via ``st.markdown``.

    Ajusta padding del contenedor principal, estilo de las metricas,
    tipografia de encabezados y clases utilitarias ``.source-line`` y
    ``.small-note``.

    Returns
    -------
    None
        Escribe estilos CSS en la pagina actual de Streamlit.
    """
    st.markdown(
        """
        <style>
        .block-container {padding-top: 1.25rem; padding-bottom: 2rem;}
        [data-testid="stMetric"] {
            border: 1px solid #3a3a3a;
            border-radius: 8px;
            padding: 12px 14px;
        }
        h1, h2, h3 {letter-spacing: 0;}
        .source-line {
            color: #5d6759;
            font-size: 0.88rem;
        }
        .small-note {
            color: #5d6759;
            font-size: 0.92rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_copy_button(text: str, key: str) -> None:
    """Renderizar un boton HTML para copiar texto al portapapeles del navegador.

    Inyecta un componente HTML con JavaScript que llama a
    ``navigator.clipboard.writeText`` al hacer clic.  El ``key`` debe ser
    unico en la pagina para evitar colisiones de IDs.

    Parameters
    ----------
    text : str
        Contenido que se copiara al portapapeles al pulsar el boton.
    key : str
        Sufijo unico que se usa para construir los IDs del boton y el
        indicador de estado en el DOM.
    """
    button_id = f"copy-llm-{key}"
    status_id = f"copy-llm-status-{key}"
    payload = json.dumps(text)
    button_label = html.escape("Copiar analisis")
    st.iframe(
            f"""
            <div style="display:flex; justify-content:flex-end; margin:0 0 0.5rem 0;">
                <button
                    id="{button_id}"
                    type="button"
                    style="border:1px solid #d0d7de; border-radius:0.5rem; background:#f6f8fa; color:#24292f; padding:0.4rem 0.8rem; font-size:0.9rem; cursor:pointer;"
                >
                    {button_label}
                </button>
                <span id="{status_id}" style="margin-left:0.5rem; font-size:0.85rem; color:#57606a;"></span>
            </div>
            <script>
            const copyButton = document.getElementById({json.dumps(button_id)});
            const status = document.getElementById({json.dumps(status_id)});
            const text = {payload};

            copyButton?.addEventListener('click', async () => {{
                try {{
                    await navigator.clipboard.writeText(text);
                    if (status) {{
                        status.textContent = 'Copiado';
                        setTimeout(() => {{ status.textContent = ''; }}, 2000);
                    }}
                }} catch (error) {{
                    if (status) {{
                        status.textContent = 'No se pudo copiar';
                    }}
                }}
            }});
            </script>
            """,
            height=42,
    )


@st.cache_data(show_spinner=False)
def load_default_data_cached() -> pd.DataFrame:
    """Cargar los datos de equipos predeterminados con cache de Streamlit.

    Returns
    -------
    pd.DataFrame
        DataFrame leido desde ``DATA_PATH``; el resultado queda cacheado
        para evitar lecturas de disco repetidas.
    """
    return load_team_data()


@st.cache_data(show_spinner=False)
def run_bracket_sample_cached(csv_text: str, params: SimParams) -> pd.DataFrame:
    """Generar el cuadro de eliminacion de una simulacion representativa con cache.

    Envuelve ``simulate_bracket_sample`` con cache de Streamlit.  El cache
    se invalida cuando cambia ``csv_text`` o ``params``.

    Parameters
    ----------
    csv_text : str
        Datos de equipos serializados como CSV (hace hashable el DataFrame).
    params : SimParams
        Hiperparametros de la simulacion.

    Returns
    -------
    pd.DataFrame
        Partidos de eliminatoria (salida de ``simulate_bracket_sample``).
    """
    df = dataframe_from_csv_text(csv_text)
    validate_team_data(df)
    return simulate_bracket_sample(df, params)


@st.cache_data(show_spinner=False)
def run_bracket_probable_cached(csv_text: str, params: SimParams) -> pd.DataFrame:
    """Generar el cuadro mas probable con cache.

    Envuelve ``simulate_bracket_most_probable`` con cache de Streamlit.  El
    cache se invalida cuando cambia ``csv_text`` o ``params``.

    Parameters
    ----------
    csv_text : str
        Datos de equipos serializados como CSV (hace hashable el DataFrame).
    params : SimParams
        Hiperparametros de la simulacion.

    Returns
    -------
    pd.DataFrame
        Cuadro mas probable calculado sobre 1000 simulaciones (salida de
        ``simulate_bracket_most_probable``).
    """
    df = dataframe_from_csv_text(csv_text)
    validate_team_data(df)
    return simulate_bracket_most_probable(df, params, n=1000)


@st.cache_data(show_spinner="Simulando torneos...")
def run_simulation_cached(csv_text: str, params: SimParams) -> pd.DataFrame:
    """Ejecutar la simulacion con cache de Streamlit.

    El cache se invalida cuando cambia ``csv_text`` o ``params``, por lo
    que cualquier edicion en los ratings o los controles laterales
    desencadena una nueva simulacion.

    Parameters
    ----------
    csv_text : str
        Datos de equipos serializados como CSV (hace hashable el DataFrame).
    params : SimParams
        Hiperparametros de la simulacion.

    Returns
    -------
    pd.DataFrame
        Resultados de la simulacion (salida de ``simulate_many``).
    """
    df = dataframe_from_csv_text(csv_text)
    validate_team_data(df)
    return simulate_many(df, params)


def build_results_export_csv(results: pd.DataFrame) -> str:
    """Construir el CSV de probabilidades finales en formato portable.

    Convierte las columnas internas en porcentaje a probabilidades 0-1 y
    conserva una fila por cada seleccion simulada.

    Parameters
    ----------
    results : pd.DataFrame
        DataFrame de resultados de la simulacion (salida de
        ``simulate_many``) con columnas ``team``, ``champion_pct``,
        ``final_pct`` y ``semifinal_pct``.

    Returns
    -------
    str
        Texto CSV con columnas ``team``, ``prob_champion``, ``prob_final``
        y ``prob_semifinal`` (probabilidades 0-1 con cuatro decimales).
    """
    export = results[
        ["team", "champion_pct", "final_pct", "semifinal_pct"]
    ].rename(
        columns={
            "champion_pct": "prob_champion",
            "final_pct": "prob_final",
            "semifinal_pct": "prob_semifinal",
        }
    ).copy()
    for column in ["prob_champion", "prob_final", "prob_semifinal"]:
        export[column] = export[column] / 100
    return export.to_csv(index=False, float_format="%.4f")


def load_persisted_outputs_once() -> None:
    """Cargar resultados guardados en disco una sola vez por sesion de Streamlit.

    Usa la clave ``_persisted_outputs_loaded`` del ``st.session_state`` como
    centinela para evitar lecturas repetidas en re-runs.  Inicializa
    ``simulation_results``, ``simulation_df`` y ``llm_answer`` si existen
    datos guardados y aun no hay valores en la sesion.

    Returns
    -------
    None
        Actualiza ``st.session_state`` con datos persistidos si existen.
    """
    if st.session_state.get("_persisted_outputs_loaded"):
        return

    try:
        persisted = load_montecarlo_results()
        if persisted is not None:
            results, teams = persisted
            st.session_state.setdefault("simulation_results", results)
            st.session_state.setdefault("simulation_df", teams)
    except Exception as exc:  # pragma: no cover - UI guardrail
        st.warning(f"No se pudo cargar la simulacion guardada: {exc}")

    try:
        bracket = load_bracket()
        if bracket is not None:
            st.session_state.setdefault("bracket", bracket)
    except Exception as exc:  # pragma: no cover - UI guardrail
        st.warning(f"No se pudo cargar el cuadro guardado: {exc}")

    try:
        bracket_probable = load_bracket_probable()
        if bracket_probable is not None:
            st.session_state.setdefault("bracket_probable", bracket_probable)
    except Exception as exc:  # pragma: no cover - UI guardrail
        st.warning(f"No se pudo cargar el cuadro mas probable guardado: {exc}")

    try:
        llm_answer = load_llm_analysis()
        if llm_answer:
            st.session_state.setdefault("llm_answer", llm_answer)
    except Exception as exc:  # pragma: no cover - UI guardrail
        st.warning(f"No se pudo cargar el analisis LLM guardado: {exc}")

    st.session_state["_persisted_outputs_loaded"] = True


def render_sidebar() -> tuple[SimParams, str]:
    """Renderizar la barra lateral con todos los controles del motor.

    Expone sliders y campos para numero de simulaciones, semilla, goles
    base, ventaja de anfitrion, ruido en eliminatorias y pesos de los
    componentes del rating.  Tambien muestra la configuracion del LLM.

    Returns
    -------
    tuple[SimParams, str]
        ``(params, model)`` donde ``params`` es el ``SimParams`` construido
        con los valores de los controles y ``model`` es el nombre del
        modelo OpenAI configurado.
    """
    st.sidebar.header("Motor")
    simulations = st.sidebar.slider("Simulaciones", 500, 50000, 5000, step=500, help="Numero de torneos simulados via Monte Carlo. Mas simulaciones = mayor precision, pero mas lento.")
    seed = st.sidebar.number_input("Semilla", min_value=1, value=2026, step=1, help="Semilla aleatoria para reproducibilidad. El mismo valor siempre genera los mismos resultados.")
    base_goals = st.sidebar.slider("Goles base por equipo", 0.8, 2.2, 1.35, step=0.05, help="Promedio de goles esperados por equipo en un partido completamente neutral. Ajusta el ritmo ofensivo global del torneo.")
    home_advantage = st.sidebar.slider("Ventaja anfitrion", 0.0, 0.25, 0.10, step=0.01, help="Multiplicador adicional de goles para los equipos anfitriones (USA, Canada, Mexico). 0.10 = +10% de goles esperados.")
    knockout_noise = st.sidebar.slider("Ruido en eliminatorias", 8.0, 30.0, 18.0, step=0.5, help="Factor de aleatoriedad en partidos eliminatorios. Valores altos favorecen sorpresas; valores bajos privilegian al mejor equipo.")

    st.sidebar.header("Pesos")
    elo_weight = st.sidebar.slider("Elo / ranking", 0.0, 1.0, 0.45, step=0.05, help="Peso del Elo/ranking FIFA en el rating compuesto de cada seleccion. Refleja historial y nivel competitivo acumulado.")
    squad_weight = st.sidebar.slider("Plantilla", 0.0, 1.0, 0.25, step=0.05, help="Peso de la calidad del plantel (valor de mercado, profundidad). Equipos con mejor banco aguantan mejor el torneo.")
    form_weight = st.sidebar.slider("Forma reciente", 0.0, 1.0, 0.15, step=0.05, help="Peso de los resultados recientes (ultimos 6-12 meses). Captura momentum y dinamica actual del equipo.")
    balance_weight = st.sidebar.slider("Ataque + defensa", 0.0, 1.0, 0.15, step=0.05, help="Peso del balance tactico (ataque vs defensa). Equipos con ratings altos en ambos extremos son mas solidos.")

    st.sidebar.header("LLM")
    model = st.sidebar.text_input("Modelo OpenAI", value=default_model(), help="Nombre del modelo de OpenAI a usar para el analisis cualitativo y busqueda de noticias. Ej: gpt-4o, gpt-5.5.")
    key_status = "detectada" if api_key_available() else "no detectada"
    st.sidebar.caption(f"OPENAI_API_KEY: {key_status}")

    params = SimParams(
        simulations=int(simulations),
        seed=int(seed),
        base_goals=float(base_goals),
        home_advantage=float(home_advantage),
        knockout_noise=float(knockout_noise),
        elo_weight=float(elo_weight),
        squad_weight=float(squad_weight),
        form_weight=float(form_weight),
        balance_weight=float(balance_weight),
    )
    return params, model


def render_probability_view(results: pd.DataFrame) -> None:
    """Renderizar la pestana de prediccion con metricas, grafico y tabla.

    Muestra cuatro metricas del favorito, un grafico de barras horizontales
    con los 12 equipos mas probables campeones coloreados por confederacion
    y una tabla completa con todas las probabilidades por etapa.

    Parameters
    ----------
    results : pd.DataFrame
        DataFrame de resultados ordenado por probabilidad de campeon
        (salida de ``simulate_many``).
    """
    top = results.head(12).copy()
    favorite = results.iloc[0]
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Favorito", favorite["team"], f"{favorite['champion_pct']:.1f}% campeon", help="Equipo con mayor probabilidad de ganar el torneo segun la simulacion Monte Carlo.")
    col2.metric("Final", favorite["team"], f"{favorite['final_pct']:.1f}%", help="Probabilidad de que el favorito llegue a la final del torneo.")
    col3.metric("Semifinal", favorite["team"], f"{favorite['semifinal_pct']:.1f}%", help="Probabilidad de que el favorito alcance al menos las semifinales.")
    col4.metric("Rating modelo", f"{favorite['overall']:.1f}", "0-100 relativo", help="Rating compuesto del favorito en escala 0-100. Combina Elo, plantilla, forma y balance tactico segun los pesos configurados.")

    sorted_top = top.sort_values("champion_pct")
    fig = px.bar(
        sorted_top,
        x="champion_pct",
        y="team",
        color="confederation",
        orientation="h",
        text=sorted_top["champion_pct"].map(lambda value: f"{value:.1f}%"),
        labels={"champion_pct": "Probabilidad de campeon", "team": "Seleccion"},
        color_discrete_sequence=CHART_COLORS,
    )
    fig.update_layout(height=520, margin=dict(l=10, r=10, t=20, b=10))
    fig.update_traces(textposition="outside", cliponaxis=False)
    st.plotly_chart(fig, width="stretch")

    st.download_button(
        "Exportar resultados CSV",
        data=build_results_export_csv(results),
        file_name="resultados_wcup2026.csv",
        mime="text/csv",
        help="Descarga todas las selecciones con columnas team, prob_champion, prob_final y prob_semifinal en escala 0-1.",
    )

    st.dataframe(
        results[
            [
                "team",
                "group",
                "confederation",
                "overall",
                "group_winner_pct",
                "round_of_32_pct",
                "round_of_16_pct",
                "quarterfinal_pct",
                "semifinal_pct",
                "final_pct",
                "champion_pct",
            ]
        ].round(2),
        width="stretch",
        hide_index=True,
    )


def render_group_view(df: pd.DataFrame, results: pd.DataFrame) -> None:
    """Renderizar la pestana de analisis por grupo.

    Permite seleccionar un grupo, muestra la tabla de ratings y
    probabilidades de sus equipos, y ofrece un comparador de duelo
    directo entre dos selecciones del grupo con metricas de xG y
    probabilidades de victoria/empate.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame original de equipos con ratings.
    results : pd.DataFrame
        DataFrame de resultados de la simulacion.
    """
    group = st.selectbox("Grupo", sorted(df["group"].unique()), help="Selecciona el grupo para ver la tabla de ratings y el comparador de duelos directos.")
    merged = df.merge(
        results[["team", "overall", "round_of_32_pct", "champion_pct"]],
        on="team",
        how="left",
    )
    view = merged.loc[merged["group"] == group].sort_values("overall", ascending=False)
    st.dataframe(
        view[
            [
                "team",
                "confederation",
                "elo",
                "attack",
                "defense",
                "squad",
                "form",
                "overall",
                "round_of_32_pct",
                "champion_pct",
            ]
        ].round(2),
        width="stretch",
        hide_index=True,
    )

    teams = view["team"].tolist()
    col1, col2 = st.columns(2)
    team_a = col1.selectbox("Equipo A", teams, index=0, help="Primera seleccion del duelo directo simulado.")
    team_b = col2.selectbox("Equipo B", teams, index=min(1, len(teams) - 1), help="Segunda seleccion del duelo directo simulado.")
    if team_a != team_b:
        params = st.session_state["params"]
        matchup = describe_matchup(team_a, team_b, df, params, samples=6000)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(f"Gana {team_a}", f"{matchup['team_a_win_pct']:.1f}%", help=f"Probabilidad de victoria de {team_a} en 6000 simulaciones del partido.")
        c2.metric("Empate", f"{matchup['draw_pct']:.1f}%", help="Probabilidad de empate al final del tiempo reglamentario.")
        c3.metric(f"Gana {team_b}", f"{matchup['team_b_win_pct']:.1f}%", help=f"Probabilidad de victoria de {team_b} en 6000 simulaciones del partido.")
        c4.metric("xG", f"{matchup['team_a_xg']:.2f} - {matchup['team_b_xg']:.2f}", help="Goles esperados promedio (xG) para cada equipo segun sus ratings de ataque y defensa.")


def render_model_view() -> None:
    """Renderizar la pestana de descripcion del modelo.

    Muestra texto explicativo sobre el funcionamiento del simulador,
    criterios de desempate, logica de eliminatorias y enlaces a las
    fuentes de datos oficiales de FIFA y OpenAI.

    Returns
    -------
    None
        Renderiza contenido informativo en la pagina Streamlit.
    """
    st.subheader("Como piensa el modelo")
    st.markdown(
        """
        - Cada seleccion tiene rating de ataque, defensa, plantilla, forma y Elo aproximado.
        - Cada partido se simula con goles Poisson a partir de ataque vs defensa y rating global.
        - La fase de grupos usa puntos, diferencia de goles, goles a favor y rating como desempate final.
        - Avanzan dos primeros de cada grupo y los ocho mejores terceros.
        - La ronda de 32 sigue el calendario FIFA; la asignacion de terceros usa las ventanas de grupos publicadas por FIFA.
        - Las eliminatorias se deciden por marcador simulado; si hay empate, se usa una probabilidad de prorroga/penales basada en rating.
        """
    )
    st.markdown(
        f"""
        <div class="source-line">
        Fuentes base:
        <a href="{FIFA_GROUPS_URL}" target="_blank">grupos FIFA</a>,
        <a href="{FIFA_SCHEDULE_URL}" target="_blank">calendario FIFA</a>,
        <a href="{OPENAI_RESPONSES_URL}" target="_blank">OpenAI Responses API</a>.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.info(
        "Los ratings del CSV son semillas editables, no una verdad oficial. La gracia del MVP es que puedas calibrarlos con datos mejores: Elo actualizado, lesiones, valor de mercado, minutos jugados y resultados recientes."
    )


def render_llm_view(results: pd.DataFrame, df: pd.DataFrame, model: str) -> None:
    """Renderizar la pestana de integracion LLM.

    Muestra ideas de uso del LLM, un area de texto para ingresar
    informacion cualitativa y un boton para generar el analisis.  Si la
    clave de API no esta disponible muestra una advertencia en su lugar.

    Parameters
    ----------
    results : pd.DataFrame
        DataFrame de resultados de la simulacion.
    df : pd.DataFrame
        DataFrame original de equipos con ratings.
    model : str
        Nombre del modelo OpenAI a usar en la llamada.
    """
    st.subheader("LLM en la ecuacion")

    if not api_key_available():
        st.warning("No detecte OPENAI_API_KEY en el entorno. Revisa que el archivo .env este en esta carpeta.")
        return

    if st.button("Buscar noticias", help="Consulta al LLM por lesiones y contexto relevante de los equipos"):
        try:
            with st.spinner("Buscando noticias relevantes..."):
                st.session_state["llm_notes"] = call_llm_news_search(model, df)
        except Exception as exc:  # pragma: no cover - UI guardrail
            st.error(f"No se pudo obtener noticias: {exc}")

    notes = st.text_area(
        "Escenario o informacion cualitativa",
        placeholder="Ejemplo: Francia llega sin su lateral titular; Mexico juega con presion local; Uruguay trae buena racha defensiva.",
        height=180,
        key="llm_notes",
        help="Agrega contexto que el modelo cuantitativo no captura: lesiones, suspensiones, clima, rivalidades, rotaciones o escenarios hipoteticos. Puedes editarlo libremente despues de usar 'Buscar noticias'.",
    )

    if st.button("Generar analisis LLM", type="primary", help="Envia los resultados de la simulacion y el escenario cualitativo al LLM para obtener un analisis integrado con recomendaciones accionables."):
        if results is None:
            st.warning("Primero pulsa **Simular torneo** para tener resultados que analizar.")
        else:
            payload = build_analysis_payload(
                results, df, notes,
                bracket_probable=st.session_state.get("bracket_probable"),
            )
            try:
                with st.spinner("Consultando al LLM..."):
                    st.session_state["llm_answer"] = call_llm_analysis(model, payload).strip()
                    save_llm_analysis(st.session_state["llm_answer"])
            except Exception as exc:  # pragma: no cover - UI guardrail
                st.error(f"No se pudo consultar el LLM: {exc}")

    if "llm_answer" in st.session_state:
        answer = st.session_state["llm_answer"]
        if answer:
            render_copy_button(answer, key="analysis")
            st.markdown(answer)
        else:
            st.warning("El LLM respondio sin texto visible. Revisa el modelo configurado o intenta de nuevo.")


def render_report_view(results: pd.DataFrame | None, df: pd.DataFrame) -> None:
    """Renderizar la pestana de generacion del reporte LaTeX/PDF.

    Muestra un boton para generar el archivo TEX y compilar el PDF.  Si
    aun no hay simulacion realizada, muestra un mensaje informativo.
    Tras la compilacion exitosa ofrece botones de descarga para el PDF
    y el TEX.

    Parameters
    ----------
    results : pd.DataFrame or None
        DataFrame de resultados de la simulacion.  Si es ``None``, se
        muestra un aviso pidiendo simular primero.
    df : pd.DataFrame
        DataFrame original de equipos con ratings.
    """
    st.subheader("Reporte")
    st.markdown(
        '<div class="small-note">Genera reporte/reporte_wcup2026.tex desde el template con placeholders y compila el PDF con pdflatex.</div>',
        unsafe_allow_html=True,
    )

    if results is None:
        st.info("Pulsa **Simular torneo** antes de generar el reporte.")
        return

    if st.button("Generar reporte", type="primary", help="Crea el archivo LaTeX final y compila reporte_wcup2026.pdf con pdflatex."):
        try:
            with st.spinner("Generando reporte LaTeX y compilando PDF..."):
                fresh_report_module = importlib.reload(report_module)
                tex_path, pdf_path = fresh_report_module.generate_report(
                    results=results,
                    teams=df,
                    llm_text=st.session_state.get("llm_answer"),
                    params=st.session_state.get("params"),
                    bracket_probable=st.session_state.get("bracket_probable"),
                    compile_pdf=True,
                )
            st.success(f"Reporte generado: {tex_path.name}")
            if pdf_path is not None:
                st.caption(f"PDF compilado: {pdf_path}")
                st.download_button(
                    "Descargar PDF",
                    data=pdf_path.read_bytes(),
                    file_name=pdf_path.name,
                    mime="application/pdf",
                )
            st.download_button(
                "Descargar TEX",
                data=tex_path.read_text(encoding="utf-8"),
                file_name=tex_path.name,
                mime="application/x-tex",
            )
        except Exception as exc:
            st.error(f"No se pudo generar el reporte: {exc}")


def render_data_editor(default_df: pd.DataFrame, model: str) -> pd.DataFrame:
    """Renderizar el editor de ratings con actualizacion via IA.

    Muestra un boton para obtener ratings actualizados usando busqueda web
    a traves del LLM, y permite editar directamente los valores de ataque,
    defensa, plantilla y forma en una tabla interactiva.

    Parameters
    ----------
    default_df : pd.DataFrame
        DataFrame predeterminado que se muestra si no se ha realizado
        ninguna actualizacion via IA.
    model : str
        Nombre del modelo OpenAI a usar para la actualizacion de ratings.

    Returns
    -------
    pd.DataFrame
        DataFrame con los ratings (posiblemente editados) que se usa para
        la simulacion.
    """
    if "working_df" not in st.session_state:
        st.session_state["working_df"] = default_df.copy()

    if api_key_available():
        if st.button(
            "Actualizar ratings con IA",
            help="Usa busqueda web para obtener ratings actualizados (Elo, forma, plantilla) de todas las selecciones y reemplaza los valores actuales.",
        ):
            try:
                with st.spinner("Buscando datos actualizados con IA..."):
                    updates = call_llm_ratings_update(model, st.session_state["working_df"])
                    st.session_state["working_df"] = apply_ratings_update(
                        st.session_state["working_df"], updates
                    )
                st.session_state["working_df"].to_csv(DATA_PATH, index=False)
                load_default_data_cached.clear()
                st.success(f"Ratings actualizados para {len(updates)} selecciones y guardados en {DATA_PATH.name}.")
                st.rerun()
            except Exception as exc:
                st.error(f"No se pudo actualizar los ratings: {exc}")
    else:
        st.caption("OPENAI_API_KEY no detectada. No es posible actualizar ratings con IA.")

    with st.expander("Editar ratings del modelo", expanded=False):
        st.markdown(
            '<div class="small-note">Edita valores de 0 a 100 para ataque, defensa, plantilla y forma. Luego vuelve a correr la simulacion.</div>',
            unsafe_allow_html=True,
        )
        return st.data_editor(
            st.session_state["working_df"],
            width="stretch",
            num_rows="fixed",
            column_config={
                "is_host": st.column_config.CheckboxColumn("is_host", help="Marca si el equipo es uno de los tres anfitriones del torneo (USA, Canada, Mexico)."),
                "attack": st.column_config.NumberColumn("attack", min_value=0, max_value=100, help="Potencial ofensivo del equipo (0-100). Afecta los goles esperados generados."),
                "defense": st.column_config.NumberColumn("defense", min_value=0, max_value=100, help="Solidez defensiva del equipo (0-100). Reduce los goles esperados del rival."),
                "squad": st.column_config.NumberColumn("squad", min_value=0, max_value=100, help="Calidad y profundidad del plantel (0-100). Equipos con mejor banco aguantan mejor torneos largos."),
                "form": st.column_config.NumberColumn("form", min_value=0, max_value=100, help="Forma reciente del equipo (0-100). Refleja los resultados de los ultimos 6-12 meses."),
            },
        )


def _build_bracket_figure(bracket: pd.DataFrame, from_round: str = "round_of_16"):
    """Construir la figura Plotly del cuadro de eliminacion como arbol vertical.

    Distribuye cada partido en columnas (octavos/cuartos/semis/final) y filas
    fijas, dibuja conexiones entre rondas y resalta al ganador con color
    turquesa.  Si el bracket esta vacio, los partidos se renderizan como TBD.

    Parameters
    ----------
    bracket : pd.DataFrame
        DataFrame con columnas ``round``, ``match_id``, ``team_a``,
        ``team_b``, ``winner`` y opcionalmente ``winner_pct``.
    from_round : str, optional
        ``"round_of_16"`` para mostrar desde octavos (configuracion
        completa) o ``"quarterfinal"`` para mostrar solo desde cuartos.
        Por defecto ``"round_of_16"``.

    Returns
    -------
    plotly.graph_objects.Figure
        Figura Plotly lista para renderizar en Streamlit.
    """
    import plotly.graph_objects as go

    if from_round == "quarterfinal":
        positions = {
            97: (0, 9.0), 98: (0, 6.0),
            99: (0, 3.0), 100: (0, 0.0),
            101: (1, 7.5), 102: (1, 1.5),
            104: (2, 4.5),
        }
        col_x = {0: 0.0, 1: 3.5, 2: 7.0}
        col_labels = {0: "Cuartos de Final", 1: "Semifinales", 2: "Final"}
        box_hh = 0.65
        y_range = [-1.5, 11.5]
        x_range = [-1.5, 9.0]
        fig_height = 500
        font_size = 11
    else:
        positions = {
            89: (0, 14.0), 90: (0, 12.0),
            93: (0, 10.0), 94: (0, 8.0),
            91: (0, 6.0),  92: (0, 4.0),
            95: (0, 2.0),  96: (0, 0.0),
            97: (1, 13.0), 98: (1, 9.0),
            99: (1, 5.0),  100: (1, 1.0),
            101: (2, 11.0), 102: (2, 3.0),
            104: (3, 7.0),
        }
        col_x = {0: 0.0, 1: 3.5, 2: 7.0, 3: 10.5}
        col_labels = {0: "Octavos", 1: "Cuartos", 2: "Semifinales", 3: "Final"}
        box_hh = 0.45
        y_range = [-1.5, 16.5]
        x_range = [-1.5, 12.5]
        fig_height = 700
        font_size = 9

    parents = {
        97: (89, 90), 98: (93, 94),
        99: (91, 92), 100: (95, 96),
        101: (97, 98), 102: (99, 100),
        104: (101, 102),
    }
    box_hw = 1.4
    match_dict = bracket.set_index("match_id").to_dict("index") if not bracket.empty else {}

    shapes = []
    annotations = []

    for child_id, (p1_id, p2_id) in parents.items():
        if child_id not in positions or p1_id not in positions or p2_id not in positions:
            continue
        child_col, child_y = positions[child_id]
        p1_col, p1_y = positions[p1_id]
        _p2_col, p2_y = positions[p2_id]
        px_right = col_x[p1_col] + box_hw
        cx_left = col_x[child_col] - box_hw
        conn_x = (px_right + cx_left) / 2
        for py in (p1_y, p2_y):
            shapes.append({"type": "line", "x0": px_right, "y0": py, "x1": conn_x, "y1": py,
                           "line": {"color": "#555", "width": 1.5}})
        shapes.append({"type": "line", "x0": conn_x, "y0": p1_y, "x1": conn_x, "y1": p2_y,
                       "line": {"color": "#555", "width": 1.5}})
        shapes.append({"type": "line", "x0": conn_x, "y0": child_y, "x1": cx_left, "y1": child_y,
                       "line": {"color": "#555", "width": 1.5}})

    for match_id, (col, y) in positions.items():
        cx = col_x[col]
        if match_id in match_dict:
            row = match_dict[match_id]
            team_a = str(row["team_a"])
            team_b = str(row["team_b"])
            winner = str(row["winner"])
            raw_pct = row.get("winner_pct")
            winner_pct = float(raw_pct) if raw_pct is not None and str(raw_pct) not in ("None", "nan", "") else None
        else:
            team_a, team_b, winner, winner_pct = "TBD", "TBD", "", None

        for slot_y0, slot_y1, team in [
            (y, y + box_hh * 2, team_a),
            (y - box_hh * 2, y, team_b),
        ]:
            fill = "#1a4d3f" if team == winner else "#1e2130"
            shapes.append({"type": "rect",
                           "x0": cx - box_hw, "y0": slot_y0,
                           "x1": cx + box_hw, "y1": slot_y1,
                           "fillcolor": fill,
                           "line": {"color": "#444", "width": 0.8}})
            color = "#4ecca3" if team == winner else "#cccccc"
            label = team
            if team == winner and winner_pct is not None:
                label = f"{team} ({winner_pct:.0f}%)"
            annotations.append({"x": cx, "y": (slot_y0 + slot_y1) / 2,
                                 "text": label, "showarrow": False,
                                 "font": {"color": color, "size": font_size},
                                 "xanchor": "center", "yanchor": "middle"})

    label_y = y_range[1] - 0.4
    for col, label in col_labels.items():
        annotations.append({"x": col_x[col], "y": label_y,
                             "text": f"<b>{label}</b>", "showarrow": False,
                             "font": {"color": "#aaaaaa", "size": 11},
                             "xanchor": "center", "yanchor": "top"})

    fig = go.Figure()
    fig.update_layout(
        shapes=shapes,
        annotations=annotations,
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        xaxis={"visible": False, "range": x_range},
        yaxis={"visible": False, "range": y_range},
        height=fig_height,
        margin={"l": 10, "r": 10, "t": 20, "b": 10},
    )
    return fig


def render_bracket_view(
    bracket: pd.DataFrame | None,
    bracket_probable: pd.DataFrame | None,
) -> None:
    """Renderizar el cuadro de eliminacion del torneo como grafico tipo arbol.

    Muestra los partidos desde octavos o cuartos de final hasta la final.
    Permite elegir entre una simulacion representativa (semilla fija) o el
    cuadro mas probable calculado sobre 1000 simulaciones, donde el
    ganador de cada posicion es el equipo que gano con mas frecuencia y
    se muestra su porcentaje de victorias.

    Parameters
    ----------
    bracket : pd.DataFrame or None
        Cuadro de una simulacion representativa.
    bracket_probable : pd.DataFrame or None
        Cuadro mas probable calculado sobre N simulaciones.
    """
    any_data = bracket is not None or bracket_probable is not None
    if not any_data:
        st.info(
            "Pulsa **Simular torneo** para generar el cuadro de eliminacion. "
            "Los resultados se guardaran en el XLSX para la proxima sesion."
        )
        return

    col_mode, col_round = st.columns([2, 2])
    with col_mode:
        mode_options = []
        if bracket_probable is not None:
            mode_options.append("Mas probable (1000 sim.)")
        if bracket is not None:
            mode_options.append("Una simulacion")
        mode = st.radio("Modo", mode_options, horizontal=True, key="bracket_mode")
    with col_round:
        start_round = st.radio(
            "Mostrar desde",
            ["Octavos de Final", "Cuartos de Final"],
            horizontal=True,
            key="bracket_start_round",
        )

    from_round = "round_of_16" if start_round == "Octavos de Final" else "quarterfinal"
    if mode == "Mas probable (1000 sim.)":
        active = bracket_probable
        st.caption("Cuadro mas probable: para cada posicion se muestra el equipo que gano con mayor frecuencia en 1000 simulaciones. El porcentaje indica la frecuencia del ganador en esa posicion.")
    else:
        active = bracket
        st.caption("Simulacion representativa (semilla fija). El ganador de cada partido aparece en verde.")

    fig = _build_bracket_figure(active, from_round)
    st.plotly_chart(fig, width='stretch')


def render_app() -> None:
    """Orquestar el flujo principal de la aplicacion Streamlit.

    Llama a todos los componentes de renderizado en orden: barra lateral,
    titulo, editor de datos, ejecucion de la simulacion y las cuatro
    pestanas de contenido (Prediccion, Grupos, Modelo, LLM).

    Returns
    -------
    None
        Renderiza la aplicacion completa y actualiza ``st.session_state``.
    """
    load_persisted_outputs_once()

    params, llm_model = render_sidebar()
    st.session_state["params"] = params

    st.title(APP_TITLE)
    st.caption(APP_CAPTION)

    default_df = load_default_data_cached()
    working_df = render_data_editor(default_df, llm_model)

    if st.button("Simular torneo", type="primary", help="Ejecuta la simulacion Monte Carlo con los ratings y parametros actuales."):
        try:
            csv_text = dataframe_to_csv_text(working_df)
            run_simulation_cached.clear()
            run_bracket_sample_cached.clear()
            run_bracket_probable_cached.clear()
            with st.spinner("Simulando torneos..."):
                st.session_state["simulation_results"] = run_simulation_cached(csv_text, params)
            with st.spinner("Generando cuadro de eliminacion..."):
                st.session_state["bracket"] = run_bracket_sample_cached(csv_text, params)
            with st.spinner("Calculando cuadro mas probable (1000 simulaciones)..."):
                st.session_state["bracket_probable"] = run_bracket_probable_cached(csv_text, params)
            st.session_state["simulation_df"] = working_df.copy()
            save_montecarlo_results(
                st.session_state["simulation_results"],
                st.session_state["simulation_df"],
                params,
                bracket=st.session_state["bracket"],
                bracket_probable=st.session_state["bracket_probable"],
            )
        except Exception as exc:
            st.error(f"No se pudo correr la simulacion: {exc}")

    results = st.session_state.get("simulation_results")
    sim_df = st.session_state.get("simulation_df", working_df)

    bracket = st.session_state.get("bracket")
    bracket_probable = st.session_state.get("bracket_probable")

    tab_pred, tab_bracket, tab_groups, tab_model, tab_llm, tab_eval, tab_report = st.tabs(
        ["Prediccion", "Bracket", "Grupos", "Modelo", "LLM", "Evaluar", "Reporte"]
    )
    with tab_pred:
        if results is None:
            st.info("Pulsa **Simular torneo** para ver las predicciones.")
        else:
            render_probability_view(results)
    with tab_bracket:
        render_bracket_view(bracket, bracket_probable)
    with tab_groups:
        if results is None:
            st.info("Pulsa **Simular torneo** para ver el analisis por grupos.")
        else:
            render_group_view(sim_df, results)
    with tab_model:
        render_model_view()
    with tab_llm:
        if results is None:
            render_llm_view(None, sim_df, llm_model)
        else:
            render_llm_view(results, sim_df, llm_model)
    with tab_eval:
        render_evaluation_view()
    with tab_report:
        render_report_view(results, sim_df)


def render_evaluation_view() -> None:
    """Renderizar la pestana de evaluacion de submissions.
    
    Muestra controles para validar los CSVs de ``evaluations/`` bajo
    demanda, calcular Brier Scores cuando existe ``ground_truth.csv`` y
    generar un reporte Excel descargable.

    Returns
    -------
    None
        Renderiza controles, tablas, graficos y botones de descarga en
        la pestana de Streamlit.
    """
    st.markdown("### Evaluación de Submissions")
    st.markdown("Valida y evalúa todos los CSVs de la carpeta `evaluations/`")
    st.markdown("")
    
    # Definir equipos válidos y columnas
    valid_teams = [
        "Algeria", "Argentina", "Australia", "Austria", "Belgium",
        "Bosnia and Herzegovina", "Brazil", "Cabo Verde", "Canada", "Colombia",
        "Congo DR", "Cote d'Ivoire", "Croatia", "Curacao", "Czechia",
        "Ecuador", "Egypt", "England", "France", "Germany",
        "Ghana", "Haiti", "IR Iran", "Iraq", "Japan",
        "Jordan", "Korea Republic", "Mexico", "Morocco", "Netherlands",
        "New Zealand", "Norway", "Panama", "Paraguay", "Portugal",
        "Qatar", "Saudi Arabia", "Scotland", "Senegal", "South Africa",
        "Spain", "Sweden", "Switzerland", "Tunisia", "Turkiye",
        "USA", "Uzbekistan", "Uruguay",
    ]
    
    required_columns = ["team", "prob_champion", "prob_final", "prob_semifinal"]
    stage_weights = {
        "prob_champion": 0.50,
        "prob_final": 0.30,
        "prob_semifinal": 0.20,
    }
    
    def calculate_brier_score(predictions: pd.Series, actuals: pd.Series) -> float:
        """Calcular el Brier Score de una serie de predicciones.

        Parameters
        ----------
        predictions : pd.Series
            Probabilidades predichas para una etapa del torneo.
        actuals : pd.Series
            Indicadores reales de ocurrencia, codificados como 0 o 1.

        Returns
        -------
        float
            Promedio de ``(prediccion - real) ** 2``.
        """
        return ((predictions - actuals) ** 2).mean()
    
    def validate_submission(df: pd.DataFrame, filename: str) -> tuple[bool, list[str]]:
        """Validar que una submission tenga el formato esperado.

        Parameters
        ----------
        df : pd.DataFrame
            Datos cargados desde el CSV de submission.
        filename : str
            Nombre del archivo, usado para contexto en mensajes.

        Returns
        -------
        tuple[bool, list[str]]
            Bandera de validez y lista de errores encontrados.
        """
        errors = []
        
        # Verificar columnas
        missing_cols = set(required_columns) - set(df.columns)
        if missing_cols:
            errors.append(f"Faltan columnas: {', '.join(missing_cols)}")
        
        extra_cols = set(df.columns) - set(required_columns)
        if extra_cols:
            errors.append(f"Columnas extra: {', '.join(extra_cols)}")
        
        if errors:
            return False, errors
        
        # Verificar cantidad de equipos
        if len(df) != 48:
            errors.append(f"Se esperan 48 equipos, se encontraron {len(df)}")
        
        # Verificar nombres de equipos
        invalid_teams = set(df["team"]) - set(valid_teams)
        if invalid_teams:
            errors.append(f"Equipos inválidos: {', '.join(sorted(invalid_teams))}")
        
        missing_teams = set(valid_teams) - set(df["team"])
        if missing_teams:
            errors.append(f"Equipos faltantes: {', '.join(sorted(missing_teams))}")
        
        # Verificar duplicados
        if df["team"].duplicated().any():
            dups = df.loc[df["team"].duplicated(), "team"].tolist()
            errors.append(f"Equipos duplicados: {', '.join(dups)}")
        
        # Verificar rangos de probabilidades
        for col in required_columns[1:]:
            invalid = ((df[col] < 0) | (df[col] > 1)).sum()
            if invalid > 0:
                errors.append(f"Columna {col}: {invalid} valores fuera de rango [0, 1]")
        
        return len(errors) == 0, errors
    
    def evaluate_submission(csv_path: Path, ground_truth: pd.DataFrame) -> dict:
        """Validar y puntuar un archivo de submission.

        Parameters
        ----------
        csv_path : Path
            Ruta del archivo CSV a evaluar.
        ground_truth : pd.DataFrame
            Resultados reales del torneo con columnas ``team``,
            ``champion``, ``final`` y ``semifinal``.

        Returns
        -------
        dict
            Resultado de validacion y scores Brier por etapa, incluyendo
            ``score_final`` cuando hay datos reales disponibles.
        """
        filename = csv_path.name
        result = {
            "file": filename,
            "valid": False,
            "errors": [],
            "brier_champion": None,
            "brier_final": None,
            "brier_semifinal": None,
            "score_final": None,
        }
        
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            result["errors"].append(f"Error al leer CSV: {str(e)}")
            return result
        
        valid, errors = validate_submission(df, filename)
        result["valid"] = valid
        result["errors"] = errors
        
        if not valid or ground_truth.empty:
            return result
        
        # Merge con resultados reales
        merged = df.merge(ground_truth, on="team", how="inner")
        
        if len(merged) != 48:
            result["errors"].append("No se pudieron hacer match todos los equipos")
            return result
        
        # Calcular Brier Scores por etapa
        result["brier_champion"] = calculate_brier_score(
            merged["prob_champion"], merged["champion"]
        )
        result["brier_final"] = calculate_brier_score(
            merged["prob_final"], merged["final"]
        )
        result["brier_semifinal"] = calculate_brier_score(
            merged["prob_semifinal"], merged["semifinal"]
        )
        
        # Calcular score final ponderado
        result["score_final"] = (
            stage_weights["prob_champion"] * result["brier_champion"]
            + stage_weights["prob_final"] * result["brier_final"]
            + stage_weights["prob_semifinal"] * result["brier_semifinal"]
        )
        
        return result

    def build_excel_report(results: list[dict], ground_truth_loaded: bool, eval_dir: Path) -> bytes:
        """Crear un reporte XLSX con formato para descargar desde Streamlit.

        Parameters
        ----------
        results : list[dict]
            Resultados producidos por ``evaluate_submission``.
        ground_truth_loaded : bool
            Indica si ``ground_truth.csv`` fue cargado correctamente.
        eval_dir : Path
            Carpeta donde se buscaron las submissions.

        Returns
        -------
        bytes
            Contenido binario del archivo XLSX generado en memoria.
        """
        from openpyxl.chart import BarChart, Reference
        from openpyxl.styles import Alignment, Font, PatternFill
        from openpyxl.utils import get_column_letter

        validation_df = pd.DataFrame(
            [
                {
                    "Submission": result["file"],
                    "Estado": "Valido" if result["valid"] else "Error",
                    "Detalles": "; ".join(result["errors"]) if result["errors"] else "OK",
                }
                for result in results
            ]
        )
        ranked_results = sorted(
            [r for r in results if r["valid"] and r["score_final"] is not None],
            key=lambda x: x["score_final"],
        )
        scores_df = pd.DataFrame(
            [
                {
                    "Posicion": i,
                    "Submission": result["file"],
                    "Champion": result["brier_champion"],
                    "Final": result["brier_final"],
                    "Semifinal": result["brier_semifinal"],
                    "Score Final": result["score_final"],
                }
                for i, result in enumerate(ranked_results, 1)
            ]
        )
        best = ranked_results[0] if ranked_results else None
        valid_count = sum(1 for result in results if result["valid"])
        summary_df = pd.DataFrame(
            [
                {"Metrica": "Carpeta evaluada", "Valor": str(eval_dir)},
                {"Metrica": "Submissions encontradas", "Valor": len(results)},
                {"Metrica": "Submissions validas", "Valor": valid_count},
                {"Metrica": "Submissions con error", "Valor": len(results) - valid_count},
                {"Metrica": "Ground truth cargado", "Valor": "Si" if ground_truth_loaded else "No"},
                {"Metrica": "Mejor submission", "Valor": best["file"] if best else "N/A"},
                {"Metrica": "Mejor score final", "Valor": best["score_final"] if best else "N/A"},
            ]
        )

        output = BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            summary_df.to_excel(writer, sheet_name="Resumen", index=False)
            validation_df.to_excel(writer, sheet_name="Validacion", index=False)
            if not scores_df.empty:
                scores_df.to_excel(writer, sheet_name="Ranking", index=False)

            workbook = writer.book
            header_fill = PatternFill("solid", fgColor="1F4E78")
            header_font = Font(color="FFFFFF", bold=True)
            summary_fill = PatternFill("solid", fgColor="D9EAF7")
            ok_fill = PatternFill("solid", fgColor="E2F0D9")
            error_fill = PatternFill("solid", fgColor="FCE4D6")
            podium_fills = {
                1: PatternFill("solid", fgColor="FFF2CC"),
                2: PatternFill("solid", fgColor="D9EAF7"),
                3: PatternFill("solid", fgColor="EADCF8"),
            }

            for worksheet in workbook.worksheets:
                worksheet.freeze_panes = "A2"
                worksheet.sheet_view.showGridLines = False
                for cell in worksheet[1]:
                    cell.fill = header_fill
                    cell.font = header_font
                    cell.alignment = Alignment(horizontal="center")
                for column_cells in worksheet.columns:
                    max_length = max(len(str(cell.value or "")) for cell in column_cells)
                    width = min(max(max_length + 3, 14), 55)
                    worksheet.column_dimensions[get_column_letter(column_cells[0].column)].width = width

            resumen = workbook["Resumen"]
            for row in resumen.iter_rows(min_row=2, max_col=2):
                row[0].font = Font(bold=True)
                row[0].fill = summary_fill

            validacion = workbook["Validacion"]
            for row in validacion.iter_rows(min_row=2):
                fill = ok_fill if row[1].value == "Valido" else error_fill
                for cell in row:
                    cell.fill = fill

            if "Ranking" in workbook.sheetnames:
                ranking = workbook["Ranking"]
                for row in ranking.iter_rows(min_row=2):
                    fill = podium_fills.get(row[0].value)
                    if fill:
                        for cell in row:
                            cell.fill = fill
                    for cell in row[2:]:
                        cell.number_format = "0.000000"

                chart = BarChart()
                chart.title = "Score final por submission"
                chart.y_axis.title = "Score final"
                chart.x_axis.title = "Submission"
                data = Reference(ranking, min_col=6, min_row=1, max_row=ranking.max_row)
                categories = Reference(ranking, min_col=2, min_row=2, max_row=ranking.max_row)
                chart.add_data(data, titles_from_data=True)
                chart.set_categories(categories)
                chart.height = 8
                chart.width = 18
                ranking.add_chart(chart, "H2")

        return output.getvalue()
    
    # Encontrar archivos
    eval_dir = Path(__file__).resolve().parents[1] / "evaluations"
    
    if not eval_dir.exists():
        st.error(f"❌ Carpeta evaluations/ no encontrada en {eval_dir}")
        return
    
    csv_files = sorted([f for f in eval_dir.glob("*.csv") if f.name != "ground_truth.csv"])

    if not csv_files:
        st.warning("No se encontraron archivos CSV para evaluar.")
        return

    ground_truth_path = eval_dir / "ground_truth.csv"
    has_ground_truth = ground_truth_path.exists()
    st.info(
        f"{len(csv_files)} submission(s) encontrada(s). "
        f"Ground truth: {'disponible' if has_ground_truth else 'no encontrado'}."
    )

    run_col, report_col, download_col = st.columns([1, 1, 1])

    with run_col:
        evaluate_clicked = st.button(
            "Evaluar submissions",
            type="primary",
            help="Valida los CSVs y calcula Brier Scores si existe ground_truth.csv.",
        )

    if evaluate_clicked:
        ground_truth = pd.DataFrame()
        ground_truth_loaded = False
        if has_ground_truth:
            try:
                ground_truth = pd.read_csv(ground_truth_path)
                ground_truth_loaded = True
            except Exception as e:
                st.warning(f"Error al cargar ground_truth.csv: {e}")
        else:
            st.warning("Archivo ground_truth.csv no encontrado.")

        with st.spinner("Evaluando submissions..."):
            st.session_state["evaluation_results"] = [
                evaluate_submission(csv_path, ground_truth) for csv_path in csv_files
            ]
            st.session_state["evaluation_ground_truth_loaded"] = ground_truth_loaded
            st.session_state.pop("evaluation_excel_report", None)

    results = st.session_state.get("evaluation_results")
    ground_truth_loaded = st.session_state.get("evaluation_ground_truth_loaded", False)

    with report_col:
        generate_report_clicked = st.button(
            "Generar reporte Excel",
            disabled=not results,
            help="Crea un XLSX con resumen, validacion, ranking y grafico.",
        )

    if generate_report_clicked and results:
        with st.spinner("Generando reporte Excel..."):
            st.session_state["evaluation_excel_report"] = build_excel_report(
                results,
                ground_truth_loaded,
                eval_dir,
            )
        st.success("Reporte Excel listo.")

    with download_col:
        report_bytes = st.session_state.get("evaluation_excel_report")
        st.download_button(
            "Descargar reporte",
            data=report_bytes or b"",
            file_name="reporte_evaluacion_wcup2026.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            disabled=report_bytes is None,
            help="Descarga el reporte generado.",
        )

    if not results:
        st.info("Pulsa **Evaluar submissions** para ver la validacion y los scores.")
        return

    st.divider()

    st.markdown("#### Validacion de Formato")
    validation_df = pd.DataFrame(
        [
            {
                "Submission": result["file"],
                "Estado": "Valido" if result["valid"] else "Error",
                "Detalles": "; ".join(result["errors"]) if result["errors"] else "OK",
            }
            for result in results
        ]
    )
    st.dataframe(validation_df, width="stretch", hide_index=True)

    st.divider()

    if ground_truth_loaded:
        st.markdown("#### Resultados de Evaluacion (Brier Score)")
        st.markdown("*Ordenado por Score Final (menor es mejor)*")

        results_sorted = sorted(
            [r for r in results if r["valid"] and r["score_final"] is not None],
            key=lambda x: x["score_final"],
        )

        if results_sorted:
            scores_df = pd.DataFrame(
                [
                    {
                        "Posicion": i,
                        "Submission": result["file"],
                        "Champion": result["brier_champion"],
                        "Final": result["brier_final"],
                        "Semifinal": result["brier_semifinal"],
                        "Score Final": result["score_final"],
                    }
                    for i, result in enumerate(results_sorted, 1)
                ]
            )

            def highlight_row(row):
                """Asignar color de fondo a las primeras posiciones.

                Parameters
                ----------
                row : pd.Series
                    Fila del DataFrame de ranking renderizado por
                    ``pandas.Styler``.

                Returns
                -------
                list[str]
                    Estilos CSS por celda para resaltar el podio.
                """
                if row["Posicion"] == 1:
                    return ["background-color: #90EE90"] * len(row)
                if row["Posicion"] == 2:
                    return ["background-color: #87CEEB"] * len(row)
                if row["Posicion"] == 3:
                    return ["background-color: #FFB6C1"] * len(row)
                return [""] * len(row)

            st.dataframe(
                scores_df.style.format({
                    "Champion": "{:.6f}",
                    "Final": "{:.6f}",
                    "Semifinal": "{:.6f}",
                    "Score Final": "{:.6f}",
                }).apply(highlight_row, axis=1),
                width="stretch",
                hide_index=True,
            )

            st.markdown("#### Comparacion de Scores Finales")
            fig = px.bar(
                scores_df,
                x="Submission",
                y="Score Final",
                title="Score Final por Submission (Brier Score Ponderado)",
                labels={"Score Final": "Score Final", "Submission": "Submission"},
                color="Score Final",
                color_continuous_scale="RdYlGn_r",
            )
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("No hay submissions validas para mostrar scores.")
    else:
        st.info("Para ver scores de evaluacion, crea archivo `ground_truth.csv` en la carpeta `evaluations/`")
        st.markdown("**Estructura esperada:**")
        st.code("team,champion,final,semifinal\nArgentina,1,1,1\nFrance,0,1,1\n...", language="csv")

    return


def main() -> None:
    """Punto de entrada de la aplicacion Streamlit.

    Configura la pagina y lanza el renderizado completo de la app.  Es
    invocado por ``app.py`` al ejecutar ``streamlit run app.py``.

    Returns
    -------
    None
        Inicializa y renderiza la aplicacion.
    """
    configure_page()
    render_app()
