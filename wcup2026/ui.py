"""Componentes de la interfaz de usuario Streamlit.

Contiene todas las funciones de renderizado de la aplicacion web:
configuracion de pagina, barra lateral de parametros, pestanas de
prediccion, grupos, descripcion del modelo, integracion LLM y editor
de datos.  El punto de entrada principal es ``main()`` llamado desde
``app.py``.
"""

from __future__ import annotations

import html
import json

import pandas as pd
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components
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
    load_llm_analysis,
    load_montecarlo_results,
    save_llm_analysis,
    save_montecarlo_results,
)
from wcup2026.report import generate_report
from wcup2026.simulator import describe_matchup, simulate_many


def configure_page() -> None:
    """Configurar la pagina Streamlit y cargar variables de entorno.

    Llama a ``st.set_page_config`` con titulo e icono de la app, carga
    el archivo ``.env`` via dotenv e inyecta los estilos CSS personalizados.
    Debe invocarse como primera instruccion Streamlit del script.
    """
    st.set_page_config(page_title=APP_TITLE, page_icon="WC26", layout="wide")
    load_dotenv()
    inject_style()


def inject_style() -> None:
    """Inyectar CSS personalizado en la pagina via ``st.markdown``.

    Ajusta padding del contenedor principal, estilo de las metricas,
    tipografia de encabezados y clases utilitarias ``.source-line`` y
    ``.small-note``.
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
        components.html(
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


def load_persisted_outputs_once() -> None:
    """Cargar resultados guardados en disco una sola vez por sesion de Streamlit.

    Usa la clave ``_persisted_outputs_loaded`` del ``st.session_state`` como
    centinela para evitar lecturas repetidas en re-runs.  Inicializa
    ``simulation_results``, ``simulation_df`` y ``llm_answer`` si existen
    datos guardados y aun no hay valores en la sesion.
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
            payload = build_analysis_payload(results, df, notes)
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
                tex_path, pdf_path = generate_report(
                    results=results,
                    teams=df,
                    llm_text=st.session_state.get("llm_answer"),
                    params=st.session_state.get("params"),
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


def render_app() -> None:
    """Orquestar el flujo principal de la aplicacion Streamlit.

    Llama a todos los componentes de renderizado en orden: barra lateral,
    titulo, editor de datos, ejecucion de la simulacion y las cuatro
    pestanas de contenido (Prediccion, Grupos, Modelo, LLM).
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
            st.session_state["simulation_results"] = run_simulation_cached(csv_text, params)
            st.session_state["simulation_df"] = working_df.copy()
            save_montecarlo_results(
                st.session_state["simulation_results"],
                st.session_state["simulation_df"],
                params,
            )
        except Exception as exc:
            st.error(f"No se pudo correr la simulacion: {exc}")

    results = st.session_state.get("simulation_results")
    sim_df = st.session_state.get("simulation_df", working_df)

    tab_pred, tab_groups, tab_model, tab_llm, tab_report = st.tabs(["Prediccion", "Grupos", "Modelo", "LLM", "Reporte"])
    with tab_pred:
        if results is None:
            st.info("Pulsa **Simular torneo** para ver las predicciones.")
        else:
            render_probability_view(results)
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
    with tab_report:
        render_report_view(results, sim_df)


def main() -> None:
    """Punto de entrada de la aplicacion Streamlit.

    Configura la pagina y lanza el renderizado completo de la app.  Es
    invocado por ``app.py`` al ejecutar ``streamlit run app.py``.
    """
    configure_page()
    render_app()
