"""Generacion del reporte LaTeX del predictor WCUP 2026.

Expone funciones para construir bloques de contenido LaTeX (resumen,
graficos TikZ, tablas), renderizar el template con placeholders y
compilar el PDF final mediante pdflatex.
"""

from __future__ import annotations

import re
import subprocess
from datetime import datetime
from pathlib import Path

import pandas as pd

from wcup2026.config import (
    REPORT_DIR,
    REPORT_PDF_PATH,
    REPORT_TEMPLATE_PATH,
    REPORT_TEX_PATH,
)
from wcup2026.parameters import SimParams


LATEX_SPECIALS = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}


def ensure_report_dir() -> None:
    """Crear la carpeta de reporte si no existe.

    Returns
    -------
    None
        La funcion solo garantiza la existencia de ``REPORT_DIR``.
    """
    REPORT_DIR.mkdir(parents=True, exist_ok=True)


def latex_escape(value: object) -> str:
    """Escapar caracteres especiales de LaTeX en una cadena de texto.

    Parameters
    ----------
    value : object
        Valor a convertir y escapar.  Si es ``None`` se trata como cadena
        vacia; cualquier otro tipo se convierte con ``str()``.

    Returns
    -------
    str
        Cadena con todos los caracteres especiales de LaTeX sustituidos
        por sus secuencias de escape equivalentes.
    """
    text = "" if value is None else str(value)
    return "".join(LATEX_SPECIALS.get(char, char) for char in text)


def _format_pct(value: object) -> str:
    """Formatear un valor numerico como porcentaje para LaTeX.

    Parameters
    ----------
    value : object
        Valor numerico (o convertible a float) a formatear.

    Returns
    -------
    str
        Cadena con un decimal y el simbolo ``\\%`` escapado para LaTeX,
        o cadena vacia si el valor no es convertible.
    """
    try:
        return f"{float(value):.1f}\\%"
    except (TypeError, ValueError):
        return ""


def _clean_markdown_inline(text: str) -> str:
    """Eliminar marcado inline de Markdown de una cadena de texto.

    Elimina negritas (``**``), cursivas (``*``), codigo inline (````)
    y enlaces ``[texto](url)``.

    Parameters
    ----------
    text : str
        Cadena que puede contener marcado Markdown inline.

    Returns
    -------
    str
        Texto plano sin marcado Markdown.
    """
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"\*(.+?)\*", r"\1", text)
    text = re.sub(r"`(.+?)`", r"\1", text)
    text = re.sub(r"\[(.+?)\]\(.+?\)", r"\1", text)
    return text.strip()


def markdown_to_latex(markdown_text: str | None) -> str:
    """Convertir markdown basico del LLM a comandos LaTeX sencillos.

    Convierte encabezados ``##``/``###`` a ``\\section*``/``\\subsection*``,
    listas con guion o asterisco a entornos ``itemize`` y parrafos de texto
    plano con ``\\par``.

    Parameters
    ----------
    markdown_text : str or None
        Texto en formato Markdown producido por el LLM.  Si es ``None`` o
        vacio devuelve un mensaje de aviso en italica.

    Returns
    -------
    str
        Cadena con los comandos LaTeX equivalentes listos para insertarse
        en el template.
    """
    if not markdown_text or not markdown_text.strip():
        return r"\emph{No hay resultado LLM guardado para este reporte.}"

    lines = markdown_text.strip().splitlines()
    output: list[str] = []
    in_items = False

    def close_items() -> None:
        """Cerrar el entorno ``itemize`` activo si esta abierto.

        Modifica ``output`` e ``in_items`` del scope envolvente para
        anadir ``\\end{itemize}`` solo cuando hay una lista en curso.

        Returns
        -------
        None
            La funcion actualiza el estado de la conversion en el scope
            envolvente.
        """
        nonlocal in_items
        if in_items:
            output.append(r"\end{itemize}")
            in_items = False

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            close_items()
            output.append("")
            continue

        heading = re.match(r"^(#{1,3})\s+(.+)$", line)
        if heading:
            close_items()
            command = "section" if len(heading.group(1)) == 2 else "subsection"
            output.append(f"\\{command}*{{{latex_escape(_clean_markdown_inline(heading.group(2)))}}}")
            continue

        bullet = re.match(r"^[-*]\s+(.+)$", line)
        if bullet:
            if not in_items:
                output.append(r"\begin{itemize}")
                in_items = True
            output.append(rf"\item {latex_escape(_clean_markdown_inline(bullet.group(1)))}")
            continue

        close_items()
        output.append(latex_escape(_clean_markdown_inline(line)) + r"\par")

    close_items()
    return "\n".join(output).strip()


def build_summary(results: pd.DataFrame, params: SimParams | None) -> str:
    """Construir el parrafo de resumen ejecutivo del reporte.

    Parameters
    ----------
    results : pd.DataFrame
        DataFrame de resultados de la simulacion ordenado por probabilidad
        de campeon (salida de ``simulate_many``).
    params : SimParams or None
        Hiperparametros de la simulacion.  Si es ``None`` o no tiene el
        atributo ``simulations``, se usa texto generico.

    Returns
    -------
    str
        Parrafo de texto plano listo para insertarse en el reporte.
    """
    favorite = results.iloc[0]
    simulations = getattr(params, "simulations", None)
    if simulations:
        simulation_text = f"{simulations:,}".replace(",", ".") + " torneos"
    else:
        simulation_text = "la ultima simulacion guardada"
    return (
        f"La simulacion Monte Carlo evaluo {simulation_text}. "
        f"El favorito del modelo es {favorite['team']} con "
        f"{float(favorite['champion_pct']):.1f}% de probabilidad de campeon, "
        f"{float(favorite['final_pct']):.1f}% de probabilidad de final y un rating "
        f"compuesto de {float(favorite['overall']):.1f}."
    )


def build_kpi_strip(results: pd.DataFrame) -> str:
    """Crear el bloque LaTeX de tarjetas KPI para la primera pagina.

    Genera una fila de cuatro tarjetas con el comando ``\\qakpicard``
    mostrando favorito, probabilidad de final, perseguidor y rating lider.

    Parameters
    ----------
    results : pd.DataFrame
        DataFrame de resultados ordenado por probabilidad de campeon.

    Returns
    -------
    str
        Fragmento LaTeX con el entorno ``tabular`` de tarjetas KPI.
    """
    favorite = results.iloc[0]
    second = results.iloc[1] if len(results) > 1 else favorite
    cards = [
        ("Favorito", favorite["team"], _format_pct(favorite["champion_pct"])),
        ("Final", favorite["team"], _format_pct(favorite["final_pct"])),
        ("Perseguidor", second["team"], _format_pct(second["champion_pct"])),
        ("Rating lider", favorite["team"], f"{float(favorite['overall']):.1f}"),
    ]
    rows = [
        r"\begin{center}",
        r"\begin{tabular}{@{}p{0.22\linewidth}p{0.22\linewidth}p{0.22\linewidth}p{0.22\linewidth}@{}}",
    ]
    rows.append(
        " & ".join(
            rf"\qakpicard{{{latex_escape(label)}}}{{{latex_escape(title)}}}{{{value}}}"
            for label, title, value in cards
        )
        + r" \\"
    )
    rows.extend([r"\end{tabular}", r"\end{center}"])
    return "\n".join(rows)


def build_champion_chart(results: pd.DataFrame, limit: int = 10) -> str:
    """Crear un grafico de barras horizontales en TikZ con los favoritos.

    Los tres primeros favoritos se colorean con ``qaturquoise``; el resto
    con ``qablue``.  La barra mas larga ocupa el 100 % del ancho.

    Parameters
    ----------
    results : pd.DataFrame
        DataFrame de resultados ordenado por probabilidad de campeon.
    limit : int, optional
        Numero maximo de equipos a mostrar.  Por defecto 10.

    Returns
    -------
    str
        Fragmento LaTeX con el entorno ``tikzpicture`` del grafico.
    """
    top = results.head(limit).iloc[::-1].reset_index(drop=True)
    max_value = max(float(top["champion_pct"].max()), 1.0)
    rows: list[str] = [
        r"\begin{tikzpicture}[x=0.12cm,y=0.62cm]",
        r"\small",
    ]
    for idx, row in top.iterrows():
        value = float(row["champion_pct"])
        width = value / max_value * 72
        team = latex_escape(row["team"])
        color = "qaturquoise" if idx >= len(top) - 3 else "qablue"
        rows.append(rf"\node[anchor=east,text=qaink] at (0,{idx}) {{{team}}};")
        rows.append(rf"\fill[qaash] (1,{idx - 0.24}) rectangle (73,{idx + 0.24});")
        rows.append(rf"\fill[{color}] (1,{idx - 0.24}) rectangle ({1 + width:.2f},{idx + 0.24});")
        rows.append(rf"\node[anchor=west,text=qaink] at ({2 + width:.2f},{idx}) {{{value:.1f}\%}};")
    rows.extend(
        [
            r"\draw[qaline] (1,-0.55) -- (1," + f"{len(top) - 0.45}" + r");",
            r"\end{tikzpicture}",
        ]
    )
    return "\n".join(rows)


def build_top_table(results: pd.DataFrame, limit: int = 12) -> str:
    """Crear la tabla LaTeX con los principales favoritos al titulo.

    La tabla incluye columnas de seleccion, rating, semifinal, final y
    campeon con filas alternadas y encabezado azul.

    Parameters
    ----------
    results : pd.DataFrame
        DataFrame de resultados ordenado por probabilidad de campeon.
    limit : int, optional
        Numero de filas a incluir.  Por defecto 12.

    Returns
    -------
    str
        Fragmento LaTeX con el entorno ``tabular`` de la tabla.
    """
    rows = [
        r"\rowcolors{2}{qasoft}{white}",
        r"\begin{tabular}{p{0.28\linewidth}rrrr}",
        r"\rowcolor{qablue}",
        r"\textcolor{white}{\textbf{Seleccion}} & \textcolor{white}{\textbf{Rating}} & \textcolor{white}{\textbf{Semifinal}} & \textcolor{white}{\textbf{Final}} & \textcolor{white}{\textbf{Campeon}} \\",
    ]
    for _, row in results.head(limit).iterrows():
        rows.append(
            " & ".join(
                [
                    latex_escape(row["team"]),
                    f"{float(row['overall']):.1f}",
                    _format_pct(row["semifinal_pct"]),
                    _format_pct(row["final_pct"]),
                    _format_pct(row["champion_pct"]),
                ]
            )
            + r" \\"
        )
    rows.extend([r"\end{tabular}"])
    return "\n".join(rows)


def build_group_table(results: pd.DataFrame, teams: pd.DataFrame) -> str:
    """Crear la tabla LaTeX compacta con los dos primeros favoritos de cada grupo.

    Parameters
    ----------
    results : pd.DataFrame
        DataFrame de resultados con columnas ``team``, ``overall``,
        ``round_of_32_pct`` y ``champion_pct``.
    teams : pd.DataFrame
        DataFrame original de equipos con columnas ``team`` y ``group``.

    Returns
    -------
    str
        Fragmento LaTeX con el entorno ``longtable`` de favoritos por grupo.
    """
    merged = teams[["team", "group"]].merge(
        results[["team", "overall", "round_of_32_pct", "champion_pct"]],
        on="team",
        how="left",
    )
    leaders = merged.sort_values(["group", "overall"], ascending=[True, False]).groupby("group").head(2)
    rows = [
        r"\rowcolors{2}{qasoft}{white}",
        r"\begin{longtable}{llrrr}",
        r"\rowcolor{qablue}",
        r"\textcolor{white}{\textbf{Grupo}} & \textcolor{white}{\textbf{Seleccion}} & \textcolor{white}{\textbf{Rating}} & \textcolor{white}{\textbf{R32}} & \textcolor{white}{\textbf{Campeon}} \\",
        r"\endhead",
    ]
    for _, row in leaders.iterrows():
        rows.append(
            " & ".join(
                [
                    latex_escape(row["group"]),
                    latex_escape(row["team"]),
                    f"{float(row['overall']):.1f}",
                    _format_pct(row["round_of_32_pct"]),
                    _format_pct(row["champion_pct"]),
                ]
            )
            + r" \\"
        )
    rows.extend([r"\end{longtable}"])
    return "\n".join(rows)


def build_bracket_chart(bracket_probable: pd.DataFrame) -> str:
    """Crear un cuadro de eliminacion top-down en TikZ desde cuartos de final.

    Dibuja cuartos, semis y final de arriba hacia abajo con lineas de
    conexion y porcentaje del ganador mas probable.

    Parameters
    ----------
    bracket_probable : pd.DataFrame
        DataFrame del cuadro mas probable con columnas ``round``,
        ``match_id``, ``team_a``, ``team_b``, ``winner``, ``winner_pct``.

    Returns
    -------
    str
        Fragmento LaTeX con el entorno ``tikzpicture`` del bracket.
    """
    if bracket_probable is None or bracket_probable.empty:
        return r"\qaempty{No hay datos de bracket disponibles.}"

    late = {"quarterfinal", "semifinal", "final"}
    bp = bracket_probable[bracket_probable["round"].isin(late)].copy()
    if bp.empty:
        return r"\qaempty{No hay datos de bracket disponibles.}"

    qf = bp[bp["round"] == "quarterfinal"].reset_index(drop=True)
    sf = bp[bp["round"] == "semifinal"].reset_index(drop=True)
    fi = bp[bp["round"] == "final"].reset_index(drop=True)

    # Posiciones x para 4 partidos de cuartos (2 por cada lado del cuadro)
    # Layout: QF(0), QF(1) -> SF(0) -> Final; QF(2), QF(3) -> SF(1) -> Final
    # x positions (cm): QF pares en 0 y 3.2, SF en 1.6, Final en centro
    xpos_qf = [0.0, 3.2, 6.4, 9.6]
    xpos_sf = [1.6, 8.0]
    xpos_fi = [4.8]

    y_qf = 0.0
    y_sf = -2.2
    y_fi = -4.4

    node_w = 2.8   # ancho caja en cm
    node_h = 0.55  # alto por equipo
    gap = 0.12     # espacio entre team_a y team_b

    lines: list[str] = [
        r"\begin{center}",
        r"\begin{tikzpicture}[x=1cm, y=1cm, font=\small]",
    ]

    def match_node(x: float, y: float, team_a: str, team_b: str, winner: str, pct: object) -> list[str]:
        """Emitir un nodo TikZ de partido con dos equipos y marcar al ganador.

        Parameters
        ----------
        x : float
            Coordenada X del centro del nodo (cm).
        y : float
            Coordenada Y del centro del nodo (cm).
        team_a : str
            Nombre del primer equipo (cuadro superior).
        team_b : str
            Nombre del segundo equipo (cuadro inferior).
        winner : str
            Nombre del equipo ganador; debe coincidir con ``team_a`` o
            ``team_b`` para que se resalte.
        pct : object
            Porcentaje de victorias del ganador.  Si no es convertible a
            ``float`` se omite.

        Returns
        -------
        list[str]
            Lista de comandos TikZ que dibujan los dos rectangulos del
            partido y las etiquetas con los nombres y porcentaje.
        """
        ta = latex_escape(team_a)
        tb = latex_escape(team_b)
        tw = latex_escape(winner)
        pct_str = ""
        try:
            pct_str = f" {float(pct):.0f}\\%"
        except (TypeError, ValueError):
            pass

        color_a = "qaturquoise" if winner == team_a else "qaink"
        color_b = "qaturquoise" if winner == team_b else "qaink"
        label_a = rf"\textbf{{{ta}}}" if winner == team_a else ta
        label_b = rf"\textbf{{{tb}}}" if winner == team_b else tb

        half_w = node_w / 2
        result = [
            # caja contenedora
            rf"\draw[draw=qaline, fill=qasoft, rounded corners=2pt] "
            rf"({x - half_w:.2f},{y + gap / 2:.2f}) rectangle ({x + half_w:.2f},{y + gap / 2 + node_h:.2f});",
            rf"\draw[draw=qaline, fill=qasoft, rounded corners=2pt] "
            rf"({x - half_w:.2f},{y - gap / 2 - node_h:.2f}) rectangle ({x + half_w:.2f},{y - gap / 2:.2f});",
            # etiquetas equipos
            rf"\node[anchor=west, text={color_a}] at ({x - half_w + 0.08:.2f},{y + gap / 2 + node_h / 2:.2f}) {{{label_a}}};",
            rf"\node[anchor=west, text={color_b}] at ({x - half_w + 0.08:.2f},{y - gap / 2 - node_h / 2:.2f}) {{{label_b}}};",
            # porcentaje ganador
            rf"\node[anchor=east, text=qamutex, font=\scriptsize] at ({x + half_w - 0.05:.2f},{y + gap / 2 + node_h / 2:.2f}) {{{pct_str}}};",
        ]
        return result

    # --- Cuartos ---
    qf_centers: list[tuple[float, float]] = []
    for i, row in qf.iterrows():
        x = xpos_qf[i] if i < len(xpos_qf) else float(i) * 3.2
        lines += match_node(x, y_qf, str(row["team_a"]), str(row["team_b"]), str(row["winner"]), row.get("winner_pct"))
        qf_centers.append((x, y_qf))

    # --- Semis ---
    sf_centers: list[tuple[float, float]] = []
    for i, row in sf.iterrows():
        x = xpos_sf[i] if i < len(xpos_sf) else xpos_sf[-1]
        lines += match_node(x, y_sf, str(row["team_a"]), str(row["team_b"]), str(row["winner"]), row.get("winner_pct"))
        sf_centers.append((x, y_sf))
        # Lineas desde los dos cuartos que alimentan esta semi
        for qi in range(2):
            qx, _ = qf_centers[i * 2 + qi] if (i * 2 + qi) < len(qf_centers) else (x, y_qf)
            lines.append(
                rf"\draw[draw=qaline] ({qx:.2f},{y_qf - (gap / 2 + node_h):.2f}) -- "
                rf"({qx:.2f},{(y_qf + y_sf) / 2:.2f}) -- "
                rf"({x:.2f},{(y_qf + y_sf) / 2:.2f}) -- "
                rf"({x:.2f},{y_sf + gap / 2 + node_h:.2f});"
            )

    # --- Final ---
    for i, row in fi.iterrows():
        x = xpos_fi[0]
        lines += match_node(x, y_fi, str(row["team_a"]), str(row["team_b"]), str(row["winner"]), row.get("winner_pct"))
        for si, (sx, _) in enumerate(sf_centers):
            lines.append(
                rf"\draw[draw=qaline] ({sx:.2f},{y_sf - (gap / 2 + node_h):.2f}) -- "
                rf"({sx:.2f},{(y_sf + y_fi) / 2:.2f}) -- "
                rf"({x:.2f},{(y_sf + y_fi) / 2:.2f}) -- "
                rf"({x:.2f},{y_fi + gap / 2 + node_h:.2f});"
            )

    lines += [r"\end{tikzpicture}", r"\end{center}"]
    return "\n".join(lines)


def render_report_tex(
    results: pd.DataFrame,
    teams: pd.DataFrame,
    llm_text: str | None,
    params: SimParams | None,
    bracket_probable: pd.DataFrame | None = None,
    output_path: Path = REPORT_TEX_PATH,
) -> Path:
    """Renderizar el template LaTeX sustituyendo todos los placeholders.

    Lee ``template_reporte_wcup2026.tex``, reemplaza cada placeholder
    ``<<...>>`` con el contenido generado y escribe el archivo final.

    Parameters
    ----------
    results : pd.DataFrame
        DataFrame de resultados de la simulacion.
    teams : pd.DataFrame
        DataFrame original de equipos con ratings.
    llm_text : str or None
        Texto Markdown del analisis LLM; si es ``None`` se muestra aviso.
    params : SimParams or None
        Hiperparametros de la simulacion para el resumen ejecutivo.
    output_path : Path, optional
        Ruta de salida del archivo TEX.  Por defecto ``REPORT_TEX_PATH``.

    Returns
    -------
    Path
        Ruta del archivo TEX generado.
    """
    ensure_report_dir()
    template = REPORT_TEMPLATE_PATH.read_text(encoding="utf-8")
    replacements = {
        "<<REPORT_TITLE>>": "Reporte WCUP 2026",
        "<<REPORT_SUBTITLE>>": "Prediccion Monte Carlo y lectura cualitativa asistida por LLM",
        "<<REPORT_DATE>>": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "<<QA_TAGLINE>>": "Simplificando la calidad y los datos para un mundo complejo",
        "<<QA_WEBSITE>>": "qualityanalytics.net",
        "<<QA_CONTACT>>": "info@qualityanalytics.net",
        "<<EXECUTIVE_SUMMARY>>": latex_escape(build_summary(results, params)),
        "<<KPI_STRIP>>": build_kpi_strip(results),
        "<<CHAMPION_CHART>>": build_champion_chart(results),
        "<<TOP_TABLE>>": build_top_table(results),
        "<<GROUP_TABLE>>": build_group_table(results, teams),
        "<<BRACKET_CHART>>": build_bracket_chart(bracket_probable),
        "<<LLM_RESULT>>": markdown_to_latex(llm_text),
    }
    tex = template
    for placeholder, value in replacements.items():
        tex = tex.replace(placeholder, value)
    output_path.write_text(tex, encoding="utf-8")
    return output_path


def compile_report(tex_path: Path = REPORT_TEX_PATH) -> Path:
    """Compilar el archivo TEX con pdflatex y devolver la ruta del PDF.

    Ejecuta pdflatex dos veces para resolver referencias internas.
    Lanza ``RuntimeError`` si la compilacion falla.

    Parameters
    ----------
    tex_path : Path, optional
        Ruta del archivo TEX a compilar.  Por defecto ``REPORT_TEX_PATH``.

    Returns
    -------
    Path
        Ruta del PDF generado (``REPORT_PDF_PATH``).

    Raises
    ------
    RuntimeError
        Si pdflatex termina con codigo de error distinto de cero.
    """
    ensure_report_dir()
    completed = None
    for _ in range(2):
        completed = subprocess.run(
            [
                "pdflatex",
                "-interaction=nonstopmode",
                f"-output-directory={REPORT_DIR}",
                str(tex_path),
            ],
            cwd=REPORT_DIR,
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            break
    if completed is None or completed.returncode != 0:
        output = "" if completed is None else completed.stdout + completed.stderr
        log_tail = "\n".join(output.splitlines()[-20:])
        raise RuntimeError(f"pdflatex fallo al compilar el reporte:\n{log_tail}")
    return REPORT_PDF_PATH


def generate_report(
    results: pd.DataFrame,
    teams: pd.DataFrame,
    llm_text: str | None,
    params: SimParams | None,
    bracket_probable: pd.DataFrame | None = None,
    compile_pdf: bool = True,
) -> tuple[Path, Path | None]:
    """Generar el archivo TEX del reporte y opcionalmente compilar el PDF.

    Combina ``render_report_tex`` y ``compile_report`` en un unico punto
    de entrada conveniente.

    Parameters
    ----------
    results : pd.DataFrame
        DataFrame de resultados de la simulacion.
    teams : pd.DataFrame
        DataFrame original de equipos con ratings.
    llm_text : str or None
        Texto Markdown del analisis LLM.
    params : SimParams or None
        Hiperparametros de la simulacion.
    bracket_probable : pd.DataFrame or None, optional
        Cuadro de eliminacion mas probable para incluir en el reporte.
    compile_pdf : bool, optional
        Si ``True`` (por defecto), compila el TEX a PDF con pdflatex.

    Returns
    -------
    tuple[Path, Path or None]
        ``(tex_path, pdf_path)`` donde ``pdf_path`` es ``None`` si
        ``compile_pdf`` es ``False``.
    """
    tex_path = render_report_tex(results, teams, llm_text, params, bracket_probable=bracket_probable)
    pdf_path = compile_report(tex_path) if compile_pdf else None
    return tex_path, pdf_path
