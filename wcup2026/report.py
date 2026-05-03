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
    """Crear la carpeta de reporte si no existe."""
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
        Cadena con un decimal y el simbolo ``\%`` escapado para LaTeX,
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

    Convierte encabezados ``##``/``###`` a ``\section*``/``\subsection*``,
    listas con guion o asterisco a entornos ``itemize`` y parrafos de texto
    plano con ``\par``.

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

    Genera una fila de cuatro tarjetas con el comando ``\qakpicard``
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


def render_report_tex(
    results: pd.DataFrame,
    teams: pd.DataFrame,
    llm_text: str | None,
    params: SimParams | None,
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
    compile_pdf : bool, optional
        Si ``True`` (por defecto), compila el TEX a PDF con pdflatex.

    Returns
    -------
    tuple[Path, Path or None]
        ``(tex_path, pdf_path)`` donde ``pdf_path`` es ``None`` si
        ``compile_pdf`` es ``False``.
    """
    tex_path = render_report_tex(results, teams, llm_text, params)
    pdf_path = compile_report(tex_path) if compile_pdf else None
    return tex_path, pdf_path
