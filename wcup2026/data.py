"""Carga, validacion y preparacion de datos de equipos.

Expone funciones para leer el CSV de ratings de equipos, validar su
estructura, y transformar los datos crudos en el diccionario de atributos
normalizado que consume el simulador.
"""

from __future__ import annotations

from io import StringIO
from typing import Any

import pandas as pd

from wcup2026.config import DATA_PATH, FEATURE_COLUMNS
from wcup2026.parameters import SimParams


def load_team_data(path=DATA_PATH) -> pd.DataFrame:
    """Cargar el CSV de ratings de equipos desde disco.

    Parameters
    ----------
    path : Path or str, optional
        Ruta al archivo CSV.  Por defecto usa ``DATA_PATH`` definido en
        ``config.py`` (``data/teams_seed.csv``).

    Returns
    -------
    pd.DataFrame
        DataFrame con una fila por equipo y columnas de atributos.
    """
    return pd.read_csv(path)


def dataframe_to_csv_text(df: pd.DataFrame) -> str:
    """Serializar un DataFrame a texto CSV sin indice.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame a serializar.

    Returns
    -------
    str
        Representacion CSV del DataFrame.
    """
    return df.to_csv(index=False)


def dataframe_from_csv_text(csv_text: str) -> pd.DataFrame:
    """Deserializar texto CSV a DataFrame.

    Parameters
    ----------
    csv_text : str
        Contenido CSV como cadena de texto.

    Returns
    -------
    pd.DataFrame
        DataFrame resultante de parsear el CSV.
    """
    return pd.read_csv(StringIO(csv_text))


def validate_team_data(df: pd.DataFrame) -> None:
    """Validar que el DataFrame de equipos cumpla el esquema requerido.

    Verifica que existan todas las columnas obligatorias, que no haya
    equipos duplicados y que el torneo tenga exactamente 12 grupos de
    4 equipos cada uno.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame de equipos a validar.

    Raises
    ------
    ValueError
        Si faltan columnas requeridas, hay equipos duplicados o la
        distribucion de grupos no es 12 x 4.
    """
    required = {
        "team",
        "group",
        "confederation",
        "is_host",
        "fifa_rank_proxy",
        "elo",
        *FEATURE_COLUMNS,
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")

    if df["team"].duplicated().any():
        duplicates = df.loc[df["team"].duplicated(), "team"].tolist()
        raise ValueError(f"Duplicated teams: {', '.join(duplicates)}")

    group_sizes = df.groupby("group")["team"].count().to_dict()
    invalid = {group: size for group, size in group_sizes.items() if size != 4}
    if len(group_sizes) != 12 or invalid:
        raise ValueError("The model expects 12 groups with 4 teams in each group.")


def prepare_teams(df: pd.DataFrame, params: SimParams) -> dict[str, dict[str, Any]]:
    """Transformar el DataFrame de equipos en un diccionario de atributos normalizado.

    Aplica limpieza de tipos, imputacion de medianas, escalado del Elo,
    calculo del rating global ponderado y computo de poderes de ataque y
    defensa compuestos.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame crudo de equipos (debe pasar ``validate_team_data``).
    params : SimParams
        Hiperparametros de la simulacion; los pesos ``elo_weight``,
        ``squad_weight``, ``form_weight`` y ``balance_weight`` controlan
        la formula del rating global.

    Returns
    -------
    dict[str, dict[str, Any]]
        Diccionario ``{nombre_equipo: {atributo: valor, ...}}`` listo para
        ser consumido por el simulador.  Incluye las claves ``overall``,
        ``attack_power``, ``defense_power``, ``elo_scaled`` e ``is_host``.
    """
    validate_team_data(df)
    clean = df.copy()
    clean["group"] = clean["group"].astype(str).str.upper().str.strip()
    clean["is_host"] = clean["is_host"].astype(int)

    for column in ["fifa_rank_proxy", "elo", *FEATURE_COLUMNS]:
        clean[column] = pd.to_numeric(clean[column], errors="coerce")
    clean = clean.fillna(clean.median(numeric_only=True))

    elo_min = clean["elo"].min()
    elo_max = clean["elo"].max()
    clean["elo_scaled"] = 100 * (clean["elo"] - elo_min) / max(1, elo_max - elo_min)
    clean["balance"] = (clean["attack"] + clean["defense"]) / 2

    total_weight = max(
        0.01,
        params.elo_weight + params.squad_weight + params.form_weight + params.balance_weight,
    )
    clean["overall"] = (
        params.elo_weight * clean["elo_scaled"]
        + params.squad_weight * clean["squad"]
        + params.form_weight * clean["form"]
        + params.balance_weight * clean["balance"]
    ) / total_weight

    clean["attack_power"] = (
        0.48 * clean["attack"]
        + 0.22 * clean["squad"]
        + 0.16 * clean["form"]
        + 0.14 * clean["elo_scaled"]
    )
    clean["defense_power"] = (
        0.54 * clean["defense"]
        + 0.20 * clean["squad"]
        + 0.12 * clean["form"]
        + 0.14 * clean["elo_scaled"]
    )

    return clean.set_index("team").to_dict(orient="index")

