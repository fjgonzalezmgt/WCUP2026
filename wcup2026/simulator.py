"""Motor de simulacion Monte Carlo del torneo.

Implementa la logica completa de simulacion: calculo de goles esperados
mediante regresion de Poisson, partidos individuales, fase de grupos,
asignacion de mejores terceros al cuadro de eliminacion y simulacion de
multiples torneos completos para estimar probabilidades.
"""

from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd

from wcup2026.bracket import R32_MATCHES, ROUND_TEMPLATES
from wcup2026.config import GROUPS, STAGE_COLUMNS
from wcup2026.data import prepare_teams
from wcup2026.parameters import SimParams


def expected_goals(
    team_a: str,
    team_b: str,
    teams: dict[str, dict[str, Any]],
    params: SimParams,
) -> tuple[float, float]:
    """Calcular los goles esperados (lambda de Poisson) para cada equipo.

    Combina la diferencia de poder de ataque vs. defensa con la diferencia
    de rating global y aplica una bonificacion de anfitrion.  Los lambdas
    resultantes se recortan al rango [0.18, 4.2].

    Parameters
    ----------
    team_a : str
        Nombre del primer equipo.
    team_b : str
        Nombre del segundo equipo.
    teams : dict[str, dict[str, Any]]
        Diccionario de atributos de equipos generado por ``prepare_teams``.
    params : SimParams
        Hiperparametros de la simulacion.

    Returns
    -------
    tuple[float, float]
        ``(lambda_a, lambda_b)`` goles esperados para equipo A y B.
    """
    a = teams[team_a]
    b = teams[team_b]
    host_a = params.home_advantage if a["is_host"] else 0.0
    host_b = params.home_advantage if b["is_host"] else 0.0

    a_edge = (a["attack_power"] - b["defense_power"]) / 38 + (a["overall"] - b["overall"]) / 95
    b_edge = (b["attack_power"] - a["defense_power"]) / 38 + (b["overall"] - a["overall"]) / 95

    lam_a = params.base_goals * np.exp(a_edge + host_a)
    lam_b = params.base_goals * np.exp(b_edge + host_b)
    return float(np.clip(lam_a, 0.18, 4.2)), float(np.clip(lam_b, 0.18, 4.2))


def simulate_match(
    team_a: str,
    team_b: str,
    teams: dict[str, dict[str, Any]],
    rng: np.random.Generator,
    params: SimParams,
    knockout: bool = False,
) -> dict[str, Any]:
    """Simular un partido entre dos equipos.

    Sortea goles con distribucion Poisson.  En fase de grupos el partido
    puede terminar en empate (``winner=None``).  En eliminatorias, si los
    goles son iguales, se decide mediante una probabilidad logistica basada
    en el rating con ruido gaussiano.

    Parameters
    ----------
    team_a : str
        Nombre del primer equipo.
    team_b : str
        Nombre del segundo equipo.
    teams : dict[str, dict[str, Any]]
        Diccionario de atributos de equipos.
    rng : np.random.Generator
        Generador de numeros aleatorios.
    params : SimParams
        Hiperparametros de la simulacion.
    knockout : bool, optional
        Si ``True``, el partido debe tener ganador (prorroga/penales).  Por
        defecto ``False``.

    Returns
    -------
    dict[str, Any]
        Diccionario con claves ``team_a``, ``team_b``, ``goals_a``,
        ``goals_b``, ``winner``, ``lambda_a`` y ``lambda_b``.
    """
    lam_a, lam_b = expected_goals(team_a, team_b, teams, params)
    goals_a = int(rng.poisson(lam_a))
    goals_b = int(rng.poisson(lam_b))
    winner = None

    if goals_a > goals_b:
        winner = team_a
    elif goals_b > goals_a:
        winner = team_b
    elif knockout:
        diff = teams[team_a]["overall"] - teams[team_b]["overall"]
        diff += params.knockout_noise * rng.normal(0, 0.18)
        prob_a = 1 / (1 + np.exp(-diff / max(1.0, params.knockout_noise)))
        winner = team_a if rng.random() < prob_a else team_b

    return {
        "team_a": team_a,
        "team_b": team_b,
        "goals_a": goals_a,
        "goals_b": goals_b,
        "winner": winner,
        "lambda_a": lam_a,
        "lambda_b": lam_b,
    }


def empty_standing(team: str) -> dict[str, Any]:
    """Crear una fila de clasificacion vacia para un equipo.

    Parameters
    ----------
    team : str
        Nombre del equipo.

    Returns
    -------
    dict[str, Any]
        Diccionario con puntos, goles a favor/en contra, diferencia,
        victorias, empates y derrotas todos inicializados a 0.
    """
    return {
        "team": team,
        "points": 0,
        "gf": 0,
        "ga": 0,
        "gd": 0,
        "wins": 0,
        "draws": 0,
        "losses": 0,
    }


def simulate_group(
    group: str,
    group_teams: list[str],
    teams: dict[str, dict[str, Any]],
    rng: np.random.Generator,
    params: SimParams,
) -> list[dict[str, Any]]:
    """Simular la fase de grupos y devolver la clasificacion ordenada.

    Juega todos los partidos de round-robin entre los cuatro equipos del
    grupo y ordena por puntos, diferencia de goles, goles a favor, rating
    global y un pequeño jitter aleatorio como ultimo desempate.

    Parameters
    ----------
    group : str
        Letra identificadora del grupo (p.ej. ``"A"``).
    group_teams : list[str]
        Lista con los cuatro nombres de equipo del grupo.
    teams : dict[str, dict[str, Any]]
        Diccionario de atributos de equipos.
    rng : np.random.Generator
        Generador de numeros aleatorios.
    params : SimParams
        Hiperparametros de la simulacion.

    Returns
    -------
    list[dict[str, Any]]
        Lista de cuatro filas de clasificacion ordenadas de primero a
        cuarto.  Cada fila incluye puntos, goles, diferencia, grupo,
        rating global y jitter.
    """
    standings = {team: empty_standing(team) for team in group_teams}

    for team_a, team_b in combinations(group_teams, 2):
        result = simulate_match(team_a, team_b, teams, rng, params, knockout=False)
        goals_a = result["goals_a"]
        goals_b = result["goals_b"]

        standings[team_a]["gf"] += goals_a
        standings[team_a]["ga"] += goals_b
        standings[team_b]["gf"] += goals_b
        standings[team_b]["ga"] += goals_a

        if goals_a > goals_b:
            standings[team_a]["points"] += 3
            standings[team_a]["wins"] += 1
            standings[team_b]["losses"] += 1
        elif goals_b > goals_a:
            standings[team_b]["points"] += 3
            standings[team_b]["wins"] += 1
            standings[team_a]["losses"] += 1
        else:
            standings[team_a]["points"] += 1
            standings[team_b]["points"] += 1
            standings[team_a]["draws"] += 1
            standings[team_b]["draws"] += 1

    rows = []
    for row in standings.values():
        row["gd"] = row["gf"] - row["ga"]
        row["group"] = group
        row["overall"] = teams[row["team"]]["overall"]
        row["jitter"] = rng.random() / 1000
        rows.append(row)

    return sorted(
        rows,
        key=lambda row: (
            row["points"],
            row["gd"],
            row["gf"],
            row["overall"],
            row["jitter"],
        ),
        reverse=True,
    )


def rank_thirds(thirds: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Ordenar la lista de terceros de grupo de mejor a peor.

    Usa el mismo criterio que la clasificacion de grupo: puntos,
    diferencia de goles, goles a favor, rating global y jitter.

    Parameters
    ----------
    thirds : list[dict[str, Any]]
        Lista de filas correspondientes a los terceros clasificados de
        cada grupo (una por grupo).

    Returns
    -------
    list[dict[str, Any]]
        Lista ordenada de mayor a menor rendimiento.
    """
    return sorted(
        thirds,
        key=lambda row: (
            row["points"],
            row["gd"],
            row["gf"],
            row["overall"],
            row["jitter"],
        ),
        reverse=True,
    )


def assign_third_slots(best_thirds: list[dict[str, Any]]) -> dict[int, str]:
    """Asignar los ocho mejores terceros a sus partidos de la ronda de 32.

    Sigue las ventanas de grupos publicadas por FIFA: cada partido de la
    ronda de 32 que enfrenta a un tercero solo acepta terceros de
    determinados grupos.  Usa backtracking para encontrar una asignacion
    valida y cae a una heuristica greedy si no la encuentra.

    Parameters
    ----------
    best_thirds : list[dict[str, Any]]
        Los ocho mejores terceros ya ordenados (salida de
        ``rank_thirds``[:8]).

    Returns
    -------
    dict[int, str]
        Mapeo ``{match_id: nombre_equipo}`` que indica que tercero juega
        en cada partido de la ronda de 32 que requiere un tercero.
    """
    remaining = {row["group"]: row["team"] for row in best_thirds}
    third_matches = {
        match_id: spec[1][1]
        for match_id, spec in R32_MATCHES.items()
        if spec[1][0] == "third"
    }

    ordered_matches = sorted(
        third_matches,
        key=lambda match_id: len(set(third_matches[match_id]).intersection(remaining)),
    )
    assignments: dict[int, str] = {}

    def backtrack(index: int, available: dict[str, str]) -> bool:
        if index == len(ordered_matches):
            return True

        match_id = ordered_matches[index]
        for group in third_matches[match_id]:
            if group not in available:
                continue
            assignments[match_id] = available[group]
            next_available = dict(available)
            next_available.pop(group)
            if backtrack(index + 1, next_available):
                return True
            assignments.pop(match_id, None)
        return False

    if backtrack(0, remaining):
        return assignments

    fallback: dict[int, str] = {}
    pool = list(remaining.items())
    for match_id in sorted(third_matches):
        allowed = third_matches[match_id]
        selected_idx = next(
            (idx for idx, (group, _team) in enumerate(pool) if group in allowed),
            0,
        )
        _group, team = pool.pop(selected_idx)
        fallback[match_id] = team
    return fallback


def resolve_slot(
    reference: tuple[str, Any],
    slots: dict[str, str],
    thirds: dict[int, str],
    match_id: int,
) -> str:
    """Resolver una referencia de cuadro al nombre real del equipo.

    Parameters
    ----------
    reference : tuple[str, Any]
        Par ``(tipo, valor)`` donde ``tipo`` es ``"slot"`` o ``"third"``.
    slots : dict[str, str]
        Diccionario ``{clave_slot: equipo}`` con ganadores y subcampeones
        de grupo (p.ej. ``{"1A": "Espana", "2B": "Francia", ...}``).
    thirds : dict[int, str]
        Asignacion de terceros por partido generada por
        ``assign_third_slots``.
    match_id : int
        ID numerico del partido que se esta resolviendo.

    Returns
    -------
    str
        Nombre del equipo que ocupa ese lugar del cuadro.

    Raises
    ------
    ValueError
        Si ``tipo`` no es ``"slot"`` ni ``"third"``.
    """
    kind, value = reference
    if kind == "slot":
        return slots[value]
    if kind == "third":
        return thirds[match_id]
    raise ValueError(f"Unknown reference kind: {kind}")


def simulate_knockout_pair(
    team_a: str,
    team_b: str,
    teams: dict[str, dict[str, Any]],
    rng: np.random.Generator,
    params: SimParams,
) -> str:
    """Simular un partido de eliminacion y devolver el ganador.

    Envuelve ``simulate_match`` con ``knockout=True`` para garantizar
    que siempre haya un ganador.

    Parameters
    ----------
    team_a : str
        Nombre del primer equipo.
    team_b : str
        Nombre del segundo equipo.
    teams : dict[str, dict[str, Any]]
        Diccionario de atributos de equipos.
    rng : np.random.Generator
        Generador de numeros aleatorios.
    params : SimParams
        Hiperparametros de la simulacion.

    Returns
    -------
    str
        Nombre del equipo ganador.
    """
    return simulate_match(team_a, team_b, teams, rng, params, knockout=True)["winner"]


def simulate_one_tournament(
    df: pd.DataFrame,
    teams: dict[str, dict[str, Any]],
    rng: np.random.Generator,
    params: SimParams,
) -> dict[str, Any]:
    """Simular un torneo completo de principio a fin.

    Ejecuta la fase de grupos de los 12 grupos, selecciona los ocho
    mejores terceros, asigna el cuadro de eliminacion y resuelve todas
    las rondas hasta la final.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame de equipos con columna ``group``.
    teams : dict[str, dict[str, Any]]
        Diccionario de atributos de equipos (salida de ``prepare_teams``).
    rng : np.random.Generator
        Generador de numeros aleatorios.
    params : SimParams
        Hiperparametros de la simulacion.

    Returns
    -------
    dict[str, Any]
        Diccionario con claves:

        - ``counts`` : ``dict[str, set[str]]`` equipos que alcanzaron cada
          etapa.
        - ``champion`` : str nombre del campeon.
        - ``runner_up`` : str nombre del subcampeon.
        - ``group_tables`` : ``dict[str, list]`` clasificaciones de grupo.
        - ``best_thirds`` : ``list`` mejores ocho terceros.
    """
    slots: dict[str, str] = {}
    thirds: list[dict[str, Any]] = []
    group_tables: dict[str, list[dict[str, Any]]] = {}

    for group in GROUPS:
        group_teams = df.loc[df["group"] == group, "team"].tolist()
        table = simulate_group(group, group_teams, teams, rng, params)
        group_tables[group] = table
        slots[f"1{group}"] = table[0]["team"]
        slots[f"2{group}"] = table[1]["team"]
        slots[f"3{group}"] = table[2]["team"]
        thirds.append(table[2])

    best_thirds = rank_thirds(thirds)[:8]
    third_assignments = assign_third_slots(best_thirds)

    counts = {stage: set() for stage in STAGE_COLUMNS}
    for group in GROUPS:
        counts["group_winner"].add(slots[f"1{group}"])
        counts["group_runner_up"].add(slots[f"2{group}"])
    for row in best_thirds:
        counts["best_third"].add(row["team"])

    r32_teams = {slots[f"1{group}"] for group in GROUPS}
    r32_teams.update(slots[f"2{group}"] for group in GROUPS)
    r32_teams.update(row["team"] for row in best_thirds)
    counts["round_of_32"].update(r32_teams)

    winners: dict[int, str] = {}
    for match_id, (left, right) in R32_MATCHES.items():
        team_a = resolve_slot(left, slots, third_assignments, match_id)
        team_b = resolve_slot(right, slots, third_assignments, match_id)
        winners[match_id] = simulate_knockout_pair(team_a, team_b, teams, rng, params)
    counts["round_of_16"].update(winners[match_id] for match_id in range(73, 89))

    for round_name in ["round_of_16", "quarterfinal", "semifinal", "final"]:
        next_stage = {
            "round_of_16": "quarterfinal",
            "quarterfinal": "semifinal",
            "semifinal": "final",
            "final": "champion",
        }[round_name]

        for match_id, (left_id, right_id) in ROUND_TEMPLATES[round_name].items():
            team_a = winners[left_id]
            team_b = winners[right_id]
            winners[match_id] = simulate_knockout_pair(team_a, team_b, teams, rng, params)

        counts[next_stage].update(winners[match_id] for match_id in ROUND_TEMPLATES[round_name])

    final_match = ROUND_TEMPLATES["final"][104]
    finalist_a = winners[final_match[0]]
    finalist_b = winners[final_match[1]]
    champion = winners[104]
    runner_up = finalist_b if champion == finalist_a else finalist_a

    return {
        "counts": counts,
        "champion": champion,
        "runner_up": runner_up,
        "group_tables": group_tables,
        "best_thirds": best_thirds,
    }


def simulate_many(df: pd.DataFrame, params: SimParams) -> pd.DataFrame:
    """Ejecutar la simulacion Monte Carlo completa y agregar probabilidades.

    Corre ``params.simulations`` torneos completos y calcula la frecuencia
    relativa con que cada equipo alcanza cada etapa del torneo.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame de equipos con ratings y metadatos.
    params : SimParams
        Hiperparametros de la simulacion, incluyendo numero de iteraciones
        y semilla aleatoria.

    Returns
    -------
    pd.DataFrame
        DataFrame con una fila por equipo ordenado por probabilidad de
        campeon descendente.  Contiene columnas de metadatos del equipo,
        ratings y columnas ``<etapa>_pct`` (en %) para cada etapa definida
        en ``STAGE_COLUMNS``.
    """
    clean = df.copy()
    clean["group"] = clean["group"].astype(str).str.upper().str.strip()
    teams = prepare_teams(clean, params)
    rng = np.random.default_rng(params.seed)

    team_names = clean["team"].tolist()
    stage_counts = {
        team: {stage: 0 for stage in STAGE_COLUMNS}
        for team in team_names
    }

    for _ in range(params.simulations):
        result = simulate_one_tournament(clean, teams, rng, params)
        for stage, stage_teams in result["counts"].items():
            for team in stage_teams:
                stage_counts[team][stage] += 1

    rows = []
    base = clean.set_index("team")
    for team in team_names:
        row = {
            "team": team,
            "group": base.loc[team, "group"],
            "confederation": base.loc[team, "confederation"],
            "is_host": int(base.loc[team, "is_host"]),
            "overall": teams[team]["overall"],
            "attack_power": teams[team]["attack_power"],
            "defense_power": teams[team]["defense_power"],
        }
        for stage in STAGE_COLUMNS:
            row[f"{stage}_pct"] = 100 * stage_counts[team][stage] / params.simulations
        rows.append(row)

    return pd.DataFrame(rows).sort_values(
        ["champion_pct", "final_pct", "semifinal_pct", "overall"],
        ascending=False,
    )


def describe_matchup(
    team_a: str,
    team_b: str,
    df: pd.DataFrame,
    params: SimParams,
    samples: int = 20000,
) -> dict[str, float]:
    """Estimar probabilidades y goles esperados para un duelo especifico.

    Simula un numero elevado de partidos en fase de grupos entre dos
    equipos y calcula estadisticas de victoria, empate y xG medios.

    Parameters
    ----------
    team_a : str
        Nombre del primer equipo.
    team_b : str
        Nombre del segundo equipo.
    df : pd.DataFrame
        DataFrame completo de equipos (necesario para ``prepare_teams``).
    params : SimParams
        Hiperparametros de la simulacion.
    samples : int, optional
        Numero de partidos a simular.  Por defecto 20 000.

    Returns
    -------
    dict[str, float]
        Diccionario con claves:

        - ``team_a_win_pct`` : probabilidad de victoria del equipo A (%).
        - ``draw_pct`` : probabilidad de empate (%).
        - ``team_b_win_pct`` : probabilidad de victoria del equipo B (%).
        - ``team_a_xg`` : goles esperados medios del equipo A.
        - ``team_b_xg`` : goles esperados medios del equipo B.
    """
    teams = prepare_teams(df, params)
    rng = np.random.default_rng(params.seed + 17)
    wins_a = 0
    draws = 0
    goals_a = 0
    goals_b = 0

    for _ in range(samples):
        result = simulate_match(team_a, team_b, teams, rng, params, knockout=False)
        goals_a += result["goals_a"]
        goals_b += result["goals_b"]
        if result["goals_a"] > result["goals_b"]:
            wins_a += 1
        elif result["goals_a"] == result["goals_b"]:
            draws += 1

    return {
        "team_a_win_pct": 100 * wins_a / samples,
        "draw_pct": 100 * draws / samples,
        "team_b_win_pct": 100 * (samples - wins_a - draws) / samples,
        "team_a_xg": goals_a / samples,
        "team_b_xg": goals_b / samples,
    }

