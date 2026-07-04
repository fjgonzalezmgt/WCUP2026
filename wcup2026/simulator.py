"""Motor de simulacion Monte Carlo del torneo.

Implementa la logica completa de simulacion: calculo de goles esperados
mediante regresion de Poisson, partidos individuales, fase de grupos,
asignacion de mejores terceros al cuadro de eliminacion y simulacion de
multiples torneos completos para estimar probabilidades.
"""

from __future__ import annotations

from collections import defaultdict, Counter
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd

from wcup2026.bracket import R32_MATCHES, ROUND_TEMPLATES
from wcup2026.config import GROUPS, STAGE_COLUMNS
from wcup2026.data import prepare_teams
from wcup2026.parameters import SimParams


ATTACK_DEFENSE_EDGE_SCALE = 60.0
OVERALL_EDGE_SCALE = 180.0
KNOCKOUT_TIEBREAK_MIN_PROB = 0.25
KNOCKOUT_TIEBREAK_MAX_PROB = 0.75


def expected_goals(
    team_a: str,
    team_b: str,
    teams: dict[str, dict[str, Any]],
    params: SimParams,
) -> tuple[float, float]:
    """Calcular los goles esperados (lambda de Poisson) para cada equipo.

    Combina la diferencia de poder de ataque vs. defensa con la diferencia
    de rating global y aplica una bonificacion de anfitrion.  Las escalas
    estan deliberadamente suavizadas para evitar que pequenas ventajas de
    rating se acumulen en probabilidades de campeon demasiado extremas. Los
    lambdas resultantes se recortan al rango [0.18, 4.2].

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

    a_edge = (
        (a["attack_power"] - b["defense_power"]) / ATTACK_DEFENSE_EDGE_SCALE
        + (a["overall"] - b["overall"]) / OVERALL_EDGE_SCALE
    )
    b_edge = (
        (b["attack_power"] - a["defense_power"]) / ATTACK_DEFENSE_EDGE_SCALE
        + (b["overall"] - a["overall"]) / OVERALL_EDGE_SCALE
    )

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
    en el rating con ruido gaussiano.  Esa probabilidad se limita entre 25%
    y 75% para reflejar que un empate en eliminatoria ya implica un alto
    componente de azar en prorroga o penales.

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
        prob_a = float(np.clip(prob_a, KNOCKOUT_TIEBREAK_MIN_PROB, KNOCKOUT_TIEBREAK_MAX_PROB))
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
        """Intentar asignar terceros a partidos restantes mediante backtracking.

        Parameters
        ----------
        index : int
            Indice del partido actual dentro de ``ordered_matches``.
        available : dict[str, str]
            Mapeo ``{grupo: equipo}`` con los terceros aun sin asignar.

        Returns
        -------
        bool
            ``True`` si se encontro una asignacion valida para todos los
            partidos restantes; ``False`` si no fue posible.
        """
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


def _select_coherent_r32_pairs(
    pair_counts: dict[int, Counter],
    candidate_limit: int = 12,
) -> dict[int, tuple[str, str]]:
    """Seleccionar pares de R32 sin equipos repetidos y con alta frecuencia.

    Construye una asignacion global de emparejamientos para la ronda de 32
    maximizando la suma de frecuencias observadas en ``pair_counts`` y
    exigiendo que cada equipo aparezca en un unico partido.

    Si no encuentra solucion completa, retorna una solucion greedy de
    respaldo.
    """
    match_ids = sorted(R32_MATCHES)
    candidates: dict[int, list[tuple[tuple[str, str], int]]] = {}
    for match_id in match_ids:
        ranked = pair_counts[match_id].most_common(candidate_limit)
        if not ranked:
            continue
        candidates[match_id] = ranked

    ordered_matches = sorted(match_ids, key=lambda match_id: len(candidates.get(match_id, [])))
    if not ordered_matches:
        return {}

    remaining_best: list[int] = []
    running = 0
    for match_id in reversed(ordered_matches):
        top = candidates.get(match_id, [((), 0)])[0][1]
        running += top
        remaining_best.append(running)
    remaining_best.reverse()

    best_score = -1
    best_assignment: dict[int, tuple[str, str]] = {}

    def backtrack(
        index: int,
        used: set[str],
        current_score: int,
        current_assignment: dict[int, tuple[str, str]],
    ) -> None:
        nonlocal best_score, best_assignment
        if index == len(ordered_matches):
            if current_score > best_score:
                best_score = current_score
                best_assignment = dict(current_assignment)
            return

        if current_score + remaining_best[index] <= best_score:
            return

        match_id = ordered_matches[index]
        for pair, count in candidates.get(match_id, []):
            team_a, team_b = pair
            if team_a in used or team_b in used:
                continue
            current_assignment[match_id] = pair
            used.add(team_a)
            used.add(team_b)
            backtrack(index + 1, used, current_score + count, current_assignment)
            used.remove(team_a)
            used.remove(team_b)
            current_assignment.pop(match_id, None)

    backtrack(0, set(), 0, {})
    if len(best_assignment) == len(match_ids):
        return best_assignment

    fallback: dict[int, tuple[str, str]] = {}
    used: set[str] = set()
    for match_id in ordered_matches:
        chosen: tuple[str, str] | None = None
        for pair, _count in candidates.get(match_id, []):
            team_a, team_b = pair
            if team_a in used or team_b in used:
                continue
            chosen = pair
            break
        if chosen is None and candidates.get(match_id):
            chosen = candidates[match_id][0][0]
        if chosen is None:
            continue
        fallback[match_id] = chosen
        used.update(chosen)
    return fallback


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


def simulate_bracket_sample(df: pd.DataFrame, params: SimParams) -> pd.DataFrame:
    """Simular un torneo completo y devolver los partidos de eliminatoria.

    Corre una unica simulacion determinista (con la semilla del params) y
    devuelve todos los partidos desde la ronda de 32 hasta la final con los
    equipos que jugaron y el ganador de cada uno.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame de equipos con ratings y metadatos.
    params : SimParams
        Hiperparametros de la simulacion.

    Returns
    -------
    pd.DataFrame
        DataFrame con columnas ``round``, ``match_id``, ``team_a``,
        ``team_b`` y ``winner``, una fila por partido de eliminatoria.
    """
    clean = df.copy()
    clean["group"] = clean["group"].astype(str).str.upper().str.strip()
    teams = prepare_teams(clean, params)
    rng = np.random.default_rng(params.seed)

    slots: dict[str, str] = {}
    thirds: list[dict[str, Any]] = []

    for group in GROUPS:
        group_teams = clean.loc[clean["group"] == group, "team"].tolist()
        table = simulate_group(group, group_teams, teams, rng, params)
        slots[f"1{group}"] = table[0]["team"]
        slots[f"2{group}"] = table[1]["team"]
        slots[f"3{group}"] = table[2]["team"]
        thirds.append(table[2])

    best_thirds = rank_thirds(thirds)[:8]
    third_assignments = assign_third_slots(best_thirds)

    rows: list[dict[str, Any]] = []
    winners: dict[int, str] = {}

    for match_id, (left, right) in R32_MATCHES.items():
        team_a = resolve_slot(left, slots, third_assignments, match_id)
        team_b = resolve_slot(right, slots, third_assignments, match_id)
        winner = simulate_knockout_pair(team_a, team_b, teams, rng, params)
        winners[match_id] = winner
        rows.append({
            "round": "round_of_32",
            "match_id": match_id,
            "team_a": team_a,
            "team_b": team_b,
            "winner": winner,
            "winner_pct": None,
        })

    for round_name in ["round_of_16", "quarterfinal", "semifinal", "final"]:
        for match_id, (left_id, right_id) in ROUND_TEMPLATES[round_name].items():
            team_a = winners[left_id]
            team_b = winners[right_id]
            winner = simulate_knockout_pair(team_a, team_b, teams, rng, params)
            winners[match_id] = winner
            rows.append({
                "round": round_name,
                "match_id": match_id,
                "team_a": team_a,
                "team_b": team_b,
                "winner": winner,
                "winner_pct": None,
            })

    return pd.DataFrame(rows)


def _normalize_r32_fixture_overrides(
    r32_fixtures: pd.DataFrame | None,
    known_teams: set[str],
) -> dict[int, tuple[str, str]] | None:
    """Validar fixtures R32 externos y devolver ``match_id -> (team_a, team_b)``."""
    if r32_fixtures is None or r32_fixtures.empty:
        return None

    required = {"match_id", "team_a", "team_b"}
    missing = required.difference(r32_fixtures.columns)
    if missing:
        raise ValueError(
            "Faltan columnas en fixtures R32: " + ", ".join(sorted(missing))
        )

    clean = r32_fixtures[["match_id", "team_a", "team_b"]].copy()
    clean["match_id"] = pd.to_numeric(clean["match_id"], errors="coerce")
    clean = clean.dropna(subset=["match_id"])
    clean["match_id"] = clean["match_id"].astype(int)
    clean = clean.loc[clean["match_id"].between(73, 88)].copy()
    clean["team_a"] = clean["team_a"].astype(str).str.strip()
    clean["team_b"] = clean["team_b"].astype(str).str.strip()
    clean = clean.loc[(clean["team_a"] != "") & (clean["team_b"] != "")].copy()

    duplicated = clean.loc[clean["match_id"].duplicated(), "match_id"].tolist()
    if duplicated:
        raise ValueError(
            "Partidos R32 duplicados en fixtures: "
            + ", ".join(map(str, sorted(set(duplicated))))
        )

    if clean.empty:
        return None

    expected_match_ids = set(R32_MATCHES)
    actual_match_ids = set(clean["match_id"].tolist())
    if actual_match_ids != expected_match_ids:
        missing_ids = sorted(expected_match_ids - actual_match_ids)
        extra_ids = sorted(actual_match_ids - expected_match_ids)
        details = []
        if missing_ids:
            details.append("faltan " + ", ".join(map(str, missing_ids)))
        if extra_ids:
            details.append("sobran " + ", ".join(map(str, extra_ids)))
        raise ValueError("Fixtures R32 incompletos: " + "; ".join(details))

    unknown_teams = sorted(
        (set(clean["team_a"]) | set(clean["team_b"])) - known_teams
    )
    if unknown_teams:
        raise ValueError(
            "Fixtures R32 incluyen equipos no encontrados en ratings: "
            + ", ".join(unknown_teams)
        )

    duplicated_teams = clean[["team_a", "team_b"]].stack()
    duplicated_teams = duplicated_teams.loc[duplicated_teams.duplicated()].tolist()
    if duplicated_teams:
        raise ValueError(
            "Equipos repetidos en fixtures R32: "
            + ", ".join(sorted(set(duplicated_teams)))
        )

    return {
        int(row.match_id): (str(row.team_a), str(row.team_b))
        for row in clean.itertuples()
    }


def _prepare_group_results_context(
    df: pd.DataFrame,
    group_results: pd.DataFrame,
    params: SimParams,
    r32_fixtures: pd.DataFrame | None = None,
) -> tuple[
    pd.DataFrame,
    dict[str, dict[str, Any]],
    dict[str, str],
    dict[int, str],
    list[dict[str, Any]],
    dict[int, tuple[str, str]] | None,
]:
    """Preparar slots de eliminatoria desde resultados finales de grupos.

    ``group_results`` debe incluir al menos ``group``, ``position`` y
    ``team``. Las columnas ``points``, ``gf`` y ``ga`` son opcionales, pero
    se usan para ordenar los mejores terceros cuando estan disponibles.
    """
    required = {"group", "position", "team"}
    missing = required.difference(group_results.columns)
    if missing:
        raise ValueError(f"Faltan columnas en resultados de grupos: {', '.join(sorted(missing))}")

    clean = df.copy()
    clean["group"] = clean["group"].astype(str).str.upper().str.strip()
    teams = prepare_teams(clean, params)
    fixture_overrides = _normalize_r32_fixture_overrides(
        r32_fixtures,
        set(clean["team"].astype(str).str.strip()),
    )
    base_group_by_team = clean.set_index("team")["group"].to_dict()

    results = group_results.copy()
    results["group"] = results["group"].astype(str).str.upper().str.strip()
    results["team"] = results["team"].astype(str).str.strip()
    results["position"] = pd.to_numeric(results["position"], errors="coerce")
    results = results.dropna(subset=["group", "team", "position"])
    results["position"] = results["position"].astype(int)
    results = results.loc[results["position"].between(1, 4)].copy()

    if results.empty:
        raise ValueError("La tabla de resultados de grupos esta vacia.")

    unknown_teams = sorted(set(results["team"]) - set(base_group_by_team))
    if unknown_teams:
        raise ValueError(f"Equipos no encontrados en ratings: {', '.join(unknown_teams)}")

    duplicated_teams = results.loc[results["team"].duplicated(), "team"].tolist()
    if duplicated_teams:
        raise ValueError(f"Equipos duplicados en resultados de grupos: {', '.join(duplicated_teams)}")

    mismatched = [
        f"{row.team} ({row.group} vs {base_group_by_team[row.team]})"
        for row in results.itertuples()
        if base_group_by_team[row.team] != row.group
    ]
    if mismatched:
        raise ValueError(
            "Hay equipos asignados a un grupo distinto al dataset base: "
            + ", ".join(mismatched)
        )

    duplicates = results.duplicated(subset=["group", "position"], keep=False)
    if duplicates.any():
        bad = results.loc[duplicates, ["group", "position"]].drop_duplicates()
        labels = [f"{row.group}-{row.position}" for row in bad.itertuples()]
        raise ValueError(f"Posiciones duplicadas por grupo: {', '.join(labels)}")

    slots: dict[str, str] = {}
    thirds: list[dict[str, Any]] = []
    for group in GROUPS:
        group_rows = results.loc[results["group"] == group].copy()
        positions = set(group_rows["position"].tolist())
        required_positions = {1, 2, 3}
        if not required_positions.issubset(positions):
            missing_positions = sorted(required_positions - positions)
            raise ValueError(
                f"Grupo {group}: faltan posiciones {', '.join(map(str, missing_positions))}."
            )

        for position in [1, 2, 3]:
            row = group_rows.loc[group_rows["position"] == position].iloc[0]
            slots[f"{position}{group}"] = row["team"]

        third = group_rows.loc[group_rows["position"] == 3].iloc[0].to_dict()
        points = pd.to_numeric(third.get("points", 0), errors="coerce")
        gf = pd.to_numeric(third.get("gf", 0), errors="coerce")
        ga = pd.to_numeric(third.get("ga", 0), errors="coerce")
        points = 0 if pd.isna(points) else int(points)
        gf = 0 if pd.isna(gf) else int(gf)
        ga = 0 if pd.isna(ga) else int(ga)
        thirds.append({
            "team": third["team"],
            "group": group,
            "points": points,
            "gf": gf,
            "ga": ga,
            "gd": gf - ga,
            "overall": teams[third["team"]]["overall"],
            "jitter": 0.0,
        })

    automatic_qualifiers = {slots[f"1{group}"] for group in GROUPS}
    automatic_qualifiers.update(slots[f"2{group}"] for group in GROUPS)
    third_by_team = {row["team"]: row for row in thirds}
    scheduled_r32_pairs = fixture_overrides
    override_teams = (
        {team for fixture in fixture_overrides.values() for team in fixture}
        if fixture_overrides is not None
        else set()
    )

    override_third_teams = override_teams - automatic_qualifiers
    overrides_are_compatible = (
        fixture_overrides is not None
        and automatic_qualifiers.issubset(override_teams)
        and len(override_third_teams) == 8
        and override_third_teams.issubset(third_by_team)
    )

    if overrides_are_compatible:
        best_thirds = [
            third_by_team[team]
            for match_id in sorted(scheduled_r32_pairs)
            for team in scheduled_r32_pairs[match_id]
            if team in override_third_teams
        ]
        third_assignments = {
            match_id: team
            for match_id, fixture in scheduled_r32_pairs.items()
            for team in fixture
            if team in override_third_teams
        }
    else:
        if fixture_overrides is not None:
            raise ValueError(
                "Los fixtures R32 cargados no son compatibles con los "
                "clasificados de la tabla de grupos."
            )
        best_thirds = rank_thirds(thirds)[:8]
        third_assignments = assign_third_slots(best_thirds)

    return clean, teams, slots, third_assignments, best_thirds, scheduled_r32_pairs


def _simulate_knockout_rows(
    teams: dict[str, dict[str, Any]],
    slots: dict[str, str],
    third_assignments: dict[int, str],
    rng: np.random.Generator,
    params: SimParams,
    scheduled_r32_pairs: dict[int, tuple[str, str]] | None = None,
    fixed_winners: dict[int, str] | None = None,
) -> list[dict[str, Any]]:
    """Simular eliminatorias desde slots fijos y devolver filas de partidos."""
    rows: list[dict[str, Any]] = []
    winners: dict[int, str] = {}
    fixed_winners = fixed_winners or {}

    def pick_winner(match_id: int, team_a: str, team_b: str) -> str:
        fixed = fixed_winners.get(match_id)
        if fixed is not None:
            if fixed not in {team_a, team_b}:
                raise ValueError(
                    f"Partido {match_id}: ganador fijo '{fixed}' no coincide con {team_a} vs {team_b}."
                )
            return fixed
        return simulate_knockout_pair(team_a, team_b, teams, rng, params)

    for match_id, (left, right) in R32_MATCHES.items():
        if scheduled_r32_pairs is not None and match_id in scheduled_r32_pairs:
            team_a, team_b = scheduled_r32_pairs[match_id]
        else:
            team_a = resolve_slot(left, slots, third_assignments, match_id)
            team_b = resolve_slot(right, slots, third_assignments, match_id)
        winner = pick_winner(match_id, team_a, team_b)
        winners[match_id] = winner
        rows.append({
            "round": "round_of_32",
            "match_id": match_id,
            "team_a": team_a,
            "team_b": team_b,
            "winner": winner,
            "winner_pct": None,
        })

    for round_name in ["round_of_16", "quarterfinal", "semifinal", "final"]:
        for match_id, (left_id, right_id) in ROUND_TEMPLATES[round_name].items():
            team_a = winners[left_id]
            team_b = winners[right_id]
            winner = pick_winner(match_id, team_a, team_b)
            winners[match_id] = winner
            rows.append({
                "round": round_name,
                "match_id": match_id,
                "team_a": team_a,
                "team_b": team_b,
                "winner": winner,
                "winner_pct": None,
            })

    return rows


def _normalize_fixed_winners(knockout_results: pd.DataFrame | None) -> dict[int, str]:
    """Convertir una tabla editable de resultados de KO a mapeo ``match_id->winner``."""
    if knockout_results is None or knockout_results.empty:
        return {}

    required = {"match_id", "winner"}
    missing = required.difference(knockout_results.columns)
    if missing:
        raise ValueError(
            "Faltan columnas en resultados de eliminatorias: " + ", ".join(sorted(missing))
        )

    clean = knockout_results[["match_id", "winner"]].copy()
    clean["match_id"] = pd.to_numeric(clean["match_id"], errors="coerce")
    clean = clean.dropna(subset=["match_id"])
    clean["match_id"] = clean["match_id"].astype(int)
    clean["winner"] = clean["winner"].astype(str).str.strip()
    clean = clean.loc[clean["winner"] != ""].copy()
    clean = clean.loc[clean["match_id"].between(73, 104)].copy()

    duplicated = clean.loc[clean["match_id"].duplicated(), "match_id"].tolist()
    if duplicated:
        raise ValueError(
            "Partidos duplicados en resultados de eliminatorias: "
            + ", ".join(map(str, sorted(set(duplicated))))
        )

    return dict(zip(clean["match_id"].tolist(), clean["winner"].tolist()))


def build_knockout_state_from_group_results(
    df: pd.DataFrame,
    group_results: pd.DataFrame,
    params: SimParams,
    knockout_results: pd.DataFrame | None = None,
    r32_fixtures: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Construir estado de llaves (equipos y ganadores fijos) desde grupos.

    Devuelve una tabla editable para capturar resultados reales de KO por
    partido. Si existen ganadores fijos, avanza esos equipos para concretar
    cruces de rondas siguientes.
    """
    _clean, _teams, slots, third_assignments, _best_thirds, scheduled_r32_pairs = _prepare_group_results_context(
        df,
        group_results,
        params,
        r32_fixtures=r32_fixtures,
    )
    fixed_winners = _normalize_fixed_winners(knockout_results)

    rows: list[dict[str, Any]] = []
    winners_or_placeholders: dict[int, str] = {}

    for match_id, (left, right) in R32_MATCHES.items():
        if scheduled_r32_pairs is not None and match_id in scheduled_r32_pairs:
            team_a, team_b = scheduled_r32_pairs[match_id]
        else:
            team_a = resolve_slot(left, slots, third_assignments, match_id)
            team_b = resolve_slot(right, slots, third_assignments, match_id)

        winner = fixed_winners.get(match_id, "")
        if winner and winner not in {team_a, team_b}:
            raise ValueError(
                f"Partido {match_id}: ganador '{winner}' no coincide con {team_a} vs {team_b}."
            )
        winners_or_placeholders[match_id] = winner if winner else f"Ganador {match_id}"
        rows.append(
            {
                "round": "round_of_32",
                "match_id": match_id,
                "team_a": team_a,
                "team_b": team_b,
                "winner": winner,
            }
        )

    for round_name in ["round_of_16", "quarterfinal", "semifinal", "final"]:
        for match_id, (left_id, right_id) in ROUND_TEMPLATES[round_name].items():
            team_a = winners_or_placeholders[left_id]
            team_b = winners_or_placeholders[right_id]
            winner = fixed_winners.get(match_id, "")

            both_concrete = not team_a.startswith("Ganador ") and not team_b.startswith("Ganador ")
            if winner:
                if not both_concrete or winner not in {team_a, team_b}:
                    raise ValueError(
                        f"Partido {match_id}: ganador '{winner}' no coincide con {team_a} vs {team_b}."
                    )
                winners_or_placeholders[match_id] = winner
            else:
                winners_or_placeholders[match_id] = f"Ganador {match_id}"

            rows.append(
                {
                    "round": round_name,
                    "match_id": match_id,
                    "team_a": team_a,
                    "team_b": team_b,
                    "winner": winner,
                }
            )

    return pd.DataFrame(rows)


def simulate_bracket_from_group_results(
    df: pd.DataFrame,
    group_results: pd.DataFrame,
    params: SimParams,
    knockout_results: pd.DataFrame | None = None,
    r32_fixtures: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Simular una llave de eliminatorias condicionada a grupos ya jugados."""
    _clean, teams, slots, third_assignments, _best_thirds, scheduled_r32_pairs = _prepare_group_results_context(
        df,
        group_results,
        params,
        r32_fixtures=r32_fixtures,
    )
    fixed_winners = _normalize_fixed_winners(knockout_results)
    rng = np.random.default_rng(params.seed + 1000)
    return pd.DataFrame(
        _simulate_knockout_rows(
            teams,
            slots,
            third_assignments,
            rng,
            params,
            scheduled_r32_pairs=scheduled_r32_pairs,
            fixed_winners=fixed_winners,
        )
    )


def simulate_knockout_projection_from_group_results(
    df: pd.DataFrame,
    group_results: pd.DataFrame,
    params: SimParams,
    n: int | None = None,
    knockout_results: pd.DataFrame | None = None,
    r32_fixtures: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Estimar probabilidades por ronda usando grupos finales como condicion fija."""
    clean, teams, slots, third_assignments, best_thirds, scheduled_r32_pairs = _prepare_group_results_context(
        df,
        group_results,
        params,
        r32_fixtures=r32_fixtures,
    )
    fixed_winners = _normalize_fixed_winners(knockout_results)
    simulations = int(n or params.simulations)
    rng = np.random.default_rng(params.seed + 2000)
    team_names = clean["team"].tolist()
    stage_counts = {
        team: {stage: 0 for stage in STAGE_COLUMNS}
        for team in team_names
    }

    for group in GROUPS:
        stage_counts[slots[f"1{group}"]]["group_winner"] = simulations
        stage_counts[slots[f"2{group}"]]["group_runner_up"] = simulations
    for row in best_thirds:
        stage_counts[row["team"]]["best_third"] = simulations

    r32_teams = {slots[f"1{group}"] for group in GROUPS}
    r32_teams.update(slots[f"2{group}"] for group in GROUPS)
    r32_teams.update(row["team"] for row in best_thirds)
    for team in r32_teams:
        stage_counts[team]["round_of_32"] = simulations

    for _ in range(simulations):
        rows = _simulate_knockout_rows(
            teams,
            slots,
            third_assignments,
            rng,
            params,
            scheduled_r32_pairs=scheduled_r32_pairs,
            fixed_winners=fixed_winners,
        )
        bracket = {row["match_id"]: row["winner"] for row in rows}
        for match_id in range(73, 89):
            stage_counts[bracket[match_id]]["round_of_16"] += 1
        for match_id in ROUND_TEMPLATES["round_of_16"]:
            stage_counts[bracket[match_id]]["quarterfinal"] += 1
        for match_id in ROUND_TEMPLATES["quarterfinal"]:
            stage_counts[bracket[match_id]]["semifinal"] += 1
        for match_id in ROUND_TEMPLATES["semifinal"]:
            stage_counts[bracket[match_id]]["final"] += 1
        stage_counts[bracket[104]]["champion"] += 1

    base = clean.set_index("team")
    rows = []
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
            row[f"{stage}_pct"] = 100 * stage_counts[team][stage] / simulations
        rows.append(row)

    return pd.DataFrame(rows).sort_values(
        ["champion_pct", "final_pct", "semifinal_pct", "overall"],
        ascending=False,
    )


def simulate_bracket_most_probable_from_group_results(
    df: pd.DataFrame,
    group_results: pd.DataFrame,
    params: SimParams,
    n: int = 1000,
    knockout_results: pd.DataFrame | None = None,
    r32_fixtures: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Calcular la llave mas probable condicionada a resultados reales de grupos."""
    _clean, teams, slots, third_assignments, _best_thirds, scheduled_r32_pairs = _prepare_group_results_context(
        df,
        group_results,
        params,
        r32_fixtures=r32_fixtures,
    )
    fixed_winners = _normalize_fixed_winners(knockout_results)
    rng = np.random.default_rng(params.seed + 3000)
    pair_counts: dict[int, Counter] = defaultdict(Counter)
    head2head_counts: dict[int, dict[tuple, Counter]] = defaultdict(lambda: defaultdict(Counter))

    for _ in range(n):
        winners: dict[int, str] = {}
        for match_id, (left, right) in R32_MATCHES.items():
            if scheduled_r32_pairs is not None and match_id in scheduled_r32_pairs:
                team_a, team_b = scheduled_r32_pairs[match_id]
            else:
                team_a = resolve_slot(left, slots, third_assignments, match_id)
                team_b = resolve_slot(right, slots, third_assignments, match_id)
            winner = fixed_winners.get(match_id)
            if winner is not None:
                if winner not in {team_a, team_b}:
                    raise ValueError(
                        f"Partido {match_id}: ganador fijo '{winner}' no coincide con {team_a} vs {team_b}."
                    )
            else:
                winner = simulate_knockout_pair(team_a, team_b, teams, rng, params)
            winners[match_id] = winner
            pair = tuple(sorted([team_a, team_b]))
            pair_counts[match_id][pair] += 1
            head2head_counts[match_id][pair][winner] += 1

        for round_name in ["round_of_16", "quarterfinal", "semifinal", "final"]:
            for match_id, (left_id, right_id) in ROUND_TEMPLATES[round_name].items():
                team_a = winners[left_id]
                team_b = winners[right_id]
                winner = fixed_winners.get(match_id)
                if winner is not None:
                    if winner not in {team_a, team_b}:
                        raise ValueError(
                            f"Partido {match_id}: ganador fijo '{winner}' no coincide con {team_a} vs {team_b}."
                        )
                else:
                    winner = simulate_knockout_pair(team_a, team_b, teams, rng, params)
                winners[match_id] = winner
                pair = tuple(sorted([team_a, team_b]))
                pair_counts[match_id][pair] += 1
                head2head_counts[match_id][pair][winner] += 1

    coherent_winner: dict[int, str] = {}
    rows: list[dict[str, Any]] = []

    selected_pairs = _select_coherent_r32_pairs(pair_counts)
    for match_id in sorted(R32_MATCHES):
        selected_pair = selected_pairs.get(match_id)
        if selected_pair is None:
            selected_pair = pair_counts[match_id].most_common(1)[0][0]
        team_a, team_b = selected_pair
        head2head = head2head_counts[match_id].get(selected_pair, Counter())
        total_h2h = sum(head2head.values())
        winner = head2head.most_common(1)[0][0] if head2head else team_a
        win_count = head2head[winner] if total_h2h else 0
        pct = round(100 * win_count / total_h2h, 1) if total_h2h else None
        coherent_winner[match_id] = winner
        rows.append({
            "round": "round_of_32",
            "match_id": match_id,
            "team_a": team_a,
            "team_b": team_b,
            "winner": winner,
            "winner_pct": pct,
        })

    for round_name in ["round_of_16", "quarterfinal", "semifinal", "final"]:
        for match_id, (left_id, right_id) in ROUND_TEMPLATES[round_name].items():
            team_a = coherent_winner[left_id]
            team_b = coherent_winner[right_id]
            pair = tuple(sorted([team_a, team_b]))
            head2head = head2head_counts[match_id].get(pair, Counter())
            total_h2h = sum(head2head.values())
            if head2head:
                winner = head2head.most_common(1)[0][0]
                pct = round(100 * head2head[winner] / total_h2h, 1)
            else:
                winner = team_a if teams[team_a]["overall"] >= teams[team_b]["overall"] else team_b
                pct = None
            coherent_winner[match_id] = winner
            rows.append({
                "round": round_name,
                "match_id": match_id,
                "team_a": team_a,
                "team_b": team_b,
                "winner": winner,
                "winner_pct": pct,
            })

    return pd.DataFrame(rows)


def simulate_bracket_most_probable(
    df: pd.DataFrame,
    params: SimParams,
    n: int = 1000,
) -> pd.DataFrame:
    """Calcular el cuadro mas probable corriendo N torneos.

    Para cada posicion del cuadro de eliminacion (match_id fijo) acumula
    que equipo gano con mayor frecuencia en ``n`` simulaciones completas y
    devuelve esos equipos como el bracket "mas probable", incluyendo el
    porcentaje de victorias del ganador en esa posicion.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame de equipos con ratings y metadatos.
    params : SimParams
        Hiperparametros de la simulacion.
    n : int, optional
        Numero de torneos a simular.  Por defecto 1000.

    Returns
    -------
    pd.DataFrame
        DataFrame con columnas ``round``, ``match_id``, ``team_a``,
        ``team_b``, ``winner`` y ``winner_pct``.  Los equipos mostrados son
        los mas frecuentes en cada posicion del cuadro.
    """
    clean = df.copy()
    clean["group"] = clean["group"].astype(str).str.upper().str.strip()
    teams = prepare_teams(clean, params)
    rng = np.random.default_rng(params.seed)

    pair_counts: dict[int, Counter] = defaultdict(Counter)
    winner_counts: dict[int, Counter] = defaultdict(Counter)
    # head2head_counts[match_id][pair] -> Counter de ganadores cuando jugaron ese par
    head2head_counts: dict[int, dict[tuple, Counter]] = defaultdict(lambda: defaultdict(Counter))

    round_map: dict[int, str] = {}
    for mid in R32_MATCHES:
        round_map[mid] = "round_of_32"
    for rname, matches in ROUND_TEMPLATES.items():
        for mid in matches:
            round_map[mid] = rname

    for _ in range(n):
        slots: dict[str, str] = {}
        thirds: list[dict[str, Any]] = []

        for group in GROUPS:
            group_teams = clean.loc[clean["group"] == group, "team"].tolist()
            table = simulate_group(group, group_teams, teams, rng, params)
            slots[f"1{group}"] = table[0]["team"]
            slots[f"2{group}"] = table[1]["team"]
            slots[f"3{group}"] = table[2]["team"]
            thirds.append(table[2])

        best_thirds = rank_thirds(thirds)[:8]
        third_assignments = assign_third_slots(best_thirds)

        winners: dict[int, str] = {}
        for match_id, (left, right) in R32_MATCHES.items():
            team_a = resolve_slot(left, slots, third_assignments, match_id)
            team_b = resolve_slot(right, slots, third_assignments, match_id)
            winner = simulate_knockout_pair(team_a, team_b, teams, rng, params)
            winners[match_id] = winner
            pair = tuple(sorted([team_a, team_b]))
            pair_counts[match_id][pair] += 1
            winner_counts[match_id][winner] += 1
            head2head_counts[match_id][pair][winner] += 1

        for round_name in ["round_of_16", "quarterfinal", "semifinal", "final"]:
            for match_id, (left_id, right_id) in ROUND_TEMPLATES[round_name].items():
                team_a = winners[left_id]
                team_b = winners[right_id]
                winner = simulate_knockout_pair(team_a, team_b, teams, rng, params)
                winners[match_id] = winner
                pair = tuple(sorted([team_a, team_b]))
                pair_counts[match_id][pair] += 1
                winner_counts[match_id][winner] += 1
                head2head_counts[match_id][pair][winner] += 1

    # Construir el bracket coherente de forma greedy:
    # el ganador mas frecuente en cada posicion avanza, y el porcentaje
    # del partido siguiente se calcula SOLO sobre las simulaciones donde
    # esos dos equipos especificos se enfrentaron.
    coherent_winner: dict[int, str] = {}
    rows: list[dict[str, Any]] = []

    # --- Ronda de 32 ---
    selected_pairs = _select_coherent_r32_pairs(pair_counts)
    for match_id in sorted(R32_MATCHES):
        selected_pair = selected_pairs.get(match_id)
        if selected_pair is None:
            selected_pair = pair_counts[match_id].most_common(1)[0][0]
        team_a, team_b = selected_pair
        head2head = head2head_counts[match_id].get(selected_pair, Counter())
        total_h2h = sum(head2head.values())
        winner = head2head.most_common(1)[0][0] if head2head else team_a
        win_count = head2head[winner] if total_h2h else 0
        pct = round(100 * win_count / total_h2h, 1) if total_h2h else None
        coherent_winner[match_id] = winner
        rows.append({
            "round": "round_of_32",
            "match_id": match_id,
            "team_a": team_a,
            "team_b": team_b,
            "winner": winner,
            "winner_pct": pct,
        })

    # --- Rondas posteriores: usar los ganadores coherentes como equipos ---
    for round_name in ["round_of_16", "quarterfinal", "semifinal", "final"]:
        for match_id, (left_id, right_id) in ROUND_TEMPLATES[round_name].items():
            team_a = coherent_winner[left_id]
            team_b = coherent_winner[right_id]
            pair = tuple(sorted([team_a, team_b]))
            head2head = head2head_counts[match_id].get(pair, Counter())
            total_h2h = sum(head2head.values())
            if head2head:
                winner = head2head.most_common(1)[0][0]
                win_count = head2head[winner]
                pct = round(100 * win_count / total_h2h, 1)
            else:
                # Fallback: el equipo con mejor rating global gana
                winner = team_a
                pct = None
            coherent_winner[match_id] = winner
            rows.append({
                "round": round_name,
                "match_id": match_id,
                "team_a": team_a,
                "team_b": team_b,
                "winner": winner,
                "winner_pct": pct,
            })

    return pd.DataFrame(rows)
