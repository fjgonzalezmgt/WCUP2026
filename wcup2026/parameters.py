"""Parametros del motor de simulacion.

Contiene el dataclass inmutable ``SimParams`` con todos los hiperparametros
necesarios para controlar el comportamiento del simulador Monte Carlo.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SimParams:
    """Hiperparametros de la simulacion Monte Carlo del torneo.

    Todos los campos son de solo lectura (``frozen=True``).  Los valores
    predeterminados corresponden a una calibracion razonable para el
    Mundial 2026.

    Parameters
    ----------
    simulations : int
        Numero de torneos completos a simular.
    seed : int
        Semilla del generador de numeros aleatorios para reproducibilidad.
    base_goals : float
        Goles esperados base por equipo y partido antes de aplicar
        diferencias de rating.
    home_advantage : float
        Incremento al lambda de goles para los paises anfitriones
        (Canada, Mexico, EE.UU.).
    knockout_noise : float
        Escala del ruido gaussiano en la probabilidad de prorroga/penales;
        valores altos aumentan el factor suerte en eliminatorias.
    elo_weight : float
        Peso del componente Elo en el rating global del equipo.
    squad_weight : float
        Peso de la calidad de plantilla en el rating global.
    form_weight : float
        Peso de la forma reciente en el rating global.
    balance_weight : float
        Peso del promedio ataque+defensa en el rating global.
    """
    simulations: int = 5000
    seed: int = 2026
    base_goals: float = 1.35
    home_advantage: float = 0.10
    knockout_noise: float = 9.0
    elo_weight: float = 0.45
    squad_weight: float = 0.25
    form_weight: float = 0.15
    balance_weight: float = 0.15

