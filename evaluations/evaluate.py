"""Script para evaluar submissions de WCUP 2026 Predictor.

Valida el formato de los CSVs, carga todos los archivos de la carpeta evaluations/,
calcula los Brier Scores por etapa y genera un reporte comparativo.

Uso:
    python evaluations/evaluate.py
    
    También acepta un CSV de resultados reales opcional:
    python evaluations/evaluate.py --results path/to/results.csv
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pandas as pd


# Todos los equipos en el torneo (orden alfabético)
VALID_TEAMS = [
    "Algeria",
    "Argentina",
    "Australia",
    "Austria",
    "Belgium",
    "Bosnia and Herzegovina",
    "Brazil",
    "Cabo Verde",
    "Canada",
    "Colombia",
    "Congo DR",
    "Cote d'Ivoire",
    "Croatia",
    "Curacao",
    "Czechia",
    "Ecuador",
    "Egypt",
    "England",
    "France",
    "Germany",
    "Ghana",
    "Haiti",
    "IR Iran",
    "Iraq",
    "Japan",
    "Jordan",
    "Korea Republic",
    "Mexico",
    "Morocco",
    "Netherlands",
    "New Zealand",
    "Norway",
    "Panama",
    "Paraguay",
    "Portugal",
    "Qatar",
    "Saudi Arabia",
    "Scotland",
    "Senegal",
    "South Africa",
    "Spain",
    "Sweden",
    "Switzerland",
    "Tunisia",
    "Turkiye",
    "USA",
    "Uzbekistan",
    "Uruguay",
]

REQUIRED_COLUMNS = ["team", "prob_champion", "prob_final", "prob_semifinal"]
STAGE_WEIGHTS = {
    "prob_champion": 0.50,
    "prob_final": 0.30,
    "prob_semifinal": 0.20,
}


def validate_submission(df: pd.DataFrame, filename: str) -> tuple[bool, list[str]]:
    """Validar que el CSV tenga el formato correcto.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame con los datos de la submission.
    filename : str
        Nombre del archivo (para mensajes de error).
    
    Returns
    -------
    tuple[bool, list[str]]
        (válido, lista_de_errores)
    """
    errors = []
    
    # Verificar columnas
    missing_cols = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing_cols:
        errors.append(f"Faltan columnas: {', '.join(missing_cols)}")
    
    extra_cols = set(df.columns) - set(REQUIRED_COLUMNS)
    if extra_cols:
        errors.append(f"Columnas extra: {', '.join(extra_cols)}")
    
    if errors:
        return False, errors
    
    # Verificar cantidad de equipos
    if len(df) != 48:
        errors.append(f"Se esperan 48 equipos, se encontraron {len(df)}")
    
    # Verificar nombres de equipos
    invalid_teams = set(df["team"]) - set(VALID_TEAMS)
    if invalid_teams:
        errors.append(f"Equipos inválidos: {', '.join(sorted(invalid_teams))}")
    
    missing_teams = set(VALID_TEAMS) - set(df["team"])
    if missing_teams:
        errors.append(f"Equipos faltantes: {', '.join(sorted(missing_teams))}")
    
    # Verificar duplicados
    if df["team"].duplicated().any():
        dups = df.loc[df["team"].duplicated(), "team"].tolist()
        errors.append(f"Equipos duplicados: {', '.join(dups)}")
    
    # Verificar rangos de probabilidades
    for col in ["prob_champion", "prob_final", "prob_semifinal"]:
        invalid = ((df[col] < 0) | (df[col] > 1)).sum()
        if invalid > 0:
            errors.append(f"Columna {col}: {invalid} valores fuera de rango [0, 1]")
    
    return len(errors) == 0, errors


def calculate_brier_score(predictions: pd.Series, actuals: pd.Series) -> float:
    """Calcular el Brier Score para una etapa.
    
    Brier Score = promedio((predicción - actual)^2)
    
    Parameters
    ----------
    predictions : pd.Series
        Probabilidades predichas [0, 1].
    actuals : pd.Series
        Resultados reales (0 o 1).
    
    Returns
    -------
    float
        Brier Score (menor es mejor).
    """
    return ((predictions - actuals) ** 2).mean()


def load_ground_truth() -> pd.DataFrame:
    """Cargar los resultados reales del torneo.
    
    Si no existen resultados reales, retorna un DataFrame vacío.
    El usuario puede proporcionar --results path/to/results.csv
    
    Returns
    -------
    pd.DataFrame
        DataFrame con columnas: team, champion, final, semifinal
    """
    # Por ahora retorna vacío - el usuario puede crear uno
    return pd.DataFrame()


def evaluate_submission(csv_path: Path, ground_truth: pd.DataFrame) -> dict[str, Any]:
    """Evaluar un archivo de submission.
    
    Parameters
    ----------
    csv_path : Path
        Ruta al archivo CSV.
    ground_truth : pd.DataFrame
        DataFrame con los resultados reales.
    
    Returns
    -------
    dict[str, Any]
        Diccionario con nombre del archivo, estado de validación, y scores.
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
        result["errors"].append("No se pudieron hacer match todos los equipos con resultados reales")
        return result
    
    # Calcular Brier Scores por etapa
    result["brier_champion"] = calculate_brier_score(
        merged["prob_champion"],
        merged["champion"],
    )
    result["brier_final"] = calculate_brier_score(
        merged["prob_final"],
        merged["final"],
    )
    result["brier_semifinal"] = calculate_brier_score(
        merged["prob_semifinal"],
        merged["semifinal"],
    )
    
    # Calcular score final ponderado
    result["score_final"] = (
        STAGE_WEIGHTS["prob_champion"] * result["brier_champion"]
        + STAGE_WEIGHTS["prob_final"] * result["brier_final"]
        + STAGE_WEIGHTS["prob_semifinal"] * result["brier_semifinal"]
    )
    
    return result


def main() -> None:
    """Ejecutar validacion y evaluacion de todas las submissions.

    Returns
    -------
    None
        Imprime en consola la validacion de formato y, si existe
        ``ground_truth.csv``, los Brier Scores ordenados.
    """
    eval_dir = Path(__file__).parent
    
    print("=" * 80)
    print("EVALUACIÓN DE SUBMISSIONS - WCUP 2026 Predictor")
    print("=" * 80)
    print()
    
    # Buscar todos los CSVs
    csv_files = sorted([f for f in eval_dir.glob("*.csv") if f.name != "ground_truth.csv"])
    
    if not csv_files:
        print("❌ No se encontraron archivos CSV para evaluar.")
        return
    
    print(f"📁 Encontrados {len(csv_files)} archivo(s) de submission")
    print()
    
    # Cargar resultados reales si existen
    ground_truth_path = eval_dir / "ground_truth.csv"
    ground_truth = pd.DataFrame()
    
    if ground_truth_path.exists():
        try:
            ground_truth = pd.read_csv(ground_truth_path)
            print(f"✅ Resultados reales cargados desde: {ground_truth_path.name}")
        except Exception as e:
            print(f"⚠️  Error al cargar resultados reales: {e}")
    else:
        print(f"⚠️  Archivo ground_truth.csv no encontrado.")
        print(f"   Usa: {ground_truth_path.name} para cargar resultados reales")
        print()
    
    print()
    
    # Validar cada submission
    print("VALIDACIÓN DE FORMATO")
    print("-" * 80)
    
    results = []
    for csv_path in csv_files:
        result = evaluate_submission(csv_path, ground_truth)
        results.append(result)
        
        status = "✅" if result["valid"] else "❌"
        print(f"{status} {result['file']}")
        
        if result["errors"]:
            for error in result["errors"]:
                print(f"   ⚠️  {error}")
    
    print()
    
    # Mostrar scores si hay resultados reales
    if not ground_truth.empty:
        print("RESULTADOS DE EVALUACIÓN")
        print("-" * 80)
        
        results_sorted = sorted(
            [r for r in results if r["valid"] and r["score_final"] is not None],
            key=lambda x: x["score_final"],
        )
        
        if results_sorted:
            for i, result in enumerate(results_sorted, 1):
                print(f"{i}. {result['file']:<30}")
                print(f"   Champion:    {result['brier_champion']:.6f}")
                print(f"   Final:       {result['brier_final']:.6f}")
                print(f"   Semifinal:   {result['brier_semifinal']:.6f}")
                print(f"   Score Final: {result['score_final']:.6f}  ⭐")
                print()
        else:
            print("⚠️  No hay submissions válidas para mostrar scores")
    else:
        print("📝 Para ver scores de evaluación, crea archivo ground_truth.csv")
        print("   Estructura: team,champion,final,semifinal")
        print("              (valores 0 o 1)")


if __name__ == "__main__":
    main()
