# WCUP 2026 Predictor

App de Streamlit para simular el Mundial 2026 con Monte Carlo, modelo Poisson y una capa opcional de analisis LLM.

## Ejecutar con conda

```powershell
conda env create -f environment.yml
conda activate wcup2026
streamlit run app.py
```

El archivo `.env` debe vivir en la raiz del proyecto y contener:

```text
OPENAI_API_KEY=tu_clave
```

La clave no se importa al codigo. La app llama `load_dotenv()` y el SDK de OpenAI lee `OPENAI_API_KEY` desde el entorno.

## Que incluye

- 48 equipos y 12 grupos de Mundial 2026.
- Avance de dos primeros y ocho mejores terceros.
- Ronda de 32 basada en el calendario FIFA.
- Ratings editables: Elo aproximado, ataque, defensa, plantilla y forma.
- Probabilidades de grupo, ronda de 32, octavos, cuartos, semifinal, final y campeon.
- Modulo LLM para explicar escenarios y sugerir ajustes cualitativos.

## Por que Monte Carlo

La idea viene de modelos publicos que han pronosticado Mundiales anteriores. FiveThirtyEight explica que primero convierte ratings de equipos en probabilidades partido por partido usando goles esperados y distribuciones Poisson; despues transforma esas probabilidades en un pronostico de torneo con simulaciones Monte Carlo. The Alan Turing Institute uso una idea parecida para 2022: modelo estadistico tipo Dixon-Coles/Bayesiano para estimar marcadores y luego corrio todo el calendario 100,000 veces para ver que equipos ganaban con mas frecuencia.

La justificacion es practica: un Mundial no depende solo de "quien es mejor", sino tambien de grupos, empates, diferencia de goles, mejores terceros, cruces de eliminatoria, penales y ruta al titulo. Monte Carlo permite simular miles de torneos completos y convertir esa incertidumbre en probabilidades faciles de leer: pasar grupo, llegar a semifinal, final y campeon. Por eso la app no da un unico ganador fijo; entrega probabilidades ajustables.

## Estructura del codigo

- `app.py`: entrada minima para Streamlit.
- `wcup2026/config.py`: rutas, URLs, grupos, columnas y colores.
- `wcup2026/parameters.py`: parametros del simulador.
- `wcup2026/data.py`: lectura, validacion y preparacion de ratings.
- `wcup2026/bracket.py`: estructura de ronda de 32 y eliminatorias.
- `wcup2026/simulator.py`: Poisson, grupos, eliminatorias y Monte Carlo.
- `wcup2026/llm.py`: payload y llamada segura al SDK de OpenAI.
- `wcup2026/ui.py`: componentes visuales y flujo de Streamlit.

## Ideas para mejorar el LLM

- Leer noticias de lesiones y convertirlas en ajustes numericos sugeridos.
- Usar RAG con reportes de scouting, convocatorias y minutos recientes.
- Generar escenarios automaticamente: localia, clima, penales, bajas clave.
- Auditar el CSV y detectar ratings fuera de rango o inconsistentes.
- Crear reportes narrativos por seleccion para publicar en newsletter.

## Fuentes base

- FIFA Final Draw: https://www.fifa.com/en/tournaments/mens/worldcup/canadamexicousa2026/articles/final-draw-results
- FIFA Match Schedule: https://www.fifa.com/en/tournaments/mens/worldcup/canadamexicousa2026/articles/match-schedule-fixtures-results-teams-stadiums
- FiveThirtyEight World Cup methodology: https://fivethirtyeight.com/features/how-our-2022-world-cup-predictions-work/
- Alan Turing Institute World Cup model: https://www.turing.ac.uk/blog/can-our-algorithm-predict-winner-2022-football-world-cup
- Opta Analyst World Cup predictions: https://theanalyst.com/articles/who-will-win-the-2022-fifa-world-cup-predictions
- OpenAI Responses API: https://platform.openai.com/docs/api-reference/responses
