# WCUP 2026 Predictor

> **Autor:** Francisco Gonzalez · [Quality Analytics](https://www.qualityanalytics.com)

![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?logo=streamlit&logoColor=white)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--5-412991?logo=openai&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2.x-150458?logo=pandas&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-5.x-3F4F75?logo=plotly&logoColor=white)
![Conda](https://img.shields.io/badge/Conda-environment-44A833?logo=anaconda&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

Aplicación de Streamlit para simular el Mundial 2026 combinando **simulación Monte Carlo**, **modelo de goles Poisson** y una capa de **inteligencia artificial** que actualiza ratings con búsqueda web, consulta noticias en tiempo real e interpreta los resultados del modelo con lenguaje natural.

---

## Aviso sobre apuestas

Este proyecto es solo para fines informativos, educativos y analíticos. Las probabilidades, simulaciones, rankings, salidas del LLM y archivos exportados no constituyen consejo financiero, recomendación de apuesta ni garantía de resultados. Cualquier decisión relacionada con apuestas deportivas debe tomarse bajo responsabilidad personal, considerando la legislación aplicable, la edad mínima legal y los riesgos de pérdida.

---

## Características principales

| Módulo | Descripción |
|---|---|
| **Simulación Monte Carlo** | Corre el torneo completo N veces (configurable) al pulsar un botón. Acumula probabilidades por ronda sin bloquear la UI. |
| **Modelo de goles Poisson** | Convierte ratings de ataque/defensa en goles esperados partido a partido, incluyendo tiempos extra y penales. |
| **Actualización de ratings con IA** | Un botón llama a la API de OpenAI con `web_search` para obtener Elo, ranking FIFA, ataque, defensa, plantilla y forma actualizados para los 48 equipos. Los valores se fusionan en el dataset base y se guardan en `data/teams_seed.csv`. |
| **LLM — Búsqueda de noticias** | Llama a la API de OpenAI con `web_search` activo para obtener noticias recientes de lesiones, bajas, convocatorias y forma de los favoritos. Las noticias se inyectan directamente en el campo de análisis cualitativo. |
| **LLM — Interpretación de resultados** | Envía el top-12 del modelo junto con los ratings y el escenario del usuario al LLM para obtener un análisis narrativo: favoritos, riesgos cualitativos y ajustes sugeridos con rangos numéricos accionables. |
| **Ratings editables** | Elo aproximado, ataque, defensa, plantilla y forma ajustables directamente en la tabla interactiva de la UI. |
| **Visualizaciones interactivas** | Gráficos de barras de Plotly por ronda, tabla de grupos y comparador de duelos directos. |
| **Exportación CSV** | Botón de descarga que genera un CSV con los resultados de la simulación, listo para subir a la [Quiniela de Modelos Predictivos WCUP 2026](https://www.kaggle.com/competitions/quiniela-de-modelos-predictivos-wcup-2026) en Kaggle. |

---

## Flujo de uso

```mermaid
flowchart LR
    A([Inicio]) --> B["① Actualizar ratings\ncon IA (opcional)"]
    B --> C["② Editar ratings\nmanualmente (opcional)"]
    C --> D["③ Configurar parámetros\nen la barra lateral"]
    D --> E["④ Pulsar\n'Simular torneo'"]
    E --> F["⑤ Explorar pestañas\nPredicción · Grupos · Modelo · LLM"]
    F --> G["⑥ Buscar noticias y\ngenerar análisis LLM (opcional)"]
    G --> H([Resultados])

    style B fill:#412991,color:#fff
    style E fill:#e03030,color:#fff
    style G fill:#412991,color:#fff
```

---

## Flujo LLM

```mermaid
flowchart TD
    U([Usuario]) --> R["Pulsa 'Actualizar ratings con IA'"]
    R --> RA["call_llm_ratings_update()"]
    RA --> RB["OpenAI Responses API\n+ web_search (alta cobertura)"]
    RB --> RC["Elo · Ranking FIFA · Ataque\nDefensa · Plantilla · Forma"]
    RC --> RD["apply_ratings_update()\n→ teams_seed.csv actualizado"]

    U --> A["Pulsa 'Buscar noticias'"]
    A --> B["call_llm_news_search()"]
    B --> C["OpenAI Responses API\n+ web_search tool"]
    C --> D["Noticias: lesiones, bajas,\nconvocatorias, forma reciente"]
    D --> E["Precarga el campo\nde notas cualitativas"]

    E --> F["Ajusta notas manualmente (opcional)"]
    F --> G["Pulsa 'Generar análisis LLM'"]
    G --> H["call_llm_analysis()"]

    subgraph payload [Payload enviado al LLM]
        P1["Top-12 del modelo\n(probabilidades por ronda)"]
        P2["Ratings de los 48 equipos\n(ataque, defensa, forma…)"]
        P3["Notas del usuario\n(noticias + escenarios)"]
    end

    H --> payload
    payload --> I["GPT — análisis narrativo"]
    I --> J["Favoritos · Riesgos · Ajustes\nsugeridos con rangos numéricos"]

    style C fill:#412991,color:#fff
    style RB fill:#412991,color:#fff
    style I fill:#412991,color:#fff
    style payload fill:#f0f4ff,stroke:#412991
```

La novedad frente a modelos clásicos es que **el LLM no solo interpreta salidas estadísticas**, sino que también **recupera y actualiza datos reales** (ratings, noticias, convocatorias y sanciones) usando `web_search` de la Responses API, unificando información cuantitativa y cualitativa en un solo flujo.

---

## Arquitectura de módulos

```mermaid
graph TD
    APP["app.py\nPunto de entrada"]

    subgraph UI["Capa de presentación"]
        ui["ui.py\nComponentes Streamlit"]
    end

    subgraph Core["Núcleo del simulador"]
        sim["simulator.py\nMonte Carlo · Poisson"]
        brk["bracket.py\nCuadro de eliminación"]
        params["parameters.py\nHiperparámetros"]
    end

    subgraph Data["Capa de datos"]
        data["data.py\nLectura · Validación · Prep."]
        cfg["config.py\nRutas · Grupos · Columnas"]
        csv[("data/\nteams_seed.csv")]
    end

    subgraph AI["Capa de inteligencia artificial"]
        llm["llm.py\nOpenAI Responses API"]
    end

    subgraph Out["Salida"]
        rep["report.py\nInforme LaTeX/PDF"]
        per["persistence.py\nEstado de sesión"]
    end

    APP --> ui
    ui --> sim
    ui --> llm
    ui --> data
    ui --> per
    sim --> brk
    sim --> data
    sim --> params
    data --> cfg
    data --> csv
    llm --> csv
    ui --> rep

    style APP fill:#e03030,color:#fff
    style sim fill:#1a73e8,color:#fff
    style llm fill:#412991,color:#fff
    style csv fill:#0b8043,color:#fff
```

---

## Flujo de simulación Monte Carlo

```mermaid
flowchart TD
    START(["Inicio simulación\n(N iteraciones)"]) --> LOAD["Cargar y preparar\nratings de equipos"]
    LOAD --> GS

    subgraph GS["Fase de grupos (12 grupos)"]
        G1["Calcular λ Poisson\npor partido"] --> G2["Sortear goles\nPoisson(λA), Poisson(λB)"]
        G2 --> G3["Acumular puntos\ny diferencia de goles"]
        G3 --> G4["Clasificar: 1°, 2° y 3°\npor grupo"]
    end

    GS --> THIRD["Asignar mejores 8\nterceros al cuadro R32"]
    THIRD --> KO

    subgraph KO["Fase eliminatoria"]
        R32["Ronda de 32\n(16 partidos)"] --> R16["Ronda de 16\n(8 partidos)"]
        R16 --> QF["Cuartos de final\n(4 partidos)"]
        QF --> SF["Semifinales\n(2 partidos)"]
        SF --> FIN["Final\n(1 partido)"]
    end

    KO --> ACC["Acumular resultado\nen contadores"]
    ACC --> NEXT{{"¿Quedan\niteraciones?"}}
    NEXT -- Sí --> GS
    NEXT -- No --> PROBA["Calcular probabilidades\npor ronda ÷ N"]
    PROBA --> VIZ["Visualizar en\nStreamlit (Plotly)"]

    style START fill:#e03030,color:#fff
    style PROBA fill:#1a73e8,color:#fff
    style VIZ fill:#0b8043,color:#fff
```

---

## Cálculo de goles esperados (Poisson)

```mermaid
flowchart LR
    ATK_A["attack_power(A)"]
    DEF_B["defense_power(B)"]
    OVR_A["overall(A)"]
    OVR_B["overall(B)"]
    HOST["is_host(A/B)\n+home_advantage"]

    ATK_A & DEF_B --> EDGE_A["a_edge = (ATK_A − DEF_B)/60\n       + (OVR_A − OVR_B)/180"]
    OVR_A & OVR_B --> EDGE_A
    HOST --> LAM_A

    EDGE_A --> LAM_A["λA = base_goals × exp(a_edge + host_A)\nclip[0.18 , 4.2]"]
    EDGE_A --> LAM_B["λB = base_goals × exp(b_edge + host_B)\nclip[0.18 , 4.2]"]

    LAM_A --> DRAW{{"λA == λB\nen eliminatoria?"}}
    LAM_B --> DRAW
    DRAW -- No --> WIN["Ganador = más goles"]
    DRAW -- Sí --> LOGIT["Probabilidad logística\n+ ruido gaussiano\nclip[0.25 , 0.75]"]
    LOGIT --> WIN

    style LAM_A fill:#1a73e8,color:#fff
    style LAM_B fill:#1a73e8,color:#fff
    style LOGIT fill:#e03030,color:#fff
```

---

## Estructura del cuadro eliminatorio FIFA 2026

```mermaid
flowchart LR
    subgraph G["12 Grupos (A–L)"]
        GRP["1°×12 · 2°×12\nmejores 3°×8"]
    end

    subgraph R32["Ronda de 32 (16 partidos)"]
        M73["73: 2A vs 2B"] & M74["74: 1E vs 3°"] & M75["75: 1F vs 2C"] & M76["76: 1C vs 2F"]
        M77["77: 1I vs 3°"] & M78["78: 2E vs 2I"] & M79["79: 1A vs 3°"] & M80["80: 1L vs 3°"]
        M81["81: 1D vs 3°"] & M82["82: 1G vs 3°"] & M83["83: 2K vs 2L"] & M84["84: 1H vs 2J"]
        M85["85: 1B vs 3°"] & M86["86: 1J vs 2H"] & M87["87: 1K vs 3°"] & M88["88: 2D vs 2G"]
    end

    subgraph R16["Ronda de 16 (8 partidos)"]
        M89["89"] & M90["90"] & M91["91"] & M92["92"]
        M93["93"] & M94["94"] & M95["95"] & M96["96"]
    end

    subgraph QF["Cuartos (4 partidos)"]
        M97["97"] & M98["98"] & M99["99"] & M100["100"]
    end

    subgraph SF["Semis (2 partidos)"]
        M101["101"] & M102["102"]
    end

    FIN["104\nFinal"]
    CHAMP(["🏆 Campeón"])

    GRP --> R32
    M74 & M77 --> M89
    M73 & M75 --> M90
    M76 & M78 --> M91
    M79 & M80 --> M92
    M83 & M84 --> M93
    M81 & M82 --> M94
    M86 & M88 --> M95
    M85 & M87 --> M96
    M89 & M90 --> M97
    M93 & M94 --> M98
    M91 & M92 --> M99
    M95 & M96 --> M100
    M97 & M98 --> M101
    M99 & M100 --> M102
    M101 & M102 --> FIN
    FIN --> CHAMP

    style CHAMP fill:#0b8043,color:#fff
    style FIN fill:#e03030,color:#fff
```

---

## Instalación y ejecución

### Requisitos previos

- [Miniconda / Anaconda](https://docs.conda.io/en/latest/miniconda.html)
- Clave de API de OpenAI

### Pasos

```powershell
# 1. Clonar o descomprimir el proyecto
cd WCUP2026

# 2. Crear el entorno conda
conda env create -f environment.yml
conda activate wcup2026

# 3. Crear el archivo de variables de entorno
# Crea un archivo .env en la raíz con el siguiente contenido:
#   OPENAI_API_KEY=tu_clave_aqui
#   OPENAI_MODEL=gpt-5   # opcional; default: gpt-5.5

# 4. Lanzar la app
streamlit run app.py
```

> La clave nunca se importa en el código. La app invoca `load_dotenv()` y el SDK de OpenAI lee `OPENAI_API_KEY` directamente desde el entorno.

---

## Cobertura del torneo

- **48 equipos** distribuidos en **12 grupos** según el sorteo FIFA 2026.
- Clasifican los **dos primeros de cada grupo** más los **ocho mejores terceros**.
- **Ronda de 32** construida siguiendo el calendario oficial FIFA.
- Probabilidades calculadas por ronda: Fase de grupos → Ronda de 32 → Octavos → Cuartos → Semifinal → Final → Campeón.

---

## Por qué Monte Carlo + Poisson

FiveThirtyEight explica que primero convierte ratings de equipos en probabilidades partido a partido usando goles esperados y distribuciones de Poisson; después transforma esas probabilidades en un pronóstico de torneo con simulaciones Monte Carlo. The Alan Turing Institute aplicó una idea parecida para 2022: un modelo estadístico tipo Dixon-Coles/Bayesiano para estimar marcadores y correr todo el calendario 100,000 veces, con el fin de ver qué equipos ganaban con más frecuencia.

La justificación es práctica: un Mundial no depende solo de quién es mejor, sino también de grupos, empates, diferencia de goles, mejores terceros, cruces de eliminatoria, penales y ruta al título. Monte Carlo permite simular miles de torneos completos y convertir esa incertidumbre en probabilidades fáciles de leer. Por eso la app no da un único ganador fijo; entrega probabilidades ajustables que el usuario puede recalibrar con información actual.

---

## Estructura del código

```
WCUP2026/
├── app.py                  # Punto de entrada Streamlit
├── environment.yml         # Entorno conda reproducible
├── requirements.txt        # Dependencias pip alternativas
├── data/
│   └── teams_seed.csv      # Ratings base de los 48 equipos (actualizable con IA)
└── wcup2026/
    ├── config.py           # Rutas, URLs, grupos, columnas y colores
    ├── parameters.py       # Parámetros del simulador (N simulaciones, etc.)
    ├── data.py             # Lectura, validación, preparación y fusión de ratings
    ├── bracket.py          # Estructura de ronda de 32 y eliminatorias
    ├── simulator.py        # Poisson, grupos, eliminatorias y Monte Carlo
    ├── llm.py              # Actualización de ratings, noticias e interpretación con OpenAI
    └── ui.py               # Componentes visuales y flujo de Streamlit
```

---

## Ideas de mejora futuras

- **RAG con scouting**: conectar un vector store con reportes de convocatorias y minutos recientes para contextualizar aún más el LLM.
- **Escenarios automáticos**: generar hipótesis de localía, clima, penales y bajas clave sin intervención manual.
- **Alertas de ratings**: detectar equipos con ratings fuera de rango o inconsistencias en el CSV.
- **Reportes narrativos**: generar un mini análisis por selección listo para publicar en newsletter.

---

## Fuentes

| Recurso | URL |
|---|---|
| FIFA Final Draw 2026 | https://www.fifa.com/en/tournaments/mens/worldcup/canadamexicousa2026/articles/final-draw-results |
| FIFA Match Schedule | https://www.fifa.com/en/tournaments/mens/worldcup/canadamexicousa2026/articles/match-schedule-fixtures-results-teams-stadiums |
| FiveThirtyEight — metodologia Mundial | https://fivethirtyeight.com/features/how-our-2022-world-cup-predictions-work/ |
| Alan Turing Institute — modelo 2022 | https://www.turing.ac.uk/blog/can-our-algorithm-predict-winner-2022-football-world-cup |
| Opta Analyst — predicciones 2022 | https://theanalyst.com/articles/who-will-win-the-2022-fifa-world-cup-predictions |
| OpenAI Responses API | https://platform.openai.com/docs/api-reference/responses |
| Quiniela de Modelos Predictivos WCUP 2026 (Kaggle) | https://www.kaggle.com/competitions/quiniela-de-modelos-predictivos-wcup-2026 |

---

*Desarrollado por **Francisco Gonzalez** para **Quality Analytics** · 2026*
