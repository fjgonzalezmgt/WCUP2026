"""Punto de entrada Streamlit del simulador WCUP 2026.

Este modulo es un wrapper minimo que delega toda la logica de UI en
``wcup2026.ui.main``.  Se ejecuta con ``streamlit run app.py``.
"""

from wcup2026.ui import main


if __name__ == "__main__":
    main()
