"""Punto de entrada Streamlit del simulador WCUP 2026.

Este modulo es un wrapper minimo que delega toda la logica de UI en
``wcup2026.ui.main``.  Se ejecuta con ``streamlit run app.py``.
"""

from __future__ import annotations

import importlib

import wcup2026.ui as ui_module


if __name__ == "__main__":
    ui_module = importlib.reload(ui_module)
    ui_module.main()
