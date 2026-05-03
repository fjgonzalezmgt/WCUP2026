@echo off
call conda activate wcup2026
streamlit run "%~dp0app.py"
