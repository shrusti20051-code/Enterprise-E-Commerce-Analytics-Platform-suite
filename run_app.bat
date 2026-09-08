@echo off
echo =====================================================================
echo  OLIST BRAZILIAN E-COMMERCE ANALYTICS PORTFOLIO
echo =====================================================================
echo Checking dependencies and launching Streamlit web application...
echo.

python -m pip install -r requirements.txt
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Failed to install dependencies. Please ensure Python is installed and added to PATH.
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo Launching Streamlit web dashboard in your default browser...
python -m streamlit run app.py

pause
