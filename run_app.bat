@echo off
echo Starting Automated Training Evaluation & Performance Analytics System...
echo.
echo Initializing database...
python init_db.py
echo.
echo Starting Flask API server...
echo The application will be available at http://localhost:5000
echo Press CTRL+C to stop the server
echo.
cd api
python app.py
pause