# Stock Direction Predictor - Project Structure

This project has two implementations:

## 1. Original CustomTkinter Implementation

- `main.py` - The original single-file CustomTkinter application
- `tooltip.py` - Supporting tooltip component for CustomTkinter
- `day.py`, `hr.py`, `month.py` - ML prediction models
- `functions.py` - Utility functions
- `YahooFinance.py` - Data fetching module

## 2. PyQt Implementation (MVC Pattern)

The new PyQt implementation uses MVC architecture with these components:

### Main Application

- `main_pyqt.py` - Entry point for the PyQt application

### Models

- `models/stock_data.py` - Handles stock data loading and manipulation
- `models/predictor.py` - Handles ML prediction logic

### Views

- `views/home_view.py` - Home screen
- `views/options_view.py` - Configuration options
- `views/results_view.py` - Prediction results display

### Controllers

- `controllers/app_controller.py` - Manages application flow

### Utilities

- `utils/chart_utils.py` - Chart generation utilities
- `utils/indicators.py` - Technical indicators calculation

## Running the Application

- Use `python main.py` to run the original CustomTkinter app
- Use `python main_pyqt.py` to run the new PyQt version

## Notes on Project Cleanup

All duplicate files have been removed:

- Removed duplicates from `frontend/pyqt/` directory
- Removed duplicates from `backend/` directory
- Removed empty structure in `stock_predictor/` directory

The project now maintains a clean structure with the original CustomTkinter implementation in the root directory and the new PyQt implementation organized in an MVC pattern.
