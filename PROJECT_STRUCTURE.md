# Stock Direction Predictor - Project Structure

This project uses a PyQt6 implementation with MVC architecture.

## Main Application

- `main.py` - Entry point for the PyQt application

## Models

- `models/stock_data.py` - Handles stock data loading and manipulation
- `models/predictor.py` - Handles ML prediction logic
- `day.py`, `hr.py`, `month.py` - ML prediction models for different time intervals
- `functions.py` - Utility functions
- `YahooFinance.py` - Data fetching module

## Views

- `views/home_view.py` - Home screen
- `views/options_view.py` - Configuration options
- `views/results_view.py` - Prediction results display

## Controllers

- `controllers/app_controller.py` - Manages application flow

## Utilities

- `utils/chart_utils.py` - Chart generation utilities
- `utils/indicators.py` - Technical indicators calculation

## Running the Application

Use `python main.py` to run the application
