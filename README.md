# CMPE257_stock_direction_predictor

A stock direction prediction application using PyQt6 with MVC architecture.

See [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) for details on the project organization.

## Environment

Python 3.10+

## Installation

```
pip install -r requirements.txt
```

### Mac SSL Fix

If you are running on Mac and the app cannot run, use the following command to fix the SSL issue:

```
sudo /Applications/Python\ 3.10/Install\ Certificates.command
```

## Running the Application

```
python main.py
```

## Dataset

This project uses Apple stock data for predictions, but can analyze any stock in the S&P 500.

## Mac Setup Notes

If you need to set up Python on Mac:

1. Download & install Python from https://www.python.org/downloads/mac-osx/ (macOS 64-bit installer)
2. Set up virtual environment:

```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Features

- Select any stock from the S&P 500 index
- Choose different time intervals for analysis (Hourly, Daily, Monthly)
- Apply various technical indicators (SMA, EMA, WMA, RSI, MACD)
- Multiple machine learning models to choose from
- Interactive charts and detailed results
