# CMPE257_stock_direction_predictor

A stock direction prediction application with two implementations:

1. Original CustomTkinter UI
2. New PyQt6 UI with improved MVC architecture

See [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) for details on the project organization.

## Environment

Python 3.10.10

## Installation

### For CustomTkinter version:

```
pip install -r requirements.txt
```

### For PyQt6 version:

```
pip install -r requirements_pyqt.txt
```

### Mac SSL Fix

If you are running on Mac and the app cannot run, use the following command to fix the SSL issue:

```
sudo /Applications/Python\ 3.10/Install\ Certificates.command
```

## Running the Application

### Original CustomTkinter version:

```
python main.py
```

### New PyQt6 version:

```
python main_pyqt.py
```

## Dataset

This project uses Apple stock data for predictions.

## Mac Setup Notes

If you need to set up Python with Tkinter on Mac:

1. Download & install Python from https://www.python.org/downloads/mac-osx/ (macOS 64-bit installer)
2. Set up virtual environment:

```
/usr/local/bin/python3.13 -m venv venv
source venv/bin/activate
pip install customtkinter pandas matplotlib seaborn yfinance mplfinance scikit-learn tooltip lxml
```

3. For PyQt6 version, additionally install:

```
pip install PyQt6 PyQt6-Charts
```
