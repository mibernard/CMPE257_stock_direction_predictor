from PyQt6.QtCore import QObject
import sys
import importlib.util
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

# We'll dynamically import day.py, hr.py, and month.py to maintain compatibility
class PredictorModel(QObject):
    def __init__(self):
        super().__init__()
        # Load the modules dynamically
        self.day_module = self._import_module('day')
        self.hr_module = self._import_module('hr')
        self.month_module = self._import_module('month')
        
    def _import_module(self, module_name):
        """Dynamically import a module from the current directory"""
        try:
            spec = importlib.util.spec_from_file_location(module_name, f"{module_name}.py")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
        except Exception as e:
            print(f"Error importing {module_name}.py: {e}")
            return None
            
    def predict(self, ticker, interval, indicators, model_type="Random Forest"):
        """Run prediction based on selected interval and model"""
        try:
            feature_cols = ["Open", "High", "Low", "Close", "Volume"]
            
            # Add selected technical indicators to features
            if "SMA" in indicators:
                feature_cols.append("SMA")
            if "EMA" in indicators:
                feature_cols.append("EMA")
            if "WMA" in indicators:
                feature_cols.append("WMA")
            if "RSI" in indicators:
                feature_cols.append("RSI")
            if "MACD" in indicators:
                feature_cols.extend(["MACD", "MACD_Signal"])
                
            # Map model_type to model index used in the original code
            model_index_map = {
                "Logistic Regression": "0",
                "SVM": "1",
                "Decision Tree": "2",
                "Random Forest": "3",
                "Gradient Boosting": "4",
                "k-NN": "5",
                "Naive Bayes": "6",
                "AdaBoost": "7",
                "Bagging": "8",
                "Extra Trees": "9"
            }
            
            # Get model index from map, default to Random Forest (3)
            model_index = model_index_map.get(model_type, "3")
            
            # Call the appropriate module based on interval
            if interval == "Hourly":
                result = self.hr_module.train_stock_hour_classifier(ticker, feature_cols, model_index)
            elif interval == "Daily":
                result = self.day_module.train_stock_day_classifier(ticker, feature_cols, model_index)
            elif interval == "Monthly":
                result = self.month_module.train_stock_month_classifier(ticker, feature_cols, model_index)
            else:
                raise ValueError(f"Invalid interval: {interval}")
                
            # Return results - now all intervals have the same return values format
            return {
                'status': 'success',
                'ticker': ticker,
                'interval': interval,
                'indicators': indicators,
                'model_type': model_type,
                'model': result[0],
                'val_accuracy': result[1],
                'test_accuracy': result[2],
                'y_val': result[3],
                'y_val_pred': result[4],
                'y_test': result[5],
                'y_test_pred': result[6],
                'df': result[7] if len(result) > 7 else None,
                'feature_cols': feature_cols  # Add feature columns to results
            }
            
        except Exception as e:
            import traceback
            print(f"Error in prediction: {e}")
            print(traceback.format_exc())
            return {
                'status': 'error',
                'message': str(e)
            } 