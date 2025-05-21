import os

from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QLabel, QPushButton,
                             QHBoxLayout, QGroupBox, QTabWidget, QTextEdit,
                             QSplitter, QSpacerItem, QSizePolicy, QFrame)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont, QPixmap

import pandas as pd
import numpy as np
from sklearn.metrics import classification_report

from utils.chart_utils import create_line_chart, create_candlestick_chart

class ResultsView(QWidget):
    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        self.chart_type = "line"  # Default chart type
        self.current_results = None
        
        # Connect to controller's analysis_complete signal
        self.controller.analysis_complete.connect(self.display_results)
        
        # Connect to theme changed signal
        self.controller.theme_changed.connect(self.refresh_chart)
        
        self.setup_ui()
        
    def setup_ui(self):
        # Main layout
        layout = QVBoxLayout()
        
        # Create a horizontal splitter for left (results) and right (charts) panels
        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Left panel - Results and controls
        left_panel = QFrame()
        left_layout = QVBoxLayout(left_panel)
        
        # Results title
        self.results_title = QLabel("Analysis Results")
        self.results_title.setFont(QFont("Arial", 20, QFont.Weight.Bold))
        self.results_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        left_layout.addWidget(self.results_title)
        
        # Accuracy labels
        self.accuracy_frame = QGroupBox("Accuracy Metrics")
        accuracy_layout = QVBoxLayout()
        
        self.val_accuracy_label = QLabel("Validation Accuracy: ")
        self.val_accuracy_label.setFont(QFont("Arial", 12))
        accuracy_layout.addWidget(self.val_accuracy_label)
        
        self.test_accuracy_label = QLabel("Test Accuracy: ")
        self.test_accuracy_label.setFont(QFont("Arial", 12))
        accuracy_layout.addWidget(self.test_accuracy_label)
        
        self.accuracy_frame.setLayout(accuracy_layout)
        left_layout.addWidget(self.accuracy_frame)
        
        # Classification reports
        self.reports_tabs = QTabWidget()
        
        # Validation report
        self.val_report_text = QTextEdit()
        self.val_report_text.setFont(QFont("Courier", 11))
        self.val_report_text.setReadOnly(True)
        self.reports_tabs.addTab(self.val_report_text, "Validation Report")

        # Load and show Confusion Matrix image in a new tab
        conf_matrix_tab = QWidget()
        conf_layout = QVBoxLayout()

        conf_label = QLabel("Confusion Matrix")
        conf_label.setFont(QFont("Arial", 12, QFont.Weight.Bold))
        conf_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        image_path = os.path.join(os.path.dirname(__file__), "../output/conf_matrix.png")
        if os.path.exists(image_path):
            pixmap = QPixmap(image_path)
            img_label = QLabel()
            img_label.setPixmap(pixmap.scaledToWidth(600, Qt.TransformationMode.SmoothTransformation))
            img_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            conf_layout.addWidget(conf_label)
            conf_layout.addWidget(img_label)
        else:
            conf_layout.addWidget(QLabel("Confusion matrix image not found."))

        conf_matrix_tab.setLayout(conf_layout)
        self.reports_tabs.addTab(conf_matrix_tab, "Confusion Matrix")

        # Test report
        self.test_report_text = QTextEdit()
        self.test_report_text.setFont(QFont("Courier", 11))
        self.test_report_text.setReadOnly(True)
        self.reports_tabs.addTab(self.test_report_text, "Test Report")
        
        # Future predictions tab
        self.predictions_text = QTextEdit()
        self.predictions_text.setFont(QFont("Courier", 11))
        self.predictions_text.setReadOnly(True)
        self.reports_tabs.addTab(self.predictions_text, "Future Predictions")
        
        left_layout.addWidget(self.reports_tabs)
        
        # Chart controls
        chart_controls = QGroupBox("Chart Controls")
        chart_layout = QHBoxLayout()
        
        self.line_chart_btn = QPushButton("Line Chart")
        self.line_chart_btn.clicked.connect(self.switch_to_line_chart)
        chart_layout.addWidget(self.line_chart_btn)
        
        self.candle_chart_btn = QPushButton("OHLC Chart")
        self.candle_chart_btn.clicked.connect(self.switch_to_candle_chart)
        chart_layout.addWidget(self.candle_chart_btn)
        
        chart_controls.setLayout(chart_layout)
        left_layout.addWidget(chart_controls)
        
        # Navigation buttons
        button_layout = QHBoxLayout()
        
        self.back_button = QPushButton("Back to Parameters")
        self.back_button.setFont(QFont("Arial", 12))
        self.back_button.clicked.connect(self.on_back_clicked)
        button_layout.addWidget(self.back_button)
        
        self.home_button = QPushButton("Back to Home")
        self.home_button.setFont(QFont("Arial", 12))
        self.home_button.clicked.connect(self.on_home_clicked)
        button_layout.addWidget(self.home_button)
        
        self.quit_button = QPushButton("Quit")
        self.quit_button.setFont(QFont("Arial", 12))
        self.quit_button.clicked.connect(self.on_quit_clicked)
        button_layout.addWidget(self.quit_button)
        
        left_layout.addLayout(button_layout)
        
        # Right panel - Charts
        self.right_panel = QFrame()
        self.right_layout = QVBoxLayout(self.right_panel)
        
        self.chart_title = QLabel("Stock Price Chart")
        self.chart_title.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        self.chart_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.right_layout.addWidget(self.chart_title)
        
        self.chart_container = QFrame()
        self.chart_layout = QVBoxLayout(self.chart_container)
        self.right_layout.addWidget(self.chart_container)
        
        # Add panels to splitter
        self.splitter.addWidget(left_panel)
        self.splitter.addWidget(self.right_panel)
        
        # Set initial sizes
        self.splitter.setSizes([400, 800])
        
        layout.addWidget(self.splitter)
        self.setLayout(layout)
        
    def display_results(self, results):
        """Display the analysis results"""
        # Store results for chart switching
        self.current_results = results
        
        print("\n=== RECEIVED RESULTS ===")
        print(f"Status: {results.get('status')}")
        print(f"Keys in results: {list(results.keys())}")
        print(f"Model type: {results.get('model_type')}")
        print(f"Model present: {'Yes' if 'model' in results and results['model'] is not None else 'No'}")
        print(f"Feature columns: {results.get('feature_cols')}")
        
        if results['status'] == 'error':
            # Show error message
            self.results_title.setText(f"Error: {results['message']}")
            return
            
        # Success - display results
        ticker = results['ticker']
        model_type = results.get('model_type', 'Unknown Model')
        self.results_title.setText(f"Analysis Results for {ticker} using {model_type}")
        
        # Set accuracy labels
        val_acc = results.get('val_accuracy', 0)
        test_acc = results.get('test_accuracy', 0)
        self.val_accuracy_label.setText(f"Validation Accuracy: {val_acc:.2f}")
        self.test_accuracy_label.setText(f"Test Accuracy: {test_acc:.2f}")
        
        # Set classification reports
        y_val = results.get('y_val')
        y_val_pred = results.get('y_val_pred')
        if y_val is not None and y_val_pred is not None:
            val_report = classification_report(y_val, y_val_pred, output_dict=True, zero_division=0)
            self.display_classification_report(val_report, self.val_report_text)
            
        y_test = results.get('y_test')
        y_test_pred = results.get('y_test_pred')
        if y_test is not None and y_test_pred is not None:
            test_report = classification_report(y_test, y_test_pred, output_dict=True, zero_division=0)
            self.display_classification_report(test_report, self.test_report_text)
            
        # Display raw prediction results
        self.display_raw_predictions(y_val, y_val_pred, y_test, y_test_pred)
            
        # Display chart if data is available
        df = results.get('df')
        print(f"DataFrame present: {'Yes' if df is not None else 'No'}")
        if df is not None:
            print(f"DataFrame shape: {df.shape}")
            print(f"DataFrame columns: {df.columns.tolist()}")
            self.display_chart(df)
            # Enable chart buttons
            self.line_chart_btn.setEnabled(True)
            self.candle_chart_btn.setEnabled(True)
        else:
            # Disable chart buttons if no data
            self.line_chart_btn.setEnabled(False)
            self.candle_chart_btn.setEnabled(False)
            # Show message in chart area
            self.chart_title.setText("Chart not available for this analysis")
            
    def display_classification_report(self, report_dict, text_widget):
        """Format and display classification report"""
        text_widget.clear()
        
        report_text = "Classification Report\n\n"
        report_text += f"{'Class':<10}{'Precision':<12}{'Recall':<12}{'F1-Score':<12}{'Support':<10}\n"
        report_text += "-" * 56 + "\n"
        
        # Add each class
        for label, metrics in report_dict.items():
            if label in ['accuracy', 'macro avg', 'weighted avg']:
                continue
                
            precision = metrics['precision']
            recall = metrics['recall']
            f1 = metrics['f1-score']
            support = metrics['support']
            
            report_text += f"{label:<10}{precision:<12.2f}{recall:<12.2f}{f1:<12.2f}{support:<10}\n"
            
        # Add averages
        report_text += "-" * 56 + "\n"
        for avg_type in ['macro avg', 'weighted avg']:
            if avg_type in report_dict:
                metrics = report_dict[avg_type]
                precision = metrics['precision']
                recall = metrics['recall']
                f1 = metrics['f1-score']
                support = metrics['support']
                
                report_text += f"{avg_type:<10}{precision:<12.2f}{recall:<12.2f}{f1:<12.2f}{support:<10}\n"
                
        # Add accuracy
        if 'accuracy' in report_dict:
            acc = report_dict['accuracy']
            report_text += f"\nAccuracy: {acc:.2f}\n"
            
        text_widget.setText(report_text)
            
    def display_chart(self, df):
        """Display chart based on current chart type"""
        try:
            print("\n=== DISPLAYING CHART ===")
            
            # Clear the current chart
            for i in reversed(range(self.chart_layout.count())): 
                widget = self.chart_layout.itemAt(i).widget()
                if widget:
                    widget.setParent(None)
            
            print("Cleared previous chart")
                
            # Get model and feature columns from current results if available
            model = None
            feature_cols = None
            
            if self.current_results:
                model = self.current_results.get('model')
                print(f"Model retrieved: {type(model).__name__ if model else 'None'}")
                
                # Get feature columns - either from the results directly or derive from indicators
                if 'feature_cols' in self.current_results:
                    feature_cols = self.current_results['feature_cols']
                    print(f"Using feature columns from results: {feature_cols}")
                elif 'indicators' in self.current_results:
                    # Derive feature columns from indicators
                    feature_cols = ["Open", "High", "Low", "Close", "Volume"]
                    indicators = self.current_results['indicators']
                    
                    # Add technical indicators to feature columns
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
                    print(f"Derived feature columns from indicators: {feature_cols}")
                else:
                    print("No feature columns found in results")
            
            # Check for model and features
            has_prediction = model is not None and feature_cols is not None
            prediction_text = " with Predictions" if has_prediction else ""
            print(f"Can show predictions: {has_prediction}")
            
            # Update chart title based on type    
            if self.chart_type == "line":
                self.chart_title.setText(f"Stock Price Line Chart{prediction_text}")
                print("Creating line chart...")
                chart, prediction_data = create_line_chart(df, parent=self.chart_container, model=model, feature_cols=feature_cols)
                print("Line chart created")
            else:
                self.chart_title.setText(f"Stock Price OHLC Chart{prediction_text}")
                print("Creating OHLC chart...")
                chart, prediction_data = create_candlestick_chart(df, parent=self.chart_container, model=model, feature_cols=feature_cols)
                print("OHLC chart created")
            
            # Store prediction data for the raw predictions tab
            if 'has_predictions' in prediction_data and prediction_data['has_predictions']:
                self.current_results['prediction_data'] = prediction_data
                # Update the raw predictions display
                self.display_future_predictions(prediction_data)
            
            # Add the chart to the layout
            self.chart_layout.addWidget(chart)
            print("Chart added to layout")
            
            # Add explanation for predictions if they're shown
            if has_prediction:
                # Add prediction explanation label below the chart
                explanation = QLabel()
                explanation.setWordWrap(True)
                explanation.setStyleSheet("padding: 5px; background-color: rgba(240, 240, 240, 0.5); border-radius: 5px;")
                explanation.setText(
                    "<b>Prediction Details:</b> The dashed line shows predicted future prices based on "
                    f"the {self.current_results.get('model_type', 'selected')} model. "
                    "Green indicates predicted price increase, red indicates predicted decrease. "
                    "<b>The shaded area represents the 95% confidence interval</b> - the range within which "
                    "the actual price is likely to fall based on historical volatility. "
                    "<i>Note: These predictions should be considered as just one of many factors in investment decisions.</i>"
                )
                self.chart_layout.addWidget(explanation)
                print("Added prediction explanation")
            
        except Exception as e:
            import traceback
            print(f"Error displaying chart: {e}")
            traceback.print_exc()
            self.chart_title.setText(f"Chart Error: {str(e)}")
            
    def switch_to_line_chart(self):
        """Switch to line chart"""
        self.chart_type = "line"
        # Get current results data
        results = getattr(self, 'current_results', None)
        if results and 'df' in results and results['df'] is not None:
            self.display_chart(results['df'])
        else:
            self.chart_title.setText("No data available for chart")
            
    def switch_to_candle_chart(self):
        """Switch to OHLC chart"""
        self.chart_type = "candle"
        # Get current results data
        results = getattr(self, 'current_results', None)
        if results and 'df' in results and results['df'] is not None:
            self.display_chart(results['df'])
        else:
            self.chart_title.setText("No data available for chart")
            
    def on_back_clicked(self):
        """Navigate back to options view"""
        # Reset state
        self.reset_view()
        self.controller.navigate_to("options")
        
    def on_home_clicked(self):
        """Navigate back to home view"""
        # Reset state
        self.reset_view()
        self.controller.navigate_to("home")
        
    def on_quit_clicked(self):
        """Quit the application"""
        self.window().close()
        
    def reset_view(self):
        """Reset the view to its initial state"""
        self.current_results = None
        self.results_title.setText("Analysis Results")
        self.val_accuracy_label.setText("Validation Accuracy: ")
        self.test_accuracy_label.setText("Test Accuracy: ")
        self.val_report_text.clear()
        self.test_report_text.clear()
        self.predictions_text.clear()
        self.predictions_text.setText("Future predictions will appear here after analysis.")
        
        # Clear chart
        for i in reversed(range(self.chart_layout.count())): 
            widget = self.chart_layout.itemAt(i).widget()
            if widget:
                widget.setParent(None)
                
        self.chart_title.setText("Stock Price Chart")
        
    def refresh_chart(self):
        """Refresh the current chart to apply theme changes"""
        if self.current_results and 'df' in self.current_results and self.current_results['df'] is not None:
            self.display_chart(self.current_results['df'])
            
    def display_raw_predictions(self, y_val=None, y_val_pred=None, y_test=None, y_test_pred=None):
        """Display the raw predictions alongside true values"""
        self.predictions_text.clear()
        
        # Create a text display showing actual vs. predicted values in a table format
        prediction_text = "Raw Model Predictions\n\n"
        
        # Get the dataset if available to extract dates
        df = None
        if self.current_results and 'df' in self.current_results:
            df = self.current_results.get('df')
        
        # Try to determine if date information is available
        has_dates = df is not None and 'Datetime' in df.columns
        date_format = '%Y-%m-%d'
        
        # Convert pandas Series to numpy arrays if needed
        import numpy as np
        import pandas as pd
        
        # Safely convert data to numpy arrays for consistent access
        if y_val is not None:
            if isinstance(y_val, pd.Series):
                y_val = y_val.values
            elif not isinstance(y_val, (list, np.ndarray)):
                y_val = np.array([y_val])
                
        if y_val_pred is not None:
            if isinstance(y_val_pred, pd.Series):
                y_val_pred = y_val_pred.values
            elif not isinstance(y_val_pred, (list, np.ndarray)):
                y_val_pred = np.array([y_val_pred])
                
        if y_test is not None:
            if isinstance(y_test, pd.Series):
                y_test = y_test.values
            elif not isinstance(y_test, (list, np.ndarray)):
                y_test = np.array([y_test])
                
        if y_test_pred is not None:
            if isinstance(y_test_pred, pd.Series):
                y_test_pred = y_test_pred.values
            elif not isinstance(y_test_pred, (list, np.ndarray)):
                y_test_pred = np.array([y_test_pred])
        
        if y_val is not None and y_val_pred is not None:
            # Display validation set predictions
            prediction_text += "Validation Set Results:\n"
            if has_dates:
                prediction_text += f"{'Index':<6}{'Date':<12}{'True':<8}{'Predicted':<9}{'Match':<6}\n"
                prediction_text += "-" * 42 + "\n"
            else:
                prediction_text += f"{'Index':<8}{'True':<10}{'Predicted':<10}{'Match':<10}\n"
                prediction_text += "-" * 38 + "\n"
            
            # Show at most 50 samples to avoid too much text
            val_samples = min(50, len(y_val))
            for i in range(val_samples):
                try:
                    true_val = y_val[i]
                    pred_val = y_val_pred[i]
                    matches = true_val == pred_val
                    match_symbol = "✓" if matches else "✗"
                    
                    if has_dates and i < len(df):
                        try:
                            # Try to get the date for this prediction
                            date_str = pd.to_datetime(df['Datetime'].iloc[i]).strftime(date_format)
                            prediction_text += f"{i:<6}{date_str:<12}{true_val:<8}{pred_val:<9}{match_symbol:<6}\n"
                        except Exception as e:
                            # Fall back if date formatting fails
                            prediction_text += f"{i:<8}{true_val:<10}{pred_val:<10}{match_symbol:<10}\n"
                    else:
                        prediction_text += f"{i:<8}{true_val:<10}{pred_val:<10}{match_symbol:<10}\n"
                except Exception as e:
                    prediction_text += f"{i:<8}{'Error accessing data':<30}\n"
            
            if len(y_val) > val_samples:
                prediction_text += f"... {len(y_val) - val_samples} more rows ...\n"
            
            prediction_text += "\n"
            
        if y_test is not None and y_test_pred is not None:
            # Display test set predictions
            prediction_text += "Test Set Results:\n"
            if has_dates:
                prediction_text += f"{'Index':<6}{'Date':<12}{'True':<8}{'Predicted':<9}{'Match':<6}\n"
                prediction_text += "-" * 42 + "\n"
            else:
                prediction_text += f"{'Index':<8}{'True':<10}{'Predicted':<10}{'Match':<10}\n"
                prediction_text += "-" * 38 + "\n"
            
            # Show at most 50 samples to avoid too much text
            test_samples = min(50, len(y_test))
            for i in range(test_samples):
                try:
                    true_val = y_test[i]
                    pred_val = y_test_pred[i]
                    matches = true_val == pred_val
                    match_symbol = "✓" if matches else "✗"
                    
                    # For test data, usually the indices are at the end of the dataframe
                    if has_dates and df is not None:
                        try:
                            # Test data is typically at the end of the dataset
                            offset = len(df) - len(y_test)
                            if offset >= 0 and i + offset < len(df):
                                date_str = pd.to_datetime(df['Datetime'].iloc[i + offset]).strftime(date_format)
                                prediction_text += f"{i:<6}{date_str:<12}{true_val:<8}{pred_val:<9}{match_symbol:<6}\n"
                            else:
                                prediction_text += f"{i:<8}{true_val:<10}{pred_val:<10}{match_symbol:<10}\n"
                        except Exception as e:
                            prediction_text += f"{i:<8}{true_val:<10}{pred_val:<10}{match_symbol:<10}\n"
                    else:
                        prediction_text += f"{i:<8}{true_val:<10}{pred_val:<10}{match_symbol:<10}\n"
                except Exception as e:
                    prediction_text += f"{i:<8}{'Error accessing data':<30}\n"
            
            if len(y_test) > test_samples:
                prediction_text += f"... {len(y_test) - test_samples} more rows ...\n"
                
        # Add summary statistics
        try:
            if y_val is not None and y_val_pred is not None:
                val_correct = sum(y_val == y_val_pred)
                val_total = len(y_val)
                val_acc = val_correct / val_total * 100 if val_total > 0 else 0
                prediction_text += f"\nValidation accuracy: {val_acc:.2f}% ({val_correct}/{val_total} correct)\n"
                
            if y_test is not None and y_test_pred is not None:
                test_correct = sum(y_test == y_test_pred)
                test_total = len(y_test)
                test_acc = test_correct / test_total * 100 if test_total > 0 else 0
                prediction_text += f"Test accuracy: {test_acc:.2f}% ({test_correct}/{test_total} correct)\n\n"
        except Exception as e:
            prediction_text += f"\nError calculating accuracy: {str(e)}\n\n"
        
        # Add a class legend
        prediction_text += "Legend for Predictions:\n"
        model_type = getattr(self, 'current_results', {}).get('model_type', 'Unknown')
        if 'random forest' in model_type.lower() or 'decision tree' in model_type.lower() or 'classification' in model_type.lower():
            prediction_text += "Classes are typically:\n"
            prediction_text += "0: Strong Positive / Uptrend\n"
            prediction_text += "1: Moderate Positive\n"
            prediction_text += "2: Neutral\n"
            prediction_text += "3: Moderate Negative\n"
            prediction_text += "4: Strong Negative / Downtrend\n"
                
        self.predictions_text.setText(prediction_text)

    def display_future_predictions(self, prediction_data):
        """Display the future price predictions in the raw predictions tab"""
        self.predictions_text.clear()
        
        if not prediction_data or not prediction_data.get('has_predictions', False):
            self.predictions_text.setText("No future predictions available.")
            return
            
        # Create a text display showing future prediction data
        prediction_text = "Future Price Predictions\n\n"
        
        # Get prediction direction and change percentage
        direction = prediction_data.get('direction', 'unknown')
        change_pct = prediction_data.get('prediction_change_pct', 0.0)
        last_price = prediction_data.get('last_known_price', 0.0)
        last_date = prediction_data.get('last_known_date', None)
        
        # Add a summary header
        if direction == 'up':
            prediction_text += f"PREDICTION: PRICE WILL LIKELY INCREASE BY {change_pct:.2f}% \n\n"
        else:
            prediction_text += f"PREDICTION: PRICE WILL LIKELY DECREASE BY {abs(change_pct):.2f}% \n\n"
            
        # Add last known price
        if last_date:
            prediction_text += f"Last known price on {last_date.strftime('%Y-%m-%d %H:%M')}: ${last_price:.2f}\n\n"
        else:
            prediction_text += f"Last known price: ${last_price:.2f}\n\n"
            
        # Create a table of predictions
        prediction_text += f"{'Date':<16}{'Predicted':<12}{'Change %':<10}{'Lower 95%':<12}{'Upper 95%':<12}\n"
        prediction_text += "-" * 62 + "\n"
        
        # Get the prediction data
        future_dates = prediction_data.get('future_dates', [])
        future_prices = prediction_data.get('future_prices', [])
        lower_prices = prediction_data.get('lower_prices', [])
        upper_prices = prediction_data.get('upper_prices', [])
        
        # Format the prediction data into a table
        for i in range(len(future_dates)):
            date_str = future_dates[i].strftime('%Y-%m-%d %H:%M')
            price = future_prices[i]
            pct_change = ((price / last_price) - 1) * 100
            lower = lower_prices[i]
            upper = upper_prices[i]
            
            # Format with colors (using ASCII color codes for simplicity)
            if pct_change >= 0:
                # Green for positive
                prediction_text += f"{date_str:<16}${price:<10.2f} +{pct_change:<8.2f}% ${lower:<10.2f} ${upper:<10.2f}\n"
            else:
                # Red for negative
                prediction_text += f"{date_str:<16}${price:<10.2f} {pct_change:<9.2f}% ${lower:<10.2f} ${upper:<10.2f}\n"
                
        # Add volatility information and disclaimer
        prediction_text += "\n\nThese predictions are based on:"
        prediction_text += "\n• Historical price patterns and volatility"
        prediction_text += "\n• Machine learning model trend direction"
        prediction_text += "\n• Statistical confidence intervals (95%)"
        
        if 'model_type' in self.current_results:
            model_type = self.current_results.get('model_type')
            prediction_text += f"\n\nModel used: {model_type}"
            
        prediction_text += "\n\nDisclaimer: These predictions are not financial advice."
        prediction_text += "\nPast performance is not indicative of future results."
        prediction_text += "\nActual prices may vary significantly from these projections."
        
        self.predictions_text.setText(prediction_text) 