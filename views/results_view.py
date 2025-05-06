from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QLabel, QPushButton, 
                             QHBoxLayout, QGroupBox, QTabWidget, QTextEdit,
                             QSplitter, QSpacerItem, QSizePolicy, QFrame)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont

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
        
        # Test report
        self.test_report_text = QTextEdit()
        self.test_report_text.setFont(QFont("Courier", 11))
        self.test_report_text.setReadOnly(True)
        self.reports_tabs.addTab(self.test_report_text, "Test Report")
        
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
            
        # Display chart if data is available
        df = results.get('df')
        if df is not None:
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
            # Clear the current chart
            for i in reversed(range(self.chart_layout.count())): 
                widget = self.chart_layout.itemAt(i).widget()
                if widget:
                    widget.setParent(None)
                
            # Update chart title based on type    
            if self.chart_type == "line":
                self.chart_title.setText("Stock Price Line Chart")
                chart = create_line_chart(df, parent=self.chart_container)
            else:
                self.chart_title.setText("Stock Price OHLC Chart")
                chart = create_candlestick_chart(df, parent=self.chart_container)
                
            self.chart_layout.addWidget(chart)
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