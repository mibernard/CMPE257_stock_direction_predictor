from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QLabel, QRadioButton, 
                             QPushButton, QHBoxLayout, QGroupBox, QCheckBox,
                             QSpacerItem, QSizePolicy, QButtonGroup, QFrame)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont

class OptionsView(QWidget):
    # Signal when parameters are selected and run is clicked
    parameters_selected = pyqtSignal(dict)
    
    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        
        # Connect to controller signals
        self.controller.analysis_started.connect(lambda: self.set_loading(True))
        self.controller.analysis_complete.connect(lambda _: self.set_loading(False))
        
        self.setup_ui()
        
    def setup_ui(self):
        # Main layout
        layout = QVBoxLayout()
        layout.setSpacing(20)
        
        # Title
        title_label = QLabel("Select Parameters")
        title_label.setFont(QFont("Arial", 24, QFont.Weight.Bold))
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title_label)
        
        # Interval selection
        interval_group = QGroupBox("Time Interval")
        interval_group.setFont(QFont("Arial", 14))
        interval_layout = QVBoxLayout()
        
        self.interval_button_group = QButtonGroup(self)
        intervals = ["Hourly", "Daily", "Monthly"]
        self.interval_buttons = {}
        
        for interval in intervals:
            radio = QRadioButton(interval)
            radio.setFont(QFont("Arial", 12))
            if interval == "Hourly":  # Default selection
                radio.setChecked(True)
            self.interval_button_group.addButton(radio)
            self.interval_buttons[interval] = radio
            interval_layout.addWidget(radio)
            
        interval_group.setLayout(interval_layout)
        layout.addWidget(interval_group)
        
        # Model selection
        model_group = QGroupBox("ML Model")
        model_group.setFont(QFont("Arial", 14))
        model_layout = QVBoxLayout()
        
        self.model_button_group = QButtonGroup(self)
        models = [
            "Logistic Regression",
            "SVM",
            "Decision Tree",
            "Random Forest",
            "Gradient Boosting",
            "k-NN",
            "Naive Bayes",
            "AdaBoost",
            "Bagging",
            "Extra Trees"
        ]
        self.model_buttons = {}
        
        for model in models:
            radio = QRadioButton(model)
            radio.setFont(QFont("Arial", 12))
            if model == "Random Forest":  # Default selection
                radio.setChecked(True)
            self.model_button_group.addButton(radio)
            self.model_buttons[model] = radio
            model_layout.addWidget(radio)
            
        model_group.setLayout(model_layout)
        layout.addWidget(model_group)
        
        # Technical indicators
        indicators_group = QGroupBox("Technical Indicators")
        indicators_group.setFont(QFont("Arial", 14))
        indicators_layout = QVBoxLayout()
        
        self.sma_checkbox = QCheckBox("SMA - Simple Moving Average")
        self.sma_checkbox.setFont(QFont("Arial", 12))
        self.sma_checkbox.setToolTip("A simple average of a stock's price over a specific period.")
        indicators_layout.addWidget(self.sma_checkbox)
        
        self.ema_checkbox = QCheckBox("EMA - Exponential Moving Average")
        self.ema_checkbox.setFont(QFont("Arial", 12))
        self.ema_checkbox.setToolTip("Gives more weight to recent prices to better capture short-term trends.")
        indicators_layout.addWidget(self.ema_checkbox)
        
        self.wma_checkbox = QCheckBox("WMA - Weighted Moving Average")
        self.wma_checkbox.setFont(QFont("Arial", 12))
        self.wma_checkbox.setToolTip("Each data point is assigned a weight, emphasizing recent data.")
        indicators_layout.addWidget(self.wma_checkbox)
        
        self.rsi_checkbox = QCheckBox("RSI - Relative Strength Index")
        self.rsi_checkbox.setFont(QFont("Arial", 12))
        self.rsi_checkbox.setToolTip("Measures the speed and change of price movements to detect overbought/oversold conditions.")
        indicators_layout.addWidget(self.rsi_checkbox)
        
        self.macd_checkbox = QCheckBox("MACD - Moving Average Convergence Divergence")
        self.macd_checkbox.setFont(QFont("Arial", 12))
        self.macd_checkbox.setToolTip("Shows the relationship between two moving averages to identify trend direction and strength.")
        indicators_layout.addWidget(self.macd_checkbox)
        
        indicators_group.setLayout(indicators_layout)
        layout.addWidget(indicators_group)
        
        # Loading status
        self.loading_label = QLabel("")
        self.loading_label.setFont(QFont("Arial", 14))
        self.loading_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.loading_label)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        self.run_button = QPushButton("Run Analysis")
        self.run_button.setFont(QFont("Arial", 14))
        self.run_button.setMinimumSize(150, 40)
        self.run_button.setStyleSheet("background-color: #4CAF50; color: white;")
        self.run_button.clicked.connect(self.on_run_clicked)
        button_layout.addWidget(self.run_button)
        
        self.back_button = QPushButton("Back")
        self.back_button.setFont(QFont("Arial", 14))
        self.back_button.setMinimumSize(150, 40)
        self.back_button.clicked.connect(self.on_back_clicked)
        button_layout.addWidget(self.back_button)
        
        self.quit_button = QPushButton("Quit")
        self.quit_button.setFont(QFont("Arial", 14))
        self.quit_button.setMinimumSize(150, 40)
        self.quit_button.clicked.connect(self.on_quit_clicked)
        button_layout.addWidget(self.quit_button)
        
        layout.addLayout(button_layout)
        self.setLayout(layout)
        
    def get_selected_interval(self):
        """Get the selected time interval"""
        for interval, button in self.interval_buttons.items():
            if button.isChecked():
                return interval
        return "Hourly"  # Default
        
    def get_selected_model(self):
        """Get the selected ML model"""
        for model, button in self.model_buttons.items():
            if button.isChecked():
                return model
        return "Random Forest"  # Default
        
    def get_selected_indicators(self):
        """Get the selected technical indicators"""
        indicators = []
        if self.sma_checkbox.isChecked():
            indicators.append("SMA")
        if self.ema_checkbox.isChecked():
            indicators.append("EMA")
        if self.wma_checkbox.isChecked():
            indicators.append("WMA")
        if self.rsi_checkbox.isChecked():
            indicators.append("RSI")
        if self.macd_checkbox.isChecked():
            indicators.append("MACD")
        return indicators
        
    def set_loading(self, is_loading):
        """Show/hide loading indicator"""
        if is_loading:
            self.loading_label.setText("Analyzing...")
            self.run_button.setEnabled(False)
            self.back_button.setEnabled(False)
        else:
            self.loading_label.setText("")
            self.run_button.setEnabled(True)
            self.back_button.setEnabled(True)
        
    def on_run_clicked(self):
        """Handle run button click"""
        interval = self.get_selected_interval()
        indicators = self.get_selected_indicators()
        model = self.get_selected_model()
        
        # Update controller with selections
        self.controller.set_interval(interval)
        self.controller.set_indicators(indicators)
        self.controller.set_model(model)
        
        # Run the analysis - loading state will be managed via signals
        self.controller.run_analysis()
        
        # Navigate to results view (will happen when analysis is complete)
        self.controller.navigate_to("results")
        
    def on_back_clicked(self):
        """Handle back button click"""
        self.controller.navigate_to("home")
        
    def on_quit_clicked(self):
        """Handle quit button click"""
        self.window().close() 