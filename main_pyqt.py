import sys
import os

# Add the current directory to the path to ensure imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from PyQt6.QtWidgets import QApplication, QMainWindow, QStackedWidget

from views.home_view import HomeView
from views.options_view import OptionsView
from views.results_view import ResultsView
from controllers.app_controller import AppController
from models.stock_data import StockDataModel
from models.predictor import PredictorModel

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Stock Direction Predictor")
        self.setMinimumSize(1200, 800)
        
        # Set up style
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background-color: #f5f5f5;
                color: #333333;
            }
            QLabel {
                color: #333333;
            }
            QPushButton {
                background-color: #4a86e8;
                color: white;
                border-radius: 4px;
                padding: 6px 12px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #3a76d8;
            }
            QPushButton:pressed {
                background-color: #2a66c8;
            }
            QGroupBox {
                border: 1px solid #cccccc;
                border-radius: 5px;
                margin-top: 1.5ex;
                padding-top: 10px;
                color: #333333;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top center;
                padding: 0 5px;
                color: #333333;
            }
            QComboBox {
                color: #333333;
                background-color: white;
                border: 1px solid #cccccc;
                border-radius: 3px;
                padding: 5px;
            }
            QComboBox QAbstractItemView {
                color: #333333;
                background-color: white;
                selection-background-color: #4a86e8;
                selection-color: white;
            }
            QRadioButton, QCheckBox {
                color: #333333;
            }
            QTextEdit {
                color: #333333;
                background-color: white;
                border: 1px solid #cccccc;
            }
            QTabWidget::pane {
                border: 1px solid #cccccc;
                background-color: white;
            }
            QTabBar::tab {
                background-color: #e0e0e0;
                color: #333333;
                border: 1px solid #cccccc;
                padding: 6px 10px;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background-color: white;
                border-bottom-color: white;
            }
            QSplitter::handle {
                background-color: #cccccc;
            }
        """)
        
        # Central stacked widget for different views
        self.stacked_widget = QStackedWidget()
        self.setCentralWidget(self.stacked_widget)
        
        # Initialize controller
        self.controller = AppController(self)
        
        # Initialize views
        self.setup_ui()
        
    def setup_ui(self):
        """Initialize and set up UI components"""
        self.home_view = HomeView(self.controller)
        self.options_view = OptionsView(self.controller) 
        self.results_view = ResultsView(self.controller)
        
        # Add views to stacked widget
        self.stacked_widget.addWidget(self.home_view)
        self.stacked_widget.addWidget(self.options_view)
        self.stacked_widget.addWidget(self.results_view)
        
        # Start with home view
        self.stacked_widget.setCurrentWidget(self.home_view)
    
    def navigate_to(self, view_name):
        """Switch to a specific view"""
        if view_name == "home":
            self.stacked_widget.setCurrentWidget(self.home_view)
        elif view_name == "options":
            self.stacked_widget.setCurrentWidget(self.options_view)
        elif view_name == "results":
            self.stacked_widget.setCurrentWidget(self.results_view)

def main():
    # Create the application
    app = QApplication(sys.argv)
    
    # Create and show the main window
    window = MainWindow()
    window.show()
    
    # Run the application event loop
    sys.exit(app.exec())

if __name__ == "__main__":
    main() 