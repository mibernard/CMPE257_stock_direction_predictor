from PyQt6.QtGui import QIcon
import sys
import os

# Add the current directory to the path to ensure imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from PyQt6.QtWidgets import QApplication, QMainWindow, QStackedWidget, QPushButton, QToolBar
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont

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
        
        # Set application icon
        icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "logo.ico")
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))
        
        # Initialize theme state
        self.dark_mode = False
        
        # Create theme toggle button in toolbar
        toolbar = QToolBar("Theme")
        self.addToolBar(toolbar)
        toolbar.setMovable(False)
        
        self.theme_button = QPushButton("Toggle Dark Mode")
        self.theme_button.setFont(QFont("Arial", 10))
        self.theme_button.setFixedSize(150, 30)
        self.theme_button.clicked.connect(self.toggle_theme)
        toolbar.addWidget(self.theme_button)
        
        # Set up initial style (light mode)
        self.apply_theme()
        
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
            
    def toggle_theme(self):
        """Toggle between light and dark mode"""
        self.dark_mode = not self.dark_mode
        self.apply_theme()
        self.theme_button.setText("Light Mode" if self.dark_mode else "Dark Mode")
        
        # Emit signal that theme has changed so views can update
        self.controller.theme_changed.emit()
        
    def apply_theme(self):
        """Apply the current theme (light or dark)"""
        if self.dark_mode:
            # Dark mode stylesheet
            self.setStyleSheet("""
                QMainWindow, QWidget {
                    background-color: #1e1e1e;
                    color: #f5f5f5;
                }
                QLabel {
                    color: #f5f5f5;
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
                    border: 1px solid #555555;
                    border-radius: 5px;
                    margin-top: 1.5ex;
                    padding-top: 10px;
                    color: #f5f5f5;
                }
                QGroupBox::title {
                    subcontrol-origin: margin;
                    subcontrol-position: top center;
                    padding: 0 5px;
                    color: #f5f5f5;
                }
                QComboBox {
                    color: #f5f5f5;
                    background-color: #2d2d2d;
                    border: 1px solid #555555;
                    border-radius: 3px;
                    padding: 5px;
                }
                QComboBox QAbstractItemView {
                    color: #f5f5f5;
                    background-color: #2d2d2d;
                    selection-background-color: #4a86e8;
                    selection-color: white;
                }
                QRadioButton, QCheckBox {
                    color: #f5f5f5;
                }
                QTextEdit {
                    color: #f5f5f5;
                    background-color: #2d2d2d;
                    border: 1px solid #555555;
                }
                QTabWidget::pane {
                    border: 1px solid #555555;
                    background-color: #2d2d2d;
                }
                QTabBar::tab {
                    background-color: #3a3a3a;
                    color: #f5f5f5;
                    border: 1px solid #555555;
                    padding: 6px 10px;
                    margin-right: 2px;
                }
                QTabBar::tab:selected {
                    background-color: #2d2d2d;
                    border-bottom-color: #2d2d2d;
                }
                QSplitter::handle {
                    background-color: #555555;
                }
                QToolBar {
                    background-color: #1e1e1e;
                    border: none;
                }
                QPushButton#theme_button {
                    background-color: #444444;
                    color: #f5f5f5;
                }
                QPushButton#theme_button:hover {
                    background-color: #555555;
                }
            """)
        else:
            # Light mode stylesheet
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
                QToolBar {
                    background-color: #f5f5f5;
                    border: none;
                }
                QPushButton#theme_button {
                    background-color: #dddddd;
                    color: #333333;
                }
                QPushButton#theme_button:hover {
                    background-color: #cccccc;
                }
            """)

def main():
    # Create the application
    app = QApplication(sys.argv)
    
    # Set application icon at the app level too
    icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "logo.ico")
    if os.path.exists(icon_path):
        app.setWindowIcon(QIcon(icon_path))
    
    # Create and show the main window
    window = MainWindow()
    window.show()
    
    # Run the application event loop
    sys.exit(app.exec())

if __name__ == "__main__":
    main() 