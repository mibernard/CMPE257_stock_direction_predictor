from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QLabel, QComboBox, 
                             QPushButton, QHBoxLayout, QSpacerItem, QSizePolicy)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont, QPixmap, QIcon, QPainter, QColor, QPen
import os

from models.stock_data import StockDataModel

# Custom ComboBox with a visible down arrow
class ArrowComboBox(QComboBox):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("""
            QComboBox {
                border: 1px solid #aaaaaa;
                border-radius: 3px;
                padding: 5px;
                padding-right: 20px;  /* Space for arrow */
                background-color: white;
                color: #333333;
            }
            QComboBox::drop-down {
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 20px;
                border-left: 1px solid #aaaaaa;
                border-top-right-radius: 3px;
                border-bottom-right-radius: 3px;
            }
            QComboBox QAbstractItemView {
                border: 1px solid #aaaaaa;
                selection-background-color: #4a86e8;
                selection-color: white;
                background-color: white;
            }
        """)
    
    def paintEvent(self, event):
        super().paintEvent(event)
        
        # Draw the triangle
        painter = QPainter(self)
        painter.setPen(QPen(QColor("#333333")))
        painter.setBrush(QColor("#333333"))
        
        # Calculate position for the triangle
        width = self.width()
        height = self.height()
        
        # Define triangle points for a down arrow
        triangle_size = 8
        x = width - 15  # Position from right edge
        y = height // 2  # Vertical center
        
        # Draw a filled triangle pointing down
        points = [
            (x, y - triangle_size//2),
            (x + triangle_size, y - triangle_size//2),
            (x + triangle_size//2, y + triangle_size//2)
        ]
        
        # Convert points to QPoint objects and draw polygon
        from PyQt6.QtCore import QPoint
        qpoints = [QPoint(p[0], p[1]) for p in points]
        painter.drawPolygon(qpoints)

class HomeView(QWidget):
    # Signal when stock is selected and next is clicked
    stock_selected = pyqtSignal(str)
    
    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        self.setup_ui()
        
    def setup_ui(self):
        # Main layout
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setSpacing(20)
        
        # Add logo at the top
        layout.addItem(QSpacerItem(20, 20, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding))
        
        # Logo
        logo_layout = QHBoxLayout()
        logo_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # Create a label for the logo
        logo_label = QLabel()
        icon_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "logo.ico")
        if os.path.exists(icon_path):
            # Set the logo image with proper scaling
            icon = QIcon(icon_path)
            pixmap = icon.pixmap(100, 100)
            logo_label.setPixmap(pixmap)
            logo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            logo_layout.addWidget(logo_label)
        
        layout.addLayout(logo_layout)
        
        # Title
        title_label = QLabel("S&P 500 Stock Direction Predictor")
        title_label.setFont(QFont("Arial", 28, QFont.Weight.Bold))
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addItem(QSpacerItem(20, 20, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed))
        layout.addWidget(title_label)
        
        # Description
        description = QLabel("Select a stock, choose parameters, and predict future movement.")
        description.setFont(QFont("Arial", 16))
        description.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(description)
        layout.addItem(QSpacerItem(20, 20, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed))
        
        # Stock selection
        stock_layout = QHBoxLayout()
        stock_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        stock_label = QLabel("Select Stock:")
        stock_label.setFont(QFont("Arial", 14))
        stock_layout.addWidget(stock_label)
        
        # Use our custom ComboBox
        self.stock_combo = ArrowComboBox()
        self.stock_combo.setMinimumWidth(200)
        self.stock_combo.setFont(QFont("Arial", 14))
        
        # Load S&P 500 tickers
        self.load_tickers()
        stock_layout.addWidget(self.stock_combo)
        
        layout.addLayout(stock_layout)
        layout.addItem(QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding))
        
        # Buttons
        button_layout = QHBoxLayout()
        button_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        self.next_button = QPushButton("Next")
        self.next_button.setFont(QFont("Arial", 14))
        self.next_button.setMinimumSize(120, 40)
        self.next_button.clicked.connect(self.on_next_clicked)
        button_layout.addWidget(self.next_button)
        
        self.quit_button = QPushButton("Quit")
        self.quit_button.setFont(QFont("Arial", 14))
        self.quit_button.setMinimumSize(120, 40)
        self.quit_button.clicked.connect(self.on_quit_clicked)
        button_layout.addWidget(self.quit_button)
        
        layout.addLayout(button_layout)
        layout.addItem(QSpacerItem(20, 40, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding))
        
        self.setLayout(layout)
        
    def load_tickers(self):
        """Load S&P 500 tickers into the combo box"""
        try:
            tickers = StockDataModel.get_sp500_tickers()
            self.stock_combo.addItems(tickers)
            # Default to AAPL
            index = self.stock_combo.findText("AAPL")
            if index >= 0:
                self.stock_combo.setCurrentIndex(index)
        except Exception as e:
            print(f"Error loading tickers: {e}")
            # Add some default tickers
            default_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
            self.stock_combo.addItems(default_tickers)
            
    def on_next_clicked(self):
        """Handle next button click"""
        selected_stock = self.stock_combo.currentText()
        # Set the selected stock in the controller
        self.controller.set_ticker(selected_stock)
        # Navigate to options view
        self.controller.navigate_to("options")
        
    def on_quit_clicked(self):
        """Handle quit button click"""
        # Close the application
        self.window().close() 