import pyqtgraph as pg
import numpy as np
from PyQt6.QtCore import Qt
import sys
import traceback

def create_line_chart(df, parent=None):
    """Create a line chart for stock price data"""
    # Configure the plot
    plot_widget = pg.PlotWidget(parent=parent)
    plot_widget.setBackground('w')
    
    # Set axis styles with black text
    label_style = {'color': '#000000', 'font-size': '12pt'}
    axis_style = {'color': '#000000', 'font-size': '10pt'}
    
    plot_widget.setLabel('left', 'Price', units='$', **label_style)
    plot_widget.setLabel('bottom', 'Date', **label_style)
    
    # Style the axes
    plot_widget.getAxis('left').setTextPen('k')  # Black text
    plot_widget.getAxis('bottom').setTextPen('k')  # Black text
    
    # Convert datetime to timestamps for x-axis
    timestamps = df['Datetime'].astype(np.int64) // 10**9
    
    # Add close price plot
    pen = pg.mkPen(color=(0, 100, 200), width=2)
    plot_widget.plot(timestamps, df['Close'].values, pen=pen, name="Close Price")
    
    # Add grid
    plot_widget.showGrid(x=True, y=True, alpha=0.3)
    
    return plot_widget

def create_candlestick_chart(df, parent=None):
    """Create a simplified OHLC chart instead of candlesticks"""
    try:
        print("Starting candlestick chart creation")
        print(f"DataFrame shape: {df.shape}")
        print(f"DataFrame columns: {df.columns.tolist()}")
        
        # Verify we have the necessary columns
        required_cols = ['Datetime', 'Open', 'High', 'Low', 'Close']
        for col in required_cols:
            if col not in df.columns:
                print(f"Missing required column: {col}")
                raise ValueError(f"DataFrame is missing required column: {col}")
        
        # Configure the plot
        plot_widget = pg.PlotWidget(parent=parent)
        plot_widget.setBackground('w')
        
        # Set axis styles with black text
        label_style = {'color': '#000000', 'font-size': '12pt'}
        axis_style = {'color': '#000000', 'font-size': '10pt'}
        
        plot_widget.setLabel('left', 'Price', units='$', **label_style)
        plot_widget.setLabel('bottom', 'Date', **label_style)
        
        # Style the axes
        plot_widget.getAxis('left').setTextPen('k')  # Black text
        plot_widget.getAxis('bottom').setTextPen('k')  # Black text
        
        # Use a simpler OHLC representation with four separate lines
        
        # 1. Plot Close prices as main line - this is most important
        close_line = pg.PlotDataItem(
            df['Datetime'].astype(np.int64) // 10**9,
            df['Close'].values,
            pen=pg.mkPen(color=(0, 100, 200), width=2),
            name="Close"
        )
        plot_widget.addItem(close_line)
        
        # 2. Plot Open prices with a dashed line
        open_line = pg.PlotDataItem(
            df['Datetime'].astype(np.int64) // 10**9,
            df['Open'].values,
            pen=pg.mkPen(color=(100, 100, 100), width=1, style=Qt.PenStyle.DashLine),
            name="Open"
        )
        plot_widget.addItem(open_line)
        
        # 3. Plot High prices with green dots
        high_line = pg.PlotDataItem(
            df['Datetime'].astype(np.int64) // 10**9,
            df['High'].values,
            pen=pg.mkPen(color=(0, 200, 0), width=1, style=Qt.PenStyle.DotLine),
            name="High"
        )
        plot_widget.addItem(high_line)
        
        # 4. Plot Low prices with red dots  
        low_line = pg.PlotDataItem(
            df['Datetime'].astype(np.int64) // 10**9,
            df['Low'].values,
            pen=pg.mkPen(color=(200, 0, 0), width=1, style=Qt.PenStyle.DotLine),
            name="Low"
        )
        plot_widget.addItem(low_line)
        
        # Add a legend
        legend = plot_widget.addLegend()
        
        # Add grid
        plot_widget.showGrid(x=True, y=True, alpha=0.3)
        
        print("Successfully created OHLC chart")
        return plot_widget
    
    except Exception as e:
        print(f"Error creating OHLC chart: {e}")
        traceback.print_exc()
        
        # Return a fallback line chart
        print("Falling back to line chart")
        return create_line_chart(df, parent)

# Keep the CandlestickItem class for backward compatibility
class CandlestickItem(pg.GraphicsObject):
    def __init__(self, x, open, high, low, close):
        pg.GraphicsObject.__init__(self)
        self.x = x
        self.open = open
        self.high = high
        self.low = low
        self.close = close
        self.width = 0.5
        self.brush = pg.mkBrush('g') if close > open else pg.mkBrush('r')
        
    def boundingRect(self):
        try:
            return pg.QtCore.QRectF(
                self.x - self.width/2,
                self.low,
                self.width,
                self.high - self.low
            )
        except Exception as e:
            print(f"Error in boundingRect: {e}")
            # Return a default rect
            return pg.QtCore.QRectF(0, 0, 1, 1)
        
    def paint(self, painter, option, widget):
        try:
            # Draw the candle body
            painter.setPen(pg.mkPen('k'))
            painter.setBrush(self.brush)
            
            # Body
            if self.open != self.close:
                painter.drawRect(
                    self.x - self.width/2,
                    self.open,
                    self.width,
                    self.close - self.open
                )
            else:
                # Draw a line if open equals close
                painter.drawLine(
                    self.x - self.width/2,
                    self.open,
                    self.x + self.width/2,
                    self.open
                )
                
            # Draw the wicks
            painter.drawLine(
                self.x,
                self.low,
                self.x,
                min(self.open, self.close)
            )
            painter.drawLine(
                self.x,
                max(self.open, self.close),
                self.x,
                self.high
            )
        except Exception as e:
            print(f"Error painting candlestick: {e}")
            traceback.print_exc()
    
    def setBrush(self, brush):
        self.brush = brush 