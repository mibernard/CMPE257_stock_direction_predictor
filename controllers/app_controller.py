import threading
from PyQt6.QtCore import QObject, pyqtSignal
from models.predictor import PredictorModel

class AppController(QObject):
    # Signal to update UI with model results
    analysis_complete = pyqtSignal(object)
    analysis_started = pyqtSignal()
    
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.current_ticker = None
        self.selected_interval = None
        self.selected_indicators = []
        self.selected_model = None
        self._analysis_thread = None
        self._analysis_running = False
        
    def navigate_to(self, view_name):
        """Navigate to a specific view"""
        # If navigating away from results, reset loading state
        if view_name in ["home", "options"]:
            self._analysis_running = False
            if self._analysis_thread and self._analysis_thread.is_alive():
                # We can't actually stop the thread, but we can ignore its results
                print("Note: Analysis is still running in background but will be ignored")
            
        self.main_window.navigate_to(view_name)
        
    def set_ticker(self, ticker):
        """Set the selected stock ticker"""
        self.current_ticker = ticker
        
    def set_interval(self, interval):
        """Set the selected time interval"""
        self.selected_interval = interval
        
    def set_indicators(self, indicators):
        """Set the selected technical indicators"""
        self.selected_indicators = indicators
        
    def set_model(self, model):
        """Set the selected ML model"""
        self.selected_model = model
        
    def run_analysis(self):
        """Run stock analysis in a background thread"""
        # Don't start a new analysis if one is already running
        if self._analysis_running:
            print("Analysis already running, ignoring request")
            return
            
        self._analysis_running = True
        self.analysis_started.emit()
        
        self._analysis_thread = threading.Thread(target=self._perform_analysis)
        self._analysis_thread.daemon = True
        self._analysis_thread.start()
        
    def _perform_analysis(self):
        """Background method to run the analysis"""
        try:
            # Get values from controller state
            ticker = self.current_ticker
            interval = self.selected_interval
            indicators = self.selected_indicators
            model = self.selected_model
            
            # Initialize predictor model
            predictor = PredictorModel()
            
            # Run prediction (actual call to ML logic)
            result = predictor.predict(ticker, interval, indicators, model)
            
            # Only emit results if analysis is still relevant
            if self._analysis_running:
                self.analysis_complete.emit(result)
                self._analysis_running = False
            
        except Exception as e:
            # Handle errors
            print(f"Error in analysis: {e}")
            import traceback
            print(traceback.format_exc())
            
            if self._analysis_running:
                self.analysis_complete.emit({'status': 'error', 'message': str(e)})
                self._analysis_running = False 