import customtkinter as ctk
import yfinance as yf
import pandas as pd


# Fetch S&P 500 stock tickers (this may be manually sourced or scraped)
def get_sp500_tickers():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tables = pd.read_html(url)
    df = tables[0]
    tickers = df['Symbol'].tolist()
    return tickers


def process_page(quit_button, process_button, stock_menu, title_label, window):
    # Update the title label with the selected stock
    title_label.configure(text=stock_menu.get())

    # Hide the quit button
    quit_button.place_forget()

    # Move "Next" button and stock menu after processing
    process_button.place(x=50, y=200)
    stock_menu.place(x=50, y=100)

    # Create the time interval selection options using radio buttons
    interval_options = ["Hourly", "Daily", "Monthly"]
    selected_interval = ctk.StringVar(value=interval_options[0])  # Set default to "Hourly"

    interval_radio_buttons = [
        ctk.CTkRadioButton(process_button.master, text=interval, value=interval, variable=selected_interval,
                           font=("Arial", 16))
        for interval in interval_options
    ]

    # Place radio buttons for time interval selection
    x_position = 150
    for radio_button in interval_radio_buttons:
        radio_button.place(x=150 + x_position, y=250)
        x_position += 150

    type_label = ctk.CTkLabel(window, text="Time series type", font=("Arial", 24))
    type_label.place(x=300, y=150)

    # Indicator selection labels
    indicator_label = ctk.CTkLabel(window, text="Select Indicators", font=("Arial", 24))
    indicator_label.place(x=300, y=350)

    # Checkboxes for each indicator
    sma_var = ctk.BooleanVar(value=False)
    ema_var = ctk.BooleanVar(value=False)
    wma_var = ctk.BooleanVar(value=False)
    rsi_var = ctk.BooleanVar(value=False)
    macd_var = ctk.BooleanVar(value=False)

    sma_checkbox = ctk.CTkCheckBox(window, text="SMA", variable=sma_var, font=("Arial", 16))
    sma_checkbox.place(x=300, y=400)

    ema_checkbox = ctk.CTkCheckBox(window, text="EMA", variable=ema_var, font=("Arial", 16))
    ema_checkbox.place(x=300, y=450)

    wma_checkbox = ctk.CTkCheckBox(window, text="WMA", variable=wma_var, font=("Arial", 16))
    wma_checkbox.place(x=300, y=500)

    rsi_checkbox = ctk.CTkCheckBox(window, text="RSI", variable=rsi_var, font=("Arial", 16))
    rsi_checkbox.place(x=300, y=550)

    macd_checkbox = ctk.CTkCheckBox(window, text="MACD", variable=macd_var, font=("Arial", 16))
    macd_checkbox.place(x=300, y=600)

    # Now you can access which indicators are selected using the .get() method of each BooleanVar
    selected_indicators = {
        "SMA": sma_var.get(),
        "EMA": ema_var.get(),
        "WMA": wma_var.get(),
        "RSI": rsi_var.get(),
        "MACD": macd_var.get()
    }

    # For example, print the selected indicators
    print("Selected Indicators:", [key for key, value in selected_indicators.items() if value])


# UI Setup with customtkinter
def create_ui():
    # Create the main window
    window = ctk.CTk()
    window.geometry("1920x1080")  # Full screen
    window.title("S&P 500 Stock Selector")

    # Make the window full screen
    window.attributes('-fullscreen', True)

    # Create a title label
    title_label = ctk.CTkLabel(window, text="Select a Stock from S&P 500", font=("Arial", 24))
    title_label.place(x=1120, y=100)

    # Get S&P 500 tickers
    sp500_tickers = get_sp500_tickers()

    # Create a drop-down menu to select a stock
    stock_menu = ctk.CTkOptionMenu(window, values=sp500_tickers, font=("Arial", 16))
    stock_menu.place(x=1200, y=250)

    # Function when a stock is selected
    def on_stock_select(event=None):
        selected_stock = stock_menu.get()
        print(f"Selected Stock: {selected_stock}")
        # You can implement more logic here, e.g., fetching data for the selected stock.

    # Bind event for selection
    stock_menu.bind("<Configure>", on_stock_select)

    # Create the "Next" button
    process_button = ctk.CTkButton(window, text="Next",
                                   command=lambda: process_page(quit_button, process_button, stock_menu, title_label,
                                                                window), font=("Arial", 16))
    process_button.place(x=1200, y=350)

    # Add a Quit button
    quit_button = ctk.CTkButton(window, text="Quit", command=window.quit, font=("Arial", 16))
    quit_button.place(x=1200, y=450)

    # Bind the "Q" key to quit the application
    window.bind('<q>', lambda event: window.quit())

    # Start the main loop
    window.mainloop()


# Run the UI
if __name__ == "__main__":
    create_ui()
