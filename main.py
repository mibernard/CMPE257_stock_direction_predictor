import customtkinter as ctk
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import threading
import time

import yfinance

import day
import hr
import month
import mplfinance as mpf
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.metrics import confusion_matrix, classification_report
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tooltip import CreateToolTip


loading_flag = False

def get_sp500_tickers():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tables = pd.read_html(url)
    df = tables[0]
    return df['Symbol'].tolist()

def plot_confusion(y_true, y_pred, parent_frame, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3, 4])
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")

    canvas = FigureCanvasTkAgg(fig, master=parent_frame)
    canvas.draw()
    widget = canvas.get_tk_widget()
    widget.pack(expand=True, fill="both", pady=10)

def animate_loading(loading_label):
    dots = ""
    while loading_flag:
        dots += "."
        if len(dots) > 3:
            dots = ""
        loading_label.configure(text=f"Analysing{dots}")
        time.sleep(0.5)

def display_classification_report(y_true, y_pred, parent_frame, title="Classification Report"):
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    df_report = pd.DataFrame(report).transpose()

    df_display = df_report[['precision', 'recall', 'f1-score']]

    textbox = ctk.CTkTextbox(parent_frame, width=400, height=200, font=("Consolas", 14))
    textbox.pack(pady=5)

    textbox.insert("end", f"{title}\n\n")
    textbox.insert("end", f"{'Label':<15}{'Precision':<10}{'Recall':<10}{'F1-Score':<10}\n")
    textbox.insert("end", "-" * 45 + "\n")

    for index, row in df_display.iterrows():
        textbox.insert("end", f"{index:<15}{row['precision']:.2f}{' ' * 5}{row['recall']:.2f}{' ' * 5}{row['f1-score']:.2f}\n")

    textbox.configure(state="disabled")

def create_ui():
    window = ctk.CTk()
    window.geometry("1400x800")
    window.title("S&P 500 Stock Selector")
    window.resizable(True, True)
    chart_type = ctk.StringVar(value="line")  # 默认折线图

    def plot_chart(df, parent_frame):
        # 清空旧图表
        for widget in parent_frame.winfo_children():
            widget.destroy()

        if chart_type.get() == "line":
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(df['Datetime'], df['Close'], label="Close Price", color="blue")
            ax.set_title("Close Price Over Time")
            ax.set_xlabel("Date")
            ax.set_ylabel("Price")
            ax.tick_params(axis='x', labelrotation=45)
            ax.legend()
        else:
            df_candle = df.set_index("Datetime")[["Open", "High", "Low", "Close"]].tail(500)
            fig, _ = mpf.plot(df_candle, type='candle', style='charles', title="Candlestick Chart", returnfig=True)

        canvas = FigureCanvasTkAgg(fig, master=parent_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(pady=10, fill="both", expand=True)

    global loading_flag



    frame_home = ctk.CTkFrame(window, fg_color="transparent")
    frame_home.pack(expand=True, fill="both")

    welcome_label = ctk.CTkLabel(frame_home, text="Welcome to S&P 500 Stock Direction Predictor", font=("Arial", 28))
    welcome_label.pack(pady=20)

    description_label = ctk.CTkLabel(frame_home, text="Select a stock, choose parameters, and predict future movement.",
                                     font=("Arial", 18))
    description_label.pack(pady=10)

    sp500_tickers = get_sp500_tickers()

    stock_menu = ctk.CTkOptionMenu(frame_home, values=sp500_tickers, font=("Arial", 16))
    stock_menu.pack(pady=20)

    next_button = ctk.CTkButton(frame_home, text="Next", font=("Arial", 16))
    next_button.pack(pady=10)

    quit_button = ctk.CTkButton(frame_home, text="Quit", command=window.quit, font=("Arial", 16))
    quit_button.pack(pady=10)

    frame_options = ctk.CTkFrame(window, fg_color="transparent")

    option_title = ctk.CTkLabel(frame_options, text="Select Parameters", font=("Arial", 24))
    option_title.pack(pady=20)

    selected_interval = ctk.StringVar(value="Hourly")
    intervals = ["Hourly", "Daily", "Monthly"]
    for interval in intervals:
        interval_radio = ctk.CTkRadioButton(frame_options, text=interval, value=interval, variable=selected_interval, font=("Arial", 16))
        interval_radio.pack(pady=5)

    indicator_label = ctk.CTkLabel(frame_options, text="Select Indicators", font=("Arial", 20))
    indicator_label.pack(pady=20)

    sma_var = ctk.BooleanVar(value=False)
    ema_var = ctk.BooleanVar(value=False)
    wma_var = ctk.BooleanVar(value=False)
    rsi_var = ctk.BooleanVar(value=False)
    macd_var = ctk.BooleanVar(value=False)

    # SMA
    sma_checkbox = ctk.CTkCheckBox(frame_options, text="SMA", variable=sma_var, font=("Arial", 16))
    sma_checkbox.pack(pady=5)
    CreateToolTip(sma_checkbox,
                  "SMA: Simple Moving Average\nA simple average of a stock’s price over a specific period.")

    # EMA
    ema_checkbox = ctk.CTkCheckBox(frame_options, text="EMA", variable=ema_var, font=("Arial", 16))
    ema_checkbox.pack(pady=5)
    CreateToolTip(ema_checkbox,
                  "EMA: Exponential Moving Average\nGives more weight to recent prices to better capture short-term trends.")

    # WMA
    wma_checkbox = ctk.CTkCheckBox(frame_options, text="WMA", variable=wma_var, font=("Arial", 16))
    wma_checkbox.pack(pady=5)
    CreateToolTip(wma_checkbox,
                  "WMA: Weighted Moving Average\nEach data point is assigned a weight, emphasizing recent data.")

    # RSI
    rsi_checkbox = ctk.CTkCheckBox(frame_options, text="RSI", variable=rsi_var, font=("Arial", 16))
    rsi_checkbox.pack(pady=5)
    CreateToolTip(rsi_checkbox,
                  "RSI: Relative Strength Index\nMeasures the speed and change of price movements to detect overbought/oversold conditions.")

    # MACD
    macd_checkbox = ctk.CTkCheckBox(frame_options, text="MACD", variable=macd_var, font=("Arial", 16))
    macd_checkbox.pack(pady=5)
    CreateToolTip(macd_checkbox,
                  "MACD: Moving Average Convergence Divergence\nShows the relationship between two moving averages to identify trend direction and strength.")

    loading_label = ctk.CTkLabel(frame_options, text="", font=("Arial", 20))
    loading_label.pack(pady=10)

    run_button = ctk.CTkButton(frame_options, text="Run", font=("Arial", 16), fg_color="green", hover_color="darkgreen")
    run_button.pack(pady=20)

    quit_button2 = ctk.CTkButton(frame_options, text="Quit", command=window.quit, font=("Arial", 16))
    quit_button2.pack(pady=10)

    frame_result = ctk.CTkFrame(window, fg_color="transparent")

    frame_result_content = ctk.CTkFrame(frame_result, fg_color="transparent")
    frame_result_content.pack(pady=20, padx=20, expand=True, fill="both")

    frame_left = ctk.CTkFrame(frame_result_content, fg_color="transparent")
    frame_left.pack(side="left", fill="y", padx=20, pady=10)

    frame_right = ctk.CTkFrame(frame_result_content, fg_color="transparent")
    frame_right.pack(side="right", fill="both", expand=True, padx=20, pady=10)

    result_label = ctk.CTkLabel(frame_left, text="Results will appear here", font=("Arial", 20))
    result_label.pack(pady=20)

    frame_val_report = ctk.CTkFrame(frame_left, fg_color="transparent")
    frame_val_report.pack(pady=10)

    frame_test_report = ctk.CTkFrame(frame_left, fg_color="transparent")
    frame_test_report.pack(pady=10)

    back_button = ctk.CTkButton(frame_left, text="Back to Home", font=("Arial", 16),
                                command=lambda: [frame_result.pack_forget(), frame_home.pack(expand=True, fill="both")])
    back_button.pack(pady=10)

    quit_button3 = ctk.CTkButton(frame_left, text="Quit", command=window.quit, font=("Arial", 16))
    quit_button3.pack(pady=10)

    frame_val_confusion = ctk.CTkFrame(frame_right, fg_color="transparent")
    frame_val_confusion.pack(pady=20)

    frame_test_confusion = ctk.CTkFrame(frame_right, fg_color="transparent")
    frame_test_confusion.pack(pady=20)

    def go_to_options():
        selected_stock = stock_menu.get()
        print(f"Selected Stock: {selected_stock}")
        frame_home.pack_forget()
        frame_options.pack(expand=True, fill="both")

    def run_selected_script():
        global loading_flag

        stock = stock_menu.get()
        interval = selected_interval.get()

        selected_features = ["Open", "High", "Low", "Close", "Volume"]
        if sma_var.get():
            selected_features.append("SMA")
        if ema_var.get():
            selected_features.append("EMA")
        if wma_var.get():
            selected_features.append("WMA")
        if rsi_var.get():
            selected_features.append("RSI")
        if macd_var.get():
            selected_features.extend(["MACD", "MACD_Signal"])

        loading_flag = True
        threading.Thread(target=animate_loading, args=(loading_label,), daemon=True).start()

        def train_and_show_result():
            global loading_flag

            stock = stock_menu.get()
            interval = selected_interval.get()

            selected_features = ["Open", "High", "Low", "Close", "Volume"]
            if sma_var.get():
                selected_features.append("SMA")
            if ema_var.get():
                selected_features.append("EMA")
            if wma_var.get():
                selected_features.append("WMA")
            if rsi_var.get():
                selected_features.append("RSI")
            if macd_var.get():
                selected_features.extend(["MACD", "MACD_Signal"])

            loading_flag = True
            threading.Thread(target=animate_loading, args=(loading_label,), daemon=True).start()

            def inner_train():
                global loading_flag

                if interval == "Hourly":
                    model, val_acc, test_acc, y_val, y_val_pred, y_test, y_test_pred, df = hr.train_stock_hour_classifier(
                        stock, selected_features)
                elif interval == "Daily":
                    model, val_acc, test_acc, y_val, y_val_pred, y_test, y_test_pred, df = day.train_stock_day_classifier(
                        stock, selected_features)
                elif interval == "Monthly":
                    model, val_acc, test_acc, y_val, y_val_pred, y_test, y_test_pred, df = month.train_stock_month_classifier(
                        stock, selected_features)
                else:
                    print("do nothing")
                    return

                loading_flag = False

                frame_options.pack_forget()
                frame_result.pack(expand=True, fill="both")


                result_label.configure(text=f"Validation Accuracy: {val_acc:.2f}\nTest Accuracy: {test_acc:.2f}")

                for container in [frame_val_confusion, frame_test_confusion, frame_val_report, frame_test_report]:
                    for widget in container.winfo_children():
                        widget.destroy()

                # plot_confusion(y_val, y_val_pred, parent_frame=frame_val_confusion,
                #                title="Validation Set Confusion Matrix")
                # plot_confusion(y_test, y_test_pred, parent_frame=frame_test_confusion,
                #                title="Test Set Confusion Matrix")
                display_classification_report(y_val, y_val_pred, parent_frame=frame_val_report,
                                              title="Validation Report")
                display_classification_report(y_test, y_test_pred, parent_frame=frame_test_report, title="Test Report")

                plot_chart(df, frame_test_confusion)

                def toggle_to_line():
                    chart_type.set("line")
                    plot_chart(df, frame_test_confusion)

                def toggle_to_candle():
                    chart_type.set("candle")
                    plot_chart(df, frame_test_confusion)

                ctk.CTkButton(frame_left, text="Line Chart", command=toggle_to_line).pack(pady=5)
                ctk.CTkButton(frame_left, text="Candlestick Chart", command=toggle_to_candle).pack(pady=5)

            threading.Thread(target=inner_train, daemon=True).start()

        threading.Thread(target=train_and_show_result, daemon=True).start()

    next_button.configure(command=go_to_options)
    run_button.configure(command=run_selected_script)

    window.mainloop()

if __name__ == "__main__":
    print(yfinance.version.version)
    create_ui()
