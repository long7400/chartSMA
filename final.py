import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton
import sys

# RUN SCRIPT : pip install -r requirements.txt

def download_stock_data(stock_symbol, start_date, end_date):
    """Tải dữ liệu từ Yahoo Finance cho một cổ phiếu từ ngày start_date đến end_date."""
    df = yf.download(stock_symbol, start=start_date, end=end_date)
    # Lưu DataFrame thành tệp CSV
    df.to_csv(f'data/{stock_symbol}.csv', index=True)  # index=False để không lưu cột chỉ mục

    # Lưu DataFrame thành tệp Excel (XLS)
    df.to_excel(f'data/{stock_symbol}.xlsx', index=True)

    # Lưu DataFrame thành tệp văn bản thông thường
    with open(f'data/{stock_symbol}.txt', 'w') as file:
        file.write(df.to_string(index=True))  # index=False để không lưu cột chỉ mục

    df.index = pd.to_datetime(df.index)
    return df

def calculate_moving_averages(data, short_window, long_window):
    """Tính toán và thêm cột MA_Short và MA_Long (Moving Averages) vào DataFrame."""
    data['MA_Short'] = data['Close'].rolling(window=short_window).mean()
    data['MA_Long'] = data['Close'].rolling(window=long_window).mean()
    return data

def set_stop_loss(data, stop_loss_pct):
    """Thêm cột Stop_Loss vào DataFrame dựa trên tỷ lệ stop_loss_pct."""
    data['Stop_Loss'] = data['Close'] * (1 - stop_loss_pct)
    return data

def generate_signals(data):
    """Tạo tín hiệu giao dịch dựa trên MA_Short và MA_Long."""
    data['SHORT_GR_LONG'] = np.where(data['MA_Short'] > data['MA_Long'], 1, 0)
    data['Signal'] = data['SHORT_GR_LONG'].diff()
    return data

def backtest(data, initial_cash=10000):
    """Thực hiện backtest và tính toán giá trị của tài khoản qua thời gian."""
    cash = initial_cash
    shares = 0
    portfolio_value = []

    for index, row in data.iterrows():
        if row['Signal'] == 1:  # Tín hiệu mua
            if cash > 0:
                shares = cash / row['Close']
                cash = 0
        elif row['Signal'] == -1:  # Tín hiệu bán
            if shares > 0:
                cash = shares * row['Close']
                shares = 0

        total_value = cash + shares * row['Close']
        portfolio_value.append(total_value)

    return portfolio_value

def plot_close_prices(data):
    """Vẽ biểu đồ giá đóng cửa."""
    plt.figure(figsize=(10, 6))
    data['Close'].plot(title='Close Prices')
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.grid(True)
    plt.show()

def plot_price_and_ma(data, best_short_window, best_long_window):
    """Vẽ biểu đồ giá đóng cửa, Moving Averages và điểm mua bán."""
    plt.figure(figsize=(20, 10))
    data['Close'].plot(label="Price", color='k')
    data['MA_Short'].plot(label=f"{best_short_window} Moving Average", color='b')
    data['MA_Long'].plot(label=f"{best_long_window} Moving Average", color='g')
    data['Stop_Loss'].plot(label="Stop Loss", color='r')
    plt.plot(data[data['Signal'] == 1].index, data['MA_Short'][data['Signal'] == 1], '^', markersize=15, color='g', label='Buy Signal')
    plt.plot(data[data['Signal'] == -1].index, data['MA_Short'][data['Signal'] == -1], 'v', markersize=15, color='r', label='Sell Signal')
    plt.legend()
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Close Price, Moving Averages, and Signals')
    plt.grid(True)
    plt.show()

def plot_portfolio_value(data, portfolio_value):
    """Vẽ biểu đồ giá trị của tài khoản qua thời gian."""
    plt.figure(figsize=(10, 6))
    plt.plot(data.index, portfolio_value)
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value ($)')
    plt.title('Portfolio Value Over Time')
    plt.grid(True)
    plt.show()

def plot_trade_signals(data):
    """Vẽ biểu đồ các điểm mua và bán."""
    buy_signals = data[data['Signal'] == 1]
    sell_signals = data[data['Signal'] == -1]
    plt.figure(figsize=(20, 6))
    plt.plot(data.index, data['Close'], label='Close Price', color='black')
    plt.scatter(buy_signals.index, buy_signals['Close'], label='Buy Signal', color='green', marker='^', alpha=1)
    plt.scatter(sell_signals.index, sell_signals['Close'], label='Sell Signal', color='red', marker='v', alpha=1)
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Buy and Sell Signals')
    plt.legend()
    plt.grid(True)
    plt.show()

def optimize_parameters(df, short_windows, long_windows, stop_loss_pcts):
    """Tối ưu hóa các tham số của chiến lược giao dịch."""
    best_performance = -np.inf
    best_short_window = 0
    best_long_window = 0
    best_stop_loss_pct = 0

    for short_window in short_windows:
        for long_window in long_windows:
            for stop_loss_pct in stop_loss_pcts:
                data = df.copy()
                data = calculate_moving_averages(data, short_window, long_window)
                data.dropna(inplace=True)

                data['SHORT_GR_LONG'] = np.where(data['MA_Short'] > data['MA_Long'], 1, 0)
                data['Signal'] = data['SHORT_GR_LONG'].diff()
                data = set_stop_loss(data, stop_loss_pct)
                data['Signal'] = np.where(
                    (data['Close'] < data['Stop_Loss']) & (data['SHORT_GR_LONG'] == 1), -1, data['Signal']
                )
                data['Position'] = data['Signal'].diff()
                data['Portfolio_Value'] = backtest(data)

                final_value = data['Portfolio_Value'].iloc[-1]
                if final_value > best_performance:
                    best_performance = final_value
                    best_short_window = short_window
                    best_long_window = long_window
                    best_stop_loss_pct = stop_loss_pct

    return best_performance, best_short_window, best_long_window, best_stop_loss_pct

class MainApp(QWidget):
    def __init__(self, df_processed):
        super().__init__()
        self.df_processed = df_processed
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        btn_close_prices = QPushButton('Close Prices', self)
        btn_close_prices.clicked.connect(self.plot_close_prices)
        layout.addWidget(btn_close_prices)

        btn_price_ma_signals = QPushButton('Price, MA, and Signals', self)
        btn_price_ma_signals.clicked.connect(self.plot_price_and_ma)
        layout.addWidget(btn_price_ma_signals)

        btn_portfolio_value = QPushButton('Portfolio Value Over Time', self)
        btn_portfolio_value.clicked.connect(self.plot_portfolio_value)
        layout.addWidget(btn_portfolio_value)

        btn_trade_signals = QPushButton('Buy and Sell Signals', self)
        btn_trade_signals.clicked.connect(self.plot_trade_signals)
        layout.addWidget(btn_trade_signals)

        self.setLayout(layout)
        self.setWindowTitle('Stock Analysis App')
        self.show()

    def plot_close_prices(self):
        plt.figure(figsize=(10, 6))
        self.df_processed['Close'].plot(title='Close Prices')
        plt.xlabel('Date')
        plt.ylabel('Close Price')
        plt.grid(True)
        plt.show()

    def plot_price_and_ma(self):
        plt.figure(figsize=(20, 10))
        self.df_processed['Close'].plot(label="Price", color='k')
        self.df_processed['MA_Short'].plot(label="Short Moving Average", color='b')
        self.df_processed['MA_Long'].plot(label="Long Moving Average", color='g')
        self.df_processed['Stop_Loss'].plot(label="Stop Loss", color='r')
        plt.plot(self.df_processed[self.df_processed['Signal'] == 1].index, self.df_processed['MA_Short'][self.df_processed['Signal'] == 1], '^', markersize=15, color='g', label='Buy Signal')
        plt.plot(self.df_processed[self.df_processed['Signal'] == -1].index, self.df_processed['MA_Short'][self.df_processed['Signal'] == -1], 'v', markersize=15, color='r', label='Sell Signal')
        plt.legend()
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.title('Close Price, Moving Averages, and Signals')
        plt.grid(True)
        plt.show()

    def plot_portfolio_value(self):
        portfolio_value = backtest(self.df_processed)
        plt.figure(figsize=(10, 6))
        plt.plot(self.df_processed.index, portfolio_value)
        plt.xlabel('Date')
        plt.ylabel('Portfolio Value ($)')
        plt.title('Portfolio Value Over Time')
        plt.grid(True)
        plt.show()

    def plot_trade_signals(self):
        plot_trade_signals(self.df_processed)


# Tải dữ liệu và xử lý chỉ một lần
def load_and_process_data(df, best_short_window, best_long_window, best_stop_loss_pct):
    df_processed = df.copy()
    df_processed = calculate_moving_averages(df, best_short_window, best_long_window)
    df_processed = generate_signals(df_processed)
    df_processed = set_stop_loss(df_processed, best_stop_loss_pct)
    return df_processed


def main():
    symbol = 'AAPL'
    start_date = '2019-01-01'
    end_date = '2024-01-01'

    df = download_stock_data(symbol, start_date, end_date)
    # Tối ưu hóa các tham số
    short_windows = range(10, 20, 30)
    long_windows = range(50, 100, 200)
    stop_loss_pcts = [0.05, 0.07, 0.08]
    best_performance, best_short_window, best_long_window, best_stop_loss_pct = optimize_parameters(df, short_windows, long_windows, stop_loss_pcts)

    # Tải dữ liệu và xử lý chỉ một lần
    df_processed = load_and_process_data(df, best_short_window, best_long_window, best_stop_loss_pct)

    # In ra thông tin về các thông số tốt nhất và hiệu suất tốt nhất
    print("Optimized Parameters:")
    print(f"Best Short Window: {best_short_window}")
    print(f"Best Long Window: {best_long_window}")
    print(f"Best Stop Loss Percentage: {best_stop_loss_pct}")
    print(f"Best Performance (Portfolio Value): ${best_performance:.2f}")

    # Hiển thị các điểm mua và bán
    buy_signals = df_processed[df_processed['Signal'] == 1]
    sell_signals = df_processed[df_processed['Signal'] == -1]
    trade_signals = pd.concat([
        buy_signals[['Close', 'Stop_Loss']].rename(columns={'Close': 'Buy_Price', 'Stop_Loss': 'Stop_Loss'}),
        sell_signals[['Close', 'Stop_Loss']].rename(columns={'Close': 'Sell_Price', 'Stop_Loss': 'Stop_Loss'})
    ], axis=1).sort_index()
    
    # Hiển thị thông tin về các điểm mua và bán dưới dạng DataFrame
    print("Trade Signals:")
    print(trade_signals)

    # Khởi chạy ứng dụng GUI và hiển thị
    app = QApplication(sys.argv)
    ex = MainApp(df_processed)
    sys.exit(app.exec_())
    s
if __name__ == "__main__":
    main()
