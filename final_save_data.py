import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton
import sys

# RUN SCRIPT : pip install -r requirements.txt

# def download_stock_data(stock_symbol, start_date, end_date):
#     """Tải dữ liệu từ Yahoo Finance cho một cổ phiếu từ ngày start_date đến end_date."""
#     df = yf.download(stock_symbol, start=start_date, end=end_date)
#     # Lưu DataFrame thành tệp CSV
#     df.to_csv(f'data/{stock_symbol}.csv', index=True)  # index=False để không lưu cột chỉ mục

#     # Lưu DataFrame thành tệp Excel (XLS)
#     df.to_excel(f'data/{stock_symbol}.xlsx', index=True)

#     # Lưu DataFrame thành tệp văn bản thông thường
#     with open(f'data/{stock_symbol}.txt', 'w') as file:
#         file.write(df.to_string(index=True))  # index=False để không lưu cột chỉ mục

#     df.index = pd.to_datetime(df.index)
#     return df

def download_stock_data(stock_symbol, start_date, end_date):
    """Đọc dữ liệu từ một tệp tin Excel cho cổ phiếu."""
    df = pd.read_excel('data/data.xlsx', parse_dates=['Date'], index_col='Date')
    
    # Thay đổi tên cột 'Price' thành 'Close'
    if 'Price' in df.columns:
        df.rename(columns={'Price': 'Close'}, inplace=True)

    # Kiểm tra xem dữ liệu đã được sắp xếp theo thứ tự tăng dần hay giảm dần
    is_descending = df.index.is_monotonic_decreasing

    # Nếu dữ liệu đang theo thứ tự giảm dần, đảo ngược lại
    if is_descending:
        df = df[::-1]

    return df

def calculate_moving_averages(data, short_window, long_window):
    """Tính toán và thêm cột MA_Short và MA_Long (Moving Averages) vào DataFrame."""
    data.loc[:, 'MA_Short'] = data['Close'].rolling(window=short_window).mean()
    data.loc[:, 'MA_Long'] = data['Close'].rolling(window=long_window).mean()
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
    portfolio = {'Cash': initial_cash, 'Shares': 0}
    portfolio_value = []
    transactions = []

    for index, row in data.iterrows():
        if row['Signal'] == 1:  # Tín hiệu mua
            if portfolio['Cash'] > 0:
                shares_to_buy = portfolio['Cash'] / row['Close']
                portfolio['Shares'] += shares_to_buy
                portfolio['Cash'] = 0
                transactions.append((index, 'BUY', row['Close'], shares_to_buy))
        elif row['Signal'] == -1:  # Tín hiệu bán
            if portfolio['Shares'] > 0:
                cash_from_sale = portfolio['Shares'] * row['Close']
                portfolio['Cash'] += cash_from_sale
                portfolio['Shares'] = 0
                transactions.append((index, 'SELL', row['Close'], 0))

        total_value = portfolio['Cash'] + portfolio['Shares'] * row['Close']
        portfolio_value.append(total_value)

    data['Portfolio_Value'] = portfolio_value
    transactions_df = pd.DataFrame(transactions, columns=['Date', 'Type', 'Price', 'Shares'])

    return data, transactions_df

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
    plt.plot(data[data['Signal'] == 1].index, data['MA_Short'][data['Signal'] == 1], '^', markersize=10, color='g', lw=0, label='Buy Signal')
    plt.plot(data[data['Signal'] == -1].index, data['MA_Short'][data['Signal'] == -1], 'v', markersize=10, color='r', lw=0, label='Sell Signal')
    plt.legend()
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Close Price, Moving Averages, and Signals')
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

# Hàm tính toán các chỉ số hiệu suất
def calculate_performance_metrics(data):
    total_return = data['Portfolio_Value'].iloc[-1] / data['Portfolio_Value'].iloc[0] - 1
    annualized_return = (1 + total_return) ** (252 / len(data)) - 1
    daily_returns = data['Portfolio_Value'].pct_change().dropna()

    # Khởi tạo các biến
    annualized_volatility = np.nan
    sharpe_ratio = np.nan
    win_ratio = np.nan

    # Tính độ biến động hàng năm nếu có lợi nhuận hàng ngày
    if len(daily_returns) > 0:
        annualized_volatility = np.std(daily_returns) * np.sqrt(252)

        if annualized_volatility != 0:
            sharpe_ratio = annualized_return / annualized_volatility

    # Tính tỷ lệ thắng
    if 'Signal' in data:
        buy_signals_count = (data['Signal'] == 1).sum()
        total_signals_count = (data['Signal'] != 0).sum()
        if total_signals_count != 0:
            win_ratio = buy_signals_count / total_signals_count

    return {
        'Annualized Return': annualized_return,
        'Annualized Volatility': annualized_volatility,
        'Sharpe Ratio': sharpe_ratio,
        'Win Ratio': win_ratio
    }
    
# Hàm tối ưu hóa các tham số
def optimize_parameters(df, short_windows, long_windows, stop_loss_pcts, log_filename):
    best_performance = -np.inf
    best_short_window = None
    best_long_window = None
    best_stop_loss_pct = None

    total_combinations = len(short_windows) * len(long_windows) * len(stop_loss_pcts)
    current_combination = 1

    log_data = []

    for short_window in short_windows:
        for long_window in long_windows:
            for stop_loss_pct in stop_loss_pcts:
                print(f"Testing combination {current_combination}/{total_combinations}: Short Window = {short_window}, Long Window = {long_window}, Stop Loss % = {stop_loss_pct}")
                current_combination += 1

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
                data, transactions_df = backtest(data)

                final_value = data['Portfolio_Value'].iloc[-1]
                performance_metrics = calculate_performance_metrics(data)
                log_entry = {
                    'Short Window': short_window,
                    'Long Window': long_window,
                    'Stop Loss %': stop_loss_pct,
                    'Final Portfolio Value': final_value
                }
                log_entry.update(performance_metrics)
                log_data.append(log_entry)

                if final_value > best_performance:
                    best_performance = final_value
                    best_short_window = short_window
                    best_long_window = long_window
                    best_stop_loss_pct = stop_loss_pct

                    # Ghi lại dữ liệu của cặp tham số tốt nhất
                    best_data = data.copy()
                    best_transactions_df = transactions_df.copy()

    # Ghi log vào file Excel
    log_df = pd.DataFrame(log_data)
    log_filename = f"log_optimize/{log_filename}_{pd.Timestamp.now().strftime('%Y%m%d')}.xlsx"
    with pd.ExcelWriter(log_filename) as writer:
        log_df.to_excel(writer, sheet_name='Optimization Log', index=False)
        best_data.to_excel(writer, sheet_name='Best Performance Data', index=False)
        best_transactions_df.to_excel(writer, sheet_name='Best Transactions', index=False)

    print(f"Optimization completed. Log saved to {log_filename}")
    return best_performance, best_short_window, best_long_window, best_stop_loss_pct, best_data, best_transactions_df

class MainApp(QWidget):
    def __init__(self, df_processed, best_short_window, best_long_window):
        super().__init__()
        self.df_processed = df_processed
        self.best_short_window = best_short_window
        self.best_long_window = best_long_window
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
        plot_close_prices(self.df_processed)

    def plot_price_and_ma(self):
        plot_price_and_ma(self.df_processed, self.best_short_window, self.best_long_window)


    def plot_portfolio_value(self):
        """Vẽ biểu đồ giá trị của tài khoản qua thời gian."""
        try:
            self.df_processed, transactions_df = backtest(self.df_processed)
            plt.figure(figsize=(10, 6))
            plt.plot(self.df_processed.index, self.df_processed['Portfolio_Value'])
            plt.xlabel('Date')
            plt.ylabel('Portfolio Value ($)')
            plt.title('Portfolio Value Over Time')
            plt.grid(True)
            plt.show()
        except Exception as e:
            print(f"An error occurred while plotting portfolio value: {e}")


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
    short_windows = range(10, 100, 20)
    long_windows = range(50, 500, 100)
    stop_loss_pcts = [0.05, 0.07, 0.08]
    log_filename = "optimization_results"

    best_performance, best_short_window, best_long_window, best_stop_loss_pct, best_data, best_transactions_df = optimize_parameters(df, short_windows, long_windows, stop_loss_pcts, log_filename)

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
    print(df_processed.head(50))
    print("Trade Signals:")
    print(trade_signals)

    # Khởi chạy ứng dụng GUI và hiển thị
    app = QApplication(sys.argv)
    ex = MainApp(df_processed, best_short_window, best_long_window)
    sys.exit(app.exec_())
    
if __name__ == "__main__":
    main()