import pandas as pd
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
from backtesting.test import SMA

# Đọc dữ liệu từ file Excel và chuyển cột 'Date' thành index
df = pd.read_excel('data/data.xlsx', parse_dates=['Date'])
df.rename(columns={'Price': 'Close'}, inplace=True)
df.set_index('Date', inplace=True)

class MyStrategy(Strategy):
    def init(self):
        # Sử dụng đường trung bình động SMA với cửa sổ 50 và 200 ngày
        self.sma_short = self.I(SMA, self.data.Close, 50)
        self.sma_long = self.I(SMA, self.data.Close, 150)
        self.stop_loss = 0.05

    def next(self):
        # Mua khi đường trung bình ngắn cắt lên trên đường trung bình dài
        if crossover(self.sma_short, self.sma_long):
            self.buy()
        # Bán khi đường trung bình ngắn cắt xuống dưới đường trung bình dài
        elif crossover(self.sma_long, self.sma_short):
            self.sell()

        # Áp dụng stop-loss
        for trade in self.trades:
            # Kiểm tra nếu giá hiện tại dưới giá mua 5% thì đóng vị thế
            if self.data.Close[-1] < trade.entry_price * (1 - self.stop_loss):
                self.position.close()

# Chạy backtest
bt = Backtest(df, MyStrategy, cash=10000, commission=.002, exclusive_orders=True)
stats = bt.run()
print(stats)
bt.plot()
