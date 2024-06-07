import pandas as pd
import numpy as np
import backtrader as bt
from datetime import datetime
import tempfile

# Define a function to create a moving average strategy
def moving_average_strategy(data, short_window=50, long_window=200):
    # Calculate the short-term moving average (50 days by default)
    data['Short_MA'] = data['Close'].rolling(window=short_window, min_periods=1).mean()
    
    # Calculate the long-term moving average (200 days by default)
    data['Long_MA'] = data['Close'].rolling(window=long_window, min_periods=1).mean()
    
    # Create a signal column: 1 means buy, 0 means sell
    data['Signal'] = 0
    data['Signal'][short_window:] = np.where(data['Short_MA'][short_window:] > data['Long_MA'][short_window:], 1, 0)
    
    # Create a position column: shows the changes in buy/sell signals
    data['Position'] = data['Signal'].diff()
    return data

# Read historical stock data from an Excel file
data = pd.read_excel('data/data_reversed.xlsx')

# Apply the moving average strategy to the data
strategy_data = moving_average_strategy(data)

# Save the strategy data to a temporary CSV file
with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp:
    strategy_data.to_csv(tmp.name, index=False)
    csv_file = tmp.name

# Define a class for the moving average strategy in Backtrader
class MovingAverageStrategy(bt.Strategy):
    params = (('short_window', 50), ('long_window', 150), ('stop_loss', 0.05),)

    def __init__(self):
        self.short_ma = bt.indicators.SimpleMovingAverage(self.data.close, period=self.params.short_window)
        self.long_ma = bt.indicators.SimpleMovingAverage(self.data.close, period=self.params.long_window)
        self.order = None
        self.buy_price = None

    def next(self):
        if self.order:
            return
        if not self.position:
            if self.short_ma[0] > self.long_ma[0]:
                self.order = self.buy()
                self.buy_price = self.data.close[0]
        else:
            if self.short_ma[0] < self.long_ma[0] or self.data.close[0] < self.buy_price * (1 - self.params.stop_loss):
                self.order = self.sell()

# Setup and run the backtest
cerebro = bt.Cerebro()
cerebro.addstrategy(MovingAverageStrategy)

# Add the stock data to Backtrader from the CSV file
data = bt.feeds.GenericCSVData(
    dataname=csv_file,
    fromdate=datetime(2019, 1, 1),
    todate=datetime(2025, 12, 31),
    dtformat='%Y-%m-%d',
    datetime=0,
    high=3,
    low=4,
    open=2,
    close=1,
    volume=5,
    openinterest=-1,
)
cerebro.adddata(data)
print(strategy_data[['Date', 'Close', 'Short_MA', 'Long_MA', 'Signal', 'Position']])

# Run the backtest and plot the results
cerebro.run()
cerebro.plot()