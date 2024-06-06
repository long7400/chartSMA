# Import necessary libraries
import pandas as pd  # Used for handling data in a table format
import numpy as np   # Used for numerical operations
import backtrader as bt  # Used for backtesting trading strategies
from datetime import datetime  # Used to handle date and time

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

# Read historical stock data from a CSV file
data = pd.read_csv('VN-Index-Historical-Data.csv')

# Apply the moving average strategy to the data
strategy_data = moving_average_strategy(data)

# Define a class for the moving average strategy in Backtrader
class MovingAverageStrategy(bt.Strategy):
    params = (('short_window', 50), ('long_window', 200), ('stop_loss', 0.05),)

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

# Add the stock data to Backtrader
data = bt.feeds.GenericCSVData(
    dataname='VN-Index-Historical-Data.csv',
    fromdate=datetime(2010, 1, 1),
    todate=datetime(2020, 12, 31),
    dtformat='%Y-%m-%d',
    datetime=0,
    high=2,
    low=3,
    open=1,
    close=4,
    volume=6,
    openinterest=-1,
    adjclose=5
)
cerebro.adddata(data)

# Run the backtest and plot the results
cerebro.run()
cerebro.plot()
