import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# Tải dữ liệu của AAPL từ Yahoo Finance
aapl = yf.download('AAPL', start='2020-01-01', end='2023-01-01')

# Vẽ biểu đồ giá cổ phiếu
aapl['Adj Close'].plot(figsize=(10, 6))
plt.title('AAPL Stock Price')
plt.xlabel('Date')
plt.ylabel('Price ($)')
plt.grid(True)
plt.show()

# Tính toán độ biến động (volatility)
aapl['Volatility'] = aapl['Adj Close'].pct_change().rolling(window=20).std() * (252**0.5)
# Vẽ biểu đồ độ biến động
aapl['Volatility'].plot(figsize=(10, 6))
plt.title('AAPL Volatility')
plt.xlabel('Date')
plt.ylabel('Volatility')
plt.grid(True)
plt.show()

# Tính toán và vẽ biểu đồ MA (Moving Average)
aapl['MA50'] = aapl['Adj Close'].rolling(window=50).mean()
aapl['MA200'] = aapl['Adj Close'].rolling(window=200).mean()
aapl[['Adj Close', 'MA50', 'MA200']].plot(figsize=(10, 6))
plt.title('AAPL Moving Averages')
plt.xlabel('Date')
plt.ylabel('Price ($)')
plt.grid(True)
plt.show()
