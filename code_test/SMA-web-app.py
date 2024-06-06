import pandas as pd
import numpy as np
import yfinance as yf
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
from backtesting.test import SMA
import plotly.graph_objects as go
from dash import Dash, dcc, html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc

datafile = yf.download('AAPL', start='2005-01-01', end='2015-12-31')
datafile.to_csv('AAPL.csv')

datafile = pd.read_csv('VN-Index-Historical-Data.csv', index_col='Date', parse_dates=True)

datafile = datafile[datafile['Volume'] >= 1000000]

def moving_average_strategy(datafile, short_window=50, long_window=200):
    datafile['Short_MA'] = datafile['Close'].rolling(window=short_window, min_periods=1).mean()
    datafile['Long_MA'] = datafile['Close'].rolling(window=long_window, min_periods=1).mean()
    datafile['Signal'] = 0
    datafile.loc[datafile['Short_MA'] > datafile['Long_MA'], 'Signal'] = 1
    datafile['Position'] = datafile['Signal'].diff()
    return datafile

datafile = moving_average_strategy(datafile)

def dualMACrossover(data, short_window=50, long_window=200, min_volume=1000000, stop_loss=0.05):
    data['Short_MA'] = data['Close'].rolling(window=short_window, min_periods=1).mean()
    data['Long_MA'] = data['Close'].rolling(window=long_window, min_periods=1).mean()

    data['BuySignalPrice'] = np.nan
    data['SellSignalPrice'] = np.nan
    data['Position'] = 0
    data['Profit/Loss'] = np.nan

    position = 0
    buy_price = 0
    print(data)
    for i in range(long_window, len(data)):
        if data['Volume'][i] >= min_volume:
            if data['Short_MA'][i] > data['Long_MA'][i] and data['Short_MA'][i - 1] <= data['Long_MA'][i - 1]:
                data['BuySignalPrice'][i] = data['Close'][i]
                position = 1
                buy_price = data['Close'][i]
            elif data['Short_MA'][i] < data['Long_MA'][i] and data['Short_MA'][i - 1] >= data['Long_MA'][i - 1]:
                data['SellSignalPrice'][i] = data['Close'][i]
                position = 0
                data['Profit/Loss'][i] = data['Close'][i] - buy_price
            elif position == 1 and data['Close'][i] < buy_price * (1 - stop_loss):
                data['SellSignalPrice'][i] = data['Close'][i]
                position = 0
                data['Profit/Loss'][i] = data['Close'][i] - buy_price
    return data

datafile = dualMACrossover(datafile)

window1 = 50
window2 = 200
data = pd.DataFrame()
data['AAPL'] = datafile['Close']
data['SMA' + str(window1)] = datafile['Short_MA']
data['SMA' + str(window2)] = datafile['Long_MA']
data['Volume'] = datafile['Volume']
data['BuySignalPrice'] = datafile['BuySignalPrice']
data['SellSignalPrice'] = datafile['SellSignalPrice']
data['Profit/Loss'] = datafile['Profit/Loss']

volume_colors = ['yellow' if i % 2 == 0 else 'purple' for i in range(len(data))]

class SmaCross(Strategy):
    n1 = 50
    n2 = 200
    stop_loss = 0.05

    def init(self):
        self.sma1 = self.I(SMA, self.data.Close, self.n1)
        self.sma2 = self.I(SMA, self.data.Close, self.n2)

    def next(self):
        # Only consider trades if volume is above 1 million shares
        if self.data.Volume[-1] >= 1000000:
            # Buy signal: when 50-day MA crosses above 200-day MA
            if crossover(self.sma1, self.sma2):
                self.buy()
            # Sell signal: when 50-day MA crosses below 200-day MA
            elif crossover(self.sma2, self.sma1):
                self.sell()

        # Implement stop-loss
        for trade in self.trades:
            if self.data.Close[-1] < trade.entry_price * (1 - self.stop_loss):
                self.position.close()

# Run Backtest
bt = Backtest(datafile, SmaCross, cash=10000, commission=.002, exclusive_orders=True)
stats = bt.run()
# bt.plot()

# Format statistics
statistics = f"""
Start: {stats['Start']}
End: {stats['End']}
Duration: {stats['Duration']}
Exposure Time [%]: {round(stats['Exposure Time [%]'], 4)}
Equity Final [$]: {round(stats['Equity Final [$]'], 4)}
Equity Peak [$]: {round(stats['Equity Peak [$]'], 4)}
Return [%]: {round(stats['Return [%]'], 4)}
Buy & Hold Return [%]: {round(stats['Buy & Hold Return [%]'], 4)}
Return (Ann.) [%]: {round(stats['Return (Ann.) [%]'], 4)}
Volatility (Ann.) [%]: {round(stats['Volatility (Ann.) [%]'], 4)}
Sharpe Ratio: {round(stats['Sharpe Ratio'], 4)}
Sortino Ratio: {round(stats['Sortino Ratio'], 4)}
Calmar Ratio: {round(stats['Calmar Ratio'], 4)}
Max. Drawdown [%]: {round(stats['Max. Drawdown [%]'], 4)}
Avg. Drawdown [%]: {round(stats['Avg. Drawdown [%]'], 4)}
Max. Drawdown Duration: {stats['Max. Drawdown Duration']}
Avg. Drawdown Duration: {stats['Avg. Drawdown Duration']}
# Trades: {stats['# Trades']}
Win Rate [%]: {round(stats['Win Rate [%]'], 4)}
Best Trade [%]: {round(stats['Best Trade [%]'], 4)}
Worst Trade [%]: {round(stats['Worst Trade [%]'], 4)}
Avg. Trade [%]: {round(stats['Avg. Trade [%]'], 4)}
Max. Trade Duration: {stats['Max. Trade Duration']}
Avg. Trade Duration: {stats['Avg. Trade Duration']}
Profit Factor: {round(stats['Profit Factor'], 4)}
Expectancy [%]: {round(stats['Expectancy [%]'], 4)}
SQN: {round(stats['SQN'], 4)}
"""

# Create Dash app
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Define layout
app.layout = html.Div([
    html.Button('Show Details', id='show-details-button', n_clicks=0, 
                style={
                    'position': 'absolute',
                    'top': '0px',
                    'right': '0px', 
                    'z-index': '1000',
                    'padding': '10px 20px', 
                    'font-size': '10px', 
                    'border-radius': '5px',
                    'background-color': '#007bff', 
                    'color': '#ffffff'
    }),
    html.Div([
        dcc.Graph(id='stock-graph', style={'height': '45vh', 'margin-top': '30px'}),
        dcc.Graph(id='volume-graph', style={'height': '45vh', 'margin-top': '30px'})
    ]),
    dbc.Modal(
        [
            dbc.ModalHeader(dbc.ModalTitle("Detailed Statistics")),
            dbc.ModalBody(id='details-div', style={'whiteSpace': 'pre-line'}),
            dbc.ModalFooter(
                dbc.Button("Close", id="close", className="ms-auto", n_clicks=0)
            ),
        ],
        id="modal",
        is_open=False,
    )
])

# Update stock graph
@app.callback(
    Output('stock-graph', 'figure'),
    Input('show-details-button', 'n_clicks')
)
def update_stock_graph(n_clicks):


    fig = go.Figure()

    fig.add_trace(go.Scatter(x=data.index, y=data['AAPL'], mode='lines', name='AAPL', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=data.index, y=data['SMA' + str(window1)], mode='lines', name='SMA' + str(window1), line=dict(color='orange')))
    fig.add_trace(go.Scatter(x=data.index, y=data['SMA' + str(window2)], mode='lines', name='SMA' + str(window2), line=dict(color='green')))

    fig.add_trace(go.Scatter(mode="markers", x=data.index, y=data.BuySignalPrice, marker_symbol='triangle-up',
                            marker_line_color="green", marker_color="green", 
                            marker_line_width=2, marker_size=10, name='Buy'))
    fig.add_trace(go.Scatter(mode="markers", x=data.index, y=data.SellSignalPrice, marker_symbol='triangle-down',
                            marker_line_color="red", marker_color="red", 
                            marker_line_width=2, marker_size=10, name='Sell'))
    
    for i in range(len(data)):
        if not np.isnan(data['Profit/Loss'][i]):
            profit_loss = data['Profit/Loss'][i]
            date = data.index[i].strftime("%Y-%m-%d")
            percentage = (profit_loss / datafile['Close'][i] * 100)
            text = f'{date}<br>Profit: {profit_loss:.2f} ({percentage:.2f}%)' if profit_loss > 0 else f'{date}<br>Loss: {profit_loss:.2f} ({percentage:.2f}%)'

            fig.add_scatter(
                x=[data.index[i]],
                y=[data['SellSignalPrice'][i]],
                mode="markers",
                marker=dict(
                    symbol="circle",
                    color="green" if profit_loss > 0 else "red",
                    size=10,
                    opacity=0
                ),
                text=text,
                hoverinfo="text",
                showlegend=False
            )

    fig.update_layout(
        title='AAPL Stock Price with Dual Moving Average Crossover Strategy',
        yaxis=dict(title='Price'),
        xaxis=dict(title='Date')
    )

    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='white')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='white')

    return fig


@app.callback(
    Output('volume-graph', 'figure'),
    Input('show-details-button', 'n_clicks')
)
def update_volume_graph(n_clicks):
    fig = go.Figure()

    fig.add_trace(go.Bar(x=data.index, y=data['Volume'], name='Volume', marker_color=volume_colors, opacity=0.7))

    fig.update_layout(
        title='AAPL Trading Volume',
        yaxis=dict(title='Volume'),
        xaxis=dict(title='Date'),
    )

    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='white')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='white')

    return fig

@app.callback(
    Output('modal', 'is_open'),
    Output('details-div', 'children'),
    Input('show-details-button', 'n_clicks'),
    Input('close', 'n_clicks'),
    State('modal', 'is_open')
)
def toggle_modal(n1, n2, is_open):
    if n1 or n2:
        return not is_open, statistics
    return is_open, statistics

if __name__ == '__main__':
    app.run_server(debug=True)