import dash
from dash import html, dcc, Input, Output
import plotly.graph_objs as go
import yfinance as yf
import pandas as pd

app = dash.Dash(__name__)
server = app.server

app.layout = html.Div(
    [
        html.H3("Painel Amanajé — Simple Price Viewer"),
        dcc.Input(id="ticker", value="AAPL", type="text", debounce=True),
        dcc.Graph(id="price-graph"),
    ],
    style={"maxWidth": "900px", "margin": "0 auto"},
)

@app.callback(Output("price-graph", "figure"), Input("ticker", "value"))
def update_graph(ticker):
    if not ticker:
        return go.Figure()
    try:
        t = yf.Ticker(ticker)
        hist = t.history(period="1mo", interval="1d")
        if hist.empty:
            return go.Figure()
        df = hist.reset_index()
        fig = go.Figure(data=[go.Scatter(x=df["Date"], y=df["Close"], mode="lines", name=f"{ticker.upper()} Close")])
        fig.update_layout(xaxis_title="Date", yaxis_title="Close")
        return fig
    except Exception:
        return go.Figure()

if __name__ == "__main__":
    app.run_server(host="0.0.0.0", port=8050, debug=False)