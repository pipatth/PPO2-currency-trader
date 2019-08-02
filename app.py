import pandas as pd
import numpy as np
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
import locale
from dash.dependencies import Input, Output
from plotly import tools
from empyrical import max_drawdown
from model import load_env_model

assets = ["USD_CAD", "USD_CHF"]
locale.setlocale(locale.LC_ALL, "")

# load environment and model
asset_data = {}
for asset in assets:
    env, model = load_env_model(asset, test=True)
    obs = env.reset()
    asset_data[asset] = {"env": env, "model": model, "obs": obs}


# flask
app = dash.Dash(
    __name__, meta_tags=[{"name": "viewport", "content": "width=device-width"}]
)
server = app.server

# take action
def next_step(asset):
    # predict
    action, _states = model.predict(asset_data[asset]["obs"])
    asset_data[asset]["obs"], rewards, done, info = asset_data[asset]["env"].step(
        action
    )
    return


# Formatting
def get_local_format(value, digits=2):
    return locale.currency(np.round(value, digits), grouping=True)


# color of Bid & Ask rates
def get_color(a, b):
    if a == b:
        return "white"
    elif a > b:
        return "#45df7e"
    else:
        return "#da5657"


# Left panel components

# Get current step of the first asset symbol
def get_steps(asset_data):
    asset = list(asset_data.keys())[0]
    env = asset_data[asset]["env"]
    return "Step: " + str(env.envs[0].current_step)


# Creates HTML Bid and Ask section
def get_row(asset, data):
    return html.Div(
        children=[
            # Summary
            html.Div(
                id=asset + "summary",
                className="row summary",
                n_clicks=0,
                children=[
                    html.Div(
                        id=asset + "row",
                        className="row",
                        children=render_row(asset, data),
                    )
                ],
            )
        ]
    )


# Render ask_bid row for currency pair with colored values
def render_row(asset, data):
    env = data["env"]
    df = env.envs[0].get_summary()
    prev_row = df.iloc[-2]
    new_row = df.iloc[-1]
    return [
        html.P(asset, id=asset, className="three-col"),  # currency pair name
        html.P(
            np.round(new_row["bid"], 5),  # Bid value
            id=asset + "bid",
            className="three-col",
            style={"color": get_color(new_row["bid"], prev_row["bid"])},
        ),
        html.P(
            np.round(new_row["ask"], 5),  # Ask value
            id=asset + "ask",
            className="three-col",
            style={"color": get_color(new_row["ask"], prev_row["ask"])},
        ),
    ]


# Top bar components

# Returns Top cell bar for header area
def get_top_bar_cell(cellTitle, cellValue, fmt=["currency", "pct"]):
    return html.Div(
        className="two-col",
        children=[
            html.P(className="p-top-bar", children=cellTitle),
            html.P(id=cellTitle, className="display-none", children=cellValue),
            html.P(
                children=(
                    (str(np.round(cellValue * 100, 2)) + "%")
                    if fmt == "pct"
                    else get_local_format(cellValue)
                )
            ),
        ],
    )


# Returns HTML Top Bar for app layout
def get_top_bar(asset_data):
    initial = 0
    nw = 0
    l_nw = []
    for asset in assets:
        env = asset_data[asset]["env"]
        initial += env.envs[0].initial
        nw += env.envs[0].net_worth
        l_nw.append(env.envs[0].get_summary()["net_worth"])
    open_pl = nw - initial
    s_nw = pd.concat(l_nw, axis=1).sum(axis=1)
    s_ret = (s_nw.diff().fillna(0) / s_nw.shift(1)).fillna(0)
    return [
        get_top_bar_cell("Initial Capital", np.round(initial, 2)),
        get_top_bar_cell("Net Worth", np.round(nw, 2)),
        get_top_bar_cell("Open P/L", np.round(open_pl, 2)),
        get_top_bar_cell("% Gain", open_pl / initial, fmt="pct"),
        get_top_bar_cell("% Max Drawdown", max_drawdown(s_ret), fmt="pct"),
    ]


# Position table components

# get position table
def get_position(asset_data):
    headers = [
        "Symbol",
        "Unit held",
        "Cost",
        "Initial Capital",
        "Net Worth",
        "Open P/L",
        "% Gain",
        "% Max Drawdown",
    ]

    rows = []
    for asset in assets:
        tr_childs = []
        env = asset_data[asset]["env"]
        summary = env.envs[0].get_summary()
        row = {
            "Symbol": asset,
            "Unit held": summary["unit_bought"].sum() - summary["unit_sold"].sum(),
            "Cost": get_local_format(summary["cost"].sum() - summary["sales"].sum()),
            "Initial Capital": get_local_format(env.envs[0].initial),
            "Net Worth": get_local_format(env.envs[0].net_worth),
            "Open P/L": get_local_format(env.envs[0].net_worth - env.envs[0].initial),
            "% Gain": str(
                np.round(
                    (env.envs[0].net_worth - env.envs[0].initial)
                    / env.envs[0].initial
                    * 100,
                    2,
                )
            )
            + "%",
            "% Max Drawdown": str(np.round(max_drawdown(summary["ret"]) * 100, 2))
            + "%",
        }
        for k in row.keys():
            tr_childs.append(html.Td(row[k]))
        # Color row based on profitability
        if row["Open P/L"][0] != "-":
            rows.append(html.Tr(className="profit", children=tr_childs))
        else:
            rows.append(html.Tr(className="no-profit", children=tr_childs))

    return html.Table(children=[html.Tr([html.Th(title) for title in headers])] + rows)


# Chart components

# colored bar
def colored_bar_trace(df):
    return go.Ohlc(
        x=df.index,
        open=df["open"],
        high=df["high"],
        low=df["low"],
        close=df["close"],
        showlegend=False,
        name="colored bar",
    )


# area
def area_trace(x, y):
    trace = go.Scatter(x=x, y=y, showlegend=False, name="area")
    return trace


# bar
def bar_trace(x, y):
    trace = go.Bar(x=x, y=y, showlegend=False, name="bar")
    return trace


# Returns graph figure
def get_fig(asset, data, n_bar=50):
    env = data["env"]
    df = env.envs[0].get_summary()[-n_bar:]
    row = 1  # number of subplots
    fig = tools.make_subplots(
        rows=row,
        shared_xaxes=True,
        shared_yaxes=True,
        cols=1,
        print_grid=False,
        vertical_spacing=0.12,
    )

    # Add main trace (style) to figure
    type_trace = "colored_bar_trace"
    fig.append_trace(eval(type_trace)(df), 1, 1)
    fig["layout"][
        "uirevision"
    ] = "The User is always right"  # Ensures zoom on graph is the same on update
    fig["layout"]["margin"] = {"t": 50, "l": 50, "b": 50, "r": 25}
    fig["layout"]["autosize"] = True
    fig["layout"]["height"] = 400
    fig["layout"]["xaxis"]["rangeslider"]["visible"] = False
    fig["layout"]["yaxis"]["showgrid"] = True
    fig["layout"]["yaxis"]["gridcolor"] = "#3E3F40"
    fig["layout"]["yaxis"]["gridwidth"] = 1
    fig["layout"].update(paper_bgcolor="#21252C", plot_bgcolor="#21252C")

    # take an action
    # print(env.envs[0].steps_left)
    next_step(asset)

    return fig


# Returns net worth graph figure
def get_fig_nw(asset, data, n_bar=50):
    env = data["env"]
    df = env.envs[0].get_summary()[-n_bar:]
    row = 1  # number of subplots
    fig = tools.make_subplots(
        rows=row,
        shared_xaxes=True,
        shared_yaxes=True,
        cols=1,
        print_grid=False,
        vertical_spacing=0.12,
    )

    # Add main trace (style) to figure
    type_trace = "area_trace"
    fig.append_trace(eval(type_trace)(df.index, df["net_worth"]), 1, 1)
    fig["layout"][
        "uirevision"
    ] = "The User is always right"  # Ensures zoom on graph is the same on update
    fig["layout"]["margin"] = {"t": 50, "l": 50, "b": 50, "r": 25}
    fig["layout"]["autosize"] = True
    fig["layout"]["height"] = 400
    fig["layout"]["xaxis"]["rangeslider"]["visible"] = False
    fig["layout"]["yaxis"]["showgrid"] = True
    fig["layout"]["yaxis"]["gridcolor"] = "#3E3F40"
    fig["layout"]["yaxis"]["gridwidth"] = 1
    fig["layout"].update(paper_bgcolor="#21252C", plot_bgcolor="#21252C")

    return fig


# Returns buy-sell figure
def get_fig_buysell(asset, data, n_bar=50):
    env = data["env"]
    df = env.envs[0].get_summary()[-n_bar:]
    row = 1  # number of subplots
    fig = tools.make_subplots(
        rows=row,
        shared_xaxes=True,
        shared_yaxes=True,
        cols=1,
        print_grid=False,
        vertical_spacing=0.12,
    )

    # Add main trace (style) to figure
    type_trace = "bar_trace"
    fig.append_trace(eval(type_trace)(df.index, df["sales"]), 1, 1)
    fig.append_trace(eval(type_trace)(df.index, -df["cost"]), 1, 1)
    fig["layout"][
        "uirevision"
    ] = "The User is always right"  # Ensures zoom on graph is the same on update
    fig["layout"]["margin"] = {"t": 50, "l": 50, "b": 50, "r": 25}
    fig["layout"]["autosize"] = True
    fig["layout"]["height"] = 400
    fig["layout"]["xaxis"]["rangeslider"]["visible"] = False
    fig["layout"]["yaxis"]["showgrid"] = True
    fig["layout"]["yaxis"]["gridcolor"] = "#3E3F40"
    fig["layout"]["yaxis"]["gridwidth"] = 1
    fig["layout"].update(
        paper_bgcolor="#21252C", plot_bgcolor="#21252C", barmode="stack"
    )

    return fig


# returns chart div
def chart_div(asset):
    return html.Div(
        id=asset + "graph_div",
        className="chart-style six columns",
        children=[
            # Chart Top Bar
            html.Div(
                className="row chart-top-bar",
                children=[
                    html.Span(
                        id=asset + "menu_button",
                        className="inline-block chart-title",
                        children=f"{asset}",
                    )
                ],
            ),
            # Graph price div
            html.Div(
                dcc.Graph(
                    id=asset + "chart",
                    className="chart-graph",
                    config={"displayModeBar": False, "scrollZoom": False},
                )
            ),
            # Graph net worth div
            html.Div(
                dcc.Graph(
                    id=asset + "chart_nw",
                    className="chart-graph",
                    config={"displayModeBar": False, "scrollZoom": False},
                )
            ),
            # Graph buy-sell div
            html.Div(
                dcc.Graph(
                    id=asset + "chart_buysell",
                    className="chart-graph",
                    config={"displayModeBar": False, "scrollZoom": False},
                )
            ),
        ],
    )


# Dash App Layout
app.layout = html.Div(
    className="row",
    children=[
        # Interval component for updates
        dcc.Interval(id="i_bis", interval=1 * 1000, n_intervals=0),
        # Left Panel Div
        html.Div(
            className="three columns div-left-panel",
            children=[
                # Div for Left Panel App Info
                html.Div(
                    className="div-info",
                    children=[
                        html.Img(
                            className="logo", src=app.get_asset_url("dash-logo-new.png")
                        ),
                        html.H6(className="title-header", children="FOREX TRADER"),
                        html.P(
                            """
                            This app continually queries gym currency trading environment (using data from OANDA),
                            updates Bid-Ask prices, and simulate trading by bots trained on a reinforcement learning
                            algorithm. You can also virtually see the profit updates.
                            """
                        ),
                    ],
                ),
                # Ask Bid Currency Div
                html.Div(
                    className="div-currency-toggles",
                    children=[
                        html.P(
                            id="live_clock",
                            className="three-col",
                            children=get_steps(asset_data),
                        ),
                        html.P(className="three-col", children="Bid"),
                        html.P(className="three-col", children="Ask"),
                        html.Div(
                            id="pairs",
                            className="div-bid-ask",
                            children=[
                                get_row(asset, asset_data[asset]) for asset in assets
                            ],
                        ),
                    ],
                ),
            ],
        ),
        # Right Panel Div
        html.Div(
            className="nine columns div-right-panel",
            children=[
                # Top Bar Div - Displays Balance, Equity, ... , Open P/L
                html.Div(
                    id="top_bar",
                    className="row div-top-bar",
                    children=get_top_bar(asset_data),
                ),
                # Charts Div
                html.Div(
                    id="charts",
                    className="row",
                    children=[chart_div(asset) for asset in assets],
                ),
                # Panel for orders
                html.Div(
                    id="bottom_panel",
                    className="row div-bottom-panel",
                    children=[
                        html.Div(id="orders_table", className="row table-orders")
                    ],
                ),
            ],
        ),
    ],
)

# Dynamic Callbacks

# Replace top bar row
def generate_top_bar_callback():
    def topbar_callback(n_i):
        return get_top_bar(asset_data)

    return topbar_callback


# Replace step
def generate_step_callback():
    def step_callback(n_i):
        return get_steps(asset_data)

    return step_callback


# Function to update position table
def generate_position_callback():
    def position_callback(n_I):
        return get_position(asset_data)

    return position_callback


# Replace bid ask row
def generate_ask_bid_row_callback(asset):
    def output_callback(n_i):
        return render_row(asset, asset_data[asset])

    return output_callback


# Function to update price Graph Figure
def generate_figure_callback(asset):
    def chart_fig_callback(n_i):
        fig = get_fig(asset, asset_data[asset])
        return fig

    return chart_fig_callback


# Function to update net worth Graph Figure
def generate_figure_nw_callback(asset):
    def chart_fig_nw_callback(n_i):
        fig = get_fig_nw(asset, asset_data[asset])
        return fig

    return chart_fig_nw_callback


# Function to update buy-sell Graph Figure
def generate_figure_buysell_callback(asset):
    def chart_fig_buysell_callback(n_i):
        fig = get_fig_buysell(asset, asset_data[asset])
        return fig

    return chart_fig_buysell_callback


# assign callbacks

# Callback to update top bar
app.callback(Output("top_bar", "children"), [Input("i_bis", "n_intervals")])(
    generate_top_bar_callback()
)

# Callback to update steps
app.callback(Output("live_clock", "children"), [Input("i_bis", "n_intervals")])(
    generate_step_callback()
)

# Callback to update Orders Table
app.callback(Output("orders_table", "children"), [Input("i_bis", "n_intervals")])(
    generate_position_callback()
)

# assign callbacks on symbol loop
for asset in assets:

    # Callback to update the ask and bid prices
    app.callback(Output(asset + "row", "children"), [Input("i_bis", "n_intervals")])(
        generate_ask_bid_row_callback(asset)
    )

    # Callback to update the price graph
    app.callback(Output(asset + "chart", "figure"), [Input("i_bis", "n_intervals")])(
        generate_figure_callback(asset)
    )

    # Callback to update the net worth graph
    app.callback(Output(asset + "chart_nw", "figure"), [Input("i_bis", "n_intervals")])(
        generate_figure_nw_callback(asset)
    )

    # Callback to update the buy-sell graph
    app.callback(
        Output(asset + "chart_buysell", "figure"), [Input("i_bis", "n_intervals")]
    )(generate_figure_buysell_callback(asset))

if __name__ == "__main__":
    app.run_server(debug=True)
