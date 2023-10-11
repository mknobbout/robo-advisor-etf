import dash
from dash import html, dcc, dash_table
from dash.dependencies import Input, Output
from optimization import ETFPortfolioOptimizer
from datetime import date
from dash.exceptions import PreventUpdate

import dash_bootstrap_components as dbc
import pandas as pd

# Load the model
portfolio_model = ETFPortfolioOptimizer.load("models/portfolio.model")

# Create a Dash web application
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
)

# Sample data for the table
data = pd.DataFrame({"Value": range(1, 101)})

# Define the layout of the app
app.layout = html.Div(
    [
        html.Div(
            [
                html.Div(
                    [
                        html.H1("Robo-Advisor"),
                    ],
                    className="banner",
                ),
                html.Div(
                    [
                        html.Label("When were you born?"),
                        html.Br(),
                        dcc.DatePickerSingle(date=date(2000, 1, 1)),
                    ]
                ),
                html.Div(
                    [
                        html.Label("Have you invested before?"),
                        dcc.Dropdown(
                            options=[
                                {"label": "I am a beginner", "value": 0},
                                {"label": "I know what investing is", "value": 1},
                                {"label": "Experienced investor", "value": 2},
                            ],
                            clearable=False,
                            value=1,
                            id="experience",
                        ),
                    ]
                ),
                html.Div(
                    [
                        html.Label("Choose how spicy you want to make your portfolio"),
                        dcc.Slider(
                            id="risk-slider",
                            min=1,
                            max=100,
                            step=1,
                            marks={
                                1: "🌶️ Mild '",
                                50: "🌶️🌶️️ Spicy",
                                100: "🌶️🌶️🌶️ Hot!",
                            },
                            value=25,
                        ),
                    ],
                    style={"white-space": "nowrap"},
                ),
                html.Div(
                    [
                        html.Label("When do you expect to see your returns?"),
                        dcc.Dropdown(
                            options=[
                                {"label": "Short term: 1 year", "value": 0},
                                {"label": "Medium term: 3 to 7 years", "value": 1},
                                {
                                    "label": "Long term: more than 7 years into the future",
                                    "value": 2,
                                },
                            ],
                            value=1,
                            clearable=False,
                            id="horizon",
                        ),
                    ]
                ),
                html.Div(
                    [
                        html.Label(
                            "Do you believe your investments can help the environment and society?"
                        ),
                        dcc.Dropdown(
                            options=[
                                {"label": "Not really", "value": 0},
                                {"label": "🌿 Make it sustainable!", "value": 0.25},
                                {
                                    "label": "🌿🌿 Make it extra sustainable!",
                                    "value": 0.5,
                                },
                            ],
                            value=0.25,
                            clearable=False,
                            id="sustainability",
                        ),
                    ]
                ),
                html.Div(
                    [
                        html.Button(
                            "Construct Portfolio", id="show-table-button", n_clicks=0
                        )
                    ]
                ),
                dcc.Checklist(
                    id="disclaimer-check",
                    options=[
                        {
                            "label": "I understand that past performance does not guarantee future results",
                            "value": "understand",
                        },
                        {
                            "label": "I agree that my data will be shared with Group 8 under GDPR regulations",
                            "value": "agree",
                        },
                    ],
                    value=[],
                    style={"font-size": "10px", "margin": "8px"},
                ),
            ],
            className="main_ui",
            id="main_ui",
        ),
        dcc.Loading(
            id="loading",
            children=[
                html.Center(html.Div(id="table-stats", style={"width": "50%"})),
                html.Center(html.Div(id="table-portfolio", style={"width": "80%"})),
            ],
            type="default",
        ),
        # html.Div(id='table-portfolio'),
    ]
)


# Callback to update the table
@app.callback(
    Output("table-portfolio", "children"),
    Output("table-stats", "children"),
    Output("main_ui", "children"),
    Input("show-table-button", "n_clicks"),
    Input("risk-slider", "value"),
    Input("sustainability", "value"),
    prevent_initial_call=True,
)
def update_table(n_clicks, target_risk, sustainability):
    if n_clicks == 0:
        raise PreventUpdate()

    # Compute portfolio weights
    portfolio_weights = portfolio_model.optimize(
        0.25 + (target_risk / 80),
        geq_constraints={"is_sustainable": sustainability},
        max_percentage_per_etf=0.12,
    )

    # Get dataframe
    portfolio_df = portfolio_model.create_portfolio_overview(portfolio_weights)
    # Only keep weights above 0.1%
    portfolio_df = portfolio_df[portfolio_df.weight >= 0.001]
    # Insert emoticons for nicer view
    portfolio_df["is_sustainable"] = portfolio_df["is_sustainable"].map(
        lambda x: "🌿" if x == 1 else ""
    )

    # Create a table
    portfolio_table = dash_table.DataTable(
        data=portfolio_df.to_dict("records"),
        columns=[
            {"name": "Symbol", "id": "symbol"},
            {"name": "Full Name", "id": "longName"},
            {
                "name": "Weight",
                "id": "weight",
                "type": "numeric",
                "format": dash_table.FormatTemplate.percentage(2),
            },
            {"name": "Category", "id": "category"},
            {"name": "Sustainable", "id": "is_sustainable"},
        ],
    )

    # Get dataframe of stats
    portfolio_stats_table = dash_table.DataTable(
        data=portfolio_model.get_statistics(portfolio_weights).to_dict("records"),
        columns=[
            {"name": "Description", "id": "description"},
            {"name": "Value", "id": "value"},
        ],
    )
    return portfolio_table, portfolio_stats_table, "Your portfolio is ready!"


if __name__ == "__main__":
    app.run_server(debug=True)

## Images taken from
# - <a href="https://www.flaticon.com/free-icons/invest" title="invest icons">Invest icons created by ultimatearm - Flaticon</a>
