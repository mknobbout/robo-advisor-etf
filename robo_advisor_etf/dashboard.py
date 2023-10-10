import dash
from dash import html, dcc, dash_table
from dash.dependencies import Input, Output
from optimization import ETFPortfolioOptimizer

import dash_bootstrap_components as dbc
import pandas as pd

# Load the model
portfolio_model = ETFPortfolioOptimizer.load('portfolio.model')

# Create a Dash web application
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP],)

# Sample data for the table
data = pd.DataFrame({
    'Value': range(1, 101)
})

# Define the layout of the app
app.layout = html.Div([
    html.Div([
        html.H1("Welcome to your personal Robo-Advisor!"),
        html.Img(src="assets/robo-advisor.png")
    ], className="banner"),
    html.H2([
        html.Label("To provide a customized experience, we need to know you better:")
    ]),
    html.Div([
        html.Label("What is your Name?"),
        dcc.Input(
            id="name",
            type="text"
        )
    ]),
    html.Div([
        html.Label("What is your age?"),
        dcc.Input(
            id="age",
            type="number"
        )
    ]),
    html.Div([
        html.Label("Have you invested before?"),
        dcc.RadioItems(
           options=[
               {'label': "Investing, what's that?", 'value': 0},
               {'label': "I know what investing is", 'value': 1},
               {'label': "Investing is my cardio", 'value': 2},
           ],
           value=1,
           id="experience"
        )
    ]),
    html.Div([
        html.Label("Move the slider towards the statement that makes you more confortable"),
        dcc.Slider(
            id='risk-slider',
            min=1,
            max=100,
            step=1,
            marks={
                1: "No risk",
                25: "Maybe a little bit of risk",
                50: "Give me just the average risk",
                75: "No pain, no gain",
                100: "Give me all risk!"
            },
            value=50
        )
    ]),
    html.Div([
        html.Label("Do you have a desired amount in mind? Leave this at 0 if you do not."),
        dcc.Input(
            id="terminal-value",
            type="number",
            value=0
        )
    ]),
    html.Div([
        html.Label("When do you expect to see your returns?"),
        dcc.RadioItems(
           options=[
               {'label': "Short term: 1 year", 'value': 0},
               {'label': "Medium term: 3 to 7 years", 'value': 1},
               {'label': "Long term: more than 7 years into the future", 'value': 2},
           ],
           value=1,
           id="horizon"
        )
    ]),
    html.Div([
        html.Label("Do you think your investing to have an impact on the environment and society in addition to your "
                   "financial returns?"),
        dcc.RadioItems(
           options=[
               {'label': "Not really", 'value': 0},
               {'label': "I do not mind", 'value': 1},
               {'label': "Yes, definitely", 'value': 2},
           ],
           value=1,
           id="sustainability"
        )
    ]),
    html.Div([
        html.Button('Construct Portfolio', id='show-table-button', n_clicks=0)
    ]),
    dcc.Loading(
        id="loading",
        children=html.Div(id='table-portfolio'),
        type="default",
    ),
    #html.Div(id='table-portfolio'),
])


# Callback to update the table
@app.callback(
    Output('table-portfolio', 'children'),
    Input('show-table-button', 'n_clicks'),
    Input('risk-slider', 'value'),
)
def update_table(n_clicks, target_risk):
    if n_clicks == 0:
        return ''

    # Compute portfolio weights
    portfolio_weights = portfolio_model.optimize(target_risk/100)

    # Get dataframe
    portfolio_df = portfolio_model.create_portfolio_overview(portfolio_weights)

    # Create a colored table
    table = dash_table.DataTable(
        data=portfolio_df.to_dict('records'),
        columns=[
            {'name': 'Symbol', 'id': 'symbol'},
            {'name': 'Full Name', 'id': 'longName'},
            {'name': 'Weight', 'id': 'weight', 'type': 'numeric', 'format': dash_table.FormatTemplate.percentage(2)},
            {'name': 'Category', 'id': 'category'},
            {'name': 'Expense Ratio', 'id': 'annualReportExpenseRatio', 'type': 'numeric'}
        #, 'id': 'Column1', 'type': 'numeric', 'format': dash_table.FormatTemplate.percentage(2)},
        ],
    )

    return table


if __name__ == '__main__':
    app.run_server(debug=True)

## Images taken from
# - <a href="https://www.flaticon.com/free-icons/invest" title="invest icons">Invest icons created by ultimatearm - Flaticon</a>