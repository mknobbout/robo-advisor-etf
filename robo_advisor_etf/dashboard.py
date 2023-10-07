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
    html.H1("Robo Advisor"),
    html.Div([
        html.Label("Select risk:"),
        dcc.Slider(
            id='risk-slider',
            min=1,
            max=100,
            step=1,
            marks=None,
            value=50
        ),
        html.Button('Construct Portfolio', id='show-table-button', n_clicks=0),
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