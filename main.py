import dash
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd


colors = {
    'background': '#A9A9A9',
    'view': '#D3D3D3',
    'text': '#23395D'
}

# Load the dataset and manage unknown values
df = pd.read_csv('dataset.csv', encoding = 'ISO-8859-1')
# nans = df.isna()
# cols = df.columns
# num_nan = len(df) - df.count()
# df = pd.DataFrame(df, columns = cols[num_nan < (len(df) * 0.5)])
# df = df.fillna(df.mean())

app = dash.Dash()
app.layout = html.Div(style = {'backgroundColor': colors['background']}, children = [
    dcc.Dropdown(
            id = 'cpu_model',
            options = [
                {'label': 'kvm', 'value': 'kvm'},
                {'label': 'atomic', 'value': 'atomic'},
                {'label': 'simple', 'value': 'simple'},
                {'label': 'o3', 'value': 'o3'},
            ],
            value = 'kvm',
            placeholder = 'Select a cpu model',
        ),
        dcc.Graph(id = 'boot_exit')
])

@app.callback(
    Output(component_id = 'boot_exit', component_property = 'figure'),
    [Input(component_id ='cpu_model', component_property = 'value')]
)
def update_v1_hist(cpu_model, range):

    mydf = df[(df['disk_name'] == 'boot-exit') & (df['param0'] == cpu_model)][['param1', 'host time', 'sim_insts']]
    mydf = mydf.groupby('param1').mean().reset_index()[['param1', 'host time', 'sim_insts']]
    plot = px.bar(histdf, x = 'param1', y = 'sim_insts')

    return plot