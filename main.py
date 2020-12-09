############### IMPORTED LIBARIES AND PACKAGES ###############

import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
import plotly.express  as px
import pandas as pd
import numpy as np
from dash.dependencies import Input, Output
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

import plotly.express as px

############### LOADING DATASET AND DATASET PRE-PROCESSING ###############

df1 = pd.read_csv('dataset3.csv', encoding='ISO-8859-1')

############## EXTRACTING DESIRED VARIABLES FOR THE REST OF THE PROGRAM ################

############## VISUALIZATION AND INTERACTION AND DATA ANALYSIS ##############

app = dash.Dash()
app.layout = html.Div([ 
  #####################################################################
  # --------------------- Div View 1 ---------------------
	html.Div([ 
        # ---------- Histograms
		html.Div([
            dcc.Input(
                id="V1_variable_selection_msg", type="text",
                value = 'Select a variable:',
                placeholder="",readOnly = True,
                style={'width': '49%','backgroundColor': '#f8f8f8', 'display': 'inline-block','border':'none','whiteSpace': 'pre-line',})
        ]),
        html.Div([
            dcc.Dropdown(
                id='V1_var_hist',
                options=[{'label': i, 'value': i} for i in df1.columns],
                value='experiment'
            )
        ], style={'width': '49%', 'display': 'inline-block'}),
        
        
        html.Div([
            dcc.Graph(id="V1_graph_histogram", style={'width': '40%', 'display': 'inline-block'})
            
        ]),
        # ---------- Div Stacked Bars
        html.Div([
            dcc.Dropdown(
                id="V1_Var1_Stacked_Bars",
                options=[{
                    'label': i,
                    'value': i
                } for i in df1.columns],
                value='result'),
            dcc.Dropdown(
                id="V1_Var2_Stacked_Bars",
                options=[{
                    'label': i,
                    'value': i
                } for i in df1.columns],
                value='result'),
            dcc.Dropdown(
                id="V1_Var3_Stacked_Bars",
                options=[{
                    'label': i,
                    'value': i
                } for i in df1.columns],
                value='result'),
            dcc.Dropdown(
                id="V1_Var4_Stacked_Bars",
                options=[{
                    'label': i,
                    'value': i
                } for i in df1.columns],
                value='result')
        ], style={'width': '25%','display': 'inline-block'}),
        
        dcc.Graph(id='V1_graph_stackedbar'),
        
        
    ])  
  
   # ---------- Div Divider
        
   #####################################################################
   # --------------------- Div View 2 ---------------------
    
  #####################################################################
   # --------------------- Div View 3 ---------------------
   
],style={'backgroundColor': '#f8f8f8'},)




#-------------------------------- View 1 Callbacks
### Histogram
@app.callback(
    Output(component_id = 'V1_graph_histogram', component_property = 'figure'),
    [Input(component_id = 'V1_var_hist',        component_property = 'value')])
def update_graph_hist(ColName):

    return {
        'data': [go.Histogram(
                    x = df1[ColName],
                    text = ColName)
            ],
        'layout': go.Layout(
            xaxis={
                'title': ColName,
            },
            yaxis={
                'title': 'Frequency',
            },
            margin={'l': 100, 'b': 30, 't': 10, 'r': 0},
            height=300,
            hovermode='closest'
        )
    }

### Stacked Bars
@app.callback(
    Output(component_id = 'V1_graph_stackedbar',  component_property = 'figure'),
    [Input(component_id = 'V1_Var1_Stacked_Bars', component_property = 'value'),
     Input(component_id = 'V1_Var2_Stacked_Bars', component_property = 'value'),
     Input(component_id = 'V1_Var3_Stacked_Bars', component_property = 'value'),
     Input(component_id = 'V1_Var4_Stacked_Bars', component_property = 'value')])
def update_graph_stackedbar(V1_Var1_Stbr, V1_Var2_Stbr, V1_Var3_Stbr, V1_Var4_Stbr):
    print(V1_Var4_Stbr)
    fig = px.bar(df1, x=V1_Var1_Stbr, y=V1_Var2_Stbr, color=V1_Var3_Stbr, hover_data=[V1_Var4_Stbr], barmode = 'group')
    return fig





if __name__ == '__main__':
    app.run_server()