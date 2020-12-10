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


colors = {
    'background': '#A9A9A9',
    'view': '#D3D3D3',
    'text': '#23395D'
}

statColor = {
    ''
}

############### LOADING DATASET AND DATASET PRE-PROCESSING ###############

df1 = pd.read_csv('dataset3.csv', encoding='ISO-8859-1')

############## EXTRACTING DESIRED VARIABLES FOR THE REST OF THE PROGRAM ################
df1['ncpus'] = df1['ncpus'].astype(object)
df1cols = df1.columns
num_cols = df1.select_dtypes(include=[np.number]).columns
cat_cols = df1.select_dtypes(exclude=[np.number]).columns
print(cat_cols)
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
        # ---------- Div Bars
        html.Div([
            dcc.Dropdown(
                id="V1_Var1_Bars",
                options=[{
                    'label': i,
                    'value': i
                } for i in cat_cols],
                value= cat_cols[0]),
            
            html.Div([
                dcc.Dropdown(
                    id="V1_Var1_lvls",
                    multi=True),
            ], style={'display': 'block'}),

            dcc.Dropdown(
                id="V1_Var2_Bars",
                options=[{
                    'label': i,
                    'value': i
                } for i in num_cols],
                value= num_cols[1]),

            dcc.Dropdown(
                id="V1_Var3_Bars",
                options=[{
                    'label': i,
                    'value': i
                } for i in cat_cols],
                value= cat_cols[1]),
            html.Div([
                dcc.Dropdown(
                    id="V1_Var3_lvls",
                    multi=True),
            ], style={'display': 'block'}),

            dcc.Dropdown(
                id="V1_Var4_Bars",
                options=[{'label': i, 'value': i} for i in cat_cols],
                value= cat_cols[2],
            ),

            html.Div([
                dcc.Dropdown(
                    id="V1_Var4_lvls",
                    multi=True),
            ], style={'display': 'block'}),

        ], style={'width': '25%','display': 'inline-block'}),
        
        dcc.Graph(id='V1_graph_bar'),
    
        
        # ---------- Div Scatter
        html.Div([
            dcc.Dropdown(
                id="V1_Var1_Scat",
                options=[{
                    'label': i,
                    'value': i
                } for i in num_cols],
                value= num_cols[2]),
            
            dcc.Dropdown(
                id="V1_Var2_Scat",
                options=[{
                    'label': i,
                    'value': i
                } for i in num_cols],
                value= num_cols[1]),

            dcc.Dropdown(
                id="V1_Var3_Scat",
                options=[{
                    'label': i,
                    'value': i
                } for i in cat_cols],
                value= cat_cols[1]),
            html.Div([
                dcc.Dropdown(
                    id="V1_Var3Scat_lvls",
                    multi=True),
            ], style={'display': 'block'}),

            dcc.Dropdown(
                id="V1_Var4_Scat",
                options=[{'label': i, 'value': i} for i in cat_cols],
                value= cat_cols[2],
            ),

            html.Div([
                dcc.Dropdown(
                    id="V1_Var4Scat_lvls",
                    multi=True),
            ], style={'display': 'block'}),

        ], style={'width': '25%','display': 'inline-block'}),
        
        dcc.Graph(id='V1_Graph_Scat'),
        
    ]),
        
   #####################################################################
   # --------------------- Div View 2 ---------------------
    
  #####################################################################
   # --------------------- Div View 3 ---------------------
   
],style={'backgroundColor': '#f8f8f8'},)




########################################### View 1 Callbacks ###########################################


##----HISTOGRAM FIGURE ----------------------------------------------------------------------##
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


##----BAR FIGURE ----------------------------------------------------------------------##
@app.callback(
    Output(component_id = 'V1_graph_bar', component_property = 'figure'),
    [Input(component_id = 'V1_Var1_Bars', component_property = 'value'),
     Input(component_id = 'V1_Var1_lvls', component_property = 'value'),
     Input(component_id = 'V1_Var2_Bars', component_property = 'value'),
     Input(component_id = 'V1_Var3_Bars', component_property = 'value'),
     Input(component_id = 'V1_Var3_lvls', component_property = 'value'),
     Input(component_id = 'V1_Var4_Bars', component_property = 'value'),
     Input(component_id = 'V1_Var4_lvls', component_property = 'value'),
     #Input(component_id = 'V1_Var5_Bars', component_property = 'value'),
     ])
def update_graph_bar(var1, lvl1, mVar, var2, lvl2, facet1, fac1Lvls):
    tmpdf = pd.DataFrame(df1, columns = [mVar, var1, var2, facet1]) 
    tmpdf = tmpdf[tmpdf[var1].isin(lvl1)]
    tmpdf = tmpdf[tmpdf[var2].isin(lvl2)]
    tmpdf = tmpdf[tmpdf[facet1].isin(fac1Lvls)]
    tmpdf = tmpdf.groupby(by = [var1, var2, facet1],as_index=False).mean()
    fig = px.bar(tmpdf, x=var1, y=mVar, color=var2, facet_col=facet1, barmode = 'group')    
    return fig
  

##----BAR VAR1 LEVELS ----------------------------------------------------------------------##
@app.callback(
    Output(component_id = 'V1_Var1_lvls',  component_property = 'options'),
    [Input(component_id = 'V1_Var1_Bars',  component_property = 'value'),
    ])
def update_V1_lvl1_opt(V1_Var1):
    opts=[{'label':lvl, 'value':lvl} for lvl in df1[V1_Var1].unique()]
    return opts

@app.callback(
    Output(component_id = 'V1_Var1_lvls',  component_property = 'value'),
    [Input(component_id = 'V1_Var1_Bars',  component_property = 'value'),
     ])
def update_V1_lvl1_val(V1_Var1):
    return df1[V1_Var1].unique()


##----BAR VAR3 LEVELS ----------------------------------------------------------------------##

@app.callback(
    Output(component_id = 'V1_Var3_lvls',  component_property = 'options'),
    [Input(component_id = 'V1_Var3_Bars',  component_property = 'value'),
     ])
def update_V1_lvl3_opt(V1_Var3):
    opts=[{'label':lvl, 'value':lvl} for lvl in df1[V1_Var3].unique()]
    return opts

@app.callback(
    Output(component_id = 'V1_Var3_lvls',  component_property = 'value'),
    [Input(component_id = 'V1_Var3_Bars',  component_property = 'value'),
     ])
def update_V1_lvl3_val(V1_Var3):
    return df1[V1_Var3].unique()


##----BAR VAR4 LEVELS ----------------------------------------------------------------------##

@app.callback(
    Output(component_id = 'V1_Var4_lvls',  component_property = 'options'),
    [Input(component_id = 'V1_Var4_Bars',  component_property = 'value'),
     ])
def update_V1_lvl4_opt(V1_Var4):
    opts=[{'label':lvl, 'value':lvl} for lvl in df1[V1_Var4].unique()]
    return opts

@app.callback(
    Output(component_id = 'V1_Var4_lvls',  component_property = 'value'),
    [Input(component_id = 'V1_Var4_Bars',  component_property = 'value'),
     ])
def update_V1_lvl4_val(V1_Var4):
    return df1[V1_Var4].unique()




#####################################################################################
#####################################################################################


##----BAR FIGURE ----------------------------------------------------------------------##
@app.callback(
   Output(component_id = 'V1_Graph_Scat', component_property = 'figure'),
   [Input(component_id = 'V1_Var1_Scat', component_property = 'value'),
    Input(component_id = 'V1_Var2_Scat', component_property = 'value'),
    Input(component_id = 'V1_Var3_Scat', component_property = 'value'),
    Input(component_id = 'V1_Var3Scat_lvls', component_property = 'value'),
    Input(component_id = 'V1_Var4_Scat', component_property = 'value'),
    Input(component_id = 'V1_Var4Scat_lvls', component_property = 'value'),
    ])
def update_graph_scatter(mVar1, mVar2, var3, lvls3, facet1, fac1Lvls):
   tmpdf = pd.DataFrame(df1, columns = [mVar1, mVar2, var3, facet1]) 
   tmpdf = tmpdf[tmpdf[var3].isin(lvls3)]
   tmpdf = tmpdf[tmpdf[facet1].isin(fac1Lvls)]
   tmpdf = tmpdf.groupby(by = [var3, facet1],as_index=False).mean()
   fig = px.scatter(tmpdf, x=mVar1, y=mVar2, color=var3, facet_col=facet1)    
   return fig
  

##----Scat VAR3 LEVELS ----------------------------------------------------------------------##

@app.callback(
    Output(component_id = 'V1_Var3Scat_lvls',  component_property = 'options'),
    [Input(component_id = 'V1_Var3_Scat',  component_property = 'value'),
     ])
def update_V1_Var3Scat_lvls_opt(V1_Var3Scat):
    opts=[{'label':lvl, 'value':lvl} for lvl in df1[V1_Var3Scat].unique()]
    return opts

@app.callback(
    Output(component_id = 'V1_Var3Scat_lvls',  component_property = 'value'),
    [Input(component_id = 'V1_Var3_Scat',  component_property = 'value'),
     ])
def update_V1_Var3Scat_val(V1_Var3Scat):
    return df1[V1_Var3Scat].unique()


##----BAR VAR4 LEVELS ----------------------------------------------------------------------##

@app.callback(
    Output(component_id = 'V1_Var4Scat_lvls', component_property = 'options'),
    [Input(component_id = 'V1_Var4_Scat',     component_property = 'value'),
     ])
def update_Var4Scat_lvls_opt(V1_Var4Scat):
    opts=[{'label':lvl, 'value':lvl} for lvl in df1[V1_Var4Scat].unique()]
    return opts

@app.callback(
    Output(component_id = 'V1_Var4Scat_lvls',  component_property = 'value'),
    [Input(component_id = 'V1_Var4_Scat',  component_property = 'value'),
     ])
def update_Var4Scat_lvls_val(V1_Var4Scat):
    return df1[V1_Var4Scat].unique()


if __name__ == '__main__':
    app.run_server()