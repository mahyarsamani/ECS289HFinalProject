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

############## VISUALIZATION AND INTERACTION AND DATA ANALYSIS ##############

app = dash.Dash()
app.layout = html.Div([ 
  
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

        ##### Div Status Board
    
        html.Div([
            dcc.Dropdown(
                id="V1_Var1_Tile",
                options=[{'label': i, 'value': i} for i in cat_cols],
                value= 'mem_sys'),
            
            html.Div([
                dcc.Dropdown(
                    id="V1_Var1Tile_lvls",
                    multi=True),
            ], style={'display': 'block'}),

            dcc.Dropdown(
                id="V1_Var2_Tile",
                options=[{'label': i, 'value': i} for i in cat_cols],
                value= 'cpu_model'),

            html.Div([
                dcc.Dropdown(
                    id="V1_Var2Tile_lvls",
                    multi=True),
            ], style={'display': 'block'}),

            dcc.Dropdown(
                id="V1_Var3_Tile",
                options=[{'label': i, 'value': i } for i in cat_cols],
                value= 'ncpus'),
            html.Div([
                dcc.Dropdown(
                    id="V1_Var3Tile_lvls",
                    multi=True),
            ], style={'display': 'block'}),

            dcc.Dropdown(
                id="V1_Var4_Tile",
                options=[{'label': i, 'value': i} for i in cat_cols],
                value= 'boot_type',
            ),

            html.Div([
                dcc.Dropdown(
                    id="V1_Var4Tile_lvls",
                    multi=True),
            ], style={'display': 'block'}),

            dcc.Dropdown(
                id="V1_Var5_Tile",
                options=[{'label': i,  'value': i } for i in cat_cols],
                value= 'result'),

            html.Div([
                dcc.Dropdown(
                    id="V1_Var5Tile_lvls",
                    multi=True),
            ], style={'display': 'block'}),

        ], style={'width': '25%','display': 'inline-block'}),
        
        dcc.Graph(id='V1_Graph_Tile'),     

        # ---------- Div Bars
        html.Div([
            dcc.Dropdown(
                id="V1_Var1_Bars",
                options=[{
                    'label': i,
                    'value': i
                } for i in cat_cols],
                value= 'cpu_model'),
            
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
                value= 'host time'),

            dcc.Dropdown(
                id="V1_Var3_Bars",
                options=[{
                    'label': i,
                    'value': i
                } for i in cat_cols],
                value= 'result'),
            html.Div([
                dcc.Dropdown(
                    id="V1_Var3_lvls",
                    multi=True),
            ], style={'display': 'block'}),

            dcc.Dropdown(
                id="V1_Var4_Bars",
                options=[{'label': i, 'value': i} for i in cat_cols],
                value= 'kernel',
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
                value= 'host time'),
            
            dcc.Dropdown(
                id="V1_Var2_Scat",
                options=[{
                    'label': i,
                    'value': i
                } for i in num_cols],
                value= 'host_inst_rate'),

            dcc.Dropdown(
                id="V1_Var3_Scat",
                options=[{
                    'label': i,
                    'value': i
                } for i in cat_cols],
                value= 'workload'),
            html.Div([
                dcc.Dropdown(
                    id="V1_Var3Scat_lvls",
                    multi=True),
            ], style={'display': 'block'}),

            dcc.Dropdown(
                id="V1_Var4_Scat",
                options=[{'label': i, 'value': i} for i in cat_cols],
                value= 'result',
            ),

            html.Div([
                dcc.Dropdown(
                    id="V1_Var4Scat_lvls",
                    multi=True),
            ], style={'display': 'block'}),

            dcc.Dropdown(
                id="V5_Var5_Scat",
                options=[{
                    'label': i,
                    'value': i
                } for i in num_cols],
                value= 'host_op_rate'),

            dcc.Checklist(
                id = 'V1_Logarithmic_Scat',
                options=[
                    {'label': 'X->Logarithmic', 'value': 'LogX'},
                    {'label': 'Y->Logarithmic', 'value': 'LogY'},
                ],
                value=['LogX']
) 

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
    return df1[V1_Var1].unique()[0:2]


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
    return df1[V1_Var3].unique()[0:2]


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
    return df1[V1_Var4].unique()[0:2]




#####################################################################################
#####################################################################################


##----SCATTER FIGURE ----------------------------------------------------------------------##
@app.callback(
   Output(component_id = 'V1_Graph_Scat',    component_property = 'figure'),
   [Input(component_id = 'V1_Var1_Scat',     component_property = 'value'),
    Input(component_id = 'V1_Var2_Scat',     component_property = 'value'),
    Input(component_id = 'V1_Var3_Scat',     component_property = 'value'),
    Input(component_id = 'V1_Var3Scat_lvls', component_property = 'value'),
    Input(component_id = 'V1_Var4_Scat',     component_property = 'value'),
    Input(component_id = 'V1_Var4Scat_lvls', component_property = 'value'),
    Input(component_id = 'V5_Var5_Scat', component_property = 'value'),
    Input(component_id = 'V1_Logarithmic_Scat', component_property = 'value')
    ])
def update_graph_scatter(mVar1, mVar2, var3, lvls3, facet1, fac1Lvls, mVar3,Log_Flag):
    tmpdf = pd.DataFrame(df1, columns = [mVar1, mVar2, var3, facet1,mVar3]) 
    tmpdf = tmpdf[tmpdf[var3].isin(lvls3)]
    tmpdf = tmpdf[tmpdf[facet1].isin(fac1Lvls)]
#    tmpdf = tmpdf.groupby(by = [var3, facet1],as_index=False).mean()
    Logx_Flag = False
    Logy_Flag = False
    if 'LogX' in Log_Flag:
        Logx_Flag = True
    if 'LogY' in Log_Flag:
        Logy_Flag = True

    fig = px.scatter(tmpdf, x=mVar1, y=mVar2, size=mVar3, color=var3, facet_col=facet1,log_x=Logx_Flag,log_y=Logy_Flag)    
    return fig
  

##----Scat VAR3 LEVELS ----------------------------------------------------------------------##

@app.callback(
    Output(component_id = 'V1_Var3Scat_lvls', component_property = 'options'),
    [Input(component_id = 'V1_Var3_Scat',     component_property = 'value'),
     ])
def update_V1_Var3Scat_lvls_opt(V1_Var3Scat):
    opts=[{'label':lvl, 'value':lvl} for lvl in df1[V1_Var3Scat].unique()]
    return opts

@app.callback(
    Output(component_id = 'V1_Var3Scat_lvls', component_property = 'value'),
    [Input(component_id = 'V1_Var3_Scat',     component_property = 'value'),
     ])
def update_V1_Var3Scat_val(V1_Var3Scat):
    return df1[V1_Var3Scat].unique()[0:2]


##----Scat VAR4 LEVELS ----------------------------------------------------------------------##

@app.callback(
    Output(component_id = 'V1_Var4Scat_lvls', component_property = 'options'),
    [Input(component_id = 'V1_Var4_Scat',     component_property = 'value'),
     ])
def update_Var4Scat_lvls_opt(V1_Var4Scat):
    opts=[{'label':lvl, 'value':lvl} for lvl in df1[V1_Var4Scat].unique()]
    return opts

@app.callback(
    Output(component_id = 'V1_Var4Scat_lvls', component_property = 'value'),
    [Input(component_id = 'V1_Var4_Scat',     component_property = 'value'),
     ])
def update_Var4Scat_lvls_val(V1_Var4Scat):
    return df1[V1_Var4Scat].unique()[0:2]

#####################################################################################
#####################################################################################

##----TILE FIGURE ----------------------------------------------------------------------##
@app.callback(
   Output(component_id = 'V1_Graph_Tile',    component_property = 'figure'),
   [Input(component_id = 'V1_Var1_Tile',     component_property = 'value'),
    Input(component_id = 'V1_Var1Tile_lvls', component_property = 'value'),
    Input(component_id = 'V1_Var2_Tile',     component_property = 'value'),
    Input(component_id = 'V1_Var2Tile_lvls', component_property = 'value'),
    Input(component_id = 'V1_Var3_Tile',     component_property = 'value'),
    Input(component_id = 'V1_Var3Tile_lvls', component_property = 'value'),
    Input(component_id = 'V1_Var4_Tile',     component_property = 'value'),
    Input(component_id = 'V1_Var4Tile_lvls', component_property = 'value'),
    Input(component_id = 'V1_Var5_Tile',     component_property = 'value'),
    Input(component_id = 'V1_Var5Tile_lvls', component_property = 'value'),
    ])
def update_graph_tile(var1, lvl1, var2, lvl2, var3, lvl3, var4, lvl4, var5, lvl5):
    tmpdf = pd.DataFrame(df1, columns = [var1, var2, var3, var4, var5, 'sim_freq'])
    if(~tmpdf[tmpdf[var1].isin(lvl1)].empty):
        tmpdf = tmpdf[tmpdf[var1].isin(lvl1)]
    if(~tmpdf[tmpdf[var2].isin(lvl2)].empty):
        tmpdf = tmpdf[tmpdf[var2].isin(lvl2)]
    if(~tmpdf[tmpdf[var3].isin(lvl3)].empty):
        tmpdf = tmpdf[tmpdf[var3].isin(lvl3)]
    if(~tmpdf[tmpdf[var4].isin(lvl4)].empty):
        tmpdf = tmpdf[tmpdf[var4].isin(lvl4)]
    if(~tmpdf[tmpdf[var5].isin(lvl5)].empty):
        tmpdf = tmpdf[tmpdf[var5].isin(lvl5)]
    #tmpdf = df1.groupby(by = [var1, var2, var3, var4, var5],as_index=False).mean()
    fig = px.treemap(tmpdf, path=[var1, var2, var3, var4, var5], values='sim_freq')
    return fig
 

##----TILE VAR1 LEVELS ----------------------------------------------------------------------##
@app.callback(
   Output(component_id = 'V1_Var1Tile_lvls', component_property = 'options'),
   [Input(component_id = 'V1_Var1_Tile',     component_property = 'value'),
   ])
def update_V1Tile_lvl_opt(V1_Var1Tile):
   opts=[{'label':lvl, 'value':lvl} for lvl in df1[V1_Var1Tile].unique()]
   return opts

@app.callback(
   Output(component_id = 'V1_Var1Tile_lvls', component_property = 'value'),
   [Input(component_id = 'V1_Var1_Tile',     component_property = 'value'),
    ])
def update_V1Tile_lvl_val(V1_Var1Tile):
   return df1[V1_Var1Tile].unique()[0:2]

##----TILE VAR2 LEVELS ----------------------------------------------------------------------##
@app.callback(
   Output(component_id = 'V1_Var2Tile_lvls', component_property = 'options'),
   [Input(component_id = 'V1_Var2_Tile',     component_property = 'value'),
   ])
def update_V2Tile_lvl_opt(V1_Var2Tile):
   opts=[{'label':lvl, 'value':lvl} for lvl in df1[V1_Var2Tile].unique()]
   return opts

@app.callback(
   Output(component_id = 'V1_Var2Tile_lvls', component_property = 'value'),
   [Input(component_id = 'V1_Var2_Tile',     component_property = 'value'),
    ])
def update_V2Tile_lvl_val(V1_Var2Tile):
   return df1[V1_Var2Tile].unique()[0:2]



##----TILE VAR3 LEVELS ----------------------------------------------------------------------##
@app.callback(
   Output(component_id = 'V1_Var3Tile_lvls', component_property = 'options'),
   [Input(component_id = 'V1_Var3_Tile',     component_property = 'value'),
    ])
def update_V1Tile_lvl3_opt(V1_Var3Tile):
   opts=[{'label':lvl, 'value':lvl} for lvl in df1[V1_Var3Tile].unique()]
   return opts

@app.callback(
   Output(component_id = 'V1_Var3Tile_lvls', component_property = 'value'),
   [Input(component_id = 'V1_Var3_Tile',     component_property = 'value'),
    ])
def update_V1Tile_lvl3_val(V1_Var3Tile):
  return df1[V1_Var3Tile].unique()[0:2]
#----TILE VAR4 LEVELS ----------------------------------------------------------------------##
@app.callback(
   Output(component_id = 'V1_Var4Tile_lvls', component_property = 'options'),
   [Input(component_id = 'V1_Var4_Tile',     component_property = 'value'),
    ])
def update_V1Tile_lvl4_opt(V1_Var4Tile):
   opts=[{'label':lvl, 'value':lvl} for lvl in df1[V1_Var4Tile].unique()]
   return opts

@app.callback(
   Output(component_id = 'V1_Var4Tile_lvls', component_property = 'value'),
   [Input(component_id = 'V1_Var4_Tile',     component_property = 'value'),
    ])
def update_V1Tile_lvl4_val(V1_Var4Tile):
   return df1[V1_Var4Tile].unique()[0:2]

#----TILE VAR5 LEVELS ----------------------------------------------------------------------##
@app.callback(
   Output(component_id = 'V1_Var5Tile_lvls', component_property = 'options'),
   [Input(component_id = 'V1_Var5_Tile',     component_property = 'value'),
    ])
def update_V1Tile_lvl5_opt(V1_Var5Tile):
   opts=[{'label':lvl, 'value':lvl} for lvl in df1[V1_Var5Tile].unique()]
   return opts

@app.callback(
   Output(component_id = 'V1_Var5Tile_lvls', component_property = 'value'),
   [Input(component_id = 'V1_Var5_Tile',     component_property = 'value'),
    ])
def update_V1Tile_lvl5_val(V1_Var5Tile):
   return df1[V1_Var5Tile].unique()[0:2]





if __name__ == '__main__':
  app.run_server()