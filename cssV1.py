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

df1 = pd.read_csv('datasetNew.csv', encoding='ISO-8859-1')

############## EXTRACTING DESIRED VARIABLES FOR THE REST OF THE PROGRAM ################

df1['Num. of CPUs'] = df1['Num. of CPUs'].astype(object)
df1cols = df1.columns
num_cols = df1.select_dtypes(include=[np.number]).columns
cat_cols = df1.select_dtypes(exclude=[np.number]).columns

print(cat_cols)
print(num_cols)
############## VISUALIZATION AND INTERACTION AND DATA ANALYSIS ##############

app = dash.Dash()
app.layout = html.Div(style = {'background-color' : '#f8f8f8'}, children = [ 

    html.Div(style = {'width' : '100%', 'height' : '10%', 'display' : 'block'}, children = [
        dcc.Markdown('DArchR/gem5 logo here', style = {'text-align' : 'center'})
    ]), 
  # --------------------- Div View 1 ---------------------
	html.Div(style = {'margin-bottom' : '5%'}, children = [ 
        # ---------- Histograms
		html.Div(style = {'width' : '10%', 'height': '100%', 'display' : 'inline-block'}, children = [  
            html.Div([
                dcc.Input(
                    id="V1_variable_selection_msg", type="text",
                    value = 'Select a variable:',
                    placeholder="",readOnly = True,
                    style={'width': '95%','backgroundColor': '#f8f8f8', 'display': 'block','border':'none'})
            ]),
            html.Div([
                dcc.Dropdown(
                    id='V1_var_hist',
                    options=[{'label': i, 'value': i} for i in df1.columns],
                    value='Kernel Version'
                )
            ], style={'width': '95%', 'display': 'block'}),
        ]),
        dcc.Graph(id="V1_graph_histogram", style={'width': '40%', 'height' : '100%', 'display': 'inline-block'})
    ]),

    html.Div(style = {'margin-bottom' : '5%'}, children = [
        ##### Div Categorical Tree
    
        html.Div(style = {'width' : '10%', 'height': '100%', 'display' : 'inline-block'}, children = [
            dcc.Dropdown(
                id="V1_Var1_Tile",
                options=[{'label': i, 'value': i} for i in cat_cols],
                value= 'Memory System'),
            
            html.Div([
                dcc.Dropdown(
                    id="V1_Var1Tile_lvls",
                    multi=True),
            ], style={'display': 'block'}),

            dcc.Dropdown(
                id="V1_Var2_Tile",
                options=[{'label': i, 'value': i} for i in cat_cols],
                value= 'CPU Model'),

            html.Div([
                dcc.Dropdown(
                    id="V1_Var2Tile_lvls",
                    multi=True),
            ], style={'display': 'block'}),

            dcc.Dropdown(
                id="V1_Var3_Tile",
                options=[{'label': i, 'value': i } for i in cat_cols],
                value= 'Num. of CPUs'),
            html.Div([
                dcc.Dropdown(
                    id="V1_Var3Tile_lvls",
                    multi=True),
            ], style={'display': 'block'}),

            dcc.Dropdown(
                id="V1_Var4_Tile",
                options=[{'label': i, 'value': i} for i in cat_cols],
                value= 'Boot Type',
            ),

            html.Div([
                dcc.Dropdown(
                    id="V1_Var4Tile_lvls",
                    multi=True),
            ], style={'display': 'block'}),

            dcc.Dropdown(
                id="V1_Var5_Tile",
                options=[{'label': i,  'value': i } for i in cat_cols],
                value= 'Sim. Status Result'),

            html.Div([
                dcc.Dropdown(
                    id="V1_Var5Tile_lvls",
                    multi=True),
            ], style={'display': 'block'}),

        ]),
        
        dcc.Graph(id='V1_Graph_Tile', style = {'width' : '80%', 'height': '100%', 'display' : 'inline-block'})
    ]),

    html.Div(style = {'margin-bottom' : '5%'}, children = [  

        # ---------- Div Bars
        html.Div(style = {'width' : '10%', 'height': '100%', 'display' : 'inline-block'}, children = [
            dcc.Dropdown(
                id="V1_Var1_Bars",
                options=[{'label': i, 'value': i} for i in cat_cols],
                value= 'CPU Model'),
            
            html.Div([
                dcc.Dropdown(
                    id="V1_Var1_lvls",
                    multi=True),
            ], style={'display': 'block'}),

            dcc.Dropdown(
                id="V1_Var2_Bars",
                options=[{'label': i,'value': i} for i in num_cols],
                value= 'Host Time'),

            dcc.Dropdown(
                id="V1_Var3_Bars",
                options=[{
                    'label': i, 'value': i} for i in cat_cols],
                value= 'Sim. Status Result'),
            html.Div([
                dcc.Dropdown(
                    id="V1_Var3_lvls",
                    multi=True),
            ], style={'display': 'block'}),

            dcc.Dropdown(
                id="V1_Var4_Bars",
                options=[{'label': i, 'value': i} for i in cat_cols],
                value= 'Kernel Version',
            ),

            html.Div([
                dcc.Dropdown(
                    id="V1_Var4_lvls",
                    multi=True),
            ], style={'display': 'block'}),

        ]),
        
        dcc.Graph(id='V1_graph_bar', style = {'width' : '80%', 'height': '100%', 'display' : 'inline-block'})
    ]),

    html.Div(style = {'margin-bottom' : '5%'}, children = [
        
        # ---------- Div Scatter
        html.Div(style = {'width' : '10%', 'height': '100%', 'display' : 'inline-block'}, children = [
            dcc.Dropdown(
                id="V1_Var1_Scat",
                options=[{'label': i, 'value': i} for i in num_cols],
                value= 'Host Time'),
            
            dcc.Dropdown(
                id="V1_Var2_Scat",
                options=[{'label': i, 'value': i} for i in num_cols],
                value= 'Host Instruction Rate'),

            dcc.Dropdown(
                id="V1_Var3_Scat",
                options=[{'label': i,'value': i} for i in cat_cols],
                value= 'Experiment Name'),
            html.Div([
                dcc.Dropdown(
                    id="V1_Var3Scat_lvls",
                    multi=True),
            ], style={'display': 'block'}),

            dcc.Dropdown(
                id="V1_Var4_Scat",
                options=[{'label': i, 'value': i} for i in cat_cols],
                value= 'Sim. Status Result',
            ),

            html.Div([
                dcc.Dropdown(
                    id="V1_Var4Scat_lvls",
                    multi=True),
            ], style={'display': 'block'}),

            dcc.Dropdown(
                id="V5_Var5_Scat",
                options=[{'label': i, 'value': i} for i in num_cols],
                value= 'Host Opcode Rate'),

            dcc.Checklist(
                id = 'V1_Logarithmic_Scat',
                options=[
                    {'label': 'X->Logarithmic', 'value': 'LogX'},
                    {'label': 'Y->Logarithmic', 'value': 'LogY'},
                ],
                value=['LogX']
            ) 

        ]),       
        dcc.Graph(id='V1_Graph_Scat', style = {'width' : '80%', 'height': '100%', 'display' : 'inline-block'})
    
    ]),
    
    # Here

    html.Div(style = {'margin-bottom' : '5%'}, children = [
		# ---------- Div General settings
		html.Div(style = {'width' : '20%', 'height': '100%', 'display' : 'inline-block'}, children = [
            
            # html.Div([
            # dcc.Textarea(id="V3_Divider",
            #     value = ['\n-----------------\n---> Views 3 <---\n-----------------\n'],
            #     readOnly = True,
            #     style={'width': '99%','backgroundColor': '#f8f8f8', 'justify': 'center', 'display': 'flex','resize': 'none','border':'none', 'height': 150, 'color': 'green', 'font-size':'25px'}),
            # ]),
            
            html.Div([
            dcc.Textarea(id="V3_txt1",
                value = ['Select Desired Group(s):'],
                readOnly = True,
                style={'width': '99%','backgroundColor': '#f8f8f8', 'justify': 'center', 'display': 'flex','resize': 'none','border':'none'}),
            ]),
            
            html.Div([
                dcc.Dropdown(
                    id='V3_groupname',
                    options=[{'label': i, 'value': i} for i in cat_cols],
                    value=cat_cols[0]
                )
            ], style={'width': '100%', 'float': 'left', 'display': 'inline-block'}),
            
            html.Div([
            dcc.Textarea(id="V3_txt2",
                value = ['Select Desired Numerical Feature(s):'],
                readOnly = True,
                style={'width': '99%','backgroundColor': '#f8f8f8', 'justify': 'center', 'display': 'flex','resize': 'none','border':'none'}),
            ]),
            
            html.Div([
                dcc.Dropdown(
                    id='V3_featconti',
                    options=[{'label': i, 'value': i} for i in num_cols],
                    value=num_cols[0:5],
                    multi=True,
                )
            ], style={'width': '100%', 'float': 'left', 'display': 'inline-block'}),
            
            html.Div([
            dcc.Textarea(id="V3_txt3",
                value = ['Select Desired Categorical Feature(s):'],
                readOnly = True,
                style={'width': '99%','backgroundColor': '#f8f8f8', 'justify': 'center', 'display': 'flex','resize': 'none','border':'none'}),
            ]),
            
            html.Div([
                dcc.Textarea(id="V3_txtyerrng",
                    value = ['Select Desired year range:'],
                    readOnly = True,
                    style={'width': '99%','backgroundColor': '#f8f8f8', 'justify': 'center', 'display': 'flex','resize': 'none','border':'none'}),
            ]),
            
            
            html.Div([
                dcc.Checklist(
                    id='V3_Preproc_Flags',
                    options=[{'label': 'Normalize', 'value': 'norm_flag'},
                             {'label': 'Enable PCA', 'value': 'pca_flag'}],
                    value=['norm_flag','pca_flag']
                )  
            ]),
            
            html.Div([
                dcc.Input(
                    id='V3_pcacompo', type='text', 
                    value = ['Select the number of components between '+ str(2) + ' and ' + str(10)+':'],
                    placeholder="",readOnly = False,
                    style={'width': '99%','backgroundColor': '#f8f8f8', 'float': 'left', 'display': 'block','border':'none','whiteSpace': 'pre-line','height': 30}
                )
            ]),
            html.Div([
                dcc.Input(
                    id="V3_pcaCompNum_txt", type="number", placeholder="", value=5,
                    min=2, max=10, step=1,
                )
            ]),
            
            html.Div([
                dcc.Textarea(
                    id="V3_datapoints",
                    value = ['Data Points:'],
                    readOnly = True,
                    style={'width': '99%', 'float': 'Center', 'display': 'inline-block','whiteSpace': 'pre-line','resize': 'none','height': 150}
                ),
            ]),
            
            html.Div([
                dcc.Textarea(
                    id="V3_txt4",
                    value = ['Select percentage of datapoints used for testing and evaluation in the slider below:'],
                    readOnly = True,
                    style={'width': '100%', 'justify': 'center', 'display': 'flex','resize': 'none','border':'none','backgroundColor': '#f8f8f8'}),
            ]),
            
            html.Div(
                dcc.Slider(
                    id='V3_testproport',
                    min = 10,
                    max = 50,
                    step=1,
                    value=30,
                    marks={str(i): str(i) for i in range(10,50)}
                ), style={'width': '99%', 'float': 'right', 'padding': '0px 20px 20px 20px','whiteSpace': 'pre-line'}
            ),
            
            
            # html.Div(
            #     dcc.Input(
            #         id='V3_accu', type='text', 
            #         value = ['Confusion Table:'],
            #         placeholder="",readOnly = True,
            #         style={'width': '99%','backgroundColor': '#f8f8f8', 'float': 'left', 'display': 'block','border':'none','whiteSpace': 'pre-line','height': 30})
            #     )
			
		]),

        # ---------- graph PCA
        html.Div([
                dcc.Graph(
                    id='V3_conftable_fig'
                )
            ], style={'width': '75%', 'display': 'inline-block', 'padding': '0 20'}),
		# ---------- graph num estimate
	]),

    # ---------- num estimate
    html.Div(style = {'margin-bottom' : '5%'}, children = [
		# ---------- Div General settings
		html.Div(style = {'width' : '20%', 'height': '100%', 'display' : 'inline-block'}, children = [

            html.Div([
                dcc.Textarea(id="V4_txt1",
                    value = ['Select Desired Measurment:'],
                    readOnly = True,
                    style={'width': '99%','backgroundColor': '#f8f8f8', 'justify': 'center', 'display': 'flex','resize': 'none','border':'none'}),
            ]),
            
            html.Div([
                dcc.Dropdown(
                    id='V4_mVar',
                    options=[{'label': i, 'value': i} for i in num_cols],
                    value=num_cols[0]
                )
            ], style={'width': '100%', 'float': 'left', 'display': 'inline-block'}),
            
            html.Div([
                dcc.Textarea(id="V4_txt2",
                    value = ['Select Desired Feature(s):'],
                    readOnly = True,
                    style={'width': '99%','backgroundColor': '#f8f8f8', 'justify': 'center', 'display': 'flex','resize': 'none','border':'none'}),
            ]),
            
            html.Div([
                dcc.Dropdown(
                    id='V4_feat',
                    options=[{'label': i, 'value': i} for i in df1.columns],
                    value=df1.columns[0:5],
                    multi=True,
                )
            ], style={'width': '100%', 'float': 'left', 'display': 'inline-block'}),               		
		]),

        # ---------- graph hist
        html.Div([
                dcc.Textarea(
                    id="V4_estimate",
                    value = ['Estimated Value:'],
                    readOnly = True,
                    style={'width': '99%', 'float': 'Center', 'display': 'inline-block','whiteSpace': 'pre-line','resize': 'none','height': 150}
                ),
            ]),

        html.Div([
                dcc.Graph(
                    id='V4_hist_fig'
                )
            ], style={'width': '75%', 'display': 'inline-block', 'padding': '0 20'}),
		# ---------- graph num estimate
	]),

])




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
    tmpdf = pd.DataFrame(df1, columns = [var1, var2, var3, var4, var5, 'Simulation Frequency'])
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
    fig = px.treemap(tmpdf, path=[var1, var2, var3, var4, var5], values='Simulation Frequency')
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

#####################################################################################
#####################################################################################

##----PCA ----------------------------------------------------------------------##

@app.callback(
    dash.dependencies.Output(component_id = 'V3_datapoints',     component_property = 'value'),
    dash.dependencies.Output(component_id = 'V3_conftable_fig',  component_property = 'figure'),
    [dash.dependencies.Input(component_id = 'V3_groupname',      component_property = 'value'),
     dash.dependencies.Input(component_id = 'V3_featconti',      component_property = 'value'),
     dash.dependencies.Input(component_id = 'V3_Preproc_Flags',  component_property = 'value'),
     dash.dependencies.Input(component_id = 'V3_pcaCompNum_txt', component_property = 'value'),
     dash.dependencies.Input(component_id = 'V3_testproport',    component_property = 'value')
      ])
def update_graph(grname, contfeat, preprocflag, Component_num, testproport):
    
    #----------------------- Select groups and year ranges
    Catlabel = grname
    ML_Cols = np.concatenate([[grname],contfeat])

    grname = df1[Catlabel].unique() 
    # df_ML = df1[(df1['iyear']>year_range[0]) & (df1['iyear']<year_range[1])]
    # df_ML = df_ML[df_ML['gname'].isin(grname)]
    df_ML = df1[ML_Cols]
    df_ML = df_ML.fillna(df_ML.mean())
    
    print(df_ML)

    df_Cont = pd.DataFrame(df_ML, columns = contfeat)
    
    labels = df_ML[Catlabel]
    
    
    if 'norm_flag' in preprocflag:
     	# ------------ Normalize
     	sc = StandardScaler()
     	df_Cont = sc.fit_transform(df_Cont)
    
    if 'pca_flag' in preprocflag:
     	# ------------ PCA
     	pca = PCA(n_components = Component_num)
     	df_Cont = pca.fit_transform(df_Cont)
     	explained_variance = pca.explained_variance_ratio_

    
    feats = df_Cont
    

    X_train, X_test, y_train, y_test = train_test_split(feats, labels, test_size=testproport/100, random_state=0)

    datapoints = df_ML.groupby([Catlabel]).size().reset_index(name="idx N")
    datapoints = pd.DataFrame(datapoints, columns = ['idx N',Catlabel])

    
    datapoints_str = ['Number of datapoints: \n'
                   + str(datapoints)]
    
    
    # ------------ Random Forest classifier 
    classifier = RandomForestClassifier(max_depth=2, random_state=0)
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)
    
    cm = confusion_matrix(y_test, y_pred)

    import plotly.figure_factory as ff

    # print(cm)
    # print(grname[len(grname)::-1])
    fig = ff.create_annotated_heatmap(cm, 
                                      font_colors=['black'], hoverinfo='text',
                                      colorscale='Viridis')
    # fig = ff.create_annotated_heatmap(cm[len(grname)::-1].T, 
    #                                   x = grname,
    #                                   y = grname[len(grname)::-1],
    #                                   font_colors=['black'], hoverinfo='text',
    #                                   colorscale='Viridis')
    fig.update_layout(title_text= 'Accuracy = '+ str(accuracy_score(y_test, y_pred)))
    
    
    return (datapoints_str
            , fig
    )

#####################################################################################
#####################################################################################

##----mVar Estimate ----------------------------------------------------------------------##
@app.callback(
    dash.dependencies.Output(component_id = 'V4_estimate', component_property = 'value'),
    dash.dependencies.Output(component_id = 'V4_hist_fig', component_property = 'figure'),
    [dash.dependencies.Input(component_id = 'V4_mVar',     component_property = 'value'),
     dash.dependencies.Input(component_id = 'V4_feat',     component_property = 'value'),
      ])
def update_graph(mVar, feat):
    tmpdf = df1
    for col in cat_cols:
        unq = tmpdf[col].unique()
        i=0
        



if __name__ == '__main__':
  app.run_server()