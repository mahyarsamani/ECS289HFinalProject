######################### IMPORTED LIBARIES AND PACKAGES #################################

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
import base64

############################### STYLE VARS ##############################################

colors = {
    'background': '#A9A9A9',
    'view': '#D3D3D3',
    'text': '#23395D'
}

statColor = {
    ''
}

###################### LOADING DATASET AND DATASET PRE-PROCESSING ######################

df1 = pd.read_csv('dataset.csv', encoding='ISO-8859-1')
df1 = df1.fillna(df1.mean())

############## EXTRACTING DESIRED VARIABLES FOR THE REST OF THE PROGRAM ################

df1['Num. of CPUs'] = df1['Num. of CPUs'].astype(object)
df1cols = df1.columns
num_cols = df1.select_dtypes(include=[np.number]).columns
cat_cols = df1.select_dtypes(exclude=[np.number]).columns
desc_cols = ['Experiment Name', 'Kernel Version', 'CPU Model', 'Num. of CPUs', 'Memory System', 'Boot Type', 'Workload', 'Input Size']

################## VISUALIZATION AND INTERACTION AND DATA ANALYSIS ####################

app = dash.Dash()
app.layout = html.Div(style = {'padding-right' : '5%', 'padding-left' : '5%'}, children = [ 

    
    ## gem5 logo
    html.Div(style = {'width' : '100%', 'display' : 'flex', 'justify-content' : 'center'}, children = [
        html.Div([
            html.Img(src = 'https://ncaagymnews.weebly.com/uploads/5/7/8/2/57829447/ucdavis-largeheader_3_orig.jpg')
        ])
    ]),

    # ------------------------------------------------------
    # --------------------- Div View 1 ---------------------
    # ------------------------------------------------------ 
    
    html.Div(style = {'margin-bottom' : '5%', 'display' : 'flex', 'align-items' : 'center'}, children = [ 
		html.Div(style = {'width' : '15%', 'height': '100%', 'display' : 'inline-block'}, children = [  
            dcc.Input(
                id="V1_cVar_selection_msg", type="text",
                value = 'Select a categorical variable to see the pie chart of its values:',
                placeholder="",readOnly = True,
                style={'width': '90%','backgroundColor': '#f8f8f8',  'border':'none', 'display': 'inline-block'}),
            dcc.Dropdown(
                id='V1_var_piechart',
                options=[{'label': i, 'value': i} for i in cat_cols],
                value='Kernel Version'),
            dcc.Input(
                id="V1_mVar_selection_msg", type="text",
                value = 'Select a measured variable to see the histogram of its distribution:',
                placeholder="",readOnly = True,
                style={'width': '90%','backgroundColor': '#f8f8f8', 'border':'none', 'display': 'inline-block'}),
            dcc.Dropdown(
                id='V1_var_hist',
                options=[{'label': i, 'value': i} for i in num_cols],
                value='Host Memory Usage')
        ]),
        dcc.Graph(id="V1_pie_chart", style={'width': '40%', 'height' : '100%', 'display': 'inline-block'}),
        dcc.Graph(id="V1_graph_histogram", style={'width': '40%', 'height' : '100%', 'display': 'inline-block' })
    ]),

    # ---------- Histogram of Measured
	# html.Div(style = {'margin-bottom' : '5%'}, children = [ 
	# 	html.Div(style = {'width' : '30%', 'height': '100%', 'display' : 'inline-block'}, children = [  
    #         html.Div([
    #             dcc.Input(
    #                 id="V1_mVar_selection_msg", type="text",
    #                 value = 'Select a measured variable to see the histogram of its distribution:',
    #                 placeholder="",readOnly = True,
    #                 style={'width': '90%','backgroundColor': '#f8f8f8', 'border':'none', 'display': 'inline-block'})
    #         ]),
    #         html.Div([
    #             dcc.Dropdown(
    #                 id='V1_var_hist',
    #                 options=[{'label': i, 'value': i} for i in num_cols],
    #                 value='Host Memory Usage'
    #             )
    #         ], style={'width': '90%', 'display': 'inline-block'}),
    #     ]),
    #     dcc.Graph(id="V1_graph_histogram", style={'width': '40%', 'height' : '100%', 'display': 'inline-block' }),
    # ]),


    # ---------- Tree Map
    html.Div(style = {'margin-bottom' : '5%'}, children = [    
        html.Div(style = {'width' : '100%', 'display' : 'flex'}, children = [
            html.Div([
                dcc.Dropdown(
                    id="V1_Var1_Tile",
                    options=[{'label': i, 'value': i} for i in cat_cols],
                    value= 'Experiment Name'),
                
                dcc.Dropdown(   
                    id="V1_Var1Tile_lvls",
                    multi=True)
                    # style = {'display' : 'inline-block'})
            ]),
            html.Div([
                dcc.Dropdown(
                    id="V1_Var2_Tile",
                    options=[{'label': i, 'value': i} for i in cat_cols],
                    value= 'Kernel Version'),

                dcc.Dropdown(
                    id="V1_Var2Tile_lvls",
                    multi=True)
            ]),

            html.Div([
                dcc.Dropdown(
                    id="V1_Var3_Tile",
                    options=[{'label': i, 'value': i} for i in cat_cols],
                    value= 'Num. of CPUs'),

                dcc.Dropdown(
                    id="V1_Var3Tile_lvls",
                    multi=True)
            ]),

            html.Div([
                dcc.Dropdown(
                    id="V1_Var4_Tile",
                    options=[{'label': i, 'value': i} for i in cat_cols],
                    value= 'Boot Type',
                ),

                dcc.Dropdown(
                    id="V1_Var4Tile_lvls",
                    multi=True)
            ]),

            html.Div([
                dcc.Dropdown(
                    id="V1_Var5_Tile",
                    options=[{'label': i, 'value': i} for i in cat_cols],
                    value= 'CPU Model'),

                dcc.Dropdown(
                    id="V1_Var5Tile_lvls",
                    multi=True),
            ]),

            html.Div([
                dcc.Dropdown(
                    id="V1_Var6_Tile",
                    options=[{'label': i, 'value': i} for i in cat_cols],
                    value= 'Memory System'),

                dcc.Dropdown(
                    id="V1_Var6Tile_lvls",
                    multi=True),
            ]),

            html.Div([
                dcc.Dropdown(
                    id="V1_Var7_Tile",
                    options=[{'label': i, 'value': i} for i in cat_cols],
                    value= 'Input Size'),

                dcc.Dropdown(
                    id="V1_Var7Tile_lvls",
                    multi=True),
            ]),

            html.Div([
                dcc.Dropdown(
                    id="V1_Var8_Tile",
                    options=[{'label': i,  'value': i} for i in cat_cols],
                    value= 'Sim. Status Result'),

                dcc.Dropdown(
                    id="V1_Var8Tile_lvls",
                    multi=True)
            ])
        ]),
        
        dcc.Graph(id='V1_Graph_Tile', style = {'width' : '100%', 'display' : 'inline-block'})
    ]),

    
    # ------------------------------------------------------
    # --------------------- Div View 2 ---------------------
    # ------------------------------------------------------

    # ---------- Bar Charts
    html.Div(style = {'margin-bottom' : '5%'}, children = [  
        html.Div(style = {'width': '100%', 'display' : 'flex'}, children = [
            html.Div([
                dcc.Textarea(
                    id="bartxt1",
                    value = ['Select X-Axis below:'],
                    readOnly = True),

                dcc.Dropdown(
                    id="V1_Var1_Bars",
                    options=[{'label': i, 'value': i} for i in cat_cols],
                    value= 'CPU Model')
            ]),

            html.Div([
                dcc.Textarea(
                    id="bartxt2",
                    value = ['Select X-Axis levels below:'],
                    readOnly = True),

                dcc.Dropdown(
                    id="V1_Var1_lvls",
                    multi=True)
            ]),

            html.Div([
                dcc.Textarea(
                    id="bartxt3",
                    value = ['Select Y-Axis below:'],
                    readOnly = True),
        
                dcc.Dropdown(
                    id="V1_Var2_Bars",
                    options=[{'label': i, 'value': i} for i in num_cols],
                    value= 'Host Time')
            ]),

            html.Div([
                dcc.Textarea(
                    id="bartxt5",
                    value = ['Select colored group variable below:'],
                    readOnly = True),

                dcc.Dropdown(
                    id="V1_Var3_Bars",
                    options=[{'label': i, 'value': i} for i in cat_cols],
                    value= 'Memory System')
            ]), 

            html.Div([
                dcc.Textarea(
                    id="bartxt6",
                    value = ['Select colored group variable levels below:'],
                    readOnly = True),

                dcc.Dropdown(
                    id="V1_Var3_lvls",
                    multi=True)
            ]),

            html.Div([
                dcc.Textarea(
                    id="bartxt7",
                    value = ['Select facet variable below:'],
                    readOnly = True),

                dcc.Dropdown(
                    id="V1_Var4_Bars",
                    options=[{'label': i, 'value': i} for i in cat_cols],
                    value= 'Kernel Version')
            ]),

            html.Div([
                dcc.Textarea(
                    id="bartxt8",
                    value = ['Select facet variable levels below:'],
                    readOnly = True),

                dcc.Dropdown(
                    id="V1_Var4_lvls",
                    multi=True)
            ]),
                 
        ]),

        dcc.Graph(id='V1_graph_bar', style = {'width' : '100%', 'display' : 'inline-block'})
    ]),
    
    html.Div(style = {'margin-bottom' : '5%'}, children = [     
        html.Div(style = {'width' : '100', 'display' : 'flex'}, children = [
            html.Div([
                dcc.Textarea(
                    id="scattxt1",
                    value = ['Select X-Axis below:'],
                    readOnly = True),

                dcc.Dropdown(
                    id="V1_Var1_Scat",
                    options=[{'label': i, 'value': i} for i in num_cols],
                    value= 'Number of Memory References')
            ]),

            html.Div([
                dcc.Textarea(
                    id="scattxt2",
                    value = ['Select Y-Axis below:'],
                    readOnly = True),
                
                dcc.Dropdown(
                    id="V1_Var2_Scat",
                    options=[{'label': i, 'value': i} for i in num_cols],
                    value= 'Host Instruction Rate')
            ]),

            html.Div([
                dcc.Textarea(
                    id="scattxt3",
                    value = ['Select color variable below:'],
                    readOnly = True),

                dcc.Dropdown(
                    id="V1_Var3_Scat",
                    options=[{'label': i, 'value': i} for i in cat_cols],
                    value= 'Memory System')

            ]),

            html.Div([
                dcc.Textarea(
                    id="scattxt4",
                    value = ['Select color variable levels below:'],
                    readOnly = True),

                dcc.Dropdown(
                id="V1_Var3Scat_lvls",
                multi=True)
            
            ]), 

            html.Div([
                dcc.Textarea(
                    id="scattxt5",
                    value = ['Select facet variable below:'],
                    readOnly = True),

                dcc.Dropdown(
                    id="V1_Var4_Scat",
                    options=[{'label': i, 'value': i} for i in cat_cols],
                    value= 'Kernel Version')
            ]),

            html.Div([
                dcc.Textarea(
                    id="scattxt6",
                    value = ['Select facet variable levels below:'],
                    readOnly = True),

                dcc.Dropdown(
                    id="V1_Var4Scat_lvls",
                    multi=True)
            ]),

            html.Div([
                dcc.Textarea(
                    id="scattxt7",
                    value = ['Select scatter dots size variable below:'],
                    readOnly = True),
                dcc.Dropdown(
                    id="V5_Var5_Scat",
                    options=[{'label': i, 'value': i} for i in num_cols],
                    value= 'Number of Load Instructions')
            ]), 

            dcc.Checklist(
                id = 'V1_Logarithmic_Scat',
                options=[
                    {'label': 'X->Logarithmic', 'value': 'LogX'},
                    {'label': 'Y->Logarithmic', 'value': 'LogY'},
                ],
                value=['LogX'])
        ]),
        
        dcc.Graph(id='V1_Graph_Scat', style = {'width' : '100%', 'display' : 'inline-block'}) 
    ]),

    # ---------- Scatter
    # html.Div(style = {'margin-bottom' : '5%'}, children = [
        
    #     html.Div(style = {'width' : '15%', 'height': '100%', 'display' : 'inline-block'}, children = [
    #         dcc.Dropdown(
    #             id="V1_Var1_Scat",
    #             options=[{'label': i, 'value': i} for i in num_cols],
    #             value= 'Host Time'),
            
    #         dcc.Dropdown(
    #             id="V1_Var2_Scat",
    #             options=[{'label': i, 'value': i} for i in num_cols],
    #             value= 'Host Instruction Rate'),

    #         dcc.Dropdown(
    #             id="V1_Var3_Scat",
    #             options=[{'label': i, 'value': i} for i in cat_cols],
    #             value= 'Experiment Name'),

    #         html.Div([
    #             dcc.Dropdown(
    #                 id="V1_Var3Scat_lvls",
    #                 multi=True),
    #         ], style={'display': 'block'}),

    #         dcc.Dropdown(
    #             id="V1_Var4_Scat",
    #             options=[{'label': i, 'value': i} for i in cat_cols],
    #             value= 'Sim. Status Result',
    #         ),

    #         html.Div([
    #             dcc.Dropdown(
    #                 id="V1_Var4Scat_lvls",
    #                 multi=True),
    #         ], style={'display': 'block'}),

    #         dcc.Dropdown(
    #             id="V5_Var5_Scat",
    #             options=[{'label': i, 'value': i} for i in num_cols],
    #             value= 'Host Opcode Rate'),

    #         dcc.Checklist(
    #             id = 'V1_Logarithmic_Scat',
    #             options=[
    #                 {'label': 'X->Logarithmic', 'value': 'LogX'},
    #                 {'label': 'Y->Logarithmic', 'value': 'LogY'},
    #             ],
    #             value=['LogX']
    #         ) 

    #     ]),       
    #     dcc.Graph(id='V1_Graph_Scat', style = {'width' : '65%', 'height': '100%', 'display' : 'inline-block'})
    
    # ]),
    
    # ------------------------------------------------------
    # --------------------- Div View 3 ---------------------
    # ------------------------------------------------------


    # -------- PCA
    html.Div(style = {'margin-bottom' : '5%', 'display' : 'flex', 'align-items' : 'center'}, children = [
		# ---------- Div General settings
		html.Div(style = {'width' : '40%', 'height': '100%', 'display' : 'inline-block', 'margin-right' : '5%'}, children = [
            dcc.Textarea(id="V3_txt1",
                value = ['Select Desired Descriptive Variable:'],
                readOnly = True,
                style={'width': '100%','backgroundColor': '#f8f8f8', 'justify': 'center', 'display': 'flex','resize': 'none','border':'none'}),
            
            dcc.Dropdown(
                id='V3_desc',
                options=[{'label': i, 'value': i} for i in cat_cols],
                value='Kernel Version'),

            dcc.Textarea(id="V3_txt2",
                value = ['Select Desired Measurement Feature(s):'],
                readOnly = True,
                style={'width': '100%','backgroundColor': '#f8f8f8', 'justify': 'center', 'display': 'flex','resize': 'none','border':'none'}),

            dcc.Dropdown(
                    id='V3_featconti',
                    options=[{'label': i, 'value': i} for i in num_cols],
                    value=num_cols[0:5],
                    multi=True),

            dcc.Textarea(id="V3_txt3",
                value = ['Select Desired Categorical Feature(s):'],
                readOnly = True,
                style={'width': '99%','backgroundColor': '#f8f8f8', 'justify': 'center', 'display': 'flex','resize': 'none','border':'none'}),
            
            dcc.Textarea(id="V3_txtyerrng",
                value = ['Select Desired year range:'],
                readOnly = True,
                style={'width': '100%','backgroundColor': '#f8f8f8', 'justify': 'center', 'display': 'flex','resize': 'none','border':'none'}),
            
            dcc.Checklist(
                id='V3_Preproc_Flags',
                options=[{'label': 'Normalize', 'value': 'norm_flag'},
                            {'label': 'Enable PCA', 'value': 'pca_flag'}],
                value=['norm_flag','pca_flag']),  

            dcc.Input(
                id='V3_pcacompo', type='text', 
                value = ['Select the number of components between '+ str(2) + ' and ' + str(10)+':'],
                placeholder="",readOnly = False,
                style={'width': '100%','backgroundColor': '#f8f8f8', 'float': 'left', 'display': 'block','border':'none','whiteSpace': 'pre-line','height': 30}),

            dcc.Input(
                id="V3_pcaCompNum_txt", type="number", placeholder="", value=3,
                min=2, max=10, step=1),

            dcc.Textarea(
                id="V3_datapoints",
                value = ['Data Points:'],
                readOnly = True,
                style={'width': '100%', 'float': 'Center', 'display': 'inline-block','whiteSpace': 'pre-line','resize': 'none','height': 150}),
            
            dcc.Textarea(
                id="V3_txt4",
                value = ['Select percentage of datapoints used for testing and evaluation in the slider below:'],
                readOnly = True,
                style={'width': '100%', 'justify': 'center', 'display': 'flex','resize': 'none','border':'none','backgroundColor': '#f8f8f8'}),
            
            dcc.Slider(
                id='V3_testproport',
                min = 10,
                max = 50,
                step=1,
                value=30,
                marks={str(i): str(i) for i in range(10,50)}),

            dcc.Textarea(id="V4_txt1",
                value = ['Select Desired Measurment:'],
                readOnly = True),
            
            dcc.Dropdown(
                id='V4_mVar',
                options=[{'label': i, 'value': i} for i in num_cols],
                value="Host Time"),
            
            dcc.Textarea(id="V4_txt2",
                value = ['Select Desired Descriptive Feature(s):'],
                readOnly = True,
                style={'width': '100%','backgroundColor': '#f8f8f8', 'justify': 'center', 'display': 'flex','resize': 'none','border':'none'}),

            dcc.Dropdown(
                id='V4_feat',
                options=[{'label': i, 'value': i} for i in desc_cols],
                value=desc_cols[0:2],
                multi=True),
     
            dcc.Input(
                id="V4_bin", type="number", placeholder="", value=5,
                min=2, max=20, step=1),

            dcc.Textarea(
                    id="V4_estimate",
                    value = ['Estimated Value:'],
                    readOnly = True,
                    style={'width': '100%', 'float': 'Center', 'display': 'inline-block','whiteSpace': 'pre-line','resize': 'none','height': 150})
		]),

        html.Div(style = {'width': '55%', 'display': 'inline-block'}, children = [
            dcc.Graph(
                id='V3_conftable_fig'
            ),

            html.Div([
                dcc.Graph(
                    id='V4_hist_fig',
                    style={'width' : '50%', 'display' : 'inline-block'}),

                dcc.Graph(
                    id='V4_hist_fig2',
                    style={'width' : '50%', 'display' : 'inline-block'})
            ])
	    ])
    ])
]) 

##********************************************************************************************************************

#     # ---------- num estimate
#     html.Div(style = {'margin-bottom' : '5%'}, children = [
# 		# ---------- Div General settings
# 		html.Div(style = {'width' : '20%', 'height': '100%', 'display' : 'inline-block'}, children = [

#             html.Div([
#                 dcc.Textarea(id="V4_txt1",
#                     value = ['Select Desired Measurment:'],
#                     readOnly = True,
#                     style={'width': '99%','backgroundColor': '#f8f8f8', 'justify': 'center', 'display': 'flex','resize': 'none','border':'none'}),
#             ]),
            
#             html.Div([
#                 dcc.Dropdown(
#                     id='V4_mVar',
#                     options=[{'label': i, 'value': i} for i in num_cols],
#                     value="Host Time"
#                 )
#             ], style={'width': '100%', 'float': 'left', 'display': 'inline-block'}),
            
#             html.Div([
#                 dcc.Textarea(id="V4_txt2",
#                     value = ['Select Desired Feature(s):'],
#                     readOnly = True,
#                     style={'width': '99%','backgroundColor': '#f8f8f8', 'justify': 'center', 'display': 'flex','resize': 'none','border':'none'}),
#             ]),
            
#             html.Div([
#                 dcc.Dropdown(
#                     id='V4_feat',
#                     options=[{'label': i, 'value': i} for i in desc_cols],
#                     value=desc_cols[0:2],
#                     multi=True,
#                 )
#             ], style={'width': '100%', 'float': 'left', 'display': 'inline-block'}),     

#             html.Div([
#                 dcc.Input(
#                     id="V4_bin", type="number", placeholder="", value=5,
#                     min=2, max=20, step=1,
#                 )
#             ]),          		
# 		]),

#         # ---------- graphs hist
#         html.Div([
#                 dcc.Textarea(
#                     id="V4_estimate",
#                     value = ['Estimated Value:'],
#                     readOnly = True,
#                     style={'width': '99%', 'float': 'Center', 'display': 'inline-block','whiteSpace': 'pre-line','resize': 'none','height': 150}
#                 ),
#             ]),

#         html.Div([
#                 dcc.Graph(
#                     id='V4_hist_fig'
#                 )
#             ], style={'width': '75%', 'display': 'inline-block', 'padding': '0 20'}),
        
#         html.Div([
#                 dcc.Graph(
#                     id='V4_hist_fig2'
#                 )
#             ], style={'width': '75%', 'display': 'inline-block', 'padding': '0 20'}),

# 	]),
# ])



###################################################################################################################
############################################### *** STATIC *** ####################################################
###################################################################################################################

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

@app.callback(
    Output(component_id = 'V1_pie_chart',    component_property = 'figure'),
    [Input(component_id = 'V1_var_piechart', component_property = 'value')])
def update_graph_pie(ColName):
    dftmp = pd.DataFrame(df1, columns = [ColName]) 
    dftmp = dftmp[dftmp[ColName] != "NAN"]
    dftmp = dftmp.groupby([ColName]).size().reset_index(name="count")
    fig = px.pie(dftmp, names = ColName, values = 'count') 
    return fig
        
###################################################################################################################
############################################# *** DYNAMIC *** #####################################################
###################################################################################################################

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
    return df1[V1_Var3Scat].unique()[1:2]


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
    return df1[V1_Var4Scat].unique()[0:1]

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
    Input(component_id = 'V1_Var6_Tile',     component_property = 'value'),
    Input(component_id = 'V1_Var6Tile_lvls', component_property = 'value'),
    Input(component_id = 'V1_Var7_Tile',     component_property = 'value'),
    Input(component_id = 'V1_Var7Tile_lvls', component_property = 'value'),
    Input(component_id = 'V1_Var8_Tile',     component_property = 'value'),
    Input(component_id = 'V1_Var8Tile_lvls', component_property = 'value'),
    ])
def update_graph_tile(var1, lvl1, var2, lvl2, var3, lvl3, var4, lvl4, var5, lvl5, var6, lvl6, var7, lvl7, var8, lvl8):
    tmpdf = pd.DataFrame(df1, columns = [var1, var2, var3, var4, var5, var6, var7, var8, 'Simulation Frequency'])
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
    if(~tmpdf[tmpdf[var6].isin(lvl6)].empty):
        tmpdf = tmpdf[tmpdf[var6].isin(lvl6)]
    if(~tmpdf[tmpdf[var7].isin(lvl7)].empty):
        tmpdf = tmpdf[tmpdf[var7].isin(lvl7)]
    if(~tmpdf[tmpdf[var8].isin(lvl8)].empty):
        tmpdf = tmpdf[tmpdf[var8].isin(lvl8)]
    #tmpdf = df1.groupby(by = [var1, var2, var3, var4, var5],as_index=False).mean()
    fig = px.treemap(tmpdf, path=[var1, var2, var3, var4, var5, var6, var7, var8], values='Simulation Frequency')
    return fig

##----TREE VAR1 LEVELS ----------------------------------------------------------------------##
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

##----TREE VAR2 LEVELS ----------------------------------------------------------------------##
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

##----TREE VAR3 LEVELS ----------------------------------------------------------------------##
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

#----TREE VAR4 LEVELS ----------------------------------------------------------------------##
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

#----TREE VAR5 LEVELS ----------------------------------------------------------------------##
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

#----TREE VAR6 LEVELS ----------------------------------------------------------------------##
@app.callback(
   Output(component_id = 'V1_Var6Tile_lvls', component_property = 'options'),
   [Input(component_id = 'V1_Var6_Tile',     component_property = 'value'),
    ])
def update_V1Tile_lvl6_opt(V1_Var6Tile):
   opts=[{'label':lvl, 'value':lvl} for lvl in df1[V1_Var6Tile].unique()]
   return opts

@app.callback(
   Output(component_id = 'V1_Var6Tile_lvls', component_property = 'value'),
   [Input(component_id = 'V1_Var6_Tile',     component_property = 'value'),
    ])
def update_V1Tile_lvl6_val(V1_Var6Tile):
   return df1[V1_Var6Tile].unique()[0:2]


#----TREE VAR7 LEVELS ----------------------------------------------------------------------##
@app.callback(
   Output(component_id = 'V1_Var7Tile_lvls', component_property = 'options'),
   [Input(component_id = 'V1_Var7_Tile',     component_property = 'value'),
    ])
def update_V1Tile_lvl7_opt(V1_Var7Tile):
   opts=[{'label':lvl, 'value':lvl} for lvl in df1[V1_Var7Tile].unique()]
   return opts

@app.callback(
   Output(component_id = 'V1_Var7Tile_lvls', component_property = 'value'),
   [Input(component_id = 'V1_Var7_Tile',     component_property = 'value'),
    ])
def update_V1Tile_lvl7_val(V1_Var7Tile):
   return df1[V1_Var7Tile].unique()[0:2]

#----TREE VAR8 LEVELS ----------------------------------------------------------------------##
@app.callback(
   Output(component_id = 'V1_Var8Tile_lvls', component_property = 'options'),
   [Input(component_id = 'V1_Var8_Tile',     component_property = 'value'),
    ])
def update_V1Tile_lvl8_opt(V1_Var8Tile):
   opts=[{'label':lvl, 'value':lvl} for lvl in df1[V1_Var8Tile].unique()]
   return opts

@app.callback(
   Output(component_id = 'V1_Var8Tile_lvls', component_property = 'value'),
   [Input(component_id = 'V1_Var8_Tile',     component_property = 'value'),
    ])
def update_V1Tile_lvl8_val(V1_Var8Tile):
   return df1[V1_Var8Tile].unique()[0:2]

###################################################################################################################
############################################ *** DATA ANALYSIS *** ################################################
###################################################################################################################

@app.callback(
    dash.dependencies.Output(component_id = 'V3_datapoints', component_property = 'value'),
    dash.dependencies.Output(component_id = 'V3_conftable_fig', component_property = 'figure'),
    [dash.dependencies.Input(component_id = 'V3_desc', component_property = 'value'),
      dash.dependencies.Input(component_id = 'V3_featconti', component_property = 'value'),
      dash.dependencies.Input(component_id = 'V3_Preproc_Flags', component_property = 'value'),
      dash.dependencies.Input(component_id = 'V3_pcaCompNum_txt', component_property = 'value'),
      dash.dependencies.Input(component_id = 'V3_testproport', component_property = 'value')
      ])
def update_graph(grname, contfeat, preprocflag, Component_num, testproport):
    
    Catlabel = grname
    ML_Cols = np.concatenate([[grname],contfeat])
    grname = df1[Catlabel].unique()
    df_ML = df1[ML_Cols]
    df_ML = df_ML.fillna(df_ML.mean())
    df_Cont = pd.DataFrame(df_ML, columns = contfeat)
    labels = df_ML[Catlabel].tolist()

    if 'norm_flag' in preprocflag:
     	# ------------ Normalize
     	sc = StandardScaler()
     	df_Cont = sc.fit_transform(df_Cont)
    
    if 'pca_flag' in preprocflag:
     	# ------------ PCA
     	pca = PCA(n_components = Component_num)
     	df_Cont = pca.fit_transform(df_Cont)
    
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

    fig = ff.create_annotated_heatmap(cm, 
                                      font_colors=['black'], hoverinfo='text',
                                      colorscale='Viridis')
    fig.update_layout(title_text= 'Accuracy = '+ str(round(accuracy_score(y_test, y_pred)*100,1)))
    return (datapoints_str, fig)


#####################################################################################
#####################################################################################

##----mVar Estimate ----------------------------------------------------------------------##
@app.callback(
   dash.dependencies.Output(component_id = 'V4_estimate',  component_property = 'value'),
   dash.dependencies.Output(component_id = 'V4_hist_fig',  component_property = 'figure'),
   dash.dependencies.Output(component_id = 'V4_hist_fig2', component_property = 'figure'),
   [dash.dependencies.Input(component_id = 'V4_mVar',      component_property = 'value'),
    dash.dependencies.Input(component_id = 'V4_feat',      component_property = 'value'),
    dash.dependencies.Input(component_id = 'V4_bin',       component_property = 'value'),
     ])
def update_histGraphs(mVar, feat, bin):
    datapoints_str = "testing"
    sortdf = df1.sort_values(by = [mVar])
    tmpdf = pd.DataFrame(sortdf, columns = [mVar])
    # tmpdf[mVar][tmpdf[mVar]>(np.mean(tmpdf[mVar])+np.std(tmpdf[mVar])/2)] = np.mean(tmpdf[mVar])+np.std(tmpdf[mVar])/2
    # tmpdf[mVar][tmpdf[mVar]>(np.mean(tmpdf[mVar])+np.std(tmpdf[mVar])/10)] = np.mean(tmpdf[mVar])+np.std(tmpdf[mVar])/10
    featdf = pd.DataFrame(sortdf, columns = feat)
    for col in feat:
        featdf[col] = featdf[col].astype('category')
        featdf[col] = featdf[col].cat.codes

    groups = np.array_split(tmpdf[mVar],bin)
    labels = []
    tmpdf["Labels"] = tmpdf[mVar]
    tmpdf["Labels"] = "0"
    for i in range(bin):
        tmpdf.loc[tmpdf[mVar].ge(np.min(groups[i])) & tmpdf[mVar].le(np.max(groups[i])),"Labels"]= "{:.1e}".format(np.min(groups[i]))+"_"+"{:.1e}".format(np.max(groups[i]))
        labels = np.append(labels,"{:.1e}".format(np.min(groups[i]))+"_"+"{:.1e}".format(np.max(groups[i])))

    fig = px.histogram(tmpdf,x = "Labels", labels = {'x':mVar, 'y':"frequency"})
    featdf = featdf.fillna(featdf.mean())
    # sc = StandardScaler()
    # featdf = sc.fit_transform(featdf)
    labels = tmpdf["Labels"]
    y_train = labels
    X_train = featdf
    # ------------ Random Forest classifier 
    classifier = RandomForestClassifier(max_depth=2, random_state=0)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_train)
    fig2 = px.histogram(y_pred)
    # cm = confusion_matrix(y_train, y_pred)
    datapoints_str = 'Accuracy = '+ str(accuracy_score(y_train, y_pred))
    return (datapoints_str,fig, fig2)




if __name__ == '__main__':
  app.run_server()