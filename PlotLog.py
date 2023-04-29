

import streamlit as st
import numpy as np
import matplotlib as mpl
import plotly.express as px
import pandas as pd

'''
# Viewer de fichier CSV

## Choisir un type de delimiter :
'''

TypeDelim = st.text_input('Type de delimiter du fichier CSV (\s+ , ou ; )', ',')


'''
## Choisir un fichier CSV :
'''
uploaded_file = st.file_uploader(" ")
if uploaded_file is not None:
    CSV = pd.read_csv(uploaded_file,na_values="-",delimiter=TypeDelim)
else:
    '''
    ## Exemple d'un fichier CSV :
    '''
    CSV = pd.read_csv('trackLog-2021-oct.-28_13-55-24.csv',na_values="-",delimiter=TypeDelim)  

'''
## Choisir les voies à afficher dans la figure 1 :
'''
CSV_num = CSV.select_dtypes(include=[float, int])

A1=CSV_num.columns
df1 = pd.DataFrame(data=A1, columns=['Channel1'])
df1['Plot1'] = True

A2=CSV_num.columns
df2 = pd.DataFrame(data=A2, columns=['Channel2'])
df2['Plot2'] = False

edited_column1 = st.experimental_data_editor(df1)
df_f1 = edited_column1[edited_column1['Plot1']]
CSV_num_f1 = CSV_num.filter(items=df_f1['Channel1'])

'''
## Choisir les voies à afficher dans la figure 2 :
'''

edited_column2 = st.experimental_data_editor(df2)
df_f2 = edited_column2[edited_column2['Plot2']]
CSV_num_f2 = CSV_num.filter(items=df_f2['Channel2'])


fig1 = px.line(CSV_num_f1, x=CSV_num_f1.index, y=CSV_num_f1.columns)
st.plotly_chart(fig1, use_container_width=True)

fig2 = px.line(CSV_num_f2, x=CSV_num_f2.index, y=CSV_num_f2.columns)
st.plotly_chart(fig2, use_container_width=True)


'''
## Tracer des signaux en fonction d'un autre :
'''
Ax=CSV_num.columns
option = st.selectbox(
    'Choisir le signal que vous souhaitez en abscisse',
    Ax)
#dfx = pd.DataFrame(data=Ax, columns=['Channelx'])
#dfx['Plotx'] = False
#edited_columnx = st.experimental_data_editor(dfx)
#df_fx = edited_columnx[edited_columnx['Plotx']]
#CSV_num_fx = CSV_num.filter(items=df_fx['Channelx'])
#df_fy = edited_columnx[edited_columnx['Plotx']==False]
#CSV_num_fy = CSV_num.filter(items=df_fy['Channelx'])


figx = px.scatter(CSV_num, x=option, y=CSV_num.columns)
st.plotly_chart(figx, use_container_width=True)


'''
## Tracer des signaux en fonction d'un autre sur un bi-histogramme :
'''
Axh=CSV_num.columns
optionxh = st.selectbox(
    'Abscisse',
    Axh)
Ayh=CSV_num.columns
optionyh = st.selectbox(
    'Ordonnée',
    Ayh)
px.density_heatmap(CSV_num, x=optionxh, y=optionyh, nbinsx=100, nbinsy=100)
