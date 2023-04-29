import streamlit as st
import numpy as np
import matplotlib as mpl
import plotly.express as px
import pandas as pd

CSV = pd.read_csv('trackLog-2021-oct.-28_13-55-24.csv',na_values="-")  
CSV_num = CSV.select_dtypes(include=[float])

A1=CSV_num.columns
df1 = pd.DataFrame(data=A1, columns=['Channel1'])
df1['Plot1'] = False

A2=CSV_num.columns
df2 = pd.DataFrame(data=A2, columns=['Channel2'])
df2['Plot2'] = False

edited_column1 = st.experimental_data_editor(df1)
df_f1 = edited_column1[edited_column1['Plot']]
CSV_num_f1 = CSV_num.filter(items=df_f1['Channel'])

edited_column2 = st.experimental_data_editor(df2)
df_f2 = edited_column2[edited_column2['Plot']]
CSV_num_f2 = CSV_num.filter(items=df_f2['Channel'])

fig1 = px.line(CSV_num_f1, x=CSV_num_f1.index, y=CSV_num_f1.columns)
st.plotly_chart(fig1, use_container_width=True)

fig2 = px.line(CSV_num_f2, x=CSV_num_f2.index, y=CSV_num_f2.columns)
st.plotly_chart(fig2, use_container_width=True)
