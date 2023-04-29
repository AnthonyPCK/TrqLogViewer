import streamlit as st
import numpy as np
import matplotlib as mpl
import plotly.express as px
import pandas as pd

CSV = pd.read_csv('trackLog-2021-oct.-28_13-55-24.csv',na_values="-")  
CSV_num = CSV.select_dtypes(include=[float])

A=CSV_num.columns
df = pd.DataFrame(data=A, columns=['Channel'])
df['Plot'] = True
edited_column = st.experimental_data_editor(df)

df_f = edited_column[edited_column['Plot']]
CSV_num_f1 = CSV_num.filter(items=df_f['Channel'])

fig = px.line(CSV_num_f1, x=CSV_num_f1.index, y=CSV_num_f1.columns)

st.plotly_chart(fig, use_container_width=True)
