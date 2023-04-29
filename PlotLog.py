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




fig = px.line(CSV_num, x=CSV_num.index, y=CSV_num.columns)

st.plotly_chart(fig, use_container_width=True)
