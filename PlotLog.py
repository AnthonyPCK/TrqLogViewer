import streamlit as st
import numpy as np
import matplotlib as mpl
import plotly.express as px
import pandas as pd


#mpl.rc('figure', max_open_warning = 0)

CSV = pd.read_csv('trackLog-2021-oct.-28_13-55-24.csv',na_values="-")  

A = CSV.columns
NbCol = len(A)

#for i in A:
#    NomCol = i
#    ValCol = CSV.get(NomCol)
#    
#    f = mpl.pyplot.figure(figsize=(8,6))
#    ax = f.add_subplot(111)#    
#    ax.plot(ValCol, '-')
#    ax.grid()
#    ax.set_title(NomCol)
#    #ax.set_xlabel("Z [N]")
#    st.pyplot(f)

import pandas as pd
df = pd.DataFrame(dict(
    x = [1, 3, 2, 4],
    y = [1, 2, 3, 4]
))

fig = px.line(df, x="Index", y="Data", color="Chanel")
#fig = px.line(CSV.get(A[3]), x="Index", y="Data", color="Chanel")

st.plotly_chart(fig, use_container_width=True)
