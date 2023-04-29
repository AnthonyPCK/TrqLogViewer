

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
figxHist = px.density_heatmap(CSV_num, x=optionxh, y=optionyh, nbinsx=100, nbinsy=100)
st.plotly_chart(figxHist, use_container_width=True)

'''
## Action :
'''
import yfinance as yf
NomAct1 = st.text_input('ID de l''action :', 'ML.PA')

def history(self, period="1mo", interval="1d",
            start=None, end=None, prepost=False, actions=True,
            auto_adjust=True, back_adjust=False,
            proxy=None, rounding=False, tz=None, timeout=None, **kwargs):
    """
    :Parameters:
        period : str
            Valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
            Either Use period parameter or use start and end
        interval : str
            Valid intervals: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
            Intraday data cannot extend last 60 days
        start: str
            Download start date string (YYYY-MM-DD) or _datetime.
            Default is 1900-01-01
        end: str
            Download end date string (YYYY-MM-DD) or _datetime.
            Default is now
        prepost : bool
            Include Pre and Post market data in results?
            Default is False
        auto_adjust: bool
            Adjust all OHLC automatically? Default is True
        back_adjust: bool
            Back-adjusted data to mimic true historical prices
        proxy: str
            Optional. Proxy server URL scheme. Default is None
        rounding: bool
            Round values to 2 decimal places?
            Optional. Default is False = precision suggested by Yahoo!
        tz: str
            Optional timezone locale for dates.
            (default data is returned as non-localized dates)
        timeout: None or float
            If not None stops waiting for a response after given number of
            seconds. (Can also be a fraction of a second e.g. 0.01)
            Default is None.
        **kwargs: dict
            debug: bool
                Optional. If passed as False, will suppress
                error message printing to console.
    """

goog = yf.Ticker(NomAct1)
data = goog.history()
st.write(data.head())
figAction = px.line(data, x="Date", y="Close")
st.plotly_chart(figAction, use_container_width=True)
