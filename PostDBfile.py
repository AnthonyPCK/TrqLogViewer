# -*- coding: utf-8 -*-
"""
Created on Sun Apr 30 11:05:36 2023

@author: antho
"""

#%reset -f
#%clear

import streamlit as st
from sqlite3 import connect
import pandas as pd
import plotly.express as px
import numpy as np
from plotly.offline import plot
import plotly.graph_objects as go
import pathlib
import uuid
from scipy.optimize import least_squares


'''
# Viewer de fichier HybridAssistant.db
## Fichier exemple Hyundai ioniq Hybrid ou charger votre fichier
'''
#@st.cache_resource
def connection_base(id):
    return connect(id)

uploaded_file = st.file_uploader("Upload a SQLite database file.", type="db")


@st.cache_data
def loadsqlite(uploaded_file):
    if uploaded_file is not None:
        fp = pathlib.Path(str(uuid.uuid4()))
        # fp = pathlib.Path("/path/to/your/tmpfile")
        try:
            fp.write_bytes(uploaded_file.getvalue())
            conn = connection_base(str(fp))
        finally:
            if fp.is_file():
                fp.unlink()
    else:   
        conn = connection_base("hybridassistant2023.db")  

    df_FastLog = pd.read_sql('SELECT * FROM FASTLOG', conn)
    df_Trips = pd.read_sql('SELECT * FROM TRIPS', conn)
    df_TripInfo = pd.read_sql('SELECT * FROM TRIPINFO', conn)
    
    return df_FastLog, df_Trips, df_TripInfo

df_FastLog, df_Trips, df_TripInfo = loadsqlite(uploaded_file)

if uploaded_file is not None:
    optionVIN = st.selectbox(
    "On selectionne la voiture que l\'on souhaite",
    pd.unique(df_TripInfo.VIN))
else:
    optionVIN = 'KMHC851CGLU177332'; # IONIQ

        
@st.cache_data
def posttreatmyvin(uploaded_file, df_FastLog, df_Trips, df_TripInfo, optionVIN):      
    # On conserve uniquement les données correspondant à un VIN
    df_Trips = df_Trips[(df_TripInfo.VIN == optionVIN)]
    df_TripInfo = df_TripInfo[(df_TripInfo.VIN == optionVIN)]
    
    
    # On garde seulement les trajet de plus d'un km (on écrase les ancien dataframe))
    df_TripInfo = df_TripInfo[(df_Trips.NKMS >275)]
    df_Trips = df_Trips[(df_Trips.NKMS >275)]
    
    # On rajoute des channels
    df_FastLog["Time_S"] = 0
    df_FastLog["diffTime_S"] = 0
    df_FastLog["PuissanceElec_kW"] = 0
    df_FastLog["Energy_Ah"] = 0
    df_FastLog["diffSOC"] = 0
    df_FastLog["NewSOC"] = 0
    df_FastLog["NewEnergy"] = 0
    df_FastLog["diffNewEnergy"] = 0
    df_FastLog["diffNewSOC"] = 0
    df_FastLog["CapaBat"] = 0
    df_FastLog["CapaBatDecharge"] = 0
    df_FastLog["CapaBatCharge"] = 0
    df_FastLog["CapaBat30"] = 0
    df_FastLog["CapaBatDecharge30"] = 0
    df_FastLog["CapaBatCharge30"] = 0
    df_FastLog["CapaBat35"] = 0
    df_FastLog["CapaBatDecharge35"] = 0
    df_FastLog["CapaBatCharge35"] = 0
    df_FastLog["CapaBat40"] = 0
    df_FastLog["CapaBatDecharge40"] = 0
    df_FastLog["CapaBatCharge40"] = 0
    df_FastLog["CapaBat45"] = 0
    df_FastLog["CapaBatDecharge45"] = 0
    df_FastLog["CapaBatCharge45"] = 0
    df_FastLog["CapaBat50"] = 0
    df_FastLog["CapaBatDecharge50"] = 0
    df_FastLog["CapaBatCharge50"] = 0
    df_FastLog["CapaBat55"] = 0
    df_FastLog["CapaBatDecharge55"] = 0
    df_FastLog["CapaBatCharge55"] = 0
    df_FastLog["CapaBat60"] = 0
    df_FastLog["CapaBatDecharge60"] = 0
    df_FastLog["CapaBatCharge60"] = 0
    df_FastLog["CapaBat65"] = 0
    df_FastLog["CapaBatDecharge65"] = 0
    df_FastLog["CapaBatCharge65"] = 0
    df_FastLog["CapaBat70"] = 0
    df_FastLog["CapaBatDecharge70"] = 0
    df_FastLog["CapaBatCharge70"] = 0
    df_FastLog["CapaBat75"] = 0
    df_FastLog["CapaBatDecharge75"] = 0
    df_FastLog["CapaBatCharge75"] = 0
    df_FastLog["CapaBat80"] = 0
    df_FastLog["CapaBatDecharge80"] = 0
    df_FastLog["CapaBatCharge80"] = 0
    df_FastLog["CapaBat85"] = 0
    df_FastLog["CapaBatDecharge85"] = 0
    df_FastLog["CapaBatCharge85"] = 0
    df_FastLog["CapaBat90"] = 0
    df_FastLog["CapaBatDecharge90"] = 0
    df_FastLog["CapaBatCharge90"] = 0
    
    # On créé un dataframe qui contient les sorties
    df_Out = pd.DataFrame(columns = ['Distance',
                                     'Kilometrage',
                                     'DateTrajet',
                                     'VitesseMoyenne_kph',
                                     'VitesseMoyenneEnRoulage_kph',
                                     'Consommation_LAu100km',
                                     'TempeAmbiante',
                                     'TempeBat',
                                     'ResistanceBat',
                                     'CapaBat',
                                     'CapaciteBatCharge',
                                     'CapaciteBatDecharge',
                                     'CapaciteBatCharge30',
                                     'CapaciteBatDecharge30',
                                     'CapaciteBatCharge35',
                                     'CapaciteBatDecharge35',
                                     'CapaciteBatCharge40',
                                     'CapaciteBatDecharge40',
                                     'CapaciteBatCharge45',
                                     'CapaciteBatDecharge45',
                                     'CapaciteBatCharge50',
                                     'CapaciteBatDecharge50',
                                     'CapaciteBatCharge55',
                                     'CapaciteBatDecharge55',
                                     'CapaciteBatCharge60',
                                     'CapaciteBatDecharge60',
                                     'CapaciteBatCharge65',
                                     'CapaciteBatDecharge65',
                                     'CapaciteBatCharge70',
                                     'CapaciteBatDecharge70',
                                     'CapaciteBatCharge75',
                                     'CapaciteBatDecharge75',
                                     'CapaciteBatCharge80',
                                     'CapaciteBatDecharge80',
                                     'CapaciteBatCharge85',
                                     'CapaciteBatDecharge85',
                                     'CapaciteBatCharge90',
                                     'CapaciteBatDecharge90'],
                                     dtype = 'float')
    
    
    bar = st.progress(0)
    for ii in df_Trips.index:
        bar.progress(ii/np.max(df_Trips.index))
        idxDeb = df_Trips.at[ii,"TSDEB"]
        idxFin = df_Trips.at[ii,"TSFIN"]
        
        idx = (df_FastLog.TIMESTAMP > idxDeb) & (df_FastLog.TIMESTAMP < idxFin)
    
        # On garde les point ou on a le signal de tension batterie HV
        max_Voltage_idx = df_FastLog[idx].HV_V.idxmax()
        max_Voltage = df_FastLog.HV_V.loc[max_Voltage_idx]
        
                
        if max_Voltage>1:

            # On rajoute des channels
            df_FastLog.loc[idx,"Time_S"] = (df_FastLog[idx].TIMESTAMP.copy() - df_FastLog[idx].TIMESTAMP.min()) / 1000
            df_FastLog[idx].diffTime_S = np.concatenate((np.array([0]),np.diff(df_FastLog[idx].Time_S.copy())))
            df_FastLog[idx].PuissanceElec_kW = df_FastLog[idx].HV_A.copy() * df_FastLog[idx].HV_V.copy() / 1000
            df_FastLog[idx].Energy = np.cumsum(df_FastLog[idx].HV_A.copy() * df_FastLog[idx].diffTime_S.copy() / 3600)
            #df_FastLog[idx].EnergyPos = np.cumsum(df_FastLog[idx].HV_A[(df_FastLog[idx].HV_A > 0)] * df_FastLog[idx].diffTime_S[(df_FastLog[idx].HV_A > 0)] / 3600)
            #df_FastLog[idx].EnergyNeg = np.cumsum(df_FastLog[idx].HV_A[(df_FastLog[idx].HV_A < 0)] * df_FastLog[idx].diffTime_S[(df_FastLog[idx].HV_A < 0)] / 3600)
            df_FastLog[idx].diffSOC = np.concatenate((np.array([0]),np.diff(df_FastLog[idx].SOC.copy())))
            
            ## On identifie la resistance moyenne de la batterie sur le trajet
            p = np.polyfit(np.diff(df_FastLog[idx].HV_A.copy()), np.diff(df_FastLog[idx].HV_V.copy()), 1)
            BatResistance = -p[0]
            #df_FastLog[idx].HV_V_cor = df_FastLog[idx].HV_V + BatResistance*df_FastLog[idx].HV_A
            
            
            
#
            #
            #
            #def EstimVolt(Bat_Capa, Bat_Rend, Bat_Res, Bat_VoltT, df_T):
            #    df_FastLog[idx].HV_A_corr = df_FastLog[idx].HV_A.copy()
            #    df_FastLog[idx].HV_A_corr[(df_FastLog[idx].HV_A < 0)] = df_FastLog[idx].HV_A_corr[(df_FastLog[idx].HV_A_corr < 0)] * Bat_Rend
            #    df_FastLog[idx].EnergyCor = np.cumsum(df_FastLog[idx].HV_A_corr.copy() * df_FastLog[idx].diffTime_S.copy() / 3600)
            #    #return df_FastLog[idx].HV_V.iloc[0] + Bat_VoltT*(df_FastLog[idx].BATTERY_TEMP - df_FastLog[idx].BATTERY_TEMP.iloc[0]) - 100*df_FastLog[idx].EnergyCor/Bat_Capa - df_FastLog[idx].HV_A*(Bat_Res + Bat_ResT*(df_FastLog[idx].BATTERY_TEMP - df_FastLog[idx].BATTERY_TEMP.mean()))
            #    return df_FastLog[idx].HV_V.iloc[0] + Bat_VoltT*(df_FastLog[idx].BATTERY_TEMP - df_FastLog[idx].BATTERY_TEMP.iloc[0]) - 100*df_FastLog[idx].EnergyCor/Bat_Capa - df_FastLog[idx].HV_A*(Bat_Res)
            #
            #def residuals(params, x, y):
            #    Bat_Capa, Bat_Rend, Bat_Res, Bat_VoltT = params
            #    return EstimVolt(Bat_Capa, Bat_Rend, Bat_Res, Bat_VoltT, df_T) - y
            #
            #params_ini = [5.0, 0.95, 0.1, 1.0]
            #result = least_squares(residuals, params_ini, args=(df_T, df_FastLog[idx].HV_V))
            #
            #Bat_Capa, Bat_Rend, Bat_Res, Bat_VoltT = result.x
            #
            #df_FastLog[idx].HV_A_corr = df_FastLog[idx].HV_A.copy()
            #df_FastLog[idx].HV_A_corr[(df_FastLog[idx].HV_A < 0)] = df_FastLog[idx].HV_A_corr[(df_FastLog[idx].HV_A_corr < 0)] * Bat_Rend
            #df_FastLog[idx].EnergyCor = np.cumsum(df_FastLog[idx].HV_A_corr.copy() * df_FastLog[idx].diffTime_S.copy() / 3600)
            #
            #df_FastLog[idx].SoCestim = df_FastLog[idx].SOC.iloc[0] - 100*df_FastLog[idx].EnergyCor/Bat_Capa
            #df_FastLog[idx].VoltageEstim = df_FastLog[idx].HV_V.iloc[0] - 100*df_FastLog[idx].EnergyCor/Bat_Capa  - df_FastLog[idx].HV_A*Bat_Res
            
            fig1 = px.scatter(df_FastLog[idx], x=df_FastLog[idx].index, y=df_FastLog[idx].columns)
            st.plotly_chart(fig1, use_container_width=True) 
            
            
    
            # On récupère les infos générales sur le trajet
            MeanAmbiantTemp = df_FastLog[idx].AMBIENT_TEMP.mean()
            MeanBatTemp = df_FastLog[idx].BATTERY_TEMP.mean()
            MeanKilometrage = df_FastLog[idx].ODO.mean()
            TrajDate = pd.to_datetime(df_FastLog[idx].TIMESTAMP.mean()/1000, unit='s')
            TrajConso = 0.001*100*df_FastLog[idx].TRIPFUEL.max()/df_FastLog[idx].TRIP_DIST.max()
            MeanSpeed = df_FastLog[idx].SPEED_OBD.mean()
            MeanRollingSpeed = df_FastLog[idx].SPEED_OBD[(df_FastLog[idx].SPEED_OBD >1)].mean()
    
            #def moving_avg(x, n):
            #    cumsum = np.cumsum(np.insert(x, 0, 0)) 
            #    return (cumsum[n:] - cumsum[:-n]) / float(n)
            #
            #FenetreMoyMobile = 5
            
            #fig22 = px.line(df_T, x=moving_avg(df_FastLog[idx].Energy,FenetreMoyMobile), y=moving_avg(df_FastLog[idx].HV_V_cor,FenetreMoyMobile))
            #fig22.add_trace(go.Scatter(x=df_FastLog[idx].Energy, y=df_FastLog[idx].HV_V))
            #st.plotly_chart(fig22, use_container_width=True)   
            #fig50 = px.density_heatmap(df_T, x=np.diff(moving_avg(df_FastLog[idx].Energy,FenetreMoyMobile)), y=np.diff(moving_avg(df_FastLog[idx].HV_V_cor,FenetreMoyMobile)))
            #fig50.update_traces(histnorm = "percent")
            #fig50.update_layout(
            #    {
            #        "coloraxis_cmin": 0,
            #        "coloraxis_cmax": 0.2,
            #    }
            #)
            #st.plotly_chart(fig50, use_container_width=True)
            #st.plotly_chart(px.line(df_T, x=df_FastLog[idx].SOC, y=df_FastLog[idx].HV_V_cor), use_container_width=True)   
    
            ## On identifie la capacité de la batterie
            df_FastLog[idx].NewSOC = df_FastLog[idx].SOC[(df_FastLog[idx].diffSOC.copy()!=0)].copy()
            df_FastLog[idx].NewEnergy = df_FastLog[idx].Energy_Ah[(df_FastLog[idx].diffSOC.copy()!=0)].copy()
            df_FastLog[idx].diffNewEnergy = df_FastLog[idx].NewEnergy.copy()
            df_FastLog[idx].diffNewEnergy[~np.isnan(df_FastLog[idx].diffNewEnergy.copy())] = np.concatenate((np.array([np.nan]),np.diff(df_FastLog[idx].NewEnergy[~np.isnan(df_FastLog[idx].diffNewEnergy.copy())].copy())))
            
            df_FastLog[idx].diffNewSOC = df_FastLog[idx].NewSOC.copy()
            df_FastLog[idx].diffNewSOC[~np.isnan(df_FastLog[idx].diffNewSOC.copy())] = np.concatenate((np.array([np.nan]),np.diff(df_FastLog[idx].NewSOC[~np.isnan(df_FastLog[idx].diffNewSOC.copy())].copy())))
            df_FastLog[idx].CapaBat = -100*df_FastLog[idx].diffNewEnergy.copy()/df_FastLog[idx].diffNewSOC.copy()
            df_FastLog[idx].CapaBatDecharge = df_FastLog[idx].CapaBat[(df_FastLog[idx].diffNewSOC.copy()<0)].copy()
            df_FastLog[idx].CapaBatCharge = df_FastLog[idx].CapaBat[(df_FastLog[idx].diffNewSOC.copy()>0)].copy()
    
            df_FastLog[idx].CapaBat30 = -100*df_FastLog[idx].diffNewEnergy[(df_FastLog[idx].NewSOC>25) & (df_FastLog[idx].NewSOC<35)].copy()/df_FastLog[idx].diffNewSOC.copy()
            df_FastLog[idx].CapaBatDecharge30 = df_FastLog[idx].CapaBat30/df_FastLog[idx].CapaBat30 * np.mean(df_FastLog[idx].CapaBat30[(df_FastLog[idx].diffNewSOC.copy()<0)].copy())
            df_FastLog[idx].CapaBatCharge30 = df_FastLog[idx].CapaBat30/df_FastLog[idx].CapaBat30 * np.mean(df_FastLog[idx].CapaBat30[(df_FastLog[idx].diffNewSOC.copy()>0)].copy())
    
            df_FastLog[idx].CapaBat35 = -100*df_FastLog[idx].diffNewEnergy[(df_FastLog[idx].NewSOC>30) & (df_FastLog[idx].NewSOC<40)].copy()/df_FastLog[idx].diffNewSOC.copy()
            df_FastLog[idx].CapaBatDecharge35 = df_FastLog[idx].CapaBat35/df_FastLog[idx].CapaBat35 * np.mean(df_FastLog[idx].CapaBat35[(df_FastLog[idx].diffNewSOC.copy()<0)].copy())
            df_FastLog[idx].CapaBatCharge35 = df_FastLog[idx].CapaBat35/df_FastLog[idx].CapaBat35 * np.mean(df_FastLog[idx].CapaBat35[(df_FastLog[idx].diffNewSOC.copy()>0)].copy())
    
            df_FastLog[idx].CapaBat40 = -100*df_FastLog[idx].diffNewEnergy[(df_FastLog[idx].NewSOC>35) & (df_FastLog[idx].NewSOC<45)].copy()/df_FastLog[idx].diffNewSOC.copy()
            df_FastLog[idx].CapaBatDecharge40 = df_FastLog[idx].CapaBat40/df_FastLog[idx].CapaBat40 * np.mean(df_FastLog[idx].CapaBat40[(df_FastLog[idx].diffNewSOC.copy()<0)].copy())
            df_FastLog[idx].CapaBatCharge40 = df_FastLog[idx].CapaBat40/df_FastLog[idx].CapaBat40 * np.mean(df_FastLog[idx].CapaBat40[(df_FastLog[idx].diffNewSOC.copy()>0)].copy())
    
            df_FastLog[idx].CapaBat45 = -100*df_FastLog[idx].diffNewEnergy[(df_FastLog[idx].NewSOC>40) & (df_FastLog[idx].NewSOC<50)].copy()/df_FastLog[idx].diffNewSOC.copy()
            df_FastLog[idx].CapaBatDecharge45 = df_FastLog[idx].CapaBat45/df_FastLog[idx].CapaBat45 * np.mean(df_FastLog[idx].CapaBat45[(df_FastLog[idx].diffNewSOC.copy()<0)].copy())
            df_FastLog[idx].CapaBatCharge45 = df_FastLog[idx].CapaBat45/df_FastLog[idx].CapaBat45 * np.mean(df_FastLog[idx].CapaBat45[(df_FastLog[idx].diffNewSOC.copy()>0)].copy())
    
            df_FastLog[idx].CapaBat50 = -100*df_FastLog[idx].diffNewEnergy[(df_FastLog[idx].NewSOC>45) & (df_FastLog[idx].NewSOC<55)].copy()/df_FastLog[idx].diffNewSOC.copy()
            df_FastLog[idx].CapaBatDecharge50 = df_FastLog[idx].CapaBat50/df_FastLog[idx].CapaBat50 * np.mean(df_FastLog[idx].CapaBat50[(df_FastLog[idx].diffNewSOC.copy()<0)].copy())
            df_FastLog[idx].CapaBatCharge50 = df_FastLog[idx].CapaBat50/df_FastLog[idx].CapaBat50 * np.mean(df_FastLog[idx].CapaBat50[(df_FastLog[idx].diffNewSOC.copy()>0)].copy())
    
            df_FastLog[idx].CapaBat55 = -100*df_FastLog[idx].diffNewEnergy[(df_FastLog[idx].NewSOC>50) & (df_FastLog[idx].NewSOC<60)].copy()/df_FastLog[idx].diffNewSOC.copy()
            df_FastLog[idx].CapaBatDecharge55 = df_FastLog[idx].CapaBat55/df_FastLog[idx].CapaBat55 * np.mean(df_FastLog[idx].CapaBat55[(df_FastLog[idx].diffNewSOC.copy()<0)].copy())
            df_FastLog[idx].CapaBatCharge55 = df_FastLog[idx].CapaBat55/df_FastLog[idx].CapaBat55 * np.mean(df_FastLog[idx].CapaBat55[(df_FastLog[idx].diffNewSOC.copy()>0)].copy())
    
            df_FastLog[idx].CapaBat60 = -100*df_FastLog[idx].diffNewEnergy[(df_FastLog[idx].NewSOC>55) & (df_FastLog[idx].NewSOC<65)].copy()/df_FastLog[idx].diffNewSOC.copy()
            df_FastLog[idx].CapaBatDecharge60 = df_FastLog[idx].CapaBat60/df_FastLog[idx].CapaBat60 * np.mean(df_FastLog[idx].CapaBat60[(df_FastLog[idx].diffNewSOC.copy()<0)].copy())
            df_FastLog[idx].CapaBatCharge60 = df_FastLog[idx].CapaBat60/df_FastLog[idx].CapaBat60 * np.mean(df_FastLog[idx].CapaBat60[(df_FastLog[idx].diffNewSOC.copy()>0)].copy())
    
            df_FastLog[idx].CapaBat65 = -100*df_FastLog[idx].diffNewEnergy[(df_FastLog[idx].NewSOC>60) & (df_FastLog[idx].NewSOC<70)].copy()/df_FastLog[idx].diffNewSOC.copy()
            df_FastLog[idx].CapaBatDecharge65 = df_FastLog[idx].CapaBat65/df_FastLog[idx].CapaBat65 * np.mean(df_FastLog[idx].CapaBat65[(df_FastLog[idx].diffNewSOC.copy()<0)].copy())
            df_FastLog[idx].CapaBatCharge65 = df_FastLog[idx].CapaBat65/df_FastLog[idx].CapaBat65 * np.mean(df_FastLog[idx].CapaBat65[(df_FastLog[idx].diffNewSOC.copy()>0)].copy())
    
            df_FastLog[idx].CapaBat70 = -100*df_FastLog[idx].diffNewEnergy[(df_FastLog[idx].NewSOC>65) & (df_FastLog[idx].NewSOC<75)].copy()/df_FastLog[idx].diffNewSOC.copy()
            df_FastLog[idx].CapaBatDecharge70 = df_FastLog[idx].CapaBat70/df_FastLog[idx].CapaBat70 * np.mean(df_FastLog[idx].CapaBat70[(df_FastLog[idx].diffNewSOC.copy()<0)].copy())
            df_FastLog[idx].CapaBatCharge70 = df_FastLog[idx].CapaBat70/df_FastLog[idx].CapaBat70 * np.mean(df_FastLog[idx].CapaBat70[(df_FastLog[idx].diffNewSOC.copy()>0)].copy())
    
            df_FastLog[idx].CapaBat75 = -100*df_FastLog[idx].diffNewEnergy[(df_FastLog[idx].NewSOC>70) & (df_FastLog[idx].NewSOC<80)].copy()/df_FastLog[idx].diffNewSOC.copy()
            df_FastLog[idx].CapaBatDecharge75 = df_FastLog[idx].CapaBat75/df_FastLog[idx].CapaBat75 * np.mean(df_FastLog[idx].CapaBat75[(df_FastLog[idx].diffNewSOC.copy()<0)].copy())
            df_FastLog[idx].CapaBatCharge75 = df_FastLog[idx].CapaBat75/df_FastLog[idx].CapaBat75 * np.mean(df_FastLog[idx].CapaBat75[(df_FastLog[idx].diffNewSOC.copy()>0)].copy())
    
            df_FastLog[idx].CapaBat80 = -100*df_FastLog[idx].diffNewEnergy[(df_FastLog[idx].NewSOC>75) & (df_FastLog[idx].NewSOC<85)].copy()/df_FastLog[idx].diffNewSOC.copy()
            df_FastLog[idx].CapaBatDecharge80 = df_FastLog[idx].CapaBat80/df_FastLog[idx].CapaBat80 * np.mean(df_FastLog[idx].CapaBat80[(df_FastLog[idx].diffNewSOC.copy()<0)].copy())
            df_FastLog[idx].CapaBatCharge80 = df_FastLog[idx].CapaBat80/df_FastLog[idx].CapaBat80 * np.mean(df_FastLog[idx].CapaBat80[(df_FastLog[idx].diffNewSOC.copy()>0)].copy())
    
            df_FastLog[idx].CapaBat85 = -100*df_FastLog[idx].diffNewEnergy[(df_FastLog[idx].NewSOC>80) & (df_FastLog[idx].NewSOC<90)].copy()/df_FastLog[idx].diffNewSOC.copy()
            df_FastLog[idx].CapaBatDecharge85 = df_FastLog[idx].CapaBat85/df_FastLog[idx].CapaBat85 * np.mean(df_FastLog[idx].CapaBat85[(df_FastLog[idx].diffNewSOC.copy()<0)].copy())
            df_FastLog[idx].CapaBatCharge85 = df_FastLog[idx].CapaBat85/df_FastLog[idx].CapaBat85 * np.mean(df_FastLog[idx].CapaBat85[(df_FastLog[idx].diffNewSOC.copy()>0)].copy())
    
            df_FastLog[idx].CapaBat90 = -100*df_FastLog[idx].diffNewEnergy[(df_FastLog[idx].NewSOC>85) & (df_FastLog[idx].NewSOC<95)].copy()/df_FastLog[idx].diffNewSOC.copy()
            df_FastLog[idx].CapaBatDecharge90 = df_FastLog[idx].CapaBat90/df_FastLog[idx].CapaBat90 * np.mean(df_FastLog[idx].CapaBat90[(df_FastLog[idx].diffNewSOC.copy()<0)].copy())
            df_FastLog[idx].CapaBatCharge90 = df_FastLog[idx].CapaBat90/df_FastLog[idx].CapaBat90 * np.mean(df_FastLog[idx].CapaBat90[(df_FastLog[idx].diffNewSOC.copy()>0)].copy())
            
            
            
            
    #
    #
            #df_FastLog[idx].NewSoCestim = df_FastLog[idx].SoCestim[(df_FastLog[idx].diffSOC.copy()!=0)].copy()
            #df_FastLog[idx].diffNewSoCestim = df_FastLog[idx].NewSoCestim.copy()
            #df_FastLog[idx].diffNewSoCestim[~np.isnan(df_FastLog[idx].diffNewSoCestim.copy())] = np.concatenate((np.array([np.nan]),np.diff(df_FastLog[idx].NewSoCestim[~np.isnan(df_FastLog[idx].diffNewSoCestim.copy())].copy())))
            #
#
            #df_FastLog[idx].SoCestim = df_FastLog[idx].SOC.iloc[0] - 100*df_FastLog[idx].EnergyCor/Bat_Capa/np.mean(df_FastLog[idx].diffNewSoCestim.copy()/df_FastLog[idx].diffNewSOC.copy())
            #st.plotly_chart(px.scatter(df_T, x=df_FastLog[idx].SoCestim, y=df_FastLog[idx].columns), use_container_width=True) 
        
        
            # fig4 = px.scatter(df_T, x=df_FastLog[idx].SOC, y=df_FastLog[idx].columns)
            # plot(fig4)
    
            df_new_row = pd.DataFrame.from_records([{'Distance' : df_Trips.at[ii,"NKMS"],
                                    'Kilometrage' : MeanKilometrage,
                                    'DateTrajet' : TrajDate,
                                    'VitesseMoyenne_kph' : MeanSpeed,
                                    'VitesseMoyenneEnRoulage_kph' : MeanRollingSpeed,
                                    'Consommation_LAu100km' : TrajConso,
                                    'TempeAmbiante' : MeanAmbiantTemp,
                                    'TempeBat' : MeanBatTemp,
                                    'ResistanceBat' : BatResistance,
                                    'CapaBat' : df_FastLog[idx].CapaBat.mean(),
                                    'CapaciteBatCharge' : df_FastLog[idx].CapaBatCharge.mean(),
                                    'CapaciteBatDecharge' : df_FastLog[idx].CapaBatDecharge.mean(),
                                    'CapaciteBatCharge30' : df_FastLog[idx].CapaBatCharge30.mean(),
                                    'CapaciteBatDecharge30' : df_FastLog[idx].CapaBatDecharge30.mean(),
                                    'CapaciteBatCharge35' : df_FastLog[idx].CapaBatCharge35.mean(),
                                    'CapaciteBatDecharge35' : df_FastLog[idx].CapaBatDecharge35.mean(),
                                    'CapaciteBatCharge40' : df_FastLog[idx].CapaBatCharge40.mean(),
                                    'CapaciteBatDecharge40' : df_FastLog[idx].CapaBatDecharge40.mean(),
                                    'CapaciteBatCharge45' : df_FastLog[idx].CapaBatCharge45.mean(),
                                    'CapaciteBatDecharge45' : df_FastLog[idx].CapaBatDecharge45.mean(),
                                    'CapaciteBatCharge50' : df_FastLog[idx].CapaBatCharge50.mean(),
                                    'CapaciteBatDecharge50' : df_FastLog[idx].CapaBatDecharge50.mean(),
                                    'CapaciteBatCharge55' : df_FastLog[idx].CapaBatCharge55.mean(),
                                    'CapaciteBatDecharge55' : df_FastLog[idx].CapaBatDecharge55.mean(),
                                    'CapaciteBatCharge60' : df_FastLog[idx].CapaBatCharge60.mean(),
                                    'CapaciteBatDecharge60' : df_FastLog[idx].CapaBatDecharge60.mean(),
                                    'CapaciteBatCharge65' : df_FastLog[idx].CapaBatCharge65.mean(),
                                    'CapaciteBatDecharge65' : df_FastLog[idx].CapaBatDecharge65.mean(),
                                    'CapaciteBatCharge70' : df_FastLog[idx].CapaBatCharge70.mean(),
                                    'CapaciteBatDecharge70' : df_FastLog[idx].CapaBatDecharge70.mean(),
                                    'CapaciteBatCharge75' : df_FastLog[idx].CapaBatCharge75.mean(),
                                    'CapaciteBatDecharge75' : df_FastLog[idx].CapaBatDecharge75.mean(),
                                    'CapaciteBatCharge80' : df_FastLog[idx].CapaBatCharge80.mean(),
                                    'CapaciteBatDecharge80' : df_FastLog[idx].CapaBatDecharge80.mean(),
                                    'CapaciteBatCharge85' : df_FastLog[idx].CapaBatCharge85.mean(),
                                    'CapaciteBatDecharge85' : df_FastLog[idx].CapaBatDecharge85.mean(),
                                    'CapaciteBatCharge90' : df_FastLog[idx].CapaBatCharge90.mean(),
                                    'CapaciteBatDecharge90' : df_FastLog[idx].CapaBatDecharge90.mean(),
                                    }])
            df_Out = pd.concat([df_Out, df_new_row], ignore_index=True)
    
    
    
    bar.empty()
    
    return df_Out, df_FastLog
    

df_Out, df_FastLog = posttreatmyvin(uploaded_file, df_FastLog, df_Trips, df_TripInfo, optionVIN)

df_Out = df_Out[(df_Out.Distance > 5)]



##################################################### Plot All vs date
fig100 = px.scatter(df_Out, x=df_Out.DateTrajet, y=df_Out.columns,title="Suivi du viellissement :")
st.plotly_chart(fig100, use_container_width=True)     
 

#
#
#
###################################################### Plot Resistance batterie
#st.plotly_chart(px.scatter(df_Out, x=df_Out.TempeBat, y=df_Out.ResistanceBat), use_container_width=True)
#
#
#
##################################################### Plot Capa locale batterie
#SoC = np.arange(30.0, 90.1, 5)
#CapaDech = np.array([df_Out.CapaciteBatDecharge30.mean(),
#                     df_Out.CapaciteBatDecharge35.mean(),
#                     df_Out.CapaciteBatDecharge40.mean(),
#                     df_Out.CapaciteBatDecharge45.mean(),
#                     df_Out.CapaciteBatDecharge50.mean(),
#                     df_Out.CapaciteBatDecharge55.mean(),
#                     df_Out.CapaciteBatDecharge60.mean(),
#                     df_Out.CapaciteBatDecharge65.mean(),
#                     df_Out.CapaciteBatDecharge70.mean(),
#                     df_Out.CapaciteBatDecharge75.mean(),
#                     df_Out.CapaciteBatDecharge80.mean(),
#                     df_Out.CapaciteBatDecharge85.mean(),
#                     df_Out.CapaciteBatDecharge90.mean()])
#fig101 = px.line(x=SoC, y=CapaDech,labels={
#                     "x": "SoC [%]",
#                     "y": "Capacité Batterie [A.h]"},
#                     range_y=[0, 1.1*np.max(CapaDech)],
#                     title="Capacité locale de la batterie :")
#st.plotly_chart(fig101, use_container_width=True)
#
#
#
#
#
#
#
##################################################### Heatmap avec Slider
#'''
#
### Bi-histogramme
#'''
#
#Sat = st.slider('Saturation couleur', 0.0, 1.0, 0.5)
#fig200 = px.density_heatmap(df, x=df.ACCELERATOR, y=df.PuissanceElec_kW)
#fig200.update_traces(histnorm = "percent")
#fig200.update_layout(
#    {
#        "coloraxis_cmin": 0,
#        "coloraxis_cmax": Sat,
#    }
#)
#st.plotly_chart(fig200, use_container_width=True)




