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
    df_TripInfo = df_TripInfo[(df_Trips.NKMS >5)]
    df_Trips = df_Trips[(df_Trips.NKMS >5)]
    
    # On rajoute des channels
    df_FastLog["Time_S"] = np.nan
    df_FastLog["diffTime_S"] = np.nan
    df_FastLog["PuissanceElec_kW"] = np.nan
    df_FastLog["Energy_Ah"] = np.nan
    df_FastLog["diffSOC"] = np.nan
    df_FastLog["NewSOC"] = np.nan
    df_FastLog["NewEnergy"] = np.nan
    df_FastLog["diffNewEnergy"] = np.nan
    df_FastLog["diffNewSOC"] = np.nan
    #df_FastLog["CapaBat"] = np.nan
    df_FastLog["CapaBatDecharge"] = np.nan
    df_FastLog["CapaBatCharge"] = np.nan
    #df_FastLog["CapaBat30"] = np.nan
    df_FastLog["CapaBatDecharge30"] = np.nan
    df_FastLog["CapaBatCharge30"] = np.nan
    #df_FastLog["CapaBat35"] = np.nan
    df_FastLog["CapaBatDecharge35"] = np.nan
    df_FastLog["CapaBatCharge35"] = np.nan
    #df_FastLog["CapaBat40"] = np.nan
    df_FastLog["CapaBatDecharge40"] = np.nan
    df_FastLog["CapaBatCharge40"] = np.nan
    #df_FastLog["CapaBat45"] = np.nan
    df_FastLog["CapaBatDecharge45"] = np.nan
    df_FastLog["CapaBatCharge45"] = np.nan
    #df_FastLog["CapaBat50"] = np.nan
    df_FastLog["CapaBatDecharge50"] = np.nan
    df_FastLog["CapaBatCharge50"] = np.nan
    #df_FastLog["CapaBat55"] = np.nan
    df_FastLog["CapaBatDecharge55"] = np.nan
    df_FastLog["CapaBatCharge55"] = np.nan
    #df_FastLog["CapaBat60"] = np.nan
    df_FastLog["CapaBatDecharge60"] = np.nan
    df_FastLog["CapaBatCharge60"] = np.nan
    #df_FastLog["CapaBat65"] = np.nan
    df_FastLog["CapaBatDecharge65"] = np.nan
    df_FastLog["CapaBatCharge65"] = np.nan
    #df_FastLog["CapaBat70"] = np.nan
    df_FastLog["CapaBatDecharge70"] = np.nan
    df_FastLog["CapaBatCharge70"] = np.nan
    #df_FastLog["CapaBat75"] = np.nan
    df_FastLog["CapaBatDecharge75"] = np.nan
    df_FastLog["CapaBatCharge75"] = np.nan
    #df_FastLog["CapaBat80"] = np.nan
    df_FastLog["CapaBatDecharge80"] = np.nan
    df_FastLog["CapaBatCharge80"] = np.nan
    #df_FastLog["CapaBat85"] = np.nan
    df_FastLog["CapaBatDecharge85"] = np.nan
    df_FastLog["CapaBatCharge85"] = np.nan
    #df_FastLog["CapaBat90"] = np.nan
    df_FastLog["CapaBatDecharge90"] = np.nan
    df_FastLog["CapaBatCharge90"] = np.nan
    #df_FastLog["CapaBat95"] = np.nan
    df_FastLog["CapaBatDecharge95"] = np.nan
    df_FastLog["CapaBatCharge95"] = np.nan
    
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
                                     'CapaciteBatDecharge90',
                                     'CapaciteBatCharge95',
                                     'CapaciteBatDecharge95'],
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
            df_FastLog.loc[idx,"Time_S"] = (df_FastLog[idx].TIMESTAMP - df_FastLog[idx].TIMESTAMP.min()) / 1000
            df_FastLog.loc[idx,"diffTime_S"] = np.concatenate((np.array([0]),np.diff(df_FastLog[idx].Time_S)))
            df_FastLog.loc[idx,"PuissanceElec_kW"] = df_FastLog[idx].HV_A * df_FastLog[idx].HV_V / 1000
            df_FastLog.loc[idx,"Energy_Ah"] = np.cumsum(df_FastLog[idx].HV_A * df_FastLog[idx].diffTime_S / 3600)
            #df_FastLog[idx].EnergyPos = np.cumsum(df_FastLog[idx].HV_A[(df_FastLog[idx].HV_A > 0)] * df_FastLog[idx].diffTime_S[(df_FastLog[idx].HV_A > 0)] / 3600)
            #df_FastLog[idx].EnergyNeg = np.cumsum(df_FastLog[idx].HV_A[(df_FastLog[idx].HV_A < 0)] * df_FastLog[idx].diffTime_S[(df_FastLog[idx].HV_A < 0)] / 3600)
            df_FastLog.loc[idx,"diffSOC"] = np.concatenate((np.array([0]),np.diff(df_FastLog[idx].SOC)))
            
            ## On identifie la resistance moyenne de la batterie sur le trajet
            p = np.polyfit(np.diff(df_FastLog[idx].HV_A), np.diff(df_FastLog[idx].HV_V), 1)
            BatResistance = -p[0]
            #df_FastLog[idx].HV_V_cor = df_FastLog[idx].HV_V + BatResistance*df_FastLog[idx].HV_A
            
            
            
#
            #
            #
            #def EstimVolt(Bat_Capa, Bat_Rend, Bat_Res, Bat_VoltT, df_T):
            #    df_FastLog[idx].HV_A_corr = df_FastLog[idx].HV_A
            #    df_FastLog[idx].HV_A_corr[(df_FastLog[idx].HV_A < 0)] = df_FastLog[idx].HV_A_corr[(df_FastLog[idx].HV_A_corr < 0)] * Bat_Rend
            #    df_FastLog[idx].EnergyCor = np.cumsum(df_FastLog[idx].HV_A_corr * df_FastLog[idx].diffTime_S / 3600)
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
            #df_FastLog[idx].HV_A_corr = df_FastLog[idx].HV_A
            #df_FastLog[idx].HV_A_corr[(df_FastLog[idx].HV_A < 0)] = df_FastLog[idx].HV_A_corr[(df_FastLog[idx].HV_A_corr < 0)] * Bat_Rend
            #df_FastLog[idx].EnergyCor = np.cumsum(df_FastLog[idx].HV_A_corr * df_FastLog[idx].diffTime_S / 3600)
            #
            #df_FastLog[idx].SoCestim = df_FastLog[idx].SOC.iloc[0] - 100*df_FastLog[idx].EnergyCor/Bat_Capa
            #df_FastLog[idx].VoltageEstim = df_FastLog[idx].HV_V.iloc[0] - 100*df_FastLog[idx].EnergyCor/Bat_Capa  - df_FastLog[idx].HV_A*Bat_Res
            
            
            
            
    
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
            idx2 = (df_FastLog.TIMESTAMP > idxDeb) & (df_FastLog.TIMESTAMP < idxFin) & (df_FastLog.diffSOC!=0)
            df_FastLog.loc[idx2,"NewSOC"] = df_FastLog.loc[idx2].SOC
            df_FastLog.loc[idx2,"NewEnergy"] = df_FastLog.loc[idx2].Energy_Ah
            df_FastLog.loc[idx2,"diffNewEnergy"] = np.concatenate((np.array([np.nan]),np.diff(df_FastLog[idx2].NewEnergy)))
            df_FastLog.loc[idx2,"diffNewSOC"] = np.concatenate((np.array([np.nan]),np.diff(df_FastLog[idx2].NewSOC)))
            
            idx21 = (df_FastLog.TIMESTAMP > idxDeb) & (df_FastLog.TIMESTAMP < idxFin) & (df_FastLog.diffSOC!=0) & (df_FastLog.diffNewSOC < 0 )
            idx22 = (df_FastLog.TIMESTAMP > idxDeb) & (df_FastLog.TIMESTAMP < idxFin) & (df_FastLog.diffSOC!=0) & (df_FastLog.diffNewSOC > 0 )
            
            df_FastLog.loc[idx21,"CapaBatDecharge"] = -100*df_FastLog[idx2].diffNewEnergy/df_FastLog[idx21].diffNewSOC
            df_FastLog.loc[idx22,"CapaBatCharge"] = -100*df_FastLog[idx2].diffNewEnergy/df_FastLog[idx22].diffNewSOC
            
            DeltaSOC = 3
            
            #idx3 = (df_FastLog.TIMESTAMP > idxDeb) & (df_FastLog.TIMESTAMP < idxFin) & (df_FastLog.diffSOC!=0) & (np.abs(df_FastLog.SOC - 30) < DeltaSOC)
            idx4 = (df_FastLog.TIMESTAMP > idxDeb) & (df_FastLog.TIMESTAMP < idxFin) & (df_FastLog.diffSOC!=0) & (np.abs(df_FastLog.SOC - 30) < DeltaSOC) & (df_FastLog.diffNewSOC < 0 )
            idx5 = (df_FastLog.TIMESTAMP > idxDeb) & (df_FastLog.TIMESTAMP < idxFin) & (df_FastLog.diffSOC!=0) & (np.abs(df_FastLog.SOC - 30) < DeltaSOC) & (df_FastLog.diffNewSOC > 0 )
            #df_FastLog.loc[idx3,"CapaBat30"] = -100*df_FastLog[idx3].diffNewEnergy/df_FastLog[idx3].diffNewSOC
            df_FastLog.loc[idx4,"CapaBatDecharge30"] = -100*df_FastLog[idx4].diffNewEnergy/df_FastLog[idx4].diffNewSOC
            df_FastLog.loc[idx5,"CapaBatCharge30"] = -100*df_FastLog[idx5].diffNewEnergy/df_FastLog[idx5].diffNewSOC
            
            #idx3 = (df_FastLog.TIMESTAMP > idxDeb) & (df_FastLog.TIMESTAMP < idxFin) & (df_FastLog.diffSOC!=0) & (np.abs(df_FastLog.SOC - 35) < DeltaSOC)
            idx4 = (df_FastLog.TIMESTAMP > idxDeb) & (df_FastLog.TIMESTAMP < idxFin) & (df_FastLog.diffSOC!=0) & (np.abs(df_FastLog.SOC - 35) < DeltaSOC) & (df_FastLog.diffNewSOC < 0 )
            idx5 = (df_FastLog.TIMESTAMP > idxDeb) & (df_FastLog.TIMESTAMP < idxFin) & (df_FastLog.diffSOC!=0) & (np.abs(df_FastLog.SOC - 35) < DeltaSOC) & (df_FastLog.diffNewSOC > 0 )
            #df_FastLog.loc[idx3,"CapaBat35"] = -100*df_FastLog[idx3].diffNewEnergy/df_FastLog[idx3].diffNewSOC
            df_FastLog.loc[idx4,"CapaBatDecharge35"] = -100*df_FastLog[idx4].diffNewEnergy/df_FastLog[idx4].diffNewSOC
            df_FastLog.loc[idx5,"CapaBatCharge35"] = -100*df_FastLog[idx5].diffNewEnergy/df_FastLog[idx5].diffNewSOC
            
            #idx3 = (df_FastLog.TIMESTAMP > idxDeb) & (df_FastLog.TIMESTAMP < idxFin) & (df_FastLog.diffSOC!=0) & (np.abs(df_FastLog.SOC - 40) < DeltaSOC)
            idx4 = (df_FastLog.TIMESTAMP > idxDeb) & (df_FastLog.TIMESTAMP < idxFin) & (df_FastLog.diffSOC!=0) & (np.abs(df_FastLog.SOC - 40) < DeltaSOC) & (df_FastLog.diffNewSOC < 0 )
            idx5 = (df_FastLog.TIMESTAMP > idxDeb) & (df_FastLog.TIMESTAMP < idxFin) & (df_FastLog.diffSOC!=0) & (np.abs(df_FastLog.SOC - 40) < DeltaSOC) & (df_FastLog.diffNewSOC > 0 )
            #df_FastLog.loc[idx3,"CapaBat40"] = -100*df_FastLog[idx3].diffNewEnergy/df_FastLog[idx3].diffNewSOC
            df_FastLog.loc[idx4,"CapaBatDecharge40"] = -100*df_FastLog[idx4].diffNewEnergy/df_FastLog[idx4].diffNewSOC
            df_FastLog.loc[idx5,"CapaBatCharge40"] = -100*df_FastLog[idx5].diffNewEnergy/df_FastLog[idx5].diffNewSOC
            
            #idx3 = (df_FastLog.TIMESTAMP > idxDeb) & (df_FastLog.TIMESTAMP < idxFin) & (df_FastLog.diffSOC!=0) & (np.abs(df_FastLog.SOC - 45) < DeltaSOC)
            idx4 = (df_FastLog.TIMESTAMP > idxDeb) & (df_FastLog.TIMESTAMP < idxFin) & (df_FastLog.diffSOC!=0) & (np.abs(df_FastLog.SOC - 45) < DeltaSOC) & (df_FastLog.diffNewSOC < 0 )
            idx5 = (df_FastLog.TIMESTAMP > idxDeb) & (df_FastLog.TIMESTAMP < idxFin) & (df_FastLog.diffSOC!=0) & (np.abs(df_FastLog.SOC - 45) < DeltaSOC) & (df_FastLog.diffNewSOC > 0 )
            #df_FastLog.loc[idx3,"CapaBat45"] = -100*df_FastLog[idx3].diffNewEnergy/df_FastLog[idx3].diffNewSOC
            df_FastLog.loc[idx4,"CapaBatDecharge45"] = -100*df_FastLog[idx4].diffNewEnergy/df_FastLog[idx4].diffNewSOC
            df_FastLog.loc[idx5,"CapaBatCharge45"] = -100*df_FastLog[idx5].diffNewEnergy/df_FastLog[idx5].diffNewSOC
            
            #idx3 = (df_FastLog.TIMESTAMP > idxDeb) & (df_FastLog.TIMESTAMP < idxFin) & (df_FastLog.diffSOC!=0) & (np.abs(df_FastLog.SOC - 50) < DeltaSOC)
            idx4 = (df_FastLog.TIMESTAMP > idxDeb) & (df_FastLog.TIMESTAMP < idxFin) & (df_FastLog.diffSOC!=0) & (np.abs(df_FastLog.SOC - 50) < DeltaSOC) & (df_FastLog.diffNewSOC < 0 )
            idx5 = (df_FastLog.TIMESTAMP > idxDeb) & (df_FastLog.TIMESTAMP < idxFin) & (df_FastLog.diffSOC!=0) & (np.abs(df_FastLog.SOC - 50) < DeltaSOC) & (df_FastLog.diffNewSOC > 0 )
            #df_FastLog.loc[idx3,"CapaBat50"] = -100*df_FastLog[idx3].diffNewEnergy/df_FastLog[idx3].diffNewSOC
            df_FastLog.loc[idx4,"CapaBatDecharge50"] = -100*df_FastLog[idx4].diffNewEnergy/df_FastLog[idx4].diffNewSOC
            df_FastLog.loc[idx5,"CapaBatCharge50"] = -100*df_FastLog[idx5].diffNewEnergy/df_FastLog[idx5].diffNewSOC
            
            #idx3 = (df_FastLog.TIMESTAMP > idxDeb) & (df_FastLog.TIMESTAMP < idxFin) & (df_FastLog.diffSOC!=0) & (np.abs(df_FastLog.SOC - 55) < DeltaSOC)
            idx4 = (df_FastLog.TIMESTAMP > idxDeb) & (df_FastLog.TIMESTAMP < idxFin) & (df_FastLog.diffSOC!=0) & (np.abs(df_FastLog.SOC - 55) < DeltaSOC) & (df_FastLog.diffNewSOC < 0 )
            idx5 = (df_FastLog.TIMESTAMP > idxDeb) & (df_FastLog.TIMESTAMP < idxFin) & (df_FastLog.diffSOC!=0) & (np.abs(df_FastLog.SOC - 55) < DeltaSOC) & (df_FastLog.diffNewSOC > 0 )
            #df_FastLog.loc[idx3,"CapaBat55"] = -100*df_FastLog[idx3].diffNewEnergy/df_FastLog[idx3].diffNewSOC
            df_FastLog.loc[idx4,"CapaBatDecharge55"] = -100*df_FastLog[idx4].diffNewEnergy/df_FastLog[idx4].diffNewSOC
            df_FastLog.loc[idx5,"CapaBatCharge55"] = -100*df_FastLog[idx5].diffNewEnergy/df_FastLog[idx5].diffNewSOC
                        
            #idx3 = (df_FastLog.TIMESTAMP > idxDeb) & (df_FastLog.TIMESTAMP < idxFin) & (df_FastLog.diffSOC!=0) & (np.abs(df_FastLog.SOC - 60) < DeltaSOC)
            idx4 = (df_FastLog.TIMESTAMP > idxDeb) & (df_FastLog.TIMESTAMP < idxFin) & (df_FastLog.diffSOC!=0) & (np.abs(df_FastLog.SOC - 60) < DeltaSOC) & (df_FastLog.diffNewSOC < 0 )
            idx5 = (df_FastLog.TIMESTAMP > idxDeb) & (df_FastLog.TIMESTAMP < idxFin) & (df_FastLog.diffSOC!=0) & (np.abs(df_FastLog.SOC - 60) < DeltaSOC) & (df_FastLog.diffNewSOC > 0 )
            #df_FastLog.loc[idx3,"CapaBat60"] = -100*df_FastLog[idx3].diffNewEnergy/df_FastLog[idx3].diffNewSOC
            df_FastLog.loc[idx4,"CapaBatDecharge60"] = -100*df_FastLog[idx4].diffNewEnergy/df_FastLog[idx4].diffNewSOC
            df_FastLog.loc[idx5,"CapaBatCharge60"] = -100*df_FastLog[idx5].diffNewEnergy/df_FastLog[idx5].diffNewSOC
            
            #idx3 = (df_FastLog.TIMESTAMP > idxDeb) & (df_FastLog.TIMESTAMP < idxFin) & (df_FastLog.diffSOC!=0) & (np.abs(df_FastLog.SOC - 65) < DeltaSOC)
            idx4 = (df_FastLog.TIMESTAMP > idxDeb) & (df_FastLog.TIMESTAMP < idxFin) & (df_FastLog.diffSOC!=0) & (np.abs(df_FastLog.SOC - 65) < DeltaSOC) & (df_FastLog.diffNewSOC < 0 )
            idx5 = (df_FastLog.TIMESTAMP > idxDeb) & (df_FastLog.TIMESTAMP < idxFin) & (df_FastLog.diffSOC!=0) & (np.abs(df_FastLog.SOC - 65) < DeltaSOC) & (df_FastLog.diffNewSOC > 0 )
            #df_FastLog.loc[idx3,"CapaBat65"] = -100*df_FastLog[idx3].diffNewEnergy/df_FastLog[idx3].diffNewSOC
            df_FastLog.loc[idx4,"CapaBatDecharge65"] = -100*df_FastLog[idx4].diffNewEnergy/df_FastLog[idx4].diffNewSOC
            df_FastLog.loc[idx5,"CapaBatCharge65"] = -100*df_FastLog[idx5].diffNewEnergy/df_FastLog[idx5].diffNewSOC
            
            #idx3 = (df_FastLog.TIMESTAMP > idxDeb) & (df_FastLog.TIMESTAMP < idxFin) & (df_FastLog.diffSOC!=0) & (np.abs(df_FastLog.SOC - 70) < DeltaSOC)
            idx4 = (df_FastLog.TIMESTAMP > idxDeb) & (df_FastLog.TIMESTAMP < idxFin) & (df_FastLog.diffSOC!=0) & (np.abs(df_FastLog.SOC - 70) < DeltaSOC) & (df_FastLog.diffNewSOC < 0 )
            idx5 = (df_FastLog.TIMESTAMP > idxDeb) & (df_FastLog.TIMESTAMP < idxFin) & (df_FastLog.diffSOC!=0) & (np.abs(df_FastLog.SOC - 70) < DeltaSOC) & (df_FastLog.diffNewSOC > 0 )
            #df_FastLog.loc[idx3,"CapaBat70"] = -100*df_FastLog[idx3].diffNewEnergy/df_FastLog[idx3].diffNewSOC
            df_FastLog.loc[idx4,"CapaBatDecharge70"] = -100*df_FastLog[idx4].diffNewEnergy/df_FastLog[idx4].diffNewSOC
            df_FastLog.loc[idx5,"CapaBatCharge70"] = -100*df_FastLog[idx5].diffNewEnergy/df_FastLog[idx5].diffNewSOC
            
            #idx3 = (df_FastLog.TIMESTAMP > idxDeb) & (df_FastLog.TIMESTAMP < idxFin) & (df_FastLog.diffSOC!=0) & (np.abs(df_FastLog.SOC - 75) < DeltaSOC)
            idx4 = (df_FastLog.TIMESTAMP > idxDeb) & (df_FastLog.TIMESTAMP < idxFin) & (df_FastLog.diffSOC!=0) & (np.abs(df_FastLog.SOC - 75) < DeltaSOC) & (df_FastLog.diffNewSOC < 0 )
            idx5 = (df_FastLog.TIMESTAMP > idxDeb) & (df_FastLog.TIMESTAMP < idxFin) & (df_FastLog.diffSOC!=0) & (np.abs(df_FastLog.SOC - 75) < DeltaSOC) & (df_FastLog.diffNewSOC > 0 )
            #df_FastLog.loc[idx3,"CapaBat75"] = -100*df_FastLog[idx3].diffNewEnergy/df_FastLog[idx3].diffNewSOC
            df_FastLog.loc[idx4,"CapaBatDecharge75"] = -100*df_FastLog[idx4].diffNewEnergy/df_FastLog[idx4].diffNewSOC
            df_FastLog.loc[idx5,"CapaBatCharge75"] = -100*df_FastLog[idx5].diffNewEnergy/df_FastLog[idx5].diffNewSOC
            
            #idx3 = (df_FastLog.TIMESTAMP > idxDeb) & (df_FastLog.TIMESTAMP < idxFin) & (df_FastLog.diffSOC!=0) & (np.abs(df_FastLog.SOC - 80) < DeltaSOC)
            idx4 = (df_FastLog.TIMESTAMP > idxDeb) & (df_FastLog.TIMESTAMP < idxFin) & (df_FastLog.diffSOC!=0) & (np.abs(df_FastLog.SOC - 80) < DeltaSOC) & (df_FastLog.diffNewSOC < 0 )
            idx5 = (df_FastLog.TIMESTAMP > idxDeb) & (df_FastLog.TIMESTAMP < idxFin) & (df_FastLog.diffSOC!=0) & (np.abs(df_FastLog.SOC - 80) < DeltaSOC) & (df_FastLog.diffNewSOC > 0 )
            #df_FastLog.loc[idx3,"CapaBat80"] = -100*df_FastLog[idx3].diffNewEnergy/df_FastLog[idx3].diffNewSOC
            df_FastLog.loc[idx4,"CapaBatDecharge80"] = -100*df_FastLog[idx4].diffNewEnergy/df_FastLog[idx4].diffNewSOC
            df_FastLog.loc[idx5,"CapaBatCharge80"] = -100*df_FastLog[idx5].diffNewEnergy/df_FastLog[idx5].diffNewSOC
            
            #idx3 = (df_FastLog.TIMESTAMP > idxDeb) & (df_FastLog.TIMESTAMP < idxFin) & (df_FastLog.diffSOC!=0) & (np.abs(df_FastLog.SOC - 85) < DeltaSOC)
            idx4 = (df_FastLog.TIMESTAMP > idxDeb) & (df_FastLog.TIMESTAMP < idxFin) & (df_FastLog.diffSOC!=0) & (np.abs(df_FastLog.SOC - 85) < DeltaSOC) & (df_FastLog.diffNewSOC < 0 )
            idx5 = (df_FastLog.TIMESTAMP > idxDeb) & (df_FastLog.TIMESTAMP < idxFin) & (df_FastLog.diffSOC!=0) & (np.abs(df_FastLog.SOC - 85) < DeltaSOC) & (df_FastLog.diffNewSOC > 0 )
            #df_FastLog.loc[idx3,"CapaBat85"] = -100*df_FastLog[idx3].diffNewEnergy/df_FastLog[idx3].diffNewSOC
            df_FastLog.loc[idx4,"CapaBatDecharge85"] = -100*df_FastLog[idx4].diffNewEnergy/df_FastLog[idx4].diffNewSOC
            df_FastLog.loc[idx5,"CapaBatCharge85"] = -100*df_FastLog[idx5].diffNewEnergy/df_FastLog[idx5].diffNewSOC
            
            #idx3 = (df_FastLog.TIMESTAMP > idxDeb) & (df_FastLog.TIMESTAMP < idxFin) & (df_FastLog.diffSOC!=0) & (np.abs(df_FastLog.SOC - 90) < DeltaSOC)
            idx4 = (df_FastLog.TIMESTAMP > idxDeb) & (df_FastLog.TIMESTAMP < idxFin) & (df_FastLog.diffSOC!=0) & (np.abs(df_FastLog.SOC - 90) < DeltaSOC) & (df_FastLog.diffNewSOC < 0 )
            idx5 = (df_FastLog.TIMESTAMP > idxDeb) & (df_FastLog.TIMESTAMP < idxFin) & (df_FastLog.diffSOC!=0) & (np.abs(df_FastLog.SOC - 90) < DeltaSOC) & (df_FastLog.diffNewSOC > 0 )
            #df_FastLog.loc[idx3,"CapaBat90"] = -100*df_FastLog[idx3].diffNewEnergy/df_FastLog[idx3].diffNewSOC
            df_FastLog.loc[idx4,"CapaBatDecharge90"] = -100*df_FastLog[idx4].diffNewEnergy/df_FastLog[idx4].diffNewSOC
            df_FastLog.loc[idx5,"CapaBatCharge90"] = -100*df_FastLog[idx5].diffNewEnergy/df_FastLog[idx5].diffNewSOC
            
            #idx3 = (df_FastLog.TIMESTAMP > idxDeb) & (df_FastLog.TIMESTAMP < idxFin) & (df_FastLog.diffSOC!=0) & (np.abs(df_FastLog.SOC - 95) < DeltaSOC)
            idx4 = (df_FastLog.TIMESTAMP > idxDeb) & (df_FastLog.TIMESTAMP < idxFin) & (df_FastLog.diffSOC!=0) & (np.abs(df_FastLog.SOC - 95) < DeltaSOC) & (df_FastLog.diffNewSOC < 0 )
            idx5 = (df_FastLog.TIMESTAMP > idxDeb) & (df_FastLog.TIMESTAMP < idxFin) & (df_FastLog.diffSOC!=0) & (np.abs(df_FastLog.SOC - 95) < DeltaSOC) & (df_FastLog.diffNewSOC > 0 )
            #df_FastLog.loc[idx3,"CapaBat95"] = -100*df_FastLog[idx3].diffNewEnergy/df_FastLog[idx3].diffNewSOC
            df_FastLog.loc[idx4,"CapaBatDecharge95"] = -100*df_FastLog[idx4].diffNewEnergy/df_FastLog[idx4].diffNewSOC
            df_FastLog.loc[idx5,"CapaBatCharge95"] = -100*df_FastLog[idx5].diffNewEnergy/df_FastLog[idx5].diffNewSOC
            
            
            fig1 = px.scatter(df_FastLog[idx], x=df_FastLog[idx].index, y=df_FastLog[idx].columns)
            st.plotly_chart(fig1, use_container_width=True) 
        
            fig2 = px.scatter(df_FastLog[idx], x=df_FastLog[idx].SOC, y=df_FastLog[idx].columns)
            st.plotly_chart(fig2, use_container_width=True) 
            
    
            df_new_row = pd.DataFrame.from_records([{'Distance' : df_Trips.at[ii,"NKMS"],
                                    'Kilometrage' : MeanKilometrage,
                                    'DateTrajet' : TrajDate,
                                    'VitesseMoyenne_kph' : MeanSpeed,
                                    'VitesseMoyenneEnRoulage_kph' : MeanRollingSpeed,
                                    'Consommation_LAu100km' : TrajConso,
                                    'TempeAmbiante' : MeanAmbiantTemp,
                                    'TempeBat' : MeanBatTemp,
                                    'ResistanceBat' : BatResistance,
                                    #'CapaBat' : df_FastLog[idx].CapaBat.mean(),
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
                                    'CapaciteBatCharge95' : df_FastLog[idx].CapaBatCharge95.mean(),
                                    'CapaciteBatDecharge95' : df_FastLog[idx].CapaBatDecharge95.mean(),
                                    }])
            df_Out = pd.concat([df_Out, df_new_row], ignore_index=True)
    
    
    
    bar.empty()
    
    return df_Out, df_FastLog
    

df_Out, df_FastLog = posttreatmyvin(uploaded_file, df_FastLog, df_Trips, df_TripInfo, optionVIN)




##################################################### Plot All vs date
fig100 = px.scatter(df_Out, x=df_Out.DateTrajet, y=df_Out.columns,title="Suivi du viellissement :")
st.plotly_chart(fig100, use_container_width=True)     
 




##################################################### Plot Resistance batterie
st.plotly_chart(px.scatter(df_Out, x=df_Out.TempeBat, y=df_Out.ResistanceBat), use_container_width=True)



#################################################### Plot Capa locale batterie
SoC = np.arange(30.0, 90.1, 5)
CapaDech = np.array([df_Out.CapaciteBatDecharge30.mean(),
                     df_Out.CapaciteBatDecharge35.mean(),
                     df_Out.CapaciteBatDecharge40.mean(),
                     df_Out.CapaciteBatDecharge45.mean(),
                     df_Out.CapaciteBatDecharge50.mean(),
                     df_Out.CapaciteBatDecharge55.mean(),
                     df_Out.CapaciteBatDecharge60.mean(),
                     df_Out.CapaciteBatDecharge65.mean(),
                     df_Out.CapaciteBatDecharge70.mean(),
                     df_Out.CapaciteBatDecharge75.mean(),
                     df_Out.CapaciteBatDecharge80.mean(),
                     df_Out.CapaciteBatDecharge85.mean(),
                     df_Out.CapaciteBatDecharge90.mean()])
fig101 = px.line(x=SoC, y=CapaDech,labels={
                     "x": "SoC [%]",
                     "y": "Capacité Batterie [A.h]"},
                     range_y=[0, 1.1*np.max(CapaDech)],
                     title="Capacité locale de la batterie :")
st.plotly_chart(fig101, use_container_width=True)







#################################################### Heatmap avec Slider
'''

## Bi-histogramme
'''

Sat = st.slider('Saturation couleur', 0.0, 1.0, 0.5)
fig200 = px.density_heatmap(df_FastLog, x=df_FastLog.ACCELERATOR, y=df_FastLog.PuissanceElec_kW)
fig200.update_traces(histnorm = "percent")
fig200.update_layout(
    {
        "coloraxis_cmin": 0,
        "coloraxis_cmax": Sat,
    }
)
st.plotly_chart(fig200, use_container_width=True)




