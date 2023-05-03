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
    df_TripInfo_MyVIN = df_TripInfo[(df_TripInfo.VIN == optionVIN)]
    df_Trips_MyVIN = df_Trips[(df_TripInfo.VIN == optionVIN)]
    
    # On garde seulement les trajet de plus d'un km (on écrase les ancien dataframe))
    df_Trips = df_Trips_MyVIN[(df_Trips_MyVIN.NKMS >5)]
    df_TripInfo = df_TripInfo_MyVIN[(df_Trips_MyVIN.NKMS >5)]
    
    # On supprime les df dont on n'a plus besoin
    del df_Trips_MyVIN, df_TripInfo_MyVIN
    
    # On créé un dataframe qui contient les sorties
    tt=1
    df_Out = pd.DataFrame(columns = ['Distance',
                                     'Kilometrage',
                                     'DateTrajet',
                                     'VitesseMoyenne_kph',
                                     'VitesseMoyenneEnRoulage_kph',
                                     'Consommation_LAu100km',
                                     'TempeAmbiante',
                                     'TempeBat',
                                     'ResistanceBat',
                                     'ResistanceBat2',
                                     'CapaBat2',
                                     'RendBat2',
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
    
    # On construit un df avec les données de roulage du véhicule choisi sur les trajet de plus d'un km
    for ii in df_Trips.index:
        idxDeb = df_Trips.at[ii,"TSDEB"]
        idxFin = df_Trips.at[ii,"TSFIN"]
    
        df_T = df_FastLog[(df_FastLog.TIMESTAMP > idxDeb) & (df_FastLog.TIMESTAMP < idxFin)].copy()
    
        # On garde les point ou on a le signal de tension batterie HV
        max_Voltage_idx = df_T.HV_V.idxmax()
        max_Voltage = df_T.HV_V.loc[max_Voltage_idx]
    
    
        if max_Voltage>1:

            # On rajoute des channels
            df_T["Time_S"] = (df_T.TIMESTAMP.copy() - df_T.TIMESTAMP.min()) / 1000
            df_T["diffTime_S"] = np.concatenate((np.array([0]),np.diff(df_T.Time_S.copy())))
            df_T["PuissanceElec_kW"] = df_T.HV_A.copy() * df_T.HV_V.copy() / 1000
            df_T["Energy"] = np.cumsum(df_T.HV_A.copy() * df_T.diffTime_S.copy() / 3600)
            df_T["EnergyPos"] = np.cumsum(df_T.HV_A[(df_T.HV_A > 0)] * df_T.diffTime_S[(df_T.HV_A > 0)] / 3600)
            df_T["EnergyNeg"] = np.cumsum(df_T.HV_A[(df_T.HV_A < 0)] * df_T.diffTime_S[(df_T.HV_A < 0)] / 3600)
            df_T["diffSOC"] = np.concatenate((np.array([0]),np.diff(df_T.SOC.copy())))
            
            ## On identifie la resistance moyenne de la batterie sur le trajet
            p = np.polyfit(np.diff(df_T.HV_A.copy()), np.diff(df_T.HV_V.copy()), 1)
            BatResistance = -p[0]
            #df_T["HV_V_cor"] = df_T.HV_V + BatResistance*df_T.HV_A
            
            
            

            
            
            def EstimVolt(Bat_Capa, Bat_Rend, Bat_Res, Bat_VoltT, df_T):
                df_T["HV_A_corr"] = df_T.HV_A.copy()
                df_T.HV_A_corr[(df_T.HV_A < 0)] = df_T.HV_A_corr[(df_T.HV_A_corr < 0)] * Bat_Rend
                df_T["EnergyCor"] = np.cumsum(df_T.HV_A_corr.copy() * df_T.diffTime_S.copy() / 3600)
                #return df_T.HV_V.iloc[0] + Bat_VoltT*(df_T.BATTERY_TEMP - df_T.BATTERY_TEMP.iloc[0]) - 100*df_T.EnergyCor/Bat_Capa - df_T.HV_A*(Bat_Res + Bat_ResT*(df_T.BATTERY_TEMP - df_T.BATTERY_TEMP.mean()))
                return df_T.HV_V.iloc[0] + Bat_VoltT*(df_T.BATTERY_TEMP - df_T.BATTERY_TEMP.iloc[0]) - 100*df_T.EnergyCor/Bat_Capa - df_T.HV_A*(Bat_Res)
            
            def residuals(params, x, y):
                Bat_Capa, Bat_Rend, Bat_Res, Bat_VoltT = params
                return EstimVolt(Bat_Capa, Bat_Rend, Bat_Res, Bat_VoltT, df_T) - y
            
            params_ini = [5.0, 0.95, 0.1, 1.0]
            result = least_squares(residuals, params_ini, args=(df_T, df_T.HV_V))
            
            Bat_Capa, Bat_Rend, Bat_Res, Bat_VoltT = result.x
            
            df_T["HV_A_corr"] = df_T.HV_A.copy()
            df_T.HV_A_corr[(df_T.HV_A < 0)] = df_T.HV_A_corr[(df_T.HV_A_corr < 0)] * Bat_Rend
            df_T["EnergyCor"] = np.cumsum(df_T.HV_A_corr.copy() * df_T.diffTime_S.copy() / 3600)
            
            df_T["SoCestim"] = df_T.SOC.iloc[0] - 100*df_T.EnergyCor/Bat_Capa
            df_T["VoltageEstim"] = df_T.HV_V.iloc[0] - 100*df_T.EnergyCor/Bat_Capa  - df_T.HV_A*Bat_Res
            
            
            #fig1 = px.scatter(df_T, x=df_T.index, y=df_T.columns)
            #st.plotly_chart(fig1, use_container_width=True) 
            
            #st.plotly_chart(px.scatter(df_T, x=df_T.SoCestim, y=df_T.columns), use_container_width=True) 
            
    
            # On récupère les infos générales sur le trajet
            MeanAmbiantTemp = df_T.AMBIENT_TEMP.mean()
            MeanBatTemp = df_T.BATTERY_TEMP.mean()
            MeanKilometrage = df_T.ODO.mean()
            TrajDate = pd.to_datetime(df_T.TIMESTAMP.mean()/1000, unit='s')
            TrajConso = 0.001*100*df_T.TRIPFUEL.max()/df_T.TRIP_DIST.max()
            MeanSpeed = df_T.SPEED_OBD.mean()
            MeanRollingSpeed = df_T.SPEED_OBD[(df_T.SPEED_OBD >1)].mean()
    
            def moving_avg(x, n):
                cumsum = np.cumsum(np.insert(x, 0, 0)) 
                return (cumsum[n:] - cumsum[:-n]) / float(n)
            
            FenetreMoyMobile = 5
            
            #fig22 = px.line(df_T, x=moving_avg(df_T.Energy,FenetreMoyMobile), y=moving_avg(df_T.HV_V_cor,FenetreMoyMobile))
            #fig22.add_trace(go.Scatter(x=df_T.Energy, y=df_T.HV_V))
            #st.plotly_chart(fig22, use_container_width=True)   
            #fig50 = px.density_heatmap(df_T, x=np.diff(moving_avg(df_T.Energy,FenetreMoyMobile)), y=np.diff(moving_avg(df_T.HV_V_cor,FenetreMoyMobile)))
            #fig50.update_traces(histnorm = "percent")
            #fig50.update_layout(
            #    {
            #        "coloraxis_cmin": 0,
            #        "coloraxis_cmax": 0.2,
            #    }
            #)
            #st.plotly_chart(fig50, use_container_width=True)
            #st.plotly_chart(px.line(df_T, x=df_T.SOC, y=df_T.HV_V_cor), use_container_width=True)   
    
            ## On identifie la capacité de la batterie
            df_T["NewSOC"] = df_T.SOC[(df_T.diffSOC.copy()!=0)].copy()
            df_T["NewEnergy"] = df_T.Energy[(df_T.diffSOC.copy()!=0)].copy()
            df_T["diffNewEnergy"] = df_T.NewEnergy.copy()
            df_T.diffNewEnergy[~np.isnan(df_T.diffNewEnergy.copy())] = np.concatenate((np.array([np.nan]),np.diff(df_T.NewEnergy[~np.isnan(df_T.diffNewEnergy.copy())].copy())))
            df_T["diffNewSOC"] = df_T.NewSOC.copy()
            df_T.diffNewSOC[~np.isnan(df_T.diffNewSOC.copy())] = np.concatenate((np.array([np.nan]),np.diff(df_T.NewSOC[~np.isnan(df_T.diffNewSOC.copy())].copy())))
            df_T["CapaBat"] = -100*df_T.diffNewEnergy.copy()/df_T.diffNewSOC.copy()
            df_T["CapaBatDecharge"] = df_T.CapaBat[(df_T.diffNewSOC.copy()<0)].copy()
            df_T["CapaBatCharge"] = df_T.CapaBat[(df_T.diffNewSOC.copy()>0)].copy()
    
            df_T["CapaBat30"] = -100*df_T.diffNewEnergy[(df_T.NewSOC>25) & (df_T.NewSOC<35)].copy()/df_T.diffNewSOC.copy()
            df_T["CapaBatDecharge30"] = df_T.CapaBat30/df_T.CapaBat30 * np.mean(df_T.CapaBat30[(df_T.diffNewSOC.copy()<0)].copy())
            df_T["CapaBatCharge30"] = df_T.CapaBat30/df_T.CapaBat30 * np.mean(df_T.CapaBat30[(df_T.diffNewSOC.copy()>0)].copy())
    
            df_T["CapaBat35"] = -100*df_T.diffNewEnergy[(df_T.NewSOC>30) & (df_T.NewSOC<40)].copy()/df_T.diffNewSOC.copy()
            df_T["CapaBatDecharge35"] = df_T.CapaBat35/df_T.CapaBat35 * np.mean(df_T.CapaBat35[(df_T.diffNewSOC.copy()<0)].copy())
            df_T["CapaBatCharge35"] = df_T.CapaBat35/df_T.CapaBat35 * np.mean(df_T.CapaBat35[(df_T.diffNewSOC.copy()>0)].copy())
    
            df_T["CapaBat40"] = -100*df_T.diffNewEnergy[(df_T.NewSOC>35) & (df_T.NewSOC<45)].copy()/df_T.diffNewSOC.copy()
            df_T["CapaBatDecharge40"] = df_T.CapaBat40/df_T.CapaBat40 * np.mean(df_T.CapaBat40[(df_T.diffNewSOC.copy()<0)].copy())
            df_T["CapaBatCharge40"] = df_T.CapaBat40/df_T.CapaBat40 * np.mean(df_T.CapaBat40[(df_T.diffNewSOC.copy()>0)].copy())
    
            df_T["CapaBat45"] = -100*df_T.diffNewEnergy[(df_T.NewSOC>40) & (df_T.NewSOC<50)].copy()/df_T.diffNewSOC.copy()
            df_T["CapaBatDecharge45"] = df_T.CapaBat45/df_T.CapaBat45 * np.mean(df_T.CapaBat45[(df_T.diffNewSOC.copy()<0)].copy())
            df_T["CapaBatCharge45"] = df_T.CapaBat45/df_T.CapaBat45 * np.mean(df_T.CapaBat45[(df_T.diffNewSOC.copy()>0)].copy())
    
            df_T["CapaBat50"] = -100*df_T.diffNewEnergy[(df_T.NewSOC>45) & (df_T.NewSOC<55)].copy()/df_T.diffNewSOC.copy()
            df_T["CapaBatDecharge50"] = df_T.CapaBat50/df_T.CapaBat50 * np.mean(df_T.CapaBat50[(df_T.diffNewSOC.copy()<0)].copy())
            df_T["CapaBatCharge50"] = df_T.CapaBat50/df_T.CapaBat50 * np.mean(df_T.CapaBat50[(df_T.diffNewSOC.copy()>0)].copy())
    
            df_T["CapaBat55"] = -100*df_T.diffNewEnergy[(df_T.NewSOC>50) & (df_T.NewSOC<60)].copy()/df_T.diffNewSOC.copy()
            df_T["CapaBatDecharge55"] = df_T.CapaBat55/df_T.CapaBat55 * np.mean(df_T.CapaBat55[(df_T.diffNewSOC.copy()<0)].copy())
            df_T["CapaBatCharge55"] = df_T.CapaBat55/df_T.CapaBat55 * np.mean(df_T.CapaBat55[(df_T.diffNewSOC.copy()>0)].copy())
    
            df_T["CapaBat60"] = -100*df_T.diffNewEnergy[(df_T.NewSOC>55) & (df_T.NewSOC<65)].copy()/df_T.diffNewSOC.copy()
            df_T["CapaBatDecharge60"] = df_T.CapaBat60/df_T.CapaBat60 * np.mean(df_T.CapaBat60[(df_T.diffNewSOC.copy()<0)].copy())
            df_T["CapaBatCharge60"] = df_T.CapaBat60/df_T.CapaBat60 * np.mean(df_T.CapaBat60[(df_T.diffNewSOC.copy()>0)].copy())
    
            df_T["CapaBat65"] = -100*df_T.diffNewEnergy[(df_T.NewSOC>60) & (df_T.NewSOC<70)].copy()/df_T.diffNewSOC.copy()
            df_T["CapaBatDecharge65"] = df_T.CapaBat65/df_T.CapaBat65 * np.mean(df_T.CapaBat65[(df_T.diffNewSOC.copy()<0)].copy())
            df_T["CapaBatCharge65"] = df_T.CapaBat65/df_T.CapaBat65 * np.mean(df_T.CapaBat65[(df_T.diffNewSOC.copy()>0)].copy())
    
            df_T["CapaBat70"] = -100*df_T.diffNewEnergy[(df_T.NewSOC>65) & (df_T.NewSOC<75)].copy()/df_T.diffNewSOC.copy()
            df_T["CapaBatDecharge70"] = df_T.CapaBat70/df_T.CapaBat70 * np.mean(df_T.CapaBat70[(df_T.diffNewSOC.copy()<0)].copy())
            df_T["CapaBatCharge70"] = df_T.CapaBat70/df_T.CapaBat70 * np.mean(df_T.CapaBat70[(df_T.diffNewSOC.copy()>0)].copy())
    
            df_T["CapaBat75"] = -100*df_T.diffNewEnergy[(df_T.NewSOC>70) & (df_T.NewSOC<80)].copy()/df_T.diffNewSOC.copy()
            df_T["CapaBatDecharge75"] = df_T.CapaBat75/df_T.CapaBat75 * np.mean(df_T.CapaBat75[(df_T.diffNewSOC.copy()<0)].copy())
            df_T["CapaBatCharge75"] = df_T.CapaBat75/df_T.CapaBat75 * np.mean(df_T.CapaBat75[(df_T.diffNewSOC.copy()>0)].copy())
    
            df_T["CapaBat80"] = -100*df_T.diffNewEnergy[(df_T.NewSOC>75) & (df_T.NewSOC<85)].copy()/df_T.diffNewSOC.copy()
            df_T["CapaBatDecharge80"] = df_T.CapaBat80/df_T.CapaBat80 * np.mean(df_T.CapaBat80[(df_T.diffNewSOC.copy()<0)].copy())
            df_T["CapaBatCharge80"] = df_T.CapaBat80/df_T.CapaBat80 * np.mean(df_T.CapaBat80[(df_T.diffNewSOC.copy()>0)].copy())
    
            df_T["CapaBat85"] = -100*df_T.diffNewEnergy[(df_T.NewSOC>80) & (df_T.NewSOC<90)].copy()/df_T.diffNewSOC.copy()
            df_T["CapaBatDecharge85"] = df_T.CapaBat85/df_T.CapaBat85 * np.mean(df_T.CapaBat85[(df_T.diffNewSOC.copy()<0)].copy())
            df_T["CapaBatCharge85"] = df_T.CapaBat85/df_T.CapaBat85 * np.mean(df_T.CapaBat85[(df_T.diffNewSOC.copy()>0)].copy())
    
            df_T["CapaBat90"] = -100*df_T.diffNewEnergy[(df_T.NewSOC>85) & (df_T.NewSOC<95)].copy()/df_T.diffNewSOC.copy()
            df_T["CapaBatDecharge90"] = df_T.CapaBat90/df_T.CapaBat90 * np.mean(df_T.CapaBat90[(df_T.diffNewSOC.copy()<0)].copy())
            df_T["CapaBatCharge90"] = df_T.CapaBat90/df_T.CapaBat90 * np.mean(df_T.CapaBat90[(df_T.diffNewSOC.copy()>0)].copy())
    
            # fig4 = px.scatter(df_T, x=df_T.SOC, y=df_T.columns)
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
                                    'ResistanceBat2' : Bat_Res,
                                    'CapaBat2' : Bat_Capa,
                                    'RendBat2' : Bat_Rend,
                                    'CapaciteBatCharge' : df_T.CapaBatCharge.mean(),
                                    'CapaciteBatDecharge' : df_T.CapaBatDecharge.mean(),
                                    'CapaciteBatCharge30' : df_T.CapaBatCharge30.mean(),
                                    'CapaciteBatDecharge30' : df_T.CapaBatDecharge30.mean(),
                                    'CapaciteBatCharge35' : df_T.CapaBatCharge35.mean(),
                                    'CapaciteBatDecharge35' : df_T.CapaBatDecharge35.mean(),
                                    'CapaciteBatCharge40' : df_T.CapaBatCharge40.mean(),
                                    'CapaciteBatDecharge40' : df_T.CapaBatDecharge40.mean(),
                                    'CapaciteBatCharge45' : df_T.CapaBatCharge45.mean(),
                                    'CapaciteBatDecharge45' : df_T.CapaBatDecharge45.mean(),
                                    'CapaciteBatCharge50' : df_T.CapaBatCharge50.mean(),
                                    'CapaciteBatDecharge50' : df_T.CapaBatDecharge50.mean(),
                                    'CapaciteBatCharge55' : df_T.CapaBatCharge55.mean(),
                                    'CapaciteBatDecharge55' : df_T.CapaBatDecharge55.mean(),
                                    'CapaciteBatCharge60' : df_T.CapaBatCharge60.mean(),
                                    'CapaciteBatDecharge60' : df_T.CapaBatDecharge60.mean(),
                                    'CapaciteBatCharge65' : df_T.CapaBatCharge65.mean(),
                                    'CapaciteBatDecharge65' : df_T.CapaBatDecharge65.mean(),
                                    'CapaciteBatCharge70' : df_T.CapaBatCharge70.mean(),
                                    'CapaciteBatDecharge70' : df_T.CapaBatDecharge70.mean(),
                                    'CapaciteBatCharge75' : df_T.CapaBatCharge75.mean(),
                                    'CapaciteBatDecharge75' : df_T.CapaBatDecharge75.mean(),
                                    'CapaciteBatCharge80' : df_T.CapaBatCharge80.mean(),
                                    'CapaciteBatDecharge80' : df_T.CapaBatDecharge80.mean(),
                                    'CapaciteBatCharge85' : df_T.CapaBatCharge85.mean(),
                                    'CapaciteBatDecharge85' : df_T.CapaBatDecharge85.mean(),
                                    'CapaciteBatCharge90' : df_T.CapaBatCharge90.mean(),
                                    'CapaciteBatDecharge90' : df_T.CapaBatDecharge90.mean(),
                                    }])
            df_Out = pd.concat([df_Out, df_new_row], ignore_index=True)
    
    
    
    
            if 'df' not in locals():
                df = df_T
            else:
                df = pd.concat([df,df_T])
            del df_T
    
    
    return df_Out, df


df_Out, df = posttreatmyvin(uploaded_file, df_FastLog, df_Trips, df_TripInfo, optionVIN)

df_Out = df_Out[(df_Out.Distance > 5)]

fig100 = px.scatter(df_Out, x=df_Out.DateTrajet, y=df_Out.columns,title="Suivi du viellissement :")
# plot(fig100)        

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
                     title="Capacité locale de la batterie :")
# plot(fig101)   


st.plotly_chart(fig100, use_container_width=True)
st.plotly_chart(px.scatter(df_Out, x=df_Out.TempeBat, y=df_Out.TempeBat), use_container_width=True)
st.plotly_chart(fig101, use_container_width=True)

'''

## Bi-histogramme
'''

#%%
Sat = st.slider('Saturation couleur', 0.0, 1.0, 0.5)

fig200 = px.density_heatmap(df, x=df.ACCELERATOR, y=df.PuissanceElec_kW)
fig200.update_traces(histnorm = "percent")
fig200.update_layout(
    {
        "coloraxis_cmin": 0,
        "coloraxis_cmax": Sat,
    }
)
plot(fig200) 
st.plotly_chart(fig200, use_container_width=True)




