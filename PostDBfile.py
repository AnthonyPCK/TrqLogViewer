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


'''
# Viewer de fichier HybridAssistant.db
## Fichier exemple Hyundai ioniq Hybrid ou charger votre fichier
'''


uploaded_file = st.file_uploader("Upload a SQLite database file.", type="db")

if uploaded_file is not None:
    fp = pathlib.Path(str(uuid.uuid4()))
    # fp = pathlib.Path("/path/to/your/tmpfile")
    try:
        fp.write_bytes(uploaded_file.getvalue())
        conn = connect(str(fp))
    finally:
        if fp.is_file():
            fp.unlink()
    
    
    
else:
    # On selectionne la voiture que l\'on souhaite
    optionVIN = 'KMHC851CGLU177332'; # IONIQ   
    conn = connect("hybridassistant2023.db")  

df_FastLog = pd.read_sql('SELECT * FROM FASTLOG', conn)
df_Trips = pd.read_sql('SELECT * FROM TRIPS', conn)
df_TripInfo = pd.read_sql('SELECT * FROM TRIPINFO', conn)

@st.cache_data
def posttreatmyvin(optionVIN):

    
    
    
    if uploaded_file is not None:
        optionVIN = st.selectbox(
        "On selectionne que la voiture que l\'on souhaite",
        pd.unique(df_TripInfo.VIN))
    
    
    
    # On conserve uniquement les données correspondant à un VIN
    df_TripInfo_MyVIN = df_TripInfo[(df_TripInfo.VIN == optionVIN)]
    df_Trips_MyVIN = df_Trips[(df_TripInfo.VIN == optionVIN)]
    
    # On garde seulement les trajet de plus d'un km (on écrase les ancien dataframe))
    df_Trips = df_Trips_MyVIN[(df_Trips_MyVIN.NKMS >1)]
    df_TripInfo = df_TripInfo_MyVIN[(df_Trips_MyVIN.NKMS >1)]
    
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
    
    
            # fig1 = px.scatter(df_T, x=df_T.index, y=df_T.columns)
            # plot(fig1)
    
            # On rajoute des channels
            df_T["Time_S"] = (df_T.TIMESTAMP.copy() - df_T.TIMESTAMP.min()) / 1000
            df_T["diffTime_S"] = np.concatenate((np.array([0]),np.diff(df_T.Time_S.copy())))
            df_T["PuissanceElec_kW"] = df_T.HV_A.copy() * df_T.HV_V.copy() / 1000
            df_T["Energy"] = np.cumsum(df_T.HV_A.copy() * df_T.diffTime_S.copy() / 3600)
            df_T["diffSOC"] = np.concatenate((np.array([0]),np.diff(df_T.SOC.copy())))
    
    
            # On récupère les infos générales sur le trajet
            MeanAmbiantTemp = df_T.AMBIENT_TEMP.mean()
            MeanBatTemp = df_T.BATTERY_TEMP.mean()
            MeanKilometrage = df_T.ODO.mean()
            TrajDate = pd.to_datetime(df_T.TIMESTAMP.mean()/1000, unit='s')
            TrajConso = 0.001*100*df_T.TRIPFUEL.max()/df_T.TRIP_DIST.max()
            MeanSpeed = df_T.SPEED_OBD.mean()
            MeanRollingSpeed = df_T.SPEED_OBD[(df_T.SPEED_OBD >1)].mean()
    
            ## On identifie la resistance moyenne de la batterie sur le trajet
            # On crée un dataframe avec la batterie qui ne débite pas de courant (moins de 1 A)
            df_LowHVA = df_T[(df_T.HV_A >-1) & (df_T.HV_A <1)]
    
            p = np.polyfit(df_LowHVA.SOC, df_LowHVA.HV_V, 2)
            SOC = np.arange(0, 100, 0.5)
            Pol = np.polyval(p,SOC)
    
            # fig2 = px.line(df_LowHVA, x=df_LowHVA.SOC, y=df_LowHVA.HV_V)
            # fig2.add_trace(go.Scatter(x=SOC, y=Pol))
            #plot(fig2)
    
            df_T["HV_V_var"] = df_T.HV_V.copy() - np.polyval(p,df_T.SOC.copy())
            # fig3 = px.scatter(df_T, x=df_T.HV_A, y=df_T.HV_V_var)
    
            p = np.polyfit(df_T.HV_A, df_T.HV_V_var, 1)
            HV_A = np.arange(-200, 200, 0.5)
            Pol = np.polyval(p,HV_A)
            # fig3.add_trace(go.Scatter(x=HV_A, y=Pol))
            # plot(fig3)
            BatResistance = -p[0]
    
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
    
    
    return df_Out


df_Out = posttreatmyvin(optionVIN)

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




