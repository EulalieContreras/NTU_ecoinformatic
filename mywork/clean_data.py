import pandas as pd
import unicodedata
import numpy as np
#test git
##### DATA IMPORTATION

### GROUND WATER 

dossier_debut = 'mywork/donnees/nappes/'
dossier_fin = '/ades_export/Quantite/chroniques.txt'
ground_columns = ['Identifiant national BSS', 'Date de la mesure', 'Profondeur relative/repère de mesure', 'Côte NGF', 'X_WGS84', 'Y_WGS84']

nappe_1 = pd.read_csv(dossier_debut + 'entraigues' + dossier_fin, sep='|', decimal='.', encoding='iso-8859-1')
nappe_2 = pd.read_csv(dossier_debut + 'monteux' + dossier_fin, sep='|', decimal='.', encoding='iso-8859-1')
nappe_3 = pd.read_csv(dossier_debut + 'thor' + dossier_fin, sep='|', decimal='.', encoding='iso-8859-1')
nappe_4 = pd.read_csv(dossier_debut + 'LePontet' + dossier_fin, sep='|', decimal='.', encoding='iso-8859-1')
dico_city_GW = {nappe_1['Identifiant national BSS'].iloc[0]: 'ENTRAIGUES', nappe_2['Identifiant national BSS'].iloc[0]: 'MONTEUX', nappe_3['Identifiant national BSS'].iloc[0]: 'THOR', nappe_4['Identifiant national BSS'].iloc[0]: 'LE PONTET'}

nappes = pd.concat([nappe_1, nappe_2, nappe_3, nappe_4], axis=0, ignore_index=True)

nappes['Date de la mesure'] = pd.to_datetime(nappes['Date de la mesure'], errors='coerce', dayfirst=True)
nappes['Date de la mesure'] = nappes['Date de la mesure'].dt.floor('D')
nappes = nappes[ground_columns]
nappes = nappes[(~nappes['Date de la mesure'].isna()) & (~nappes['Identifiant national BSS'].isna())]
nappes = nappes[(~nappes['Côte NGF'].isna()) | (~nappes['Profondeur relative/repère de mesure'].isna())]
nappes.columns = ['station_ID', 'date', 'relative_depth', 'NGF_depth', 'X_coordonate', 'Y_coordonate'] #depth is in meters, NGF is the French national height reference system
nappes['station_ID'] = nappes['station_ID'].astype(str).str.strip()
nappes.drop_duplicates(subset=['station_ID','date'], inplace=True)
nappes['city'] = nappes['station_ID'].map(dico_city_GW)

groundwater1 = nappes[(nappes['date'] >= '2002-04-03') & (nappes['date'] <= '2013-11-27')]
groundwater2 = nappes[(nappes['date'] >= '2016-04-03') & (nappes['date'] <= '2024-12-31')]
groundwater2 = groundwater2[~(groundwater2['city']=='ENTRAIGUES')] 

missing_rows = []
for station in groundwater1['station_ID'].unique():
    sous_df = groundwater1[groundwater1['station_ID'] == station]
    date_range = pd.date_range(start=sous_df['date'].min(), end=sous_df['date'].max(), freq='D')
    missing_dates = date_range[~date_range.isin(sous_df['date'])]
    missing_rows.append(pd.DataFrame({'station_ID': station, 'date': missing_dates, 'relative_depth': np.nan, 'NGF_depth': np.nan, 'X_coordonate': sous_df['X_coordonate'].iloc[0], 'Y_coordonate': sous_df['Y_coordonate'].iloc[0]}))
groundwater1 = pd.concat([groundwater1] + missing_rows, axis=0, ignore_index=True)
groundwater1 = groundwater1.sort_values(by=['station_ID', 'date']).reset_index(drop=True)

groundwater1.to_csv('mywork/donnees/groundwater_02-13.csv', index=False, sep=';')

missing_rows = []
for station in groundwater2['station_ID'].unique():
    sous_df = groundwater2[groundwater2['station_ID'] == station]
    date_range = pd.date_range(start=sous_df['date'].min(), end=sous_df['date'].max(), freq='D')
    missing_dates = date_range[~date_range.isin(sous_df['date'])]
    missing_rows.append(pd.DataFrame({'station_ID': station, 'date': missing_dates, 'relative_depth': np.nan, 'NGF_depth': np.nan, 'X_coordonate': sous_df['X_coordonate'].iloc[0], 'Y_coordonate': sous_df['Y_coordonate'].iloc[0]}))
groundwater2 = pd.concat([groundwater2] + missing_rows, axis=0, ignore_index=True)
groundwater2 = groundwater2.sort_values(by=['station_ID', 'date']).reset_index(drop=True)

groundwater2.to_csv('mywork/donnees/groundwater_16-24.csv', index=False, sep=';')

###SURFACE WATER

surface_columns = ['Code de la station hydrométrique', 'Date d\'observation élaborée hydrométrique', 'Résultat de l\'observation élaborée hydrométrique']
dico_city_rivers = {'V615502001' : 'FONTAINE DE VAUCLUSE 1', 'V615502002' : 'FONTAINE DE VAUCLUSE 2', 'V615562101' : 'BEDARRIDES'}

debit_max = pd.read_csv('mywork/donnees/riviere/debit_max_02-13/export_hydro_series.csv', sep=';', skiprows = 1)
debit_max = debit_max[surface_columns]
debit_max.columns = ['station_ID', 'date', 'max_flow']
debit_moyen = pd.read_csv('mywork/donnees/riviere/debit_moyen_02-13/export_hydro_series.csv', sep=';', skiprows = 1)
debit_moyen = debit_moyen[surface_columns]
debit_moyen.columns = ['station_ID', 'date', 'mean_flow']
debit_min = pd.read_csv('mywork/donnees/riviere/debit_min_02-13/export_hydro_series.csv', sep=';', skiprows = 1)
debit_min = debit_min[surface_columns]
debit_min.columns = ['station_ID', 'date', 'min_flow']
hauteur_max = pd.read_csv('mywork/donnees/riviere/hauteur_max_02-13/export_hydro_series.csv', sep=';', skiprows = 1)
hauteur_max = hauteur_max[surface_columns]
hauteur_max.columns = ['station_ID', 'date', 'max_height']
hauteur_min = pd.read_csv('mywork/donnees/riviere/hauteur_min_02-13/export_hydro_series.csv', sep=';', skiprows = 1)
hauteur_min = hauteur_min[surface_columns]
hauteur_min.columns = ['station_ID', 'date', 'min_height']

for df in (debit_moyen, debit_min, debit_max, hauteur_min, hauteur_max):
    df['date'] = pd.to_datetime(df['date'], errors='coerce', format='%Y-%m-%d %H:%M:%S')
    df['date'] = df['date'].dt.floor('D')
    df.drop_duplicates(subset=['station_ID','date'], inplace=True)
    df['station_ID'] = df['station_ID'].astype(str).str.strip()

rivers = (
    debit_moyen
    .merge(debit_min,   on=['station_ID','date'], how='outer')
    .merge(debit_max,   on=['station_ID','date'], how='outer')
    .merge(hauteur_min, on=['station_ID','date'], how='outer')
    .merge(hauteur_max, on=['station_ID','date'], how='outer')
)

val_cols = ['mean_flow','min_flow','max_flow','min_height','max_height'] #in m3/s for flow and in m for height
rivers = rivers[rivers[val_cols].notna().any(axis=1)].reset_index(drop=True)
rivers = rivers[rivers[['station_ID', 'date']].notna().all(axis=1)].reset_index(drop=True)
rivers.drop_duplicates(subset=['station_ID','date'], inplace=True)
rivers['city'] = rivers['station_ID'].map(dico_city_rivers)

rivers1 = rivers[(rivers['date'] >= '2002-04-03') & (rivers['date'] <= '2013-11-27')]
rivers1 = rivers1[~(rivers1['city']=='BEDARRIDES')]
rivers2 = rivers[(rivers['date'] >= '2016-03-04') & (rivers['date'] <= '2024-12-31')]

missing_rows = []
for station in rivers1['station_ID'].unique():
    sous_df = rivers1[rivers1['station_ID'] == station]
    date_range = pd.date_range(start=sous_df['date'].min(), end=sous_df['date'].max(), freq='D')
    missing_dates = date_range[~date_range.isin(sous_df['date'])]
    missing_rows.append(pd.DataFrame({'station_ID': station, 'date': missing_dates, 'max_flow': np.nan, 'min_flow': np.nan, 'mean_flow': np.nan, 'min_height': np.nan, 'max_height': np.nan}))
rivers1 = pd.concat([rivers1] + missing_rows, axis=0, ignore_index=True)
rivers1 = rivers1.sort_values(by=['station_ID', 'date']).reset_index(drop=True)

rivers1.to_csv('mywork/donnees/rivers_02-13.csv', index=False, sep=';')

missing_rows = []
for station in rivers2['station_ID'].unique():
    sous_df = rivers2[rivers2['station_ID'] == station]
    date_range = pd.date_range(start=sous_df['date'].min(), end=sous_df['date'].max(), freq='D')
    missing_dates = date_range[~date_range.isin(sous_df['date'])]
    missing_rows.append(pd.DataFrame({'station_ID': station, 'date': missing_dates, 'max_flow': np.nan, 'min_flow': np.nan, 'mean_flow': np.nan, 'min_height': np.nan, 'max_height': np.nan}))
rivers2 = pd.concat([rivers2] + missing_rows, axis=0, ignore_index=True)
rivers2 = rivers2.sort_values(by=['station_ID', 'date']).reset_index(drop=True)

rivers2.to_csv('mywork/donnees/rivers_16-24.csv', index=False, sep=';')

### METEO

def norm_city(s):
    if pd.isna(s):
        return pd.NA
    s2 = ''.join(ch for ch in unicodedata.normalize('NFD', str(s)) if unicodedata.category(ch) != 'Mn')
    return s2.upper().strip()


meteo_0 = pd.read_csv('mywork/donnees/meteo/precipitation_bedarrides/export_meteo_series.csv', sep=';', skiprows = 1)
meteo_0 = meteo_0[['Code du site météorologique', 'Date de l\'observation météorologique', 'Résultat de l\'observation météorologique']]
meteo_0.columns = ['station_ID', 'date', 'precipitation']
meteo_0['date'] = pd.to_datetime(meteo_0['date'], errors='coerce', format='%Y-%m-%d %H:%M:%S')
meteo_0['date'] = meteo_0['date'].dt.floor('D')
meteo_0['station_ID'] = meteo_0['station_ID'].astype(str).str.strip()
doc = {meteo_0['station_ID'].iloc[0]: 'BEDARRIDES'}
meteo_0['city'] = meteo_0['station_ID'].map(doc)

meteo_1 = pd.read_csv('mywork/donnees/meteo/meteo_RRTVent.csv', sep=';', compression='gzip')
meteo_1 = meteo_1[['NUM_POSTE', 'NOM_USUEL', 'LAT', 'LON', 'ALTI', 'AAAAMMJJ', 'RR', 'TNTXM', 'TAMPLI', 'TNSOL', 'DRR']]
meteo_1.columns = ['station_ID', 'city', 'latitude', 'longitude', 'altitude', 'date', 'precipitation', 'mean_temp', 'temp_amplitude', 'temp_at10cmaboveground', 'duration_precipitation'] #in mm for precipitation, in °C for temperatures and in mn for duration of precipitation
meteo_1['date'] = pd.to_datetime(meteo_1['date'].astype(str), format='%Y%m%d', errors='coerce')
meteo_1['date'] = meteo_1['date'].dt.floor('D')
meteo_1['station_ID'] = meteo_1['station_ID'].astype(str).str.strip()
meteo_1['city'] = meteo_1['city'].apply(norm_city)
meteo_1 = meteo_1[(meteo_1['city'] == 'ISLE SUR SORGUE') | (meteo_1['city'] == 'JONQUERETTES')]

meteo_2 = pd.read_csv('mywork/donnees/meteo/meteo.csv', sep=';', compression='gzip')
meteo_2 = meteo_2[['NUM_POSTE', 'NOM_USUEL', 'LAT', 'LON', 'ALTI', 'AAAAMMJJ', 'ETPMON', 'ETPGRILLE', 'ORAG', 'PMERM', 'UM']]
meteo_2.columns = ['station_ID', 'city', 'latitude', 'longitude', 'altitude', 'date', 'calculated_evapotranspiration', 'measured_evapotranspiration', 'storm_occurrence', 'sea_mean_pressure', 'mean_relative_humidity'] #in mm for evapotranspirations, 0 or 1 for orage occurence, in hPa for sea mean pressure and in % for relative humidity
meteo_2['date'] = pd.to_datetime(meteo_2['date'].astype(str), format='%Y%m%d', errors='coerce')
meteo_2['date'] = meteo_2['date'].dt.floor('D')
meteo_2['station_ID'] = meteo_2['station_ID'].astype(str).str.strip()
meteo_2['city'] = meteo_2['city'].apply(norm_city)
meteo_2 = meteo_2[(meteo_2['city'] == 'ISLE SUR SORGUE') | (meteo_2['city'] == 'JONQUERETTES')]


cols = [ 'precipitation', 'duration_precipitation', 'mean_temp', 'temp_amplitude', 'temp_at10cmaboveground', 'calculated_evapotranspiration', 'measured_evapotranspiration', 'storm_occurrence', 'sea_mean_pressure', 'mean_relative_humidity']
# attention après verificatino sur toutes ces donnes seules les colonnes 'precipitation', 'measured evapotranspiration' and 'storm_occurrence' ont des valeurs non nulles, les autres sont toutes nulles, donc on peut les ignorer pour l'instant

#technique de copilot pour éviter que ca utilise trop de mémoire et que ca plante: on déduplique les deux tables avant de les fusionner, puis on fusionne uniquement sur les clés réelles (station_ID + date), puis on coalesce les champs de localisation (city, latitude, longitude, altitude) pour éviter d'avoir des colonnes en double

# dédupliquer pour éviter des jointures many-to-many
meteo_1 = meteo_1.drop_duplicates(subset=['station_ID', 'date'])
meteo_2 = meteo_2.drop_duplicates(subset=['station_ID', 'date'])

# fusionner uniquement sur les clés réelles (station_ID + date)
meteo = pd.merge(meteo_1, meteo_2, on=['station_ID', 'date'], how='outer', suffixes=('_1','_2'))

# coalescer les champs de localisation: préférer _1 puis _2
for c in ['city','latitude','longitude','altitude']:
    c1 = f"{c}_1"
    c2 = f"{c}_2"
    if c1 in meteo.columns and c2 in meteo.columns:
        meteo[c] = meteo[c1].combine_first(meteo[c2])
        meteo.drop([c1, c2], axis=1, inplace=True)

#fin technique copilot

meteo = pd.concat([meteo, meteo_0], axis=0, ignore_index=True)

meteo = meteo[meteo[['station_ID', 'date']].notna().all(axis=1)].reset_index(drop=True)

meteo1 = meteo[(meteo['date'] >= '2002-04-03') & (meteo['date'] <= '2013-11-27')]
meteo1 = meteo1[~(meteo1['city']=='BEDARRIDES')]
meteo2 = meteo[(meteo['date'] >= '2016-04-03') & (meteo['date'] <= '2024-12-31')]
meteo2 = meteo2[~(meteo2['city']=='JONQUERETTES')]

missing_rows = []
for station in meteo1['station_ID'].unique():
    sous_df = meteo1[meteo1['station_ID'] == station]
    date_range = pd.date_range(start=sous_df['date'].min(), end=sous_df['date'].max(), freq='D')
    missing_dates = date_range[~date_range.isin(sous_df['date'])]
    missing_rows.append(pd.DataFrame({'station_ID': station, 'date': missing_dates, 'city': sous_df['city'].iloc[0], 'longitude': sous_df['longitude'].iloc[0], 'latitude': sous_df['latitude'].iloc[0], 'altitude': sous_df['altitude'].iloc[0], 'precipitation': np.nan, 'duration_precipitation': np.nan, 'mean_temp': np.nan, 'temp_amplitude': np.nan, 'temp_at10cmaboveground': np.nan, 'calculated_evapotranspiration': np.nan, 'measured_evapotranspiration': np.nan, 'storm_occurrence': np.nan, 'sea_mean_pressure': np.nan, 'mean_relative_humidity': np.nan}))
meteo1 = pd.concat([meteo1] + missing_rows, axis=0, ignore_index=True)
meteo1 = meteo1.sort_values(by=['station_ID', 'date']).reset_index(drop=True)

meteo1.to_csv('mywork/donnees/meteo_02-13.csv', index=False, sep=';')

missing_rows = []
for station in meteo2['station_ID'].unique():
    sous_df = meteo2[meteo2['station_ID'] == station]
    date_range = pd.date_range(start=sous_df['date'].min(), end=sous_df['date'].max(), freq='D')
    missing_dates = date_range[~date_range.isin(sous_df['date'])]
    missing_rows.append(pd.DataFrame({'station_ID': station, 'date': missing_dates, 'city': sous_df['city'].iloc[0], 'longitude': sous_df['longitude'].iloc[0], 'latitude': sous_df['latitude'].iloc[0], 'altitude': sous_df['altitude'].iloc[0], 'precipitation': np.nan, 'duration_precipitation': np.nan, 'mean_temp': np.nan, 'temp_amplitude': np.nan, 'temp_at10cmaboveground': np.nan, 'calculated_evapotranspiration': np.nan, 'measured_evapotranspiration': np.nan, 'storm_occurrence': np.nan, 'sea_mean_pressure': np.nan, 'mean_relative_humidity': np.nan}))
meteo2 = pd.concat([meteo2] + missing_rows, axis=0, ignore_index=True)
meteo2 = meteo2.sort_values(by=['station_ID', 'date']).reset_index(drop=True)

meteo2.to_csv('mywork/donnees/meteo_16-24.csv', index=False, sep=';')




