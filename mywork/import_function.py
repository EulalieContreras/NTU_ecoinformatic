import pandas as pd
import numpy as np
from numpy.random import default_rng
import pandas as pd
from pandas.plotting import autocorrelation_plot
import scipy.stats as st
from scipy.signal import detrend
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from pyEDM import *
from scipy.stats import kendalltau
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed


### Parameters
E_values = range(2, 9)
window_ = 60
theta_values = np.arange(0, 3.1, 0.1)

### Functions of preprocessing

def normalize_series(series):
    array = np.array(series) # Convert the input series to a numpy array
    scaler = StandardScaler() # StandardScaler normalizes the data to have mean 0 and variance 1
    array = scaler.fit_transform(array.reshape(-1, 1)).flatten() # returns a 1D array after normalization
    return pd.Series(array, index=series.index) # Return a pandas Series with the same index as the input series

def deseason_mult(serie, period_):
    series = serie.reset_index(drop=True)
    lenght = 365//period_
    n = len(series)

    idx = np.arange(n)
    period_idx = idx // lenght
    position_idx = idx % lenght
    n_season = period_idx.max() + 1

    period_mean = np.array([series[period_idx == _].mean(skipna=True) for _ in range(n_season)])

    season_index = []
    for i in series.index:
        season_index.append(series[i]/period_mean[period_idx[i]])
    season_index = np.array(season_index)

    mean_season_index = []
    for p in range(lenght):
        mean_season_index.append(np.nanmean(season_index[position_idx == p])) 
    mean_season_index = np.array(mean_season_index)

    deseasoned = []
    for j in series.index:
       deseasoned.append(series[j]/mean_season_index[position_idx[j]])
    deseasoned = pd.Series(deseasoned, index=series.index)

    return deseasoned

def detrend_add(series):
    rolling_mean = series.rolling(window=window_, center=True).mean()
    detrended = series - rolling_mean
    return detrended 

def deseason_add(serie, period_):
    series = serie.reset_index(drop=True)
    lenght = 365//period_
    n = len(series)

    idx = np.arange(n)
    period_idx = idx // lenght
    position_idx = idx % lenght
    n_season = period_idx.max() + 1

    period_mean = np.array([series[period_idx == _].mean(skipna=True) for _ in range(n_season)])

    season_index = []
    for i in series.index:
        season_index.append(series[i] - period_mean[period_idx[i]])
    season_index = np.array(season_index)

    mean_season_index = []
    for p in range(lenght):
        mean_season_index.append(np.nanmean(season_index[position_idx == p]))
    mean_season_index = np.array(mean_season_index)

    deseasoned = []
    for j in series.index:
       deseasoned.append(series[j] - mean_season_index[position_idx[j]])
    deseasoned = pd.Series(deseasoned, index=series.index)

    return deseasoned

def detrend_mult(series):
    rolling_mean = series.rolling(window=window_, center=True).mean()
    detrended = series / rolling_mean
    return detrended 

### Functions of EDM

def run_simplex(ts, E, lib, pred, Tp=1):
    df = pd.DataFrame({'Time': np.arange(1, len(ts) + 1), 'X': ts})
    preds = Simplex(
        dataFrame=df,
        lib=lib,
        pred=pred,
        E=E,
        columns="X",
        target="X",
        showPlot=False, 
        Tp=Tp
    )
    # Verifying the columns exist
    if 'Observations' not in preds or 'Predictions' not in preds:
        return np.nan

    valid = ~preds['Observations'].isna() # ~preds['Observations'].isna() creates a boolean mask where True indicates valid observations
    if valid.sum() == 0:
        return np.nan

    rho = preds.loc[valid, ['Observations', 'Predictions']].corr().iloc[0, 1]
    return rho

def evaluate_dimension(ts, lib, pred, Tp=1):
    data = []
    for E in E_values:
        rho = run_simplex(ts, E, lib, pred, Tp=Tp)
        data.append({'E': E, 'rho': rho})
    return pd.DataFrame(data)

def run_smap(ts, E, lib, pred, theta_values=theta_values, Tp=1):
    ts = pd.Series(ts).dropna().reset_index(drop=True)
    df = pd.DataFrame({'Time': np.arange(1, len(ts) + 1), 'X': ts})    
    results = []
    for theta in theta_values:
        res = SMap(
            dataFrame=df,

            lib=lib,
            pred=pred,
            E=E,
            columns="X",
            target="X",
            theta=theta,
            showPlot=False,
            Tp=Tp
        )
        # SMap may return a DataFrame or a dict-like object; handle both
        if isinstance(res, pd.DataFrame):
            preds = res
        else:
            preds = None
            try:
                preds = res.get('predictions')
            except Exception:
                preds = None

        if preds is None:
            rho = np.nan
        else:
            if 'Observations' not in preds.columns or 'Predictions' not in preds.columns:
                rho = np.nan
            else:
                valid = ~preds['Observations'].isna()
                if valid.sum() == 0:
                    rho = np.nan
                else:
                    rho = preds.loc[valid, ['Observations', 'Predictions']].corr().iloc[0, 1]
        results.append({'theta': theta, 'rho': rho})
    return pd.DataFrame(results)

def find_best_E_ccm(df, col_lib, col_target, seed=42): # Tp = -1 to avoid overfitting ? (make more research on that)
    df = df.copy().reset_index(drop=True)
    if 'date' not in df.columns:
        df.insert(0, 'date', np.arange(len(df)))
    df = df[['date', col_lib, col_target]].dropna(subset=[col_lib, col_target]).reset_index(drop=True)

    results = []
    col_name = f"{col_lib}:{col_target}"
    lib_size = max(50, min(len(df) // 2, 1000))
    for E in range(2, 9):
        out = CCM(
            dataFrame=df,
            columns=col_lib,
            target=col_target,
            E=E,
            Tp=-1,
            libSizes=str(lib_size),  # libSize min max step
            sample=5, #comment choisir cette valeur ? 
            showPlot=False, 
            seed = seed
        )

        if col_name not in out.columns:
            alt_name = f"{col_target}:{col_lib}"
            if alt_name in out.columns:
                col_name = alt_name
            else:
                raise ValueError(f"Expected column '{col_name}' not found. Got: {out.columns}")

        rho = out[col_name].iloc[-1]
        results.append((E, rho))

    return max(results, key=lambda x: x[1])[0]

def find_best_E(df, col_lib, col_target, lib, pred, Tp=1):
    df = _normalize_pyedm_time(df)
    results = []

    for E in range(2,9):
        preds = Simplex(
            dataFrame=df,
            lib=lib,
            pred=pred,
            E=E,
            columns=col_lib,
            target=col_target,
            showPlot=False, 
            Tp=Tp
        )
        # Verifying the columns exist
        if 'Observations' not in preds or 'Predictions' not in preds:
            return np.nan

        valid = ~preds['Observations'].isna() # ~preds['Observations'].isna() creates a boolean mask where True indicates valid observations
        if valid.sum() == 0:
            return np.nan

        rho = preds.loc[valid, ['Observations', 'Predictions']].corr().iloc[0, 1]    
        results.append((E, rho))

    return max(results, key=lambda x: x[1])[0]

def run_ccm_bootstrap(df, col_lib, col_trg, E, libs, samples=10, seed=2301, Tp=0):
    df = df.copy().reset_index(drop=True)
    if 'date' not in df.columns:
        df.insert(0, 'date', np.arange(len(df)))
    df = df[['date', col_lib, col_trg]].dropna(subset=[col_lib, col_trg]).reset_index(drop=True)

    out = CCM(
        dataFrame=df, columns=col_lib, target=col_trg,
        E=E, Tp=Tp, libSizes=" ".join(map(str, libs)),
        sample=samples, seed=seed,
        showPlot=False, includeData=True
    )
    return out["PredictStats1"]  # or PredictStats2, depends on the causality's direction

def summarize_trend(df_rho, col_name):
    stats = df_rho.groupby('LibSize')[col_name].quantile([0.25, 0.5, 0.75]).unstack()
    tau = {}
    for q in stats.columns:
        tau[q] = kendalltau(stats.index.astype(float), stats[q]).correlation
    return stats, tau

def final_ccm(df, col_lib, col_target, libs, Tp=0, samples=10, seed_E = 42, seed_finale = 2301):
    # pyEDM requires the first column to be a time/index column, so we keep date first.
    df_clean = df.copy().reset_index(drop=True)
    if 'date' not in df_clean.columns:
        df_clean.insert(0, 'date', np.arange(len(df_clean)))
    df_clean = df_clean[['date', col_lib, col_target]].dropna(subset=[col_lib, col_target]).reset_index(drop=True)
    E_best = find_best_E_ccm(df_clean, col_lib, col_target, seed=seed_E)
    df_rho = run_ccm_bootstrap(df_clean, col_lib, col_target, E_best, libs, samples=samples, Tp=Tp, seed=seed_finale)
    stats, tau = summarize_trend(df_rho, "rho")
    return stats, tau

def surrogate(df, col_lib, col_target, num, libs, comparaison=None, Tp=0, seed_ccm=2301, seed_E=42, seed_surrogate=0):
    df_sur = pd.DataFrame({'date': df['date'], f'{col_lib}': df[col_lib]})

    sur = SurrogateData(df_sur, col_lib, numSurrogates=num, method='seasonal', alpha=0, smooth=0.8, seed=seed_surrogate)
    rho = []
    sur = sur.merge(df[['date', col_target]], on='date', how='outer')

    for i in range(1, num+1):
        stats, tau = final_ccm(sur, f'{col_lib}_{i}', col_target, libs=libs, Tp=Tp, seed_finale=seed_ccm, seed_E=seed_E)
        last_lib = float(libs[-1])
        rho_med = float(stats.loc[last_lib, 0.5])
        rho.append(rho_med)

    surrogate_series = pd.Series(rho)

    resume = {
        0.25: surrogate_series.quantile(0.25),
        0.5: surrogate_series.median(),
        0.75: surrogate_series.quantile(0.75)}

    pvalue = (sum(r > comparaison for r in rho) + 1) / (len(rho) + 1)
    return surrogate_series, resume, pvalue

def _ccm_rho_for_one_surrogate(date_values, surrogate_values, target_values, surr_name, col_target, libs, Tp=0, seed_finale=2301, seed_E=42):
    d = pd.DataFrame({
        'date': date_values,
        surr_name: surrogate_values,
        col_target: target_values
    }).dropna().reset_index(drop=True)

    sur = pd.DataFrame({
        'date': date_values,
        surr_name: surrogate_values
    }).dropna().reset_index(drop=True)

    temp = pd.DataFrame({
        'date': d['date'].to_numpy(),
        surr_name: sur[surr_name].to_numpy(),
        col_target: d[col_target].to_numpy()
    }).dropna().reset_index(drop=True)

    stats, tau = final_ccm(temp, surr_name, col_target, libs=libs, Tp=Tp, seed_finale=seed_finale, seed_E=seed_E)

    last_lib = float(libs[-1])
    rho_med = float(stats.loc[last_lib, 0.5])
    return rho_med

def surrogate_fast(df, col_lib, col_target, num, libs, comparaison=None, Tp=0, maxworker=2, seed_ccm=2301, seed_E=42, seed_surrogate=0, method='seasonal'): ## fontion écrite en coopération avec copilot pour accélerer le processus ## l'idée avec les seed d'avoir une pour faire tourner le choix de E_best puis une différent pour le ccm final mais toujours le même entre les surrogates pour avoir des résultats comparables entre eux

    """
    Version rapide et stable :
    - utilise SurrogateData
    - garde la forme de sortie de la fonction surrogate
    - évite la fragmentation de DataFrame
    """
    start_total = time.perf_counter()

    if isinstance(libs, str):
        libs = [int(x) for x in libs.split()]

    # on garde seulement les colonnes utiles, dans l'ordre attendu par SurrogateData
    d = df[['date', col_lib, col_target]].dropna().reset_index(drop=True)

    # important : la 1ère colonne est le temps, 2e la série surrogatée
    sur = SurrogateData(
        dataFrame = d[['date', col_lib]].copy(),
        columns=col_lib,
        numSurrogates=num,
        methode=method,
        alpha=0,
        seed = seed_surrogate
    )

    tasks = []
    for i in range(1, num + 1):
        surr_name = sur.columns[i]
        tasks.append((
            d['date'].to_numpy(),
            sur[surr_name].to_numpy(),
            d[col_target].to_numpy(),
            surr_name,
            col_target,
            libs,
            Tp,
            seed_ccm,
            seed_E
        ))

    surrogate_rhos = []

    with ProcessPoolExecutor(maxworker) as executor:
        futures = [executor.submit(_ccm_rho_for_one_surrogate, *task) for task in tasks]

    for idx, fut in enumerate(as_completed(futures), start=1):
        surrogate_rhos.append(fut.result())
        if idx % 25 == 0 or idx == len(tasks):
            print(f"[surrogate_fast] processed {idx}/{len(tasks)} surrogates")

    surrogate_series = pd.Series(surrogate_rhos, name='rho')

    resume = {
        0.25: surrogate_series.quantile(0.25),
        0.5: surrogate_series.median(),
        0.75: surrogate_series.quantile(0.75),
    }

    elapsed = time.perf_counter() - start_total
    print(f"[surrogate_fast] finished in {elapsed:.2f}s")


    if comparaison is None:
        return surrogate_series, resume

    pvalue = (sum(r > comparaison for r in surrogate_rhos)) / (len(surrogate_rhos)) #j'enleve le +1 car peu de surrogate et pour tester la consistance des résultats
    return surrogate_series, resume, pvalue

def _normalize_pyedm_time(df):
    """Ensure the date column is the first column and in ISO format expected by pyEDM."""
    if 'date' not in df.columns:
        return df.copy()

    out = df.copy()
    out['date'] = pd.to_datetime(out['date'], errors='coerce')
    out = out.dropna(subset=['date']).sort_values('date').reset_index(drop=True)
    out['date'] = out['date'].dt.strftime('%Y-%m-%d')
    cols = ['date'] + [c for c in out.columns if c != 'date']
    return out[cols]

def simplex_prediction(df, col_lib, col_target, lib, pred, Tp=1):
    df = _normalize_pyedm_time(df)
    n = len(col_lib.split())
    if n < 2:
        E_best = find_best_E(df=df, col_lib=col_lib, col_target=col_target, Tp=Tp, lib=lib, pred=pred)
        preds = Simplex(
            dataFrame=df,
            lib=str(lib),
            pred=str(pred),
            E=E_best,
            columns=col_lib,
            target=col_target,
            showPlot=False,
            Tp=Tp
        )
    else:
        E_best = n
        preds = Simplex(
            dataFrame=df,
            lib=str(lib),
            pred=str(pred),
            E=E_best,
            columns=col_lib,
            target=col_target,
            showPlot=False,
            Tp=Tp, 
            embedded=True
        )
    return preds, E_best

def find_besttheta(df, col_lib, col_target, lib, pred, E, Tp=0):
    df = _normalize_pyedm_time(df)
    results = []
    for theta in theta_values:
        res = SMap(
            dataFrame=df,
            lib=lib,
            pred=pred,
            E=E,
            columns=col_lib,
            target=col_target,
            theta=theta,
            showPlot=False, Tp=Tp, embedded=True
        )
        # SMap may return a DataFrame or a dict-like object; handle both
        if isinstance(res, pd.DataFrame):
            preds = res
        else:
            preds = None
            try:
                preds = res.get('predictions')
            except Exception:
                preds = None

        if preds is None:
            rho = np.nan
        else:
            if 'Observations' not in preds.columns or 'Predictions' not in preds.columns:
                rho = np.nan
            else:
                valid = ~preds['Observations'].isna()
                if valid.sum() == 0:
                    rho = np.nan
                else:
                    rho = preds.loc[valid, ['Observations', 'Predictions']].corr().iloc[0, 1]
        results.append((theta, rho))
    return max(results, key=lambda x: x[1])[0]

def Smap_prediction(df, col_lib, col_target, lib, pred, Tp=1):
    df = _normalize_pyedm_time(df)
    n = len(col_lib.split())
    if n < 2:
        E = find_best_E(df=df, col_lib=col_lib, col_target=col_target, lib=lib, pred=pred, Tp=Tp)
        theta_best = find_besttheta(df, col_lib, col_target, lib, pred, E, Tp=Tp)
        preds = SMap(datFrame=df, columns=col_lib, target=col_target, lib=lib, pred=pred, E=E, Tp=Tp, theta=theta_best)
    else:
        E = n
        theta_best = find_besttheta(df, col_lib, col_target, lib, pred, E, Tp=Tp)
        preds = SMap(dataFrame=df, columns=col_lib, target=col_target, lib=lib, pred=pred, E=E, Tp=Tp, theta=theta_best, embedded=True)
    return preds, theta_best