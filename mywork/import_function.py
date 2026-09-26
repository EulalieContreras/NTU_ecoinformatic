import pandas as pd
import numpy as np
from numpy.random import default_rng
import pandas as pd
from pandas.plotting import autocorrelation_plot
import scipy.stats as st
from scipy.signal import detrend
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from pyEDM import *
from scipy.stats import kendalltau
from scipy.interpolate import UnivariateSpline


# Parameters
lib = "1 500"
pred = "501 1000"
E_values = range(2, 9)
window_ = 60
theta_values = np.arange(0, 3.1, 0.1)

# Functions of preprocessing

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

# Functions of EDM

def run_simplex(ts, E, lib_range, pred_range):
    df = pd.DataFrame({'Time': np.arange(1, len(ts) + 1), 'X': ts})
    preds = Simplex(
        dataFrame=df,
        lib=lib_range,
        pred=pred_range,
        E=E,
        columns="X",
        target="X",
        showPlot=False
    )
    # Verifying the columns exist
    if 'Observations' not in preds or 'Predictions' not in preds:
        return np.nan

    valid = ~preds['Observations'].isna() # ~preds['Observations'].isna() creates a boolean mask where True indicates valid observations
    if valid.sum() == 0:
        return np.nan

    rho = preds.loc[valid, ['Observations', 'Predictions']].corr().iloc[0, 1]
    return rho

def evaluate_dimension(ts, lib=lib, pred=pred):
    data = []
    for E in E_values:
        rho = run_simplex(ts, E, lib, pred)
        data.append({'E': E, 'rho': rho})
    return pd.DataFrame(data)


def run_smap(ts, E, lib=lib, pred=pred, theta_values=theta_values):
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
            showPlot=False
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

def find_best_E(df, col_lib, col_target, Tp=0, libSizes="10 100 10"):
    results = []
    col_name = f"{col_lib}:{col_target}"

    for E in range(2, 9):
        out = CCM(
            dataFrame=df,
            columns=col_lib,
            target=col_target,
            E=E,
            Tp=Tp,
            libSizes=libSizes,  # libSize min max step
            sample=1,
            showPlot=False
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

def run_ccm_bootstrap(df, col_lib, col_trg, E, libs = lib, samples=200, seed=2301, Tp=0):
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

def final_ccm(df, col_lib, col_target, libs=lib, Tp=0):
    # Drop rows with missing values in the two columns to avoid issues with pyEDM
    df_clean = df[[col_lib, col_target]].dropna().reset_index(drop=True)
    E_best = find_best_E(df_clean, col_lib, col_target, Tp=Tp)
    df_rho = run_ccm_bootstrap(df_clean, col_lib, col_target, E_best, libs, Tp=Tp)
    stats, tau = summarize_trend(df_rho, "rho")
    return stats, tau

def surrogate(df, col_lib, col_target, num=200, libs=lib, comparaison=None):
    df_sur = pd.DataFrame({'date': df['date'], f'{col_lib}': df[col_lib]})

    sur = SurrogateData(df_sur, col_lib, numSurrogates=num, method='seasonal', alpha=0, smooth=0.8)
    rho = []
    sur = sur.merge(df[['date', col_target]], on='date', how='outer')

    for i in range(1, num+1):
        stats, tau = final_ccm(sur, f'{col_lib}_{i}', col_target, libs=libs, Tp=0)
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



def surrogate_fast(df, col_lib, col_target, num=200, libs=lib, comparaison=None): ## fontion écrite en coopération avec copilot pour accélerer le processus 

    """
    Version rapide et stable :
    - utilise SurrogateData
    - garde la forme de sortie de la fonction surrogate
    - évite la fragmentation de DataFrame
    """

    if isinstance(libs, str):
        libs = [int(x) for x in libs.split()]

    # on garde seulement les colonnes utiles, dans l'ordre attendu par SurrogateData
    d = df[['date', col_lib, col_target]].dropna().reset_index(drop=True)

    # important : la 1ère colonne est le temps, 2e la série surrogatée
    sur = SurrogateData(
        d[['date', col_lib]].copy(),
        column=col_lib,
        numSurrogates=num,
        method='seasonal',
        alpha=0,
        smooth=0.8
    )

    surrogate_rhos = []

    # la première colonne de sur est "date", les surrogates commencent à 1
    for i in range(1, num + 1):
        surr_name = sur.columns[i]

        temp = pd.DataFrame({
            'date': d['date'].to_numpy(),
            surr_name: sur[surr_name].to_numpy(),
            col_target: d[col_target].to_numpy()
        }).dropna().reset_index(drop=True)

        stats, tau = final_ccm(temp, surr_name, col_target, libs=libs, Tp=0)

        last_lib = float(libs[-1])
        rho_med = float(stats.loc[last_lib, 0.5])
        surrogate_rhos.append(rho_med)

    surrogate_series = pd.Series(surrogate_rhos, name='rho')

    resume = {
        0.25: surrogate_series.quantile(0.25),
        0.5: surrogate_series.median(),
        0.75: surrogate_series.quantile(0.75),
    }

    if comparaison is None:
        return surrogate_series, resume

    pvalue = (sum(r > comparaison for r in surrogate_rhos) + 1) / (len(surrogate_rhos) + 1)
    return surrogate_series, resume, pvalue