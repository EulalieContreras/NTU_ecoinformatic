import import_function as fun
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from pyEDM import *

if __name__ == '__main__': ## nécessaire pour le multiprocess de ccm mais doit être enlevé  dans le notebook

    ### PREPROCESS 

    #river

    fdv1 = pd.read_csv('mywork/donnees/rivers_02-13.csv', sep=';', parse_dates=['date'])
    fdv1 = fdv1[(fdv1['city']=='FONTAINE DE VAUCLUSE 1') & (fdv1['date'] > '2004-01-01')]
    mean_flow_fdv1 = fdv1['mean_flow']
    river_flow = fun.normalize_series(mean_flow_fdv1)

    deseason_mean_flow_fdv1 = fun.deseason_add(mean_flow_fdv1, 1)
    detrend_mean_flow_fdv1 = fun.detrend_add(deseason_mean_flow_fdv1)
    normalize_mean_flow_fdv1_a = fun.normalize_series(detrend_mean_flow_fdv1)
    river_flow_a = normalize_mean_flow_fdv1_a
    plt.figure()
    plt.plot(fdv1['date'], fun.normalize_series(mean_flow_fdv1), label = 'no preprocess', color='r')
    plt.plot(fdv1['date'], normalize_mean_flow_fdv1_a, label = 'additive model', color='b')


    deseason_mean_flow_fdv1 = fun.deseason_mult(mean_flow_fdv1, 1)
    detrend_mean_flow_fdv1 = fun.detrend_mult(deseason_mean_flow_fdv1)
    normalize_mean_flow_fdv1_m = fun.normalize_series(detrend_mean_flow_fdv1)
    river_flow_m = normalize_mean_flow_fdv1_m
    plt.plot(fdv1['date'], normalize_mean_flow_fdv1_m, label = 'multiplicative model', color='c')


    plt.title('mean flow for FONTAINE DE VAUCLUSE 1')
    plt.xlabel('Date')
    plt.ylabel('mean flow (m³/s)')
    plt.legend()

    # precipitation

    meteo = pd.read_csv('mywork/donnees/meteo_02-13.csv', sep = ';', parse_dates=['date'])
    meteo = meteo[(meteo['city'] == 'ISLE SUR SORGUE') & (meteo['date'] > '2004-01-01')]
    rain = meteo['precipitation']

    rain_ = fun.normalize_series(rain)

    deseason_rain= fun.deseason_add(rain, 1)
    detrend_rain = fun.detrend_add(deseason_rain)
    normalize_rain = fun.normalize_series(detrend_rain)
    rain_a = normalize_rain
    plt.figure()
    plt.plot(meteo['date'], rain_, label = 'no preprocess', color = 'r')
    plt.plot(meteo['date'], rain_a, label = 'additive model', color ='b')


    deseason_rain= fun.deseason_mult(rain, 1)
    detrend_rain = fun.detrend_mult(deseason_rain)
    normalize_rain = fun.normalize_series(detrend_rain)
    rain_m = normalize_rain
    plt.plot(meteo['date'], rain_m, label = 'multiplicative model', color ='c')

    plt.title('rain for ISLES SUR SORGUE')
    plt.xlabel('Date')
    plt.ylabel('precipitation (mm)')
    plt.legend()

    '''
    ### SIMPLEX ALGORITHM

    result = fun.evaluate_dimension(fun.normalize_series(mean_flow_fdv1))
    result_a = fun.evaluate_dimension(normalize_mean_flow_fdv1_a)
    result_m = fun.evaluate_dimension(normalize_mean_flow_fdv1_m)

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(result['E'], result['rho'], label = 'no preprocess', color='r')
    ax2.plot(result_a['E'], result_a['rho'], label = 'additive model', color = 'b')
    ax2.plot(result_m['E'], result_m['rho'], label = 'multiplicative model', color='c')
    fig.suptitle('Simplex Evaluation for mean_flow FONTAINE DE VAUCLUSE 1')
    ax1.set_xlabel('Embedding Dimension (E)')
    ax1.set_ylabel('correlation coefficient', color='r')
    ax2.set_ylabel('correlation coefficient', color='c')
    ax1.tick_params(axis='y', labelcolor='r')
    ax2.tick_params(axis='y', labelcolor='b')
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    result_rain = fun.evaluate_dimension(rain_)
    result_rain_a = fun.evaluate_dimension(rain_a)
    result_rain_m = fun.evaluate_dimension(rain_m)

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(result_rain['E'], result_rain['rho'], label = 'no preprocess', color='r')
    ax2.plot(result_rain_a['E'], result_rain_a['rho'], label = 'additive model', color = 'b')
    ax2.plot(result_rain_m['E'], result_rain_m['rho'], label = 'multiplicative model', color='c')
    fig.suptitle('Simplex Evaluation for rain in ISLES SUR SORGUE')
    ax1.set_xlabel('Embedding Dimension (E)')
    ax1.set_ylabel('correlation coefficient rho', color='r')
    ax2.set_ylabel('correlation coefficient rho', color='c')
    ax1.tick_params(axis='y', labelcolor='r')
    ax2.tick_params(axis='y', labelcolor='b')
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    ### S-MAP ALGORITHM

    s_map = fun.run_smap(river_flow, 2)
    s_map_m = fun.run_smap(river_flow_m, 3)
    s_map_a = fun.run_smap(river_flow_a, 2)

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(s_map['theta'], s_map['rho'], label = 'no preprocess', color='r')
    ax2.plot(s_map_a['theta'], s_map_a['rho'], label = 'additive model', color='b')
    ax2.plot(s_map_m['theta'], s_map_m['rho'], label = 'multiplicative model', color='c')
    fig.suptitle('S-map evaluation for mean_flow FONTAINE DE VAUCLUSE 1')
    ax1.set_xlabel('non linear coefficient theta')
    ax1.set_ylabel('correlation coefficient rho', color='r')
    ax2.set_ylabel('correlation coefficient rho', color='c')
    ax1.tick_params(axis='y', labelcolor='r')
    ax2.tick_params(axis='y', labelcolor='b')
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    s_map_rain = fun.run_smap(river_flow, 3)
    s_map_rain_m = fun.run_smap(river_flow_m, 6)
    s_map_rain_a = fun.run_smap(river_flow_a, 5)

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(s_map_rain['theta'], s_map_rain['rho'], label = 'no preprocess', color='r')
    ax2.plot(s_map_rain_a['theta'], s_map_rain_a['rho'], label = 'additive model', color='b')
    ax2.plot(s_map_rain_m['theta'], s_map_rain_m['rho'], label = 'multiplicative model', color='c')
    fig.suptitle('S-map evaluation for rain for ISLES SUR SORGUE')
    ax1.set_xlabel('non linear coefficient theta')
    ax1.set_ylabel('correlation coefficient rho', color='r')
    ax2.set_ylabel('correlation coefficient rho', color='c')
    ax1.tick_params(axis='y', labelcolor='r')
    ax2.tick_params(axis='y', labelcolor='b')
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    ### CCM ALGORITHM (on utilise les données brutes plutot que preprocess ici)

    df = pd.merge(meteo, fdv1, on='date', how='outer', suffixes = ('_meteo', '_riviere'))
    E_best_causal = fun.find_best_E(df, 'mean_flow', 'precipitation')
    E_best_noncausal = fun.find_best_E(df, 'precipitation', 'mean_flow')

    libs = list(range(20, 81, 20)) + list(range(100, 1001, 100))

    causal = fun.run_ccm_bootstrap(df, 'mean_flow', 'precipitation', E_best_causal, libs=libs)
    non_causal = fun.run_ccm_bootstrap(df, 'precipitation', 'mean_flow', E_best_noncausal, libs=libs)


    final_causal0 = fun.final_ccm(df, 'mean_flow', 'precipitation', libs=libs, Tp=0)
    final_causal1 = fun.final_ccm(df, 'mean_flow', 'precipitation', libs=libs, Tp=-1)
    final_causal2 = fun.final_ccm(df, 'mean_flow', 'precipitation', libs=libs, Tp=-2)
    final_causal3 = fun.final_ccm(df, 'mean_flow', 'precipitation', libs=libs, Tp=-3)
    final_causal5 = fun.final_ccm(df, 'mean_flow', 'precipitation', libs=libs, Tp=-5)
    final_causal10 = fun.final_ccm(df, 'mean_flow', 'precipitation', libs=libs, Tp=-10)
    final_causal30 = fun.final_ccm(df, 'mean_flow', 'precipitation', libs=libs, Tp=-30)
    final_noncausal = fun.final_ccm(df, 'precipitation', 'mean_flow', libs=libs)


    stat_causal0 = final_causal0[0]
    stat_causal1 = final_causal1[0]
    stat_causal2 = final_causal2[0]
    stat_causal3 = final_causal3[0]
    stat_causal5 = final_causal5[0]
    stat_causal10 = final_causal10[0]
    stat_causal30 = final_causal30[0]
    stat_noncausal = final_noncausal[0]
    
    libsize = stat_causal0.index

    corr0 = df['mean_flow'].corr(df['precipitation'])

    plt.figure()
    plt.plot(libsize, stat_causal0[0.5], color='r', label= 'causal (precipitation->mean_flow) 0 lag' + " median")
    plt.fill_between(libsize, stat_causal0[0.25], stat_causal0[0.75], color='r', alpha=0.3)
    plt.plot(libsize, stat_causal1[0.5], label= 'causal (precipitation->mean_flow) 1 lag' + " median")
    plt.plot(libsize, stat_causal2[0.5], label= 'causal (precipitation->mean_flow) 2 lag' + " median")
    plt.plot(libsize, stat_causal3[0.5], label= 'causal (precipitation->mean_flow) 3 lag' + " median")
    plt.plot(libsize, stat_causal5[0.5], label= 'causal (precipitation->mean_flow) 5 lag' + " median")
    plt.plot(libsize, stat_causal10[0.5],label= 'causal (precipitation->mean_flow) 10 lag' + " median")
    plt.plot(libsize, stat_causal30[0.5],label= 'causal (precipitation->mean_flow) 30 lag' + " median")
    plt.plot(libsize, stat_noncausal[0.5], color='g', label= 'noncausal (mean_flow->precipitation)' + " median")
    plt.fill_between(libsize, stat_noncausal[0.25], stat_noncausal[0.75], color='g', alpha=0.3)
    plt.axhline(corr0, color='b', linestyle='--', label=f"corr(mean_flow, precipitation)")
    plt.title('CCM between flow and rain')
    plt.xlabel('LibSize')
    plt.ylabel('Correlation of Pearson (rho)')
    plt.legend()
    plt.ylim(0,1)


    df = pd.DataFrame(data = {'date' : meteo[meteo['date']==fdv1['date']]['date'], 'mean_flow' : river_flow_a, 'precipitation' : rain_a})
    libs = list(range(20, 81, 20)) + list(range(100, 465, 100))

    final_causal0 = fun.final_ccm(df, 'mean_flow', 'precipitation', libs=libs, Tp=0)
    final_causal1 = fun.final_ccm(df, 'mean_flow', 'precipitation', libs=libs, Tp=-1)
    final_causal2 = fun.final_ccm(df, 'mean_flow', 'precipitation', libs=libs, Tp=-2)
    final_causal3 = fun.final_ccm(df, 'mean_flow', 'precipitation', libs=libs, Tp=-3)
    final_causal5 = fun.final_ccm(df, 'mean_flow', 'precipitation', libs=libs, Tp=-5)
    final_causal10 = fun.final_ccm(df, 'mean_flow', 'precipitation', libs=libs, Tp=-10)
    final_causal30 = fun.final_ccm(df, 'mean_flow', 'precipitation', libs=libs, Tp=-30)
    final_noncausal = fun.final_ccm(df, 'precipitation', 'mean_flow', libs=libs)

    stat_causal0 = final_causal0[0]
    stat_causal1 = final_causal1[0]
    stat_causal2 = final_causal2[0]
    stat_causal3 = final_causal3[0]
    stat_causal5 = final_causal5[0]
    stat_causal10 = final_causal10[0]
    stat_causal30 = final_causal30[0]


    stat_noncausal = final_noncausal[0]
    libsize = stat_causal0.index

    corr0 = df['mean_flow'].corr(df['precipitation'])

    plt.figure()
    plt.plot(libsize, stat_causal0[0.5], color='r', label= 'causal (precipitation->mean_flow) 0 lag' + " median")
    plt.fill_between(libsize, stat_causal0[0.25], stat_causal0[0.75], color='r', alpha=0.3)
    plt.plot(libsize, stat_causal1[0.5], label= 'causal (precipitation->mean_flow) 1 lag' + " median")
    plt.plot(libsize, stat_causal2[0.5], label= 'causal (precipitation->mean_flow) 2 lag' + " median")
    plt.plot(libsize, stat_causal3[0.5], label= 'causal (precipitation->mean_flow) 3 lag' + " median")
    plt.plot(libsize, stat_causal5[0.5], label= 'causal (precipitation->mean_flow) 5 lag' + " median")
    plt.plot(libsize, stat_causal10[0.5], label= 'causal (precipitation->mean_flow) 10 lag' + " median")
    plt.plot(libsize, stat_causal30[0.5], label= 'causal (precipitation->mean_flow) 30 lag' + " median")
    plt.plot(libsize, stat_noncausal[0.5], color='g', label= 'noncausal (mean_flow->precipitation)' + " median")
    plt.fill_between(libsize, stat_noncausal[0.25], stat_noncausal[0.75], color='g', alpha=0.3)
    plt.axhline(corr0, color='b', linestyle='--', label=f"corr(mean_flow, precipitation)")
    plt.title('CCM between flow and rain being preprocess with additive model')
    plt.xlabel('LibSize')
    plt.ylabel('Correlation of Pearson (rho)')
    plt.legend()
    plt.ylim(0,1)


    df = pd.DataFrame(data = {'date' : meteo[meteo['date']==fdv1['date']]['date'], 'mean_flow' : river_flow_m, 'precipitation' : rain_m})
    
    final_causal0 = fun.final_ccm(df, 'mean_flow', 'precipitation', libs=libs, Tp=0)
    final_causal1 = fun.final_ccm(df, 'mean_flow', 'precipitation', libs=libs, Tp=-1)
    final_causal2 = fun.final_ccm(df, 'mean_flow', 'precipitation', libs=libs, Tp=-2)
    final_causal3 = fun.final_ccm(df, 'mean_flow', 'precipitation', libs=libs, Tp=-3)
    final_causal5 = fun.final_ccm(df, 'mean_flow', 'precipitation', libs=libs, Tp=-5)
    final_causal10 = fun.final_ccm(df, 'mean_flow', 'precipitation', libs=libs, Tp=-10)
    final_causal30 = fun.final_ccm(df, 'mean_flow', 'precipitation', libs=libs, Tp=-30)

    final_noncausal = fun.final_ccm(df, 'precipitation', 'mean_flow', libs=libs, Tp=0)

    stat_causal0 = final_causal0[0]
    stat_causal1 = final_causal1[0]
    stat_causal2 = final_causal2[0]
    stat_causal3 = final_causal3[0]
    stat_causal5 = final_causal5[0]
    stat_causal10 = final_causal10[0]
    stat_causal30 = final_causal30[0]

    stat_noncausal = final_noncausal[0]
    libsize = stat_causal0.index

    corr0 = df['mean_flow'].corr(df['precipitation'])

    plt.figure()
    plt.plot(libsize, stat_causal0[0.5], color='r', label= 'causal (precipitation->mean_flow) 0 lag' + " median")
    plt.fill_between(libsize, stat_causal0[0.25], stat_causal0[0.75], color='r', alpha=0.3)
    plt.plot(libsize, stat_causal1[0.5], label= 'causal (precipitation->mean_flow) 1 lag' + " median")
    plt.plot(libsize, stat_causal2[0.5], label= 'causal (precipitation->mean_flow) 2 lag' + " median")
    plt.plot(libsize, stat_causal3[0.5], label= 'causal (precipitation->mean_flow) 3 lag' + " median")
    plt.plot(libsize, stat_causal5[0.5], label= 'causal (precipitation->mean_flow) 5 lag' + " median")
    plt.plot(libsize, stat_causal10[0.5], label= 'causal (precipitation->mean_flow) 10 lag' + " median")
    plt.plot(libsize, stat_causal30[0.5], label= 'causal (precipitation->mean_flow) 30 lag' + " median")

    plt.plot(libsize, stat_noncausal[0.5], color='g', label= 'noncausal (mean_flow->precipitation)' + " median")
    plt.fill_between(libsize, stat_noncausal[0.25], stat_noncausal[0.75], color='g', alpha=0.3)
    plt.axhline(corr0, color='b', linestyle='--', label=f"corr(mean_flow, precipitation)")
    plt.title('CCM between flow and rain being preprocess with multiplicative model')
    plt.xlabel('LibSize')
    plt.ylabel('Correlation of Pearson (rho)')
    plt.legend()
    plt.ylim(0,1)

    ## this is very strange because it seems that there is a causality mean_flow --> precipitation instead of precipitation --> mean_flow
    # even though we are using different kind of preprocessing
    # i am going to evaluate surrogate correlation to see if we get something different 
    # an explanation could be : artefact/error in the code, lack in the data (particularly a lot of 0 in precipitation data) or delay between precipitation and variation of mean_flow
    # so I can also try to implement delay to but I am not sure how to do it --> need to read the last article

    #avec un lag de 3 jours et sans preprocess on semble obtenir quelque chose de satisfaisant ! il faut maintenant le quantifier :)
'''
    ## surrogate
    
    df = pd.merge(meteo, fdv1, on='date', how='outer', suffixes = ('_meteo', '_riviere'))
    libs = list(range(20, 81, 20)) + list(range(100, 465, 100))

    final_causal = fun.final_ccm(df, 'mean_flow', 'precipitation', libs=libs, Tp=-3)
    final_noncausal = fun.final_ccm(df, 'precipitation', 'mean_flow', libs=libs, Tp=0)

    stat_causal = final_causal[0]
    total_causal = stat_causal[0.5].iloc[-1]
    stat_noncausal = final_noncausal[0]
    total_noncausal = stat_noncausal[0.5].iloc[-1]
    libsize = stat_causal.index

    corr0 = df['mean_flow'].corr(df['precipitation'])

    z1 = np.atanh(corr0)
    z2 = np.atanh(total_causal)
    n1 = len(df[['mean_flow', 'precipitation']].dropna())
    n2 = n1    
    zobs = (z1-z2) / np.sqrt( 1 / (n1-3) + 1 / (n2-3) )

    pval = 2 * stats.norm.cdf(-abs(zobs))

    pvalue = None
    Z_score = None 

    print(f'Fisher Z comparison : Z_score = {Z_score} and pvalue = {pvalue}')

    surr_causal = fun.surrogate_fast(df, 'mean_flow', 'precipitation', comparaison = total_causal, libs=libs)
    surr_noncausal = fun.surrogate_fast(df, 'precipitation', 'mean_flow', comparaison = total_noncausal, libs=libs)

    pvalue_causal = surr_causal[2]
    pvalue_noncausal = surr_noncausal[2]

    surresume_causal = surr_causal[1]
    surresume_noncausal = surr_noncausal[1]

    plt.figure()
    plt.plot(libsize, stat_causal[0.5], color='r', label= 'causal (precipitation->mean_flow) 3 lag' + " median")
    plt.fill_between(libsize, stat_causal[0.25], stat_causal[0.75], color='r', alpha=0.3)
    plt.plot(libsize, stat_noncausal[0.5], color='g', label= 'noncausal (mean_flow->precipitation) no lag' + " median")
    plt.fill_between(libsize, stat_noncausal[0.25], stat_noncausal[0.75], color='g', alpha=0.3)
    plt.axhline(corr0, color='b', linestyle='--', label=f"corr(mean_flow, precipitation)")
    plt.boxplot(surr_causal[0], positions=[0], widths=0.4, patch_artist=True,
           boxprops=dict(facecolor='lightblue'),
           medianprops=dict(color='red', linewidth=2))
    plt.scatter(0, total_causal, color='green', marker='X', s=150, zorder=3)
    plt.text(0.15, total_causal - 0.03, f"pvalue = {pvalue_causal:.3f}", va='top', fontsize=9)    
    plt.boxplot(surr_noncausal[0], positions=[0], widths=0.4, patch_artist=True,
           boxprops=dict(facecolor='lightblue'),
           medianprops=dict(color='blue', linewidth=2))
    plt.scatter(0, total_noncausal, color='green', marker='X', s=150, zorder=3)
    plt.text(0.15, total_noncausal - 0.03, f"pvalue = {pvalue_noncausal:.3f}", va='top', fontsize=9)    
    plt.title('CCM between flow and rain being preprocess with multiplicative model')
    plt.xlabel('LibSize')
    plt.ylabel('Correlation of Pearson (rho)')
    plt.legend()
    plt.ylim(0,1)


    ### DIFFERENT EMBEDDING

    ### INTERACTION STRENGTH 

    ### PARAMETER EXPLORATION 

    plt.show()

