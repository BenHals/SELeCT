#%%
import pandas as pd
import numpy as np
import pathlib
import seaborn as sns
import matplotlib.pyplot as plt
import os
import json
import seaborn as sns

#%%
from eval_utils import get_classifier_matched_results
get_ipython().run_line_magic('matplotlib', 'ipympl')


# %%

def ts_to_ranges(ts):
    ranges = []
    prev_val = None
    for i, v in enumerate(ts.values):
        if np.isnan(v):
            continue
        if prev_val != v:
            if prev_val != None:
                ranges[-1][1] = i-1
            ranges.append([i, None, int(v)])
        prev_val = v
    ranges[-1][1] = ts.shape[0]
    print(ranges)
    return ranges

def plot_probabilities(prob_column, save_file=False, run_individual=False, smoothing=True):
    probs = df[prob_column].str.split(';', expand=True)
    active_model = df['active_model']
    gt_model = df['ground_truth_concept']
    active_model_ranges = ts_to_ranges(active_model)
    gt_model_ranges = ts_to_ranges(gt_model)
    del_cols = []
    val_nas = {}
    for c in probs.columns:

        col_id = probs[c].str.rsplit(':').str[0].astype('float').unique()[-1]
        print(probs[c].str.rsplit(':').str[0].astype('float').unique())
        probs[c] = probs[c].str.rsplit(':').str[-1].astype('float')
        probs[f"v{col_id}"] = probs[c]
        val_nas[c] = probs[c].isna().sum()
        # if probs[c].isna().sum() > probs.shape[0] * 0.5:
        #     del_cols.append(f"v{col_id}")
        del_cols.append(c)
    print(val_nas)
    probs = probs.drop(del_cols, axis=1)
    del_cols = []
    for c in probs.columns:
        probs[int(float(c[1:]))] = probs[c]
        del_cols.append(c)
    probs = probs.drop(del_cols, axis=1)
    probs['ts'] = probs.index
    m_df = pd.melt(probs.iloc[::10, :], id_vars='ts')
    m_df['variable'] = m_df['variable'].astype('category')
    m_df['value'] = m_df['value'].replace({-1:0})
    # m_df['value'] = m_df['value'].rolling(window=int(probs.shape[0]*0.001)).mean()
    if smoothing:
        m_df['value'] = m_df['value'].rolling(window=7).mean()

    sns.lineplot(data=m_df[m_df['variable'] != '-1'], x='ts', y='value', hue='variable')
    if save_file:
        plt.savefig(f"explore_probabilities/{experiment_name}-{data_name}.pdf")
    
    ax = plt.gca()
    handles, labels = ax.get_legend_handles_labels()
    # colorsplt.rcParams["axes.prop_cycle"].by_key()["color"]
    print(labels)
    for rs, re, v in active_model_ranges:
        if str(v) in labels:
            color = handles[labels.index(str(v))].get_color()
            plt.hlines(y = -0.05, xmin = rs, xmax = re, colors = [color])
        else:
            plt.hlines(y = -0.01, xmin = rs, xmax = re, colors=["red"])
    for rs, re, v in gt_model_ranges:
        if str(v) in labels:
            color = handles[labels.index(str(v))].get_color()
            plt.hlines(y = -0.1, xmin = rs, xmax = re, colors = [color])
        else:
            plt.hlines(y = -0.1, xmin = rs, xmax = re, colors=["red"])
    plt.xlim((0, probs.shape[0]))
    plt.title(f"{data_name}-{prob_column}")
    plt.legend().remove()
    plt.show()
    if not run_individual:
        return
    for c in m_df['variable'].unique():
        sns.lineplot(data=m_df[m_df['variable'] == c], x='ts', y='value', hue='variable')
        if save_file:
            plt.savefig(f"explore_probabilities/{experiment_name}-{data_name}-{c}.pdf")
        plt.show()

# %%
# experiment_name =  "profilingTLikelihoodM3"
# experiment_name =  "sensNectar"
# experiment_name =  "sensitiivtyVSestimatorriskatESR0_5"
# experiment_name =  "windowsizeT"
# experiment_name =  "windowsizeVsimgap"
# experiment_name = "windowsizeVminwindowratio"
# experiment_name = "windowsizeVfpgap"
# experiment_name = "windowsizeVnafpgap"
# experiment_name = "windowsizeVobgap"
# experiment_name = "sensVsimgap"
# experiment_name = "sensVfpgap"
# experiment_name = "sensVser"
# experiment_name = "gptVgpwm"
# experiment_name = "sensAllT2"
# experiment_name = "bufferratio"
# experiment_name = "defaultT1"
# experiment_name = "T5"
# experiment_name = "T11"
# experiment_name = "T13"
# experiment_name = "T13-2"
# experiment_name = "T13-6"
# experiment_name = "TV4-6"
# experiment_name = "T14-8"
# experiment_name = "TV4-6"
# experiment_name = "T14-7"
# experiment_name = "TV4-8"
# experiment_name = "TV4-13"
# experiment_name = "TV4-6"
# experiment_name = "TV4"
# experiment_name = "TV5-5"
# experiment_name = "TV5-fpgap"
# experiment_name = "defaultT1mimwindowratio"
# experiment_name = "defaultT1minwindowratio2"
# experiment_name = "defaultT1obgap"
# experiment_name = "defaultT1fp_gap"
# experiment_name = "defaultT1nafpgap"
# experiment_name = "defaultT1minsimstdev"
# experiment_name = "fingerprintgraceperiod"
# experiment_name = "T"
# experiment_name = "nectarAQTempsensitivityVesr"
# experiment_name = "contlikelihoodVdriftlikelihood"
# experiment_name = "minlikelihoodVdriftlikelihood"
# experiment_name = "defaultV4"
# experiment_name = "defaultV4gpt"
# experiment_name = "defaultV4likelihoods"
# experiment_name = "defaultV4minimumconceptlikelihood"
# experiment_name = "defaultV4gpwm"
# experiment_name = "defaultV4fingerprintgraceperiod"
# experiment_name = "defaultV4"
# experiment_name = "defaultV5"
# experiment_name = "defaultV8"
# experiment_name = "defaultV8ifpacf"
# experiment_name = "defaultV8ifpacfS2"
# experiment_name = "defaultV8ifpacf-bufferratio"
# experiment_name = "defaultV8ifpacf-minsimstdev"
# experiment_name = "defaultV8ifpacf-maxsimstdev"
# experiment_name = "defaultV8ifpacf-obgap"
# experiment_name = "defaultV8ifpacf-minestimatedposteriorthreshold2"
# experiment_name = "defaultV8ifpacf-simgap"
# experiment_name = "defaultV8ifpacf-mindriftlikelihoodthreshold"
# experiment_name = "defaultV8ifpacf-minimumconceptlikelihood"
# experiment_name = "defaultV8ifpacf-stateestimatorswaprisk"
# experiment_name = "defaultV8ifpacf-thresholds"
# experiment_name = "defaultV8ifpacf-mindriftlikelihoodthreshold"
# experiment_name = "defaultV8ifacf"
# experiment_name = "defaultV6"
# experiment_name = "defaultV7-2nopacf"
# experiment_name = "defaultV7nopacf"
# experiment_name = "defaultV7"
# experiment_name = "defaultV5sens"
# experiment_name = "defaultV4-2"
# experiment_name = "SV6-fpgap"
# experiment_name = "TV4"
# experiment_name = "TV7"
# experiment_name = "T14-8"
# experiment_name = "TV8-12-7t-nopacf"
# experiment_name = "TV8-12-7-nopacf"
# experiment_name = "Tmerge"
# experiment_name = "Tmerge5"
# experiment_name = "Tmerge5-S0-HThresh"
# experiment_name = "tD-5"
# experiment_name = "tD-2"
# experiment_name = "Tmerge4-3"    
# experiment_name = "defaultV8ifpacf-bypassgraceperiodthreshold"
# experiment_name = "defaultV8ifpacf-bypassgraceperiodthreshold2"
# experiment_name = "defaultV8ifpacf-stateestimatorrisk"
# experiment_name = "defaultV8ifpacf-mergethreshold"
# experiment_name = "defaultV8ifpacf-stategraceperiodwindowmultiplier"
# experiment_name = "defaultV8ifpacf-fingerprintgraceperiod"
# experiment_name = "defaultV8ifpacf-thresholds2"
# experiment_name = "defaultV8ifpacf-minwindowratio"
# experiment_name = "TD-6"
# experiment_name = "TD-13"
# experiment_name = "TD-11-2"
# experiment_name = "TD-2-11"
# experiment_name = "TD-2-10"
# experiment_name = "bounds_test"
# experiment_name = "defaultV8ifpacf-windowsize"
# experiment_name = "defaultV85-thresholds"
# experiment_name = "tD-swap_risks_05"
# experiment_name = "tD-swap_risks_065"
# experiment_name = "tD-swap_risks_075"
# experiment_name = "tD-swap_risks_2_075"
# experiment_name = "tD-swap_risks_2_04"
# experiment_name = "tD-swap_risks_085"
# experiment_name = "tD-thresholds"
# experiment_name = "testSettings-V3"
# experiment_name = "testSettings-V0"
# experiment_name = "testSettings-V3-7"
# experiment_name = "testSettings-V3-6"
# experiment_name = "testSettings-V3-P0"
# experiment_name = "testSettings-V3-P11"
# experiment_name = "testSettings-V3-P13-2"
# experiment_name = "testSettings-V3-P12-3"
# experiment_name = "testSettings-V3-P12-10"
# experiment_name = "testSettings-V3-P12-2"
# experiment_name = "testSettings-V3-P12-11"
# experiment_name = "testSettings-V3-P12-12"
# experiment_name = "testSettings-V3-P12-8"
# experiment_name = "testSettings-V3-P13"
# experiment_name = "rtreehard_test3"
# experiment_name = "testSettings-V3-P13-3"
# experiment_name = "testSettings-V3-P13-7"
# experiment_name = "testSettings-V3-P13-8-2"
# experiment_name = "testSettings-V3-P13-8-12-8"
# experiment_name = "testSettings-D10-B2-1"
# experiment_name = "testSettings-D10-B3-3-27"
# experiment_name = "testSettings-D10-B3-T-4-11-6-2"
# experiment_name =  "noise-3"
# experiment_name =  "5rep"
# experiment_name =  "5rep-2"
# experiment_name =  "ablation-basicprior"
# experiment_name =  "ablation5-baset3"
# experiment_name =  "ablation5-basicprior2"
# experiment_name =  "test-clock-2"
# experiment_name =  "noise"
# experiment_name =  "test_diff"
experiment_name =  "prior-complexity"
# experiment_name =  "ablation_selection"




# experiment_name = "testSettings-V3"
# experiment_name = "tD-T-backgroundstatepriormultiplier"
# experiment_name = "tD-T"
# experiment_name = "tD-T-2"
# experiment_name = "tD-T-4-backgroundstatepriormultiplier"
# experiment_name = "tD-T-7-backgroundstatepriormultiplier"
# experiment_name = "tD-T-6"
# experiment_name = "tD-T-7-bufferratio"
# experiment_name = "tD-T-7-maxsimstdev"
# experiment_name = "tD-T-7-obgap"
# experiment_name = "tD-T-9-nafpgap"
# experiment_name = "tD-T-7"
# experiment_name = "tD-T-9-maxsimstdev2"
# experiment_name = "tD-T-9-4-multihoppenalty"
# experiment_name = "tD-T-10-5"
# experiment_name = "tD-T-D10-B2-9-multihoppenalty"
# experiment_name = "tD-T-D10-B2-13"
# experiment_name = "tD-T-D10-B2-10"
# experiment_name = "tD-T-10"
# experiment_name = "tD-T-9-10-multihoppenalty"
# experiment_name = "D10-B3-12-prevstateprior"
# experiment_name = "D10-B3-23-zeroprobminimum"
# experiment_name = "D10-B3-26-2-m"
# experiment_name = "D10-B3-28-m"
# experiment_name = "D10-B3-20"
# experiment_name = "D10-B3-25"
# experiment_name = "param-prevstateprior"
# experiment_name = "param-multihoppenalty"
# experiment_name = "param-zeroprobminimum"
# experiment_name = "param-mergethreshold"
# experiment_name = "param-backgroundstatepriormultipler"
# experiment_name = "param-bufferratio"
# experiment_name = "param-maxsimstdev"
# experiment_name = "param-minsimstdev"
# experiment_name = "param-obgap"
# experiment_name = "param-nafpgap"
# experiment_name = "param-fpgap"
# experiment_name = "param-simgap"
# experiment_name = "param-minestimatedposteriorthreshold"
# experiment_name = "param-mindriftlikelihoodthreshold"
# experiment_name = "param-minimumconceptlikelihood"
# experiment_name = "param-stateestimatorswaprisk"
# experiment_name = "param-stateestimatorrisk"
# experiment_name = "param-bypassgraceperiodthreshold"
# experiment_name = "param-fingerprintgraceperiod"
# experiment_name = "param-stategraceperiodwindowmultiplier"
# experiment_name = "param-minwindowratio"
# experiment_name = "param-windowsize"
# experiment_name = "param-sensitivity"
# experiment_name = "Defaults2-width"
# experiment_name = "Defaults2-noise"
# experiment_name = "Defaults2-TMnoise"
# experiment_name = "BaseL4"
experiment_name = "BaseL3_wAblation"
# experiment_name = "concept_diff"

base_path = pathlib.Path(__file__).absolute().parents[1] / 'output' / experiment_name
# base_path = pathlib.Path("//TOMSK/shared_results") / experiment_name
# base_path = pathlib.Path(r"C:\Users\Ben\Documents\shared_results") / experiment_name

# experiment_name = "TV8-4-nopacf"
# experiment_name = "TV8-12-7-nopacf"
# experiment_name = "TV7-2"
# experiment_name = "TV7-3t"
# experiment_name = "onlineT5Orig"
# experiment_name = "onlineT5-kas-3-acf0"
# experiment_name = "SV6-highsens"
# experiment_parameters = ["sensitivity", "state_estimator_risk"]
# experiment_parameters = ["window_size", "similarity_gap"]
# experiment_parameters = ["window_size", "min_window_ratio"]
# experiment_parameters = ["window_size", "fp_gap"]
# experiment_parameters = ["window_size", "nonactive_fp_gap"]
# experiment_parameters = ["window_size", "observation_gap"]
# experiment_parameters = ["sensitivity", "similarity_gap"]

# experiment_parameters = ["sensitivity", "fp_gap"]
# experiment_parameters = ["sensitivity", ""]
# experiment_parameters = ["sensitivity", "state_estimator_risk"]
# experiment_parameters = ["state_estimator_risk", "state_estimator_swap_risk", "data_name", "classifier"]
# experiment_parameters = ["data_name", "state_estimator_swap_risk", "classifier", "state_estimator_risk"]
# experiment_parameters = ["bypass_grace_period_threshold", "state_grace_period_window_multiplier"]
# experiment_parameters = ["min_drift_likelihood_threshold", "min_estimated_posterior_threshold", "data_name"]
# experiment_parameters = ["fp_gap", "observation_gap"]
# experiment_parameters = ["minimum_concept_likelihood", "min_drift_likelihood_threshold"]
# experiment_parameters = ["sensitivity", "buffer_ratio"]
# experiment_parameters = ["data_name", "FICSUM"]
# experiment_parameters = ["data_name", "classifier"]
# experiment_parameters = ["data_name", "background_state_prior_multiplier", "classifier"]
# experiment_parameters = ["data_name", "buffer_ratio", "classifier"]
# experiment_parameters = ["data_name", "pcpu", "cpu", "classifier"]

# experiment_parameters = ["FICSUM", "buffer_ratio"]
# experiment_parameters = ["FICSUM", "similarity_stdev_min"]
# experiment_parameters = ["data_name", "buffer_ratio"]
# experiment_parameters = ["data_name", "min_window_ratio"]
# experiment_parameters = ["data_name", "observation_gap"]
# experiment_parameters = ["data_name", "fp_gap"]
# experiment_parameters = ["data_name", "nonactive_fp_gap"]
# experiment_parameters = ["data_name", "similarity_stdev_min"]
# experiment_parameters = ["data_name", "similarity_stdev_max"]
# experiment_parameters = ["data_name", "bypass_grace_period_threshold"]
# experiment_parameters = ["data_name", "minimum_concept_likelihood"]
# experiment_parameters = ["data_name", "state_grace_period_window_multiplier"]
# experiment_parameters = ["data_name", "fingerprint_grace_period"]
# experiment_parameters = ["data_name", "sensitivity"]
# experiment_parameters = ["data_name", "min_estimated_posterior_threshold"]
# experiment_parameters = ["data_name", "similarity_gap"]
# experiment_parameters = ["data_name", "min_drift_likelihood_threshold"]
# experiment_parameters = ["data_name", "minimum_concept_likelihood"]
# experiment_parameters = ["data_name", "state_estimator_swap_risk"]
# experiment_parameters = ["data_name", "state_estimator_risk"]
# experiment_parameters = ["data_name", "merge_threshold"]
# experiment_parameters = ["data_name", "window_size"]
# experiment_parameters = ["data_name", "similarity_stdev_min", "classifier"]
# experiment_parameters = ["data_name", "similarity_stdev_max", "classifier"]
# experiment_parameters = ["data_name", "multihop_penalty", "classifier"]
# experiment_parameters = ["data_name", "TMnoise", "sensitivity", "zero_prob_minimum", "prev_state_prior", "multihop_penalty", "classifier"]
# experiment_parameters = ["data_name", "prev_state_prior", "classifier"]
# experiment_parameters = ["data_name", "multihop_penalty", "classifier"]
# experiment_parameters = ["data_name", "zero_prob_minimum", "classifier"]
# experiment_parameters = ["data_name", "merge_threshold", "classifier"]
# experiment_parameters = ["data_name", "background_state_prior_multiplier", "classifier"]
# experiment_parameters = ["data_name", "buffer_ratio", "classifier"]
# experiment_parameters = ["data_name", "similarity_stdev_max", "classifier"]
# experiment_parameters = ["data_name", "similarity_stdev_min", "classifier"]
# experiment_parameters = ["data_name", "observation_gap", "classifier"]
# experiment_parameters = ["data_name", "nonactive_fp_gap", "classifier"]
# experiment_parameters = ["data_name", "fp_gap", "classifier"]
# experiment_parameters = ["data_name", "similarity_gap", "classifier"]
# experiment_parameters = ["data_name", "min_estimated_posterior_threshold", "classifier"]
# experiment_parameters = ["data_name", "min_drift_likelihood_threshold", "classifier"]
# experiment_parameters = ["data_name", "minimum_concept_likelihood", "classifier"]
# experiment_parameters = ["data_name", "state_estimator_swap_risk", "classifier"]
# experiment_parameters = ["data_name", "state_estimator_risk", "classifier"]
# experiment_parameters = ["data_name", "bypass_grace_period_threshold", "classifier"]
# experiment_parameters = ["data_name", "fingerprint_grace_period", "classifier"]
# experiment_parameters = ["data_name", "state_grace_period_window_multiplier", "classifier"]
# experiment_parameters = ["data_name", "min_window_ratio", "classifier"]
# experiment_parameters = ["data_name", "window_size", "classifier"]
# experiment_parameters = ["data_name", "sensitivity", "classifier"]
# experiment_parameters = ["data_name", "drift_width", "classifier"]
# experiment_parameters = ["data_name", "noise", "classifier"]
# experiment_parameters = ["data_name", "conceptdifficulty", "classifier"]
# experiment_parameters = ["data_name", "TMnoise", "classifier"]
# experiment_parameters = ["data_name", "TMforward", "classifier"]

experiment_parameters = ["data_name", "classifier"]

# experiment_parameters = ["data_name", "observation_gap", "classifier"]
# experiment_parameters = ["data_name", "nonactive_fp_gap", "classifier"]

# experiment_parameters = ["sensitivity", "fingerprint_grace_period"]
# experiment_IVs = ['overall_accuracy', 'GT_mean_f1', 'nomerge-GT_mean_f1', 'm-GT_mean_f1', 'r-GT_mean_f1', "overall_time"]
# experiment_IVs = ['overall_accuracy', 'GT_mean_f1', 'kappa', 'kappa_m', 'kappa_t', "overall_time"]
# experiment_IVs = ['overall_accuracy', 'nomerge-GT_mean_f1', 'kappa', 'kappa_m', 'kappa_t', "overall_time"]
# experiment_IVs = ['overall_accuracy', 'overall_time']
experiment_IVs = ['overall_time', 'overall_mem', 'peak_fingerprint_mem']
# experiment_name =  "expDefault"



data_name = ''
# data_name = 'cmc'
# data_name = 'Arabic'
# data_name = 'STAGGERS'
# data_name = 'STAGGER'
# data_name = 'Arabic'
# data_name = 'AQSex'
# data_name = 'RTREE'
# data_name = 'RTREESAMPLE'
# data_name = 'AQTemp'
# print(base_path.resolve())
# results_paths = list(base_path.rglob(f"results_run_*.txt"))
# results_paths = [x for x in results_paths if data_name in str(x)]
# results = [json.load(rp.open()) for rp in results_paths]
# print(len(results))
# df = pd.DataFrame(results)
# if 'GT_mean_f1' not in df.columns:
#     df['GT_mean_f1'] = df['m-GT_mean_f1']
# if df.shape[0] == 0:
#     raise ValueError("Data not loaded - maybe path is wrong")
# # seeds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
# # seeds = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
# # seeds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
seeds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45]
# # seeds = [1, 2, 3, 4, 5]
# # seeds = [6, 7, 8, 9, 10]
# # seeds = [11, 12, 13, 14, 15]
# # seeds = [16, 17, 18, 19, 20]
# # seeds = [2]
# df = df[df['seed'].isin(seeds) ]
# # df = df[df['state_estimator_swap_risk'].isin([0.25])]
# # df = df[df['state_estimator_risk'].isin([0.5])]
# # df = df[df['state_grace_period_window_multiplier'].isin([8])]
# # df = df[df['min_drift_likelihood_threshold'].isin([0.1])]
# # df = df[df['sensitivity'].isin([0.1])]
# # df = df[df['sensitivity'].isin([0.07, 0.09])]
# # df = df[df['merge_threshold'].isin([0.85])]
# df = df[df['merge_threshold'].isin([0.9])]
restrictions = []
# restrictions = [

#     ("merge_threshold", [0.9])
# ]
selected, groups, grouped = get_classifier_matched_results(base_path, experiment_name, parameters_of_interest=experiment_parameters, IVs_of_interest=experiment_IVs, seeds=seeds, restrictions=restrictions)
# grouped = df[[*experiment_IVs, *experiment_parameters]]
# selected = df[[*experiment_IVs, *experiment_parameters]]
# groups = df.groupby(experiment_parameters)
# sizes = groups.size()
# grouped['size'] = sizes
# grouped = groups.mean()[experiment_IVs]
# grouped=grouped.loc[(sizes > 4).values]
pd.set_option('display.max_rows', None)
grouped = grouped.reset_index()

grouped
# grouped[grouped['classifier'] == 'CC_nomerge'][['data_name', 'nomerge-GT_mean_f1']]
#%%
fig = plt.figure()
sns.lineplot(data=selected, x=experiment_parameters[1], y=experiment_IVs[0], hue=experiment_parameters[0])
plt.show()
fig = plt.figure()
sns.lineplot(data=selected, x=experiment_parameters[1], y=experiment_IVs[1], hue=experiment_parameters[0])
plt.show()
#%%
%matplotlib inline
fig = plt.figure()
sns.scatterplot(data=grouped, x=experiment_parameters[1], y=experiment_parameters[0], hue=experiment_IVs[0], s=100, palette="rocket")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()
# %%
# %matplotlib inline
fig = plt.figure()
sns.scatterplot(data=grouped, x=experiment_parameters[1], y=experiment_parameters[0], hue=experiment_IVs[1], s=100, palette="rocket")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
#%%
%matplotlib widget
fig = plt.figure()
ax = fig.gca(projection='3d')
surf=ax.plot_trisurf(grouped[experiment_parameters[1]], grouped[experiment_parameters[0]], grouped[experiment_IVs[0]], cmap=plt.cm.viridis, linewidth=0.2)
fig.colorbar( surf, shrink=0.5, aspect=5)
fig_name = f"{data_name} seeds {seeds} - {experiment_IVs[0]} 3D"
plt.title(fig_name)
plt.savefig(f"parameters/{fig_name.replace(' ', '_')}.pdf")

# %%
# %matplotlib ipympl
fig = plt.figure()
# ax = fig.gca(projection='3d')
ax = fig.add_subplot(111, projection='3d')
surf=ax.plot_trisurf(grouped[experiment_parameters[1]], grouped[experiment_parameters[0]], grouped[experiment_IVs[1]], cmap=plt.cm.viridis, linewidth=0.2)
fig.colorbar( surf, shrink=0.5, aspect=5, ax=ax)
fig_name = f"{data_name} seeds {seeds} - {experiment_IVs[1]} 3D"
plt.title(fig_name)
plt.savefig(f"parameters/{fig_name.replace(' ', '_')}.pdf")
# %%
%matplotlib widget
# %%
df_hm = selected.groupby(experiment_parameters).mean()[experiment_IVs[0]]
df_hm = df_hm.unstack(experiment_parameters[1])
sns.heatmap(df_hm.iloc[::-1], cmap="rocket")
fig_name = f"{data_name} seeds {seeds} - {experiment_IVs[0]}"
plt.title(fig_name)
plt.savefig(f"parameters/{fig_name.replace(' ', '_')}.pdf")


# %%
df_hm = selected.groupby(experiment_parameters).mean()[experiment_IVs[1]]
df_hm = df_hm.unstack(experiment_parameters[1])
sns.heatmap(df_hm.iloc[::-1], cmap="rocket")
fig_name = f"{data_name} seeds {seeds} - {experiment_IVs[1]}"
plt.title(fig_name)
plt.savefig(f"parameters/{fig_name.replace(' ', '_')}.pdf")

# %%
grouped.to_latex(f"{experiment_name}_table.tex", index=False, float_format="{:0.2%}".format)
# %%
grouped[grouped['data_name'].isin(['AQSex', 'AQTemp', 'Arabic', 'cmc', 'RTREESAMPLE_HARD', 'STAGGERS'])].to_latex(f"{data_name}_table.tex", index=False, float_format="{:0.2%}".format)


#%%
groups
# %%
def str_stdev(x):
    return f"{np.mean(x):.2%} ({np.std(x):.2%})"

g = groups.aggregate(str_stdev)[experiment_IVs]
g['size'] = groups.size()
# %%
g.reset_index().to_latex(f"{experiment_name}_table.tex", index=False, float_format="{:0.2%}".format)

# %%
plt.rcParams["font.family"] = "Times New Roman"
rc('text', usetex=False)
sns.set_context('talk')
plt.rcParams['font.size'] = 10
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['xtick.major.size'] = 6
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['ytick.major.size'] = 6
plt.rcParams['axes.linewidth'] = 1
plt.rcParams['lines.linewidth'] = 2
data_name_map = {
    "AQSex": "AQSex",
    "AQTemp": "AQTemp",
    "RTREESAMPLE_HARD": "TREE-H",
    "Arabic": "Arabic",
    "cmc": "CMC",
    "RTREESAMPLE_Diff": "TREE"
}

classifier_name_map = {
    "CC": "SELeCT",
    "arf": "ARF",
    "dwm": "DWM",
    "cpf": "CPF",
    "dynse": "DYNSE",
    "ficsum": "FICSUM",
    "lower_bound": "Lower",
    "upper_bound": "Upper"
}

dashes = {k:'' for k in classifier_name_map.values()}
dashes['Lower'] = (3, 1)
dashes['Upper'] = (3, 1)
dashes['ARF'] = (1, 1)
print(dashes)


# parameter = 'noise'
parameter = "conceptdifficulty"
parameter_alias = "Complexity"
# parameter = "TMnoise"
# parameter = "TMforward"
# data_sets = ['AQSex', 'AQTemp', 'RTREESAMPLE_HARD']
# data_sets = ['Arabic', 'AQTemp', 'cmc']
data_sets = ['RTREESAMPLE_Diff']
# y = "overall_accuracy"
# y_alias = "Accuracy"
y = "kappa"
y_alias = "Kappa"
# data_name = "RTREESAMPLE_HARD"
# fig = plt.figure()
# fig, ax = plt.subplots(2, 2, figsize=(10,8), sharey=True, sharex=True)
fig, ax = plt.subplots(1, 2, figsize=(6.5,2.5), sharey=True, sharex=True)
fig.set_size_inches(6.5,2.5)
# last = 3
last = 1
for i, data_name in enumerate(data_sets):
    r = i // 2
    c = i % 2
    # axis = ax[r, c]
    axis = ax[i]
    s = selected[selected['data_name'] == data_name]
    s['classifier'] = s['classifier'].map(classifier_name_map)
    sns.lineplot(data=s, x=parameter, y=y, hue='classifier', ax=axis, legend=True, style='classifier', dashes=dashes)
    handles, labels = axis.get_legend_handles_labels()
    axis.get_legend().remove()
    axis.set_title(data_name_map[data_name])
    axis.set_ylabel(y_alias)
    axis.set_xlabel(parameter_alias)

i = last
r = i // 2
c = i % 2
# axis = ax[r, c]
axis = ax[i]
print(labels)
axis.legend(handles, labels, loc='upper left', bbox_to_anchor=(0., 1.2), frameon=False)
axis.set_axis_off()
# fig.legend(handles, labels, loc='upper right')
plt.savefig(f"{parameter}5.pdf", bbox_inches='tight')
plt.show()
# # %%
# data_name_map = {
#     "AQSex": "AQSex",
#     "AQTemp": "AQTemp",
#     "RTREESAMPLE_HARD": "TREE-H",
#     "Arabic": "Arabic",
#     "cmc": "CMC",
#     "RTREESAMPLE_Diff": "TREE",
#     "qg": "QG",
#     "RBFMed": "RBF",
#     "UCI-Wine": "Wine"
# }

# classifier_name_map = {
#     "CC": "SELeCT",
#     "arf": "ARF",
#     "ficsum": "FICSUM",
#     "lower_bound": "lower",
#     "upper_bound": "upper"
# }

# column_name_map = {
#     "GT_mean_f1": "C-F1",
#     "drift_width": "Width",
#     "kappa": "$\kappa$",
#     "noise": "Noise",
# }



# # parameter = "Width"
# parameter_values = [0, 100, 500, 2500]
# parameter = "Noise"
# parameter_values = [0, 0.05, 0.1, 0.25]
# performance = ["$\kappa$", "C-F1"]
# data_sets = ['AQSex', 'AQTemp', 'TREE-H']

# def mean_stdev(X):
#     return f"{np.mean(X):.2f}({np.std(X):.2f})"

# def map_list(X):
#     return [x if x not in column_name_map else column_name_map[x] for x in X]

# t = selected.copy()
# # t = t[t['noise'] == 0.0]
# t.columns = map_list(t.columns)
# t['data_name'] = t['data_name'].map(data_name_map)
# t= t.dropna()
# t['classifier'] = t['classifier'].map(classifier_name_map)
# # t['data_name']

# t = t[t['data_name'].isin(data_sets)]
# t = t[t['classifier'].isin(['SELeCT', 'FICSUM', 'lower', 'upper'])]
# t = t[t[parameter].isin(parameter_values)]
# tg = t.groupby(["classifier", parameter, "data_name"]).agg(mean_stdev)[performance].unstack('data_name')
# # t.columns = pd.MultiIndex.from_tuples([(d, p) for d in data_sets for p in performance])
# tg.columns = tg.columns.swaplevel(0, 1)
# tg = tg.reindex(data_sets, axis=1, level=0)
# tg.to_latex(f"{parameter}_datatable.txt", escape=False)
# tg

#%%
data_name_map = {
    "AQSex": "AQSex",
    "AQTemp": "AQTemp",
    "RTREESAMPLE_HARD": "TREE-H",
    "Arabic": "Arabic",
    "cmc": "CMC",
    "RTREESAMPLE_Diff": "TREE",
    "qg": "QG",
    "RBFMed": "RBF",
    "UCI-Wine": "Wine",
    "STAGGERS": "STAGGER",
    "WINDSIM": "WINDSIM"
}

# classifier_name_map = {
#     "CC": "SELeCT",
#     "CC_basicprior": "$S_p$",
#     "CC_MAP": "$S_{MAP}$",
#     "CC_nomerge": "$S_{m}$",
# }
classifier_name_map = {
    "CC": "SELeCT",
    "CC_basicprior": "$S_p$",
    "CC_MAP": "$S_{MAP}$",
    "CC_nomerge": "$S_{m}$",
    "arf": "ARF",
    "ficsum": "FICSUM",
    "lower_bound": "lower",
    "upper_bound": "upper",
    "dwm": "DWM",
    "cpf": "CPF",
    "rcd": "RCD",
    "dynse": "DYNSE",
}

column_name_map = {
    "GT_mean_f1": "C-F1",
    "drift_width": "Width",
    "kappa": "$\kappa$",
    "noise": "Noise",
    "TMnoise": "Transition Noise",
    "overall_time": "Runtime (s)",
    "peak_fingerprint_mem": "Peak Memory (Mb)"
}



# parameter = "Width"
# parameter_values = [0, 100, 500, 2500]
# parameter = "Noise"
# parameter = "Transition Noise"
parameter_values = [0, 0.05, 0.1, 0.25]
parameter = None
# performance = ["$\kappa$", "C-F1"]
# performance = ["$\kappa$", "C-F1", "Runtime (m)"]
performance = ["Runtime (s)", "Peak Memory (Mb)"]
# data_sets = ['AQSex', 'AQTemp', 'TREE-H']
data_sets = ['AQSex', 'AQTemp', 'Arabic', 'CMC', 'TREE-H', 'STAGGER', "WINDSIM"]

def mean_stdev(X):
    return f"{np.mean(X):.2f}({np.std(X):.2f})"

def map_list(X):
    return [x if x not in column_name_map else column_name_map[x] for x in X]

t = selected.copy()
# t = t[t['noise'] == 0.0]
t['Runtime (m)'] = t['overall_time'] / 60
t.columns = map_list(t.columns)
t['data_name'] = t['data_name'].map(data_name_map)
t= t.dropna()
t['classifier'] = t['classifier'].map(classifier_name_map)
# t['data_name']

t = t[t['data_name'].isin(data_sets)]
t = t[t['classifier'].isin(['SELeCT', "$S_p$", "$S_{MAP}$", "$S_{m}$", 'FICSUM', 'lower', 'upper', 'DWM', 'CPF', 'RCD', 'DYNSE'])]
if parameter:
    t = t[t[parameter].isin(parameter_values)]
    tg = t.groupby(["classifier", parameter, "data_name"]).agg(mean_stdev)[performance].unstack('data_name')
else:
    tg = t.groupby(["classifier", "data_name"]).agg(mean_stdev)[performance].unstack('data_name')
# t.columns = pd.MultiIndex.from_tuples([(d, p) for d in data_sets for p in performance])
tg.columns = tg.columns.swaplevel(0, 1)
tg = tg.reindex(data_sets, axis=1, level=0)
# tg = tg.transpose()[['FICSUM', 'RCD',  'CPF', 'DWM', 'DYNSE','SELeCT', 'lower', 'upper', "$S_p$", "$S_{MAP}$", "$S_{m}$" ]]
# tg = tg.transpose()[['FICSUM', 'RCD',  'CPF', 'DWM', 'DYNSE','SELeCT', 'lower', 'upper']]
tg = tg.transpose()[['SELeCT', "$S_p$", "$S_{MAP}$", "$S_{m}$" ]]
tg = tg.swaplevel()
tg = tg.sort_index()
# tg.to_latex(f"{parameter}_datatable.txt", escape=False, sparsify=True, multirow=True)
# tg.to_latex(f"{parameter}_datatable_ablation.txt", escape=False, sparsify=True, multirow=True)
tg.to_latex(f"{parameter}_datatable_ablation_time.txt", escape=False, sparsify=True, multirow=True)
# tg.to_latex(f"{parameter}_datatable_time.txt", escape=False, sparsify=True, multirow=True)
tg
# %%
selected['data_name']
# %%
noise_df = selected.copy()
# %%
width_df = selected.copy()

#%%
tmnoise_df = selected.copy()

#%%
tmnoise_df['param'] = 'Transition Noise'
tmnoise_df['param_value'] = tmnoise_df['TMnoise']
width_df['param'] = 'Drift Width'
width_df['param_value'] = width_df['drift_width']
noise_df['param'] = 'Noise'
noise_df['param_value'] = noise_df['noise']

#%%
width_df_s = width_df[width_df['drift_width'].isin([0, 500, 2500])]
noise_df_s = noise_df[noise_df['noise'].isin([0, 0.10, 0.25])]
tmnoise_df_s = tmnoise_df[tmnoise_df['TMnoise'].isin([0, 0.10, 0.25])]
full_df = pd.concat([width_df_s, noise_df_s, tmnoise_df_s])
full_df

#%%
def mean_stdev(X):
    return f"{np.mean(X):.2f}({np.std(X):.2f})"
# g = full_df.groupby(['data_name', 'classifier', 'param', 'param_value']).mean()[['kappa', 'GT_mean_f1']]
g = full_df.groupby(['data_name', 'classifier', 'param', 'param_value']).agg(mean_stdev)[['kappa', 'GT_mean_f1']]
# full_df.groupby(['data_name', 'classifier', 'param', 'param_value']).mean().columns
#%%
gt = g.unstack(['param', 'param_value'])
# gt.columns = gt.columns.swaplevel(0, 2)
# gt.columns = gt.columns.swaplevel(0, 1)
gt = gt.stack(0)
#%%
gt.index = gt.index.swaplevel(0, 2)
gt.index = gt.index.swaplevel(1, 2)
# gt.index = gt.index.sortlevel(0)
#%%
gt = gt.sort_index(0, ascending=False)
# %%
gt = gt.loc[( slice(None), ['AQSex', 'AQTemp', 'RTREESAMPLE_HARD'], ['CC', 'ficsum', 'lower_bound', 'upper_bound']), :]
# %%
gt.to_latex('param_table.txt', sparsify=True, multirow=True)
# %%

table_path = pathlib.Path('param_table.txt')
table_str = table_path.open().read()
table_str = table_str.replace(r'RTREESAMPLE\_HARD', 'TREE')
table_str = table_str.replace(r'upper\_bound', 'Upper')
table_str = table_str.replace(r'lower\_bound', 'Lower')
table_str = table_str.replace(r'GT\_mean\_f1', 'C-F1')
table_path.open('w').write(table_str)

# %%
