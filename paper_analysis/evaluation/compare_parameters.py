#%%
import pandas as pd
import numpy as np
import pathlib
import seaborn as sns
import matplotlib.pyplot as plt
import os
import json
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'ipympl')

#%%
def make_df(experiment_name, experiment_parameters, experiment_IVs, data_name, seeds, restrictions, base_df):
    base_path = pathlib.Path(__file__).absolute().parents[1] / 'output' / experiment_name
    print(base_path.resolve())
    results_paths = list(base_path.rglob(f"results_run_*.txt"))
    results_paths = [x for x in results_paths if data_name in str(x)]
    results = [json.load(rp.open()) for rp in results_paths]
    print(len(results))
    df = pd.DataFrame(results)
    if 'classifier' not in df.columns:
        df['classifier'] = "CC"
    if 'merge_threshold' not in df.columns:
        df['merge_threshold'] = -1
    if 'correlation_merge' not in df.columns:
        df['correlation_merge'] = False
    if df.shape[0] == 0:
        raise ValueError("Data not loaded - maybe path is wrong")
    df = df[df['seed'].isin(seeds) ]
    for restricted_param, restricted_vals in restrictions:
        df = df[df[restricted_param].isin(restricted_vals)]
    selected = df[[*experiment_IVs, *experiment_parameters, 'seed']]
    if base_df is None:
        base_df = selected
    compare = base_df.join(selected.groupby([*experiment_parameters, 'seed']).mean(), on=[*experiment_parameters, 'seed'], how='inner', lsuffix="_base")
    for iv in experiment_IVs:
        compare[f"c_{iv}"] = compare[f"{iv}"] - compare[f"{iv}_base"]
    groups = df.groupby(experiment_parameters)
    sizes = groups.size()
    grouped = groups.mean()[experiment_IVs]
    grouped['size'] = sizes
    grouped = grouped.reset_index()
    return grouped, selected, compare

# %%
experiment_name1 = None
experiment_name2 = None
# experiment_name1 = "testSettings-V3-P0"
# experiment_name1 = "testSettings-V3-P12-2"
# experiment_name1 = "testSettings-V3-P13-8-12-4"
# experiment_name1 = "testSettings-D10"
# experiment_name1 = "testSettings-D10"
# experiment_name1 = "testSettings-D10"
# experiment_name1 = "testSettings-D10-B2-21"
# experiment_name1 = "testSettings-D10-B3-3-36-2"
# experiment_name1 = "testSettings-D10-B3-T-4-7-2"
# experiment_name1 = "testSettings-D10-B3-T-4-11-4"
# experiment_name1 = "testSettings-D10-B3-T-4-11-5-2"
# experiment_name1 = "testSettings-D10-B3-T-5-19"
experiment_name1 = "testSettings-D10-B3-T-4-11-8"
# experiment_name1 = "testSettings-V3-P13"
# experiment_name1 = "defaultV8ifpacf"
# experiment_name1 = "defaultV8ifpacf"
# experiment_name2 = "defaultV8ifpacfS6"
# experiment_name1 = "defaultV5"
# experiment_name1 = "defaultV8"
# experiment_name1 = "defaultV8ifpacf"
# experiment_name1 = "TV8-nopacf"
# experiment_name1 = "TV8-12-7-nopacf"

# experiment_name2 = "TV8-11-nopacf"
# experiment_name2 = "TV8-12-nopacf"
# experiment_name2 = "TV8-12-13-nopacf"
# experiment_name2 = "TV8-16-nopacf"
# experiment_name2 = "TV8-16-nopacf"
# experiment_name1 = "Tmerge4-3"
# experiment_name1 = "Tmerge4-10"
# experiment_name2 = "Tmerge4-13"
# experiment_name1 = "TD-5"
# experiment_name2 = "TD-9"
# experiment_name2 = "testSettings-V3-P10"
# experiment_name2 = "testSettings-V3-P12-4"
# experiment_name2 = "testSettings-V3-P12-15"
# experiment_name2 = "testSettings-V3-P13-8-13-3"
# experiment_name2 = "testSettings-D10-B2-1"
# experiment_name2 = "testSettings-D10-B2-22"
# experiment_name2 = "testSettings-D10-5-3"
# experiment_name2 = "bounds_test"
# experiment_name2 = "testSettings-D10-B3-T-4-6"
# experiment_name2 = "testSettings-D10-B3-T-4-11-8-3"
# experiment_name2 = "testSettings-D10-B3-T-5-12"
# experiment_name2 = "testSettings-D10-B3-T-6-4"
experiment_name2 = "noise-1"


# experiment_parameters = ["sensitivity", "state_estimator_risk"]
# experiment_parameters = ["window_size", "similarity_gap"]
# experiment_parameters = ["window_size", "min_window_ratio"]
# experiment_parameters = ["window_size", "fp_gap"]
# experiment_parameters = ["window_size", "nonactive_fp_gap"]
# experiment_parameters = ["window_size", "observation_gap"]
# experiment_parameters = ["sensitivity", "similarity_gap"]

# experiment_parameters = ["sensitivity", "fp_gap"]
# experiment_parameters = ["sensitivity", "state_estimator_swap_risk"]
# experiment_parameters = ["sensitivity", "state_estimator_risk"]
# experiment_parameters = ["bypass_grace_period_threshold", "state_grace_period_window_multiplier"]
# experiment_parameters = ["min_drift_likelihood_threshold", "min_estimated_posterior_threshold"]
# experiment_parameters = ["fp_gap", "observation_gap"]
# experiment_parameters = ["minimum_concept_likelihood", "min_drift_likelihood_threshold"]
# experiment_parameters = ["sensitivity", "buffer_ratio"]
experiment_parameters = ["data_name", "sensitivity", "classifier"]
# experiment_parameters = ["FICSUM", "buffer_ratio"]
# experiment_parameters = ["FICSUM", "similarity_stdev_min"]
# experiment_parameters = ["data_name", "buffer_ratio"]
# experiment_parameters = ["data_name", "min_window_ratio"]
# experiment_parameters = ["data_name", "observation_gap"]
# experiment_parameters = ["data_name", "fp_gap"]
# experiment_parameters = ["data_name", "nonactive_fp_gap"]
# experiment_parameters = ["data_name", "similarity_stdev_min"]
# experiment_parameters = ["data_name", "bypass_grace_period_threshold"]
# experiment_parameters = ["data_name", "minimum_concept_likelihood"]
# experiment_parameters = ["data_name", "state_grace_period_window_multiplier"]
# experiment_parameters = ["data_name", "fingerprint_grace_period"]
# experiment_parameters = ["data_name", "sensitivity"]

# experiment_parameters = ["sensitivity", "fingerprint_grace_period"]
experiment_IVs = ['overall_accuracy', 'GT_mean_f1', "overall_time"]
# experiment_IVs = ['overall_accuracy', 'overall_time']
# experiment_name =  "expDefault"

df = None
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
# seeds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
# seeds = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
# seeds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
seeds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45]
# seeds = [1, 2, 3, 4, 5]
# seeds = [6, 7, 8, 9, 10]
# seeds = [11, 12, 13, 14, 15]
# seeds = [16, 17, 18, 19, 20]
# seeds = [2]

g, s, c = make_df(experiment_name1, experiment_parameters, experiment_IVs, data_name, seeds, {}, None)
g2, s2, c2 = make_df(experiment_name2, experiment_parameters, experiment_IVs, data_name, seeds, {}, s)

groups = c2.groupby(experiment_parameters)
sizes = groups.size()
c2_g = c2.groupby(experiment_parameters).mean()
c2_g['size'] = sizes
c2
# c2[c2['c_overall_accuracy'] != 0.0]
#%%
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

# %%

