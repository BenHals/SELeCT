#%%
import pandas as pd
import numpy as np
import pathlib
import seaborn as sns
import matplotlib.pyplot as plt
import os

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
# experiment_name =  "posteriorProbTestRE29-H"
experiment_name =  "profilingTLikelihoodM3"
# experiment_name =  "expDefault"
base_path = pathlib.Path(__file__).absolute().parents[1] / 'output' / experiment_name
data_name = ''
# data_name = 'cmc'
# data_name = 'STAGGERS'
# data_name = 'STAGGER'
data_name = 'Arabic'
# data_name = 'AQSex'
# data_name = 'RTREE'
# data_name = 'RTREESAMPLE'
# data_name = 'AQTemp'
# print(base_path.resolve())
# csv_paths = list(base_path.rglob(f"run_*.csv"))
# csv_paths = [x for x in csv_paths if data_name in str(x)]
# try:
#     print(csv_paths[0])
#     path_to_csv = csv_paths[0]
# except:
#     pass
# path_to_csv = pathlib.Path(r'S:\PhD\Packages\SELeCT\output\profilingTLikelihoodM3\SELeCT-stateBuffers-3a4fc59\Arabic\1\run_-4906345892244427007_0.csv')
# path_to_csv = pathlib.Path(r'H:\PhD\PythonPackages\SELeCT\output\swapT\SELeCT-stateBuffers-0f306c6\Arabic\1\run_7720719684052207006_0.csv')
# path_to_csv = pathlib.Path(r'H:\PhD\PythonPackages\SELeCT\output\swapT\SELeCT-stateBuffers-0f306c6\Arabic\1\run_7456533350545879809_0.csv')
# path_to_csv = pathlib.Path(r'H:\PhD\PythonPackages\SELeCT\output\swapT\SELeCT-stateBuffers-0f306c6\Arabic\1\run_7774380090789620276_0.csv')
# path_to_csv = pathlib.Path(r'S:\PhD\Packages\SELeCT\output\onlineT\SELeCT-stateBuffers-c4f8009\cmc\1\run_-7475330257824156815_0.csv')
# path_to_csv = pathlib.Path(r'S:\PhD\Packages\SELeCT\output\onlineT\SELeCT-stateBuffers-c4f8009\cmcBaseline\1\run_380236928070498745_0.csv')
# path_to_csv = pathlib.Path(r'S:\PhD\Packages\SELeCT\output\TV8-12-8-nopacf\SELeCT-stateBuffers-fc7c4ba\Arabic\9\run_8156343151782563766_0.csv')
path_to_csv = pathlib.Path(r'S:\PhD\Packages\SELeCT\output\TV8-16-nopacf\SELeCT-stateBuffers-2902d24\Arabic\9\run_7106462244228926706_0.csv')
path_to_csv = pathlib.Path(r'S:\PhD\Packages\SELeCT\output\test\SELeCT-stateBuffers-2902d24\Arabic\9O2\run_-1482282508415240976_0.csv')

df = pd.read_csv(path_to_csv)
df.head()
# %%

likelihoods = df['adwin_likelihood_estimate'].str.split(';', expand=True)
drop_cols = []
for c in likelihoods.columns:
    data = likelihoods[c].str.split(':', expand=True)
    concept = data[0].dropna()
    if concept.shape[0] > 0:
        concept_name = f"c-{concept.unique()[0]}"
        likelihood = data[1]
        likelihoods[concept_name] = likelihood
        drop_cols.append(c)
drop_cols.append('c--1')
likelihoods = likelihoods.drop(drop_cols, axis=1)
for i, c1 in enumerate(likelihoods.columns):
    for c2 in likelihoods.columns[i+1:]:
        l = likelihoods[[c1, c2]].dropna()
        l1 = pd.to_numeric(l.iloc[:, 0])
        l2 = pd.to_numeric(l.iloc[:, 1])
        # print(np.corrcoef(l1.values, l2.values))
        correlation = np.corrcoef(l1.values, l2.values)[0][1]
        if correlation > 0.7:
            print(f"merge {c1}, {c2}")
        print(f"{c1}-{c2}: {correlation}")
#%%


# %%
plot_probabilities('concept_priors')
plot_probabilities('concept_likelihoods')
# plot_probabilities('concept_likelihoods_smoothed', smoothing=False)
plot_probabilities('adwin_likelihood_estimate', smoothing=False)
# plot_probabilities('adwin_likelihood_delta_0.01', smoothing=False)
# plot_probabilities('adwin_likelihood_delta_0.05', smoothing=False)
# plot_probabilities('adwin_likelihood_delta_0.15000000000000002', smoothing=False)
# plot_probabilities('adwin_likelihood_delta_0.25', smoothing=False)
# plot_probabilities('adwin_likelihood_delta_0.5', smoothing=False)
plot_probabilities('concept_posteriors')
# plot_probabilities('concept_posteriors_smoothed', smoothing=False)
plot_probabilities('adwin_posterior_estimate', smoothing=False)
# plot_probabilities('adwin_posterior_delta_0.95', smoothing=False)
# plot_probabilities('concept_posterior_mh')
# plot_probabilities('concept_posterior_mh_exp')

# %%
