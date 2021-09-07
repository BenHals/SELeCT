#%%
import pandas as pd
import numpy as np
import pathlib
import seaborn as sns
import matplotlib.pyplot as plt
import os

#%%
def run_task(cmd_str, condaenv):
    cmd = cmd_str.replace('../run_experiment.py', str(pathlib.Path(__file__).absolute().parents[1] / 'run_experiment.py'))
    cmd = cmd.replace('../RawData', str(pathlib.Path(__file__).absolute().parents[3] / 'CCRawData'))
    cmd = cmd.replace('testoutput/', str(pathlib.Path(__file__).absolute().parents[1] / 'output/'))
    cmd = cmd.replace('testlog/', str(pathlib.Path(__file__).absolute().parents[1] / 'experimentlog/'))
    print(cmd)
    if condaenv:
        os.system(f'conda activate {condaenv} && {cmd}')
    else:
        os.system(f'{cmd}')

experiment_name = "posteriorProbTestRE29-H"

run_cmds = [
# f'python ../run_experiment.py --forcegitcheck --seeds 1 --seedaction list --experimentname {experiment_name} --datalocation ../RawData --datasets STAGGERS --outputlocation testoutput/ --loglocation testlog/  --ifeatures IMF MI  --single --fsmethod fisher_overall --min_sim_stdev 0.02 --sim_gap 3 --ob_gap 25 --fp_gap 25 --na_fp_gap 100 --window_size 75 --repeatproportion 1.0 --logging',
f'python ../run_experiment.py --forcegitcheck --seeds 1 --seedaction list --experimentname {experiment_name} --datalocation ../RawData --datasets cmc --outputlocation testoutput/ --loglocation testlog/  --ifeatures IMF MI  --single --fsmethod fisher_overall --min_sim_stdev 0.01 --sim_gap 3 --ob_gap 25 --fp_gap 25 --na_fp_gap 100 --window_size 75 --repeatproportion 1.0 --logging',
f'python ../run_experiment.py --forcegitcheck --seeds 1 --seedaction list --experimentname {experiment_name} --datalocation ../RawData --datasets Arabic --outputlocation testoutput/ --loglocation testlog/  --ifeatures IMF MI  --single --fsmethod fisher_overall --min_sim_stdev 0.01 --sim_gap 3 --ob_gap 25 --fp_gap 25 --na_fp_gap 100 --window_size 75 --repeatproportion 1.0 --logging',
f'python ../run_experiment.py --forcegitcheck --seeds 1 --seedaction list --experimentname {experiment_name} --datalocation ../RawData --datasets AQTemp --outputlocation testoutput/ --loglocation testlog/  --ifeatures IMF MI  --single --fsmethod fisher_overall --min_sim_stdev 0.01 --sim_gap 3 --ob_gap 25 --fp_gap 25 --na_fp_gap 100 --window_size 75 --repeatproportion 1.0 --logging',
f'python ../run_experiment.py --forcegitcheck --seeds 1 --seedaction list --experimentname {experiment_name} --datalocation ../RawData --datasets STAGGER --outputlocation testoutput/ --loglocation testlog/  --ifeatures IMF MI  --single --fsmethod fisher_overall --min_sim_stdev 0.01 --sim_gap 3 --ob_gap 25 --fp_gap 25 --na_fp_gap 100 --window_size 75 --repeatproportion 1.0 --logging',
f'python ../run_experiment.py --forcegitcheck --seeds 1 --seedaction list --experimentname {experiment_name} --datalocation ../RawData --datasets RTREESAMPLE --outputlocation testoutput/ --loglocation testlog/  --ifeatures IMF MI  --single --fsmethod fisher_overall --min_sim_stdev 0.01 --sim_gap 3 --ob_gap 25 --fp_gap 25 --na_fp_gap 100 --window_size 75 --repeatproportion 1.0 --logging',
f'python ../run_experiment.py --forcegitcheck --seeds 1 --seedaction list --experimentname {experiment_name} --datalocation ../RawData --datasets AQSex --outputlocation testoutput/ --loglocation testlog/  --ifeatures IMF MI  --single --fsmethod fisher_overall --min_sim_stdev 0.01 --sim_gap 3 --ob_gap 25 --fp_gap 25 --na_fp_gap 100 --window_size 75 --repeatproportion 1.0 --logging',
]

for cmd_str in run_cmds:
    run_task(cmd_str, None)


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
    # merges = {int(d.split(':')[0]):int(d.split(':')[1]) for d in df.iloc[df.shape[0]-1, :]["merges"].split(';')}
    # print(merges)
    # df['merge_model'] = df['active_model'].copy()
    # for m_init in merges:
    #     m_to = merges[m_init]
    #     while m_to in merges:
    #         m_from = m_to
    #         m_to = merges[m_from]
    #     df['merge_model'] = df['merge_model'].replace(m_init, m_to)
    merge_model = df['merge_model'] if 'merge_model' in df.columns else None
    repair_model = df['repair_model'] if 'repair_model' in df.columns else None
    gt_model = df['ground_truth_concept']
    active_model_ranges = ts_to_ranges(active_model)
    merge_model_ranges = ts_to_ranges(merge_model) if merge_model is not None else None
    repair_model_ranges = ts_to_ranges(repair_model) if repair_model is not None else None
    gt_model_ranges = ts_to_ranges(gt_model)
    del_cols = []
    val_nas = {}
    dms = df['deletions'].dropna().values
    deleted_models = []
    for dm in dms:
        try:
            deleted_models.append(int(float(dm)))
        except:
            ids = dm.split(';')
            # print(ids)
            for id in ids:
                if len(id) > 0:
                    deleted_models.append(int(float(id)))
    for c in probs.columns:

        # col_id = probs[c].str.rsplit(':').str[0].astype('float').unique()[-1]
        unique_ids = probs[c].str.rsplit(':').str[0].astype('float').unique()
        col_ids = probs[c].str.rsplit(':').str[0].astype('float')
        prob_vals = probs[c].str.rsplit(':').str[-1].astype('float')
        for u_id in unique_ids:
            if np.isnan(u_id):
                continue
            indexes = col_ids == u_id
            if f"v{u_id}" not in probs.columns:
                probs[f"v{u_id}"] = np.nan
            probs.loc[indexes, f"v{u_id}"] = prob_vals.loc[indexes]
            if u_id in deleted_models:
                del_cols.append(f"v{u_id}")
        # print(probs[c].str.rsplit(':').str[0].astype('float').unique())
        probs[c] = probs[c].str.rsplit(':').str[-1].astype('float')
        # probs[f"v{col_id}"] = probs[c]
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
    print(probs)
    m_df = pd.melt(probs.iloc[::10, :], id_vars='ts')
    m_df['variable'] = m_df['variable'].astype('category')
    m_df['value'] = m_df['value'].replace({-1:0})
    # m_df['value'] = m_df['value'].rolling(window=int(probs.shape[0]*0.001)).mean()
    if smoothing:
        m_df['value'] = m_df['value'].rolling(window=7).mean()
    print(m_df)
    fig, ax = plt.subplots()
    sns.lineplot(data=m_df[m_df['variable'] != '-1'], x='ts', y='value', hue='variable', ax=ax, linewidth=0.5)
    # if f"{prob_column}_background" in df.columns:
    #     plt.plot()

    handles, labels = ax.get_legend_handles_labels()
    # colorsplt.rcParams["axes.prop_cycle"].by_key()["color"]
    print(labels)
    for rs, re, v in active_model_ranges:
        if str(v) in labels:
            color = handles[labels.index(str(v))].get_color()
            ax.hlines(y = -0.02, xmin = rs, xmax = re, colors = [color])
        else:
            ax.hlines(y = -0.015, xmin = rs, xmax = re, colors=["red"])
    if merge_model_ranges is not None:
        for rs, re, v in merge_model_ranges:
            if str(v) in labels:
                color = handles[labels.index(str(v))].get_color()
                ax.hlines(y = -0.03, xmin = rs, xmax = re, colors = [color])
            else:
                ax.hlines(y = -0.025, xmin = rs, xmax = re, colors=["red"])
    if repair_model_ranges is not None:
        for rs, re, v in repair_model_ranges:
            if str(v) in labels:
                color = handles[labels.index(str(v))].get_color()
                ax.hlines(y = -0.04, xmin = rs, xmax = re, colors = [color])
            else:
                ax.hlines(y = -0.035, xmin = rs, xmax = re, colors=["red"])
    for rs, re, v in gt_model_ranges:
        if str(v) in labels:
            color = handles[labels.index(str(v))].get_color()
            ax.hlines(y = -0.1, xmin = rs, xmax = re, colors = [color])
        else:
            ax.hlines(y = -0.1, xmin = rs, xmax = re, colors=["red"])
    plt.xlim((0, probs.shape[0]))
    plt.title(f"{data_name}-{prob_column}")
    plt.legend().remove()
    plt.show()
    if save_file:
        plt.savefig(f"explore_probabilities/{experiment_name}-{data_name}.pdf")
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
print(base_path.resolve())
csv_paths = list(base_path.rglob(f"run_*.csv"))
csv_paths = [x for x in csv_paths if data_name in str(x)]
try:
    print(csv_paths[0])
    path_to_csv = csv_paths[0]
except:
    pass
path_to_csv = pathlib.Path(r'S:\PhD\Packages\SELeCT\output\profilingTLikelihoodM3\SELeCT-stateBuffers-3a4fc59\Arabic\1\run_-4906345892244427007_0.csv')
path_to_csv = pathlib.Path(r'H:\PhD\PythonPackages\SELeCT\output\swapT\SELeCT-stateBuffers-0f306c6\Arabic\1\run_7720719684052207006_0.csv')
path_to_csv = pathlib.Path(r'H:\PhD\PythonPackages\SELeCT\output\swapT\SELeCT-stateBuffers-0f306c6\Arabic\1\run_7456533350545879809_0.csv')
path_to_csv = pathlib.Path(r'H:\PhD\PythonPackages\SELeCT\output\swapT\SELeCT-stateBuffers-0f306c6\Arabic\1\run_7774380090789620276_0.csv')
path_to_csv = pathlib.Path(r'S:\PhD\Packages\SELeCT\output\onlineT\SELeCT-stateBuffers-c4f8009\cmc\1\run_-7475330257824156815_0.csv')
path_to_csv = pathlib.Path(r'S:\PhD\Packages\SELeCT\output\onlineT\SELeCT-stateBuffers-c4f8009\cmcBaseline\1\run_380236928070498745_0.csv')
path_to_csv = pathlib.Path(r'S:\PhD\Packages\SELeCT\output\TV8-12-8-nopacf\SELeCT-stateBuffers-fc7c4ba\Arabic\9\run_8156343151782563766_0.csv')
path_to_csv = pathlib.Path(r'S:\PhD\Packages\SELeCT\output\TV8-16-nopacf\SELeCT-stateBuffers-2902d24\Arabic\9\run_7106462244228926706_0.csv')
path_to_csv = pathlib.Path(r'S:\PhD\Packages\SELeCT\output\test\SELeCT-stateBuffers-2902d24\Arabic\9O2\run_-1482282508415240976_0.csv')
path_to_csv = pathlib.Path(r'S:\PhD\Packages\SELeCT\output\test\SELeCT-stateBuffers-2902d24\Arabic\9\run_-1286977174350703795_0.csv')
path_to_csv = pathlib.Path(r'S:\PhD\Packages\SELeCT\output\test\SELeCT-stateBuffers-a9d795f\Arabic\9\run_-2334222087965642595_0.csv')
path_to_csv = pathlib.Path(r'S:\PhD\Packages\SELeCT\output\Tmerge3\SELeCT-stateBuffers-2720df5\AQTemp\3\run_-414814470583489287_0.csv')
path_to_csv = pathlib.Path(r'S:\PhD\Packages\SELeCT\output\testSettings-V3-P6\SELeCT-stateBuffers-649ba89\AQTemp\1\run_14990155563209172_0.csv')
path_to_csv = pathlib.Path(r'S:\PhD\Packages\SELeCT\output\testSettings-V3-P6\SELeCT-stateBuffers-649ba89\AQTemp\1\run_14990155563209172_0.csv')
path_to_csv = pathlib.Path(r'S:\PhD\Packages\SELeCT\output\testSettings-V3-P0\SELeCT-stateBuffers-6682033\AQTemp\1\run_8795191060781276537_0.csv')
path_to_csv = pathlib.Path(r'S:\PhD\Packages\SELeCT\output\testSettings-V3-P12\SELeCT-stateBuffers-ec3116a\Arabic\1\run_-2427937433353169728_0.csv')
path_to_csv = pathlib.Path(r'S:\PhD\Packages\SELeCT\output\testSettings-V3-P0\SELeCT-stateBuffers-6682033\Arabic\1\run_1289982109998869422_0.csv')
path_to_csv = pathlib.Path(r'S:\PhD\Packages\SELeCT\output\testSettings-V3-P12-2\SELeCT-stateBuffers-16c5a50\AQTemp\2\run_-8648117680525027462_0.csv')
path_to_csv = pathlib.Path(r'S:\PhD\Packages\SELeCT\output\testSettings-V3-P12\SELeCT-stateBuffers-ec3116a\AQTemp\2\run_401743281064339815_0.csv')
path_to_csv = pathlib.Path(r'S:\PhD\Packages\SELeCT\output\testSettings-V3-P12-2\SELeCT-stateBuffers-16c5a50\Arabic\1\run_-7021534754439845788_0.csv')
path_to_csv = pathlib.Path(r'S:\PhD\Packages\SELeCT\output\testSettings-D10-3-3\SELeCT-statebuffers-db3b40a\cmc\10\run_1116232907565486398_0.csv')
# path_to_csv = pathlib.Path(r'S:\PhD\Packages\SELeCT\output\testSettings-D10-2\SELeCT-statebuffers-8f0804b\cmc\10\run_-7792760104552520916_0.csv')
path_to_csv = pathlib.Path(r'S:\PhD\Packages\SELeCT\output\testSettings-D10-3-4\SELeCT-statebuffers-db3b40a\cmc\10\run_-2782813622978549440_0.csv')
path_to_csv = pathlib.Path(r'S:\PhD\Packages\SELeCT\output\testSettings-D10-4-2\SELeCT-statebuffers-db3b40a\cmc\10\run_7378665546162938027_0.csv')
path_to_csv = pathlib.Path(r'S:\PhD\Packages\SELeCT\output\testSettings-D10-4-5\SELeCT-statebuffers-4ddf7dc\cmc\10\run_-77474826253634745_0.csv')
path_to_csv = pathlib.Path(r'S:\PhD\Packages\SELeCT\output\testSettings-D10-4-5\SELeCT-statebuffers-4ddf7dc\Arabic\10\run_-5834284137516759901_0.csv')
path_to_csv = pathlib.Path(r'S:\PhD\Packages\SELeCT\output\testSettings-D10-B2-1\SELeCT-priors-be7a6c9\AQTemp\6\run_-6898402652529479959_0.csv')
path_to_csv = pathlib.Path(r'\\TOMSK\shared_results\tD-T-10\SELeCT-statebuffers-ab94293\AQTemp\6\run_4008170250407915953_0.csv')
path_to_csv = pathlib.Path(r'\\TOMSK\shared_results\tD-T-D10-B2-8\SELeCT-priors-42deb3a\AQTemp\6\run_467715493666154432_0.csv')
path_to_csv = pathlib.Path(r'S:\PhD\Packages\SELeCT\output\testSettings-D10-B2-14\SELeCT-priors-42deb3a\AQTemp\6\run_5013787337043374538_0.csv')
path_to_csv = pathlib.Path(r'S:\PhD\Packages\SELeCT\output\testSettings-D10-B2-2\SELeCT-priors-3d79483\AQTemp\6\run_-5094303040414714410_0.csv')
path_to_csv = pathlib.Path(r'S:\PhD\Packages\SELeCT\output\testSettings-D10-B2-3\SELeCT-priors-095027b\AQTemp\6\run_4372363773004147061_0.csv')
path_to_csv = pathlib.Path(r'S:\PhD\Packages\SELeCT\output\testSettings-D10-B2-4\SELeCT-priors-74c0993\AQTemp\6\run_-7000716186798840050_0.csv')
path_to_csv = pathlib.Path(r'S:\PhD\Packages\SELeCT\output\testSettings-D10-B2-5\SELeCT-priors-24aee31\AQTemp\6\run_-3117678569588092678_0.csv')
path_to_csv = pathlib.Path(r'S:\PhD\Packages\SELeCT\output\testSettings-D10-B2-15\SELeCT-priors-3d5ef52\Arabic\6\run_8073661786443797523_0.csv')
path_to_csv = pathlib.Path(r'S:\PhD\Packages\SELeCT\output\testSettings-D10-B2-14\SELeCT-priors-42deb3a\Arabic\6\run_-8412819785534724605_0.csv')
path_to_csv = pathlib.Path(r'S:\PhD\Packages\SELeCT\output\testSettings-D10-B2-15\SELeCT-priors-3d5ef52\AQTemp\6\run_-917543459569217241_0.csv')
path_to_csv = pathlib.Path(r'S:\PhD\Packages\SELeCT\output\testSettings-D10-B2-17\SELeCT-priors-7c6f3ea\AQTemp\6\run_-1703348946972639225_0.csv')
path_to_csv = pathlib.Path(r'S:\PhD\Packages\SELeCT\output\testSettings-D10-B2-17\SELeCT-priors-7c6f3ea\AQTemp\6\run_-1703348946972639225_0.csv')
path_to_csv = pathlib.Path(r'S:\PhD\Packages\SELeCT\output\testSettings-D10-B2-17\SELeCT-priors-7c6f3ea\cmc\6\run_-8405429129304182241_0.csv')
path_to_csv = pathlib.Path(r'S:\PhD\Packages\SELeCT\output\testSettings-D10-B2-18\SELeCT-priors-fb28d49\cmc\6\run_8104631862684623703_0.csv')
path_to_csv = pathlib.Path(r'S:\PhD\Packages\SELeCT\output\testSettings-D10-B2-18\SELeCT-priors-fb28d49\AQTemp\6\run_-1263815857352348788_0.csv')
path_to_csv = pathlib.Path(r'S:\PhD\Packages\SELeCT\output\testSettings-D10-B2-17\SELeCT-priors-7c6f3ea\AQTemp\6\run_-1703348946972639225_0.csv')
path_to_csv = pathlib.Path(r'S:\PhD\Packages\SELeCT\output\testSettings-D10-B2-19\SELeCT-priors-fb28d49\Arabic\6\run_-7297231075461819370_0.csv')
path_to_csv = pathlib.Path(r'S:\PhD\Packages\SELeCT\output\testSettings-D10-B2-20\SELeCT-priors-3820088\Arabic\6\run_7332856957806996537_0.csv')
path_to_csv = pathlib.Path(r'S:\PhD\Packages\SELeCT\output\testSettings-D10-B2-18\SELeCT-priors-fb28d49\Arabic\6\run_-79656820910170731_0.csv')
path_to_csv = pathlib.Path(r'S:\PhD\Packages\SELeCT\output\testSettings-D10-B2-21\SELeCT-priors-3820088\Arabic\6\run_-582482443632789739_0.csv')
path_to_csv = pathlib.Path(r'S:\PhD\Packages\SELeCT\output\testSettings-D10-B2-21\SELeCT-priors-3820088\AQTemp\6\run_-701395743448462604_0.csv')
path_to_csv = pathlib.Path(r'S:\PhD\Packages\SELeCT\output\testSettings-D10-B2-17\SELeCT-priors-7c6f3ea\AQTemp\6\run_-1703348946972639225_0.csv')
path_to_csv = pathlib.Path(r'S:\PhD\Packages\SELeCT\output\testSettings-D10-B2-23\SELeCT-priors-401768b\Arabic\6\run_-3609093992919286257_0.csv')
path_to_csv = pathlib.Path(r'S:\PhD\Packages\SELeCT\output\testSettings-D10-B2-23\SELeCT-priors-401768b\Arabic\6\run_-3609093992919286257_0.csv')
path_to_csv = pathlib.Path(r'S:\PhD\Packages\SELeCT\output\testSettings-D10-B3-3-7\SELeCT-priors2-dacd993\AQTemp\6\run_8637657323661532754_0.csv')
path_to_csv = pathlib.Path(r'S:\PhD\Packages\SELeCT\output\testSettings-D10-B3-3-9\SELeCT-priors2-9cde3e3\AQTemp\6\run_-681288597296053517_0.csv')
path_to_csv = pathlib.Path(r'S:\PhD\Packages\SELeCT\output\testSettings-D10-B3-3-11\SELeCT-priors2-3cd698f\AQTemp\6\run_2013101890507892326_0.csv')
path_to_csv = pathlib.Path(r'S:\PhD\Packages\SELeCT\output\testSettings-D10-B3-3-12\SELeCT-priors2-d186101\AQTemp\6\run_-775087376686089259_0.csv')
path_to_csv = pathlib.Path(r'S:\PhD\Packages\SELeCT\output\testSettings-D10-B3-3-11\SELeCT-priors2-3cd698f\Arabic\6\run_973593372072442564_0.csv')
path_to_csv = pathlib.Path(r'S:\PhD\Packages\SELeCT\output\testSettings-D10-B3-3-13\SELeCT-priors2-04a3adf\Arabic\6\run_-6272884049477198139_0.csv')
path_to_csv = pathlib.Path(r'S:\PhD\Packages\SELeCT\output\testSettings-D10-B3-3-14\SELeCT-priors2-28ab8cb\Arabic\6\run_-5640337934485237672_0.csv')
path_to_csv = pathlib.Path(r'S:\PhD\Packages\SELeCT\output\testSettings-D10-B3-3-14\SELeCT-priors2-28ab8cb\AQTemp\6\run_3384380694627715630_0.csv')
path_to_csv = pathlib.Path(r'S:\PhD\Packages\SELeCT\output\testSettings-D10-B3-3-11\SELeCT-priors2-3cd698f\cmc\7\run_1481759200453017880_0.csv')
path_to_csv = pathlib.Path(r'S:\PhD\Packages\SELeCT\output\testSettings-D10-B3-3-11\SELeCT-priors2-3cd698f\cmc\7\run_1481759200453017880_0.csv')
path_to_csv = pathlib.Path(r'S:\PhD\Packages\SELeCT\output\testSettings-D10-B3-3-16\SELeCT-priors2-d38e676\cmc\7\run_-8588173513788286142_0.csv')
path_to_csv = pathlib.Path(r'S:\PhD\Packages\SELeCT\output\testSettings-D10-B3-3-17\SELeCT-priors2-9ba69c3\cmc\7\run_379625529657455990_0.csv')
# path_to_csv = pathlib.Path(r'S:\PhD\Packages\SELeCT\output\testSettings-D10-B3-3-16-3\SELeCT-priors2-d38e676\cmc\7\run_-4973503737647182706_0.csv')
path_to_csv = pathlib.Path(r'S:\PhD\Packages\SELeCT\output\testSettings-D10-B3-3-18\SELeCT-priors2-9a56b47\cmc\7\run_7632237673092749779_0.csv')
path_to_csv = pathlib.Path(r'S:\PhD\Packages\SELeCT\output\testSettings-D10-B3-3-18\SELeCT-priors2-9a56b47\cmc\9\run_2957224928435201167_0.csv')
path_to_csv = pathlib.Path(r'S:\PhD\Packages\SELeCT\output\testSettings-D10-B3-3-11\SELeCT-priors2-3cd698f\cmc\9\run_1527794014708273912_0.csv')
path_to_csv = pathlib.Path(r'S:\PhD\Packages\SELeCT\output\testSettings-D10-B3-3-36-2\SELeCT-priors2-ef00b19\cmc\6\run_-3676759946911310081_0.csv')
path_to_csv = pathlib.Path(r'S:\PhD\Packages\SELeCT\output\testSettings-D10-B3-4-1\SELeCT-priors2-4f5dd6e\cmc\6\run_-993268999481376650_0.csv')
path_to_csv = pathlib.Path(r'S:\PhD\Packages\SELeCT\output\testSettings-D10-B3-4-3\SELeCT-priors2-8df75d3\cmc\6\run_3055389198645447149_0.csv')
path_to_csv = pathlib.Path(r'S:\PhD\Packages\SELeCT\output\testSettings-D10-B3-4-4\SELeCT-priors2-26d8575\cmc\6\run_-3308486596650530687_0.csv')
path_to_csv = pathlib.Path(r'S:\PhD\Packages\SELeCT\output\testSettings-D10-B3-4-5\SELeCT-priors2-ca77f5a\cmc\6\run_5823875966638452969_0.csv')
path_to_csv = pathlib.Path(r'S:\PhD\Packages\SELeCT\output\testSettings-D10-B3-4-5-2\SELeCT-priors2-27f3d00\cmc\6\run_7595168293497642963_0.csv')
path_to_csv = pathlib.Path(r'S:\PhD\Packages\SELeCT\output\testSettings-D10-B3-4-5-4\SELeCT-priors2-27f3d00\cmc\6\run_1582632060928794641_0.csv')
path_to_csv = pathlib.Path(r'S:\PhD\Packages\SELeCT\output\testSettings-D10-B3-4-6\SELeCT-priors2-27f3d00\cmc\6\run_-3797495793661116334_0.csv')
path_to_csv = pathlib.Path(r'S:\PhD\Packages\SELeCT\output\testSettings-D10-B3-4-6\SELeCT-priors2-27f3d00\AQTemp\6\run_2020185689280716127_0.csv')
path_to_csv = pathlib.Path(r'S:\PhD\Packages\SELeCT\output\testSettings-D10-B3-4-6-2\SELeCT-priors2-3256705\cmc\6\run_-7278267033953540716_0.csv')
path_to_csv = pathlib.Path(r'S:\PhD\Packages\SELeCT\output\testSettings-D10-B3-T\SELeCT-priors2-220d2ae\cmc\6\run_-8693024440947840213_0.csv')
path_to_csv = pathlib.Path(r'S:\PhD\Packages\SELeCT\output\testSettings-D10-B3-T\SELeCT-priors2-220d2ae\Arabic\6\run_-6775324196196218961_0.csv')
path_to_csv = pathlib.Path(r'S:\PhD\Packages\SELeCT\output\testSettings-D10-B3-T-2\SELeCT-priors2-466c527\cmc\6\run_7710471615099305981_0.csv')
path_to_csv = pathlib.Path(r'S:\PhD\Packages\SELeCT\output\testSettings-D10-B3-T-2\SELeCT-priors2-466c527\Arabic\6\run_-8817753182036057214_0.csv')
path_to_csv = pathlib.Path(r'S:\PhD\Packages\SELeCT\output\testSettings-D10-B3-T-3\SELeCT-priors2-878f90f\cmc\6\run_8938625320450220018_0.csv')
path_to_csv = pathlib.Path(r'S:\PhD\Packages\SELeCT\output\testSettings-D10-B3-T-3\SELeCT-priors2-878f90f\Arabic\6\run_5579877236041522461_0.csv')
path_to_csv = pathlib.Path(r'S:\PhD\Packages\SELeCT\output\testSettings-D10-B3-T-4\SELeCT-priors2-f8031ea\cmc\6\run_-739959059111144128_0.csv')
path_to_csv = pathlib.Path(r'S:\PhD\Packages\SELeCT\output\testSettings-D10-B3-T-4\SELeCT-priors2-f8031ea\Arabic\6\run_949132301387092141_0.csv')
path_to_csv = pathlib.Path(r'S:\PhD\Packages\SELeCT\output\testSettings-D10-B3-T-4-2\SELeCT-priors2-f8031ea\Arabic\6\run_8814689318318436074_0.csv')
path_to_csv = pathlib.Path(r'S:\PhD\Packages\SELeCT\output\testSettings-D10-B3-T-4-3\SELeCT-priors2-ea6d7ad\Arabic\6\run_2926970076925571197_0.csv')
path_to_csv = pathlib.Path(r'S:\PhD\Packages\SELeCT\output\testSettings-D10-B3-T-4-4\SELeCT-priors2-3bdf29e\Arabic\6\run_8708524116069839000_0.csv')
path_to_csv = pathlib.Path(r'S:\PhD\Packages\SELeCT\output\testSettings-D10-B3-T-4-5\SELeCT-priors2-3bdf29e\Arabic\6\run_3715514906986226031_0.csv')
path_to_csv = pathlib.Path(r'S:\PhD\Packages\SELeCT\output\testSettings-D10-B3-T-4-7\SELeCT-priors2-3bdf29e\cmc\6\run_-6777813348691598762_0.csv')
path_to_csv = pathlib.Path(r'S:\PhD\Packages\SELeCT\output\testSettings-D10-B3-T-4-7-2\SELeCT-priors2-3bdf29e\cmc\6\run_4731751777945250470_0.csv')
path_to_csv = pathlib.Path(r'S:\PhD\Packages\SELeCT\output\testSettings-D10-B3-T-4-7-2\SELeCT-priors2-3bdf29e\Arabic\6\run_-3656433076419362402_0.csv')
path_to_csv = pathlib.Path(r'S:\PhD\Packages\SELeCT\output\test\SELeCT-priors2-1b7eb01\Arabic\6\run_-2926814714350918692_0.csv')
path_to_csv = pathlib.Path(r'S:\PhD\Packages\SELeCT\output\testSettings-D10-B3-T-4-6-2\SELeCT-priors2-3bdf29e\Arabic\6\run_2856454305910319954_0.csv')
path_to_csv = pathlib.Path(r'S:\PhD\Packages\SELeCT\output\testSettings-D10-B3-T-4-8-2\SELeCT-priors2-1b7eb01\Arabic\6\run_-1206288495553886818_0.csv')
path_to_csv = pathlib.Path(r'S:\PhD\Packages\SELeCT\output\testSettings-D10-B3-T-4-8-2\SELeCT-priors2-1b7eb01\AQTemp\9\run_-7808891639350108241_0.csv')
path_to_csv = pathlib.Path(r'S:\PhD\Packages\SELeCT\output\testSettings-D10-B3-T-4-6-2\SELeCT-priors2-3bdf29e\AQTemp\9\run_8900057978001149137_0.csv')
path_to_csv = pathlib.Path(r'S:\PhD\Packages\SELeCT\output\test\SELeCT-priors2-775fe61\Arabic\6\run_2641186024789872503_0.csv')
path_to_csv = pathlib.Path(r'S:\PhD\Packages\SELeCT\output\test\SELeCT-priors2-2caebe2\Arabic\6\run_8188025696606856962_0.csv')
path_to_csv = pathlib.Path(r'S:\PhD\Packages\SELeCT\output\testSettings-D10-B3-T-4-10\SELeCT-priors2-84e2272\Arabic\6\run_2331798105968368434_0.csv')
path_to_csv = pathlib.Path(r'S:\PhD\Packages\SELeCT\output\test\SELeCT-priors2-84e2272\Arabic\6\run_-1378383428424223329_0.csv')
path_to_csv = pathlib.Path(r'S:\PhD\Packages\SELeCT\output\test\SELeCT-priors2-84e2272\Arabic\6\run_2267801262864978425_0.csv')
path_to_csv = pathlib.Path(r'S:\PhD\Packages\SELeCT\output\test\SELeCT-priors2-84e2272\Arabic\6\run_563613841599681077_0.csv')
path_to_csv = pathlib.Path(r'\\TOMSK\shared_results\D10-B3-24\SELeCT-priors2-1b7eb01\AQSex\6\run_-3757298497491527154_0.csv')
path_to_csv = pathlib.Path(r'\\TOMSK\shared_results\D10-B3-26\SELeCT-priors2-92a04e1\AQSex\6\run_-2859552812262852118_0.csv')
path_to_csv = pathlib.Path(r'S:\PhD\Packages\SELeCT\output\noise-1\SELeCT-master-bbac8ef\AQTemp\1\run_-3593869472345843843_0.csv')
df = pd.read_csv(path_to_csv)
df.head()
# %%
plot_probabilities('concept_priors')
plot_probabilities('concept_priors_2h')
plot_probabilities('concept_priors_3h')
plot_probabilities('concept_priors_combined')
plot_probabilities('concept_likelihoods')
# %%
%matplotlib inline
plot_probabilities('concept_priors')
plot_probabilities('concept_likelihoods')
# plot_probabilities('concept_likelihoods_smoothed', smoothing=False)
plot_probabilities('adwin_likelihood_estimate', smoothing=False)
plot_probabilities('adwin_posterior_estimate', smoothing=False)
# plot_probabilities('adwin_likelihood_delta_0.01', smoothing=False)
# plot_probabilities('adwin_likelihood_delta_0.05', smoothing=False)
# plot_probabilities('adwin_likelihood_delta_0.15000000000000002', smoothing=False)
# plot_probabilities('adwin_likelihood_delta_0.25', smoothing=False)
# plot_probabilities('adwin_likelihood_delta_0.5', smoothing=False)
plot_probabilities('concept_posteriors')
# plot_probabilities('concept_posteriors_smoothed', smoothing=False)
# plot_probabilities('adwin_posterior_delta_0.95', smoothing=False)
# plot_probabilities('concept_posterior_mh')
# plot_probabilities('concept_posterior_mh_exp')
#%%
plot_probabilities('adwin_posterior_estimate', smoothing=False, save_file=True)
plot_probabilities('adwin_likelihood_estimate', smoothing=False, save_file=True)
plt.show()
# %%


# deleted_models

# probs = df['adwin_posterior_estimate'].str.split(';', expand=True)
# c = probs.columns[1]
# # for c in probs:
# unique_ids = probs[c].str.rsplit(':').str[0].astype('float').unique()
# col_ids = probs[c].str.rsplit(':').str[0].astype('float')
# prob_vals = probs[c].str.rsplit(':').str[-1].astype('float')
# # u_id = 1.0
# # indexes = col_ids == u_id
# # prob_vals.loc[indexes]
# # prob_vals
# #%%
# for u_id in unique_ids:
#     if np.isnan(u_id):
#         continue
#     if f"v{u_id}" not in probs.columns:
#         probs[f"v{u_id}"] = np.nan
#     indexes = col_ids == u_id
#     probs.loc[indexes, f"v{u_id}"] = prob_vals.loc[indexes]
# probs

#%%

# %%
%matplotlib inline
plot_probabilities('concept_priors_2h')
# %%
