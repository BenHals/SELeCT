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
#Goal is to find a case in AQTemp where FICSUM and SELeCT perform significantly differently.
#Then analyse what caused this difference, and what the effect on performance was.

#Find dataset with significant differences

experiment_name = "BaseL3_wAblation"
base_path = pathlib.Path(__file__).absolute().parents[1] / 'output' / experiment_name
experiment_parameters = ["data_name", "classifier"]
experiment_IVs = ['overall_accuracy', 'GT_mean_f1']

#%%
desc_paths = list(base_path.rglob(f"desc*.txt"))
for d_path in desc_paths:
    print(d_path.open().read())
results_paths = list(base_path.rglob(f"results_run_*.txt"))
results = []
for rp in results_paths:
    try:
        results.append(json.load(rp.open()))
    except:
        print(rp)
# results = [json.load(rp.open()) for rp in results_paths]
df = pd.DataFrame(results)
#%%
# df[['data_name', 'seed', 'classifier', *experiment_IVs]]
data_name = 'AQTemp'
s = df[df['classifier'].isin(['CC', 'ficsum']) & df['data_name'].isin([data_name])]
g = s.groupby(['data_name', 'seed', 'classifier'])[[*experiment_IVs]].mean()
g.unstack('classifier')

#%%
# seed = 4
# for seed in [1, 2, 3, 4, 5]:
for seed in [5]:
    found_paths = []
    for rp in results_paths:
        try:
            data = json.load(rp.open())
            if data['data_name'] == data_name and data['seed'] == seed and data['classifier'] in ['CC', 'ficsum']:
                found_paths.append((rp, data))
        except:
            print(rp)
    found_paths.sort(key = lambda x: x[1]['classifier'])
    res_path = found_paths[1]
    for res_path in found_paths:
        res_name = res_path[0].stem
        run_name = res_path[0].parent / f"{res_name.split('results_')[1]}.csv"
        run_df = pd.read_csv(run_name)
        run_df.head()
        rolling_acc = run_df['is_correct'].rolling(500).mean().to_numpy()
        sns.lineplot(data=rolling_acc[::1], label=res_path[1]['classifier'])
    plt.title(f"Dataset: {data_name}, Seed: {seed}")
    plt.ylabel(f"Rolling Accuracy")
    plt.xlabel(f"Observation")
    ax = plt.gca()
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, map({"CC": "SELeCT", "ficsum": "FICSUM"}.get, labels))
    plt.savefig(fr"case_study\{data_name}-{seed}_rolling_acc.pdf")
    plt.show()

print(len(found_paths))
#%%
for seed in [5]:
    found_paths = []
    for rp in results_paths:
        try:
            data = json.load(rp.open())
            if data['data_name'] == data_name and data['seed'] == seed and data['classifier'] in ['CC', 'ficsum']:
                found_paths.append((rp, data))
        except:
            print(rp)
    found_paths.sort(key = lambda x: x[1]['classifier'])
    res_path = found_paths[1]
    for res_path in found_paths:
        res_name = res_path[0].stem
        run_name = res_path[0].parent / f"{res_name.split('results_')[1]}.csv"
        run_df = pd.read_csv(run_name)
        # run_df.head()
        # cd = run_df.iloc[17500:22500][['change_detected']].reset_index()
        # rolling_acc = run_df[['is_correct']].rolling(100).mean().iloc[17500:22500]
        cd = run_df.iloc[19500:20500][['change_detected']].reset_index()
        rolling_acc = run_df[['is_correct']].rolling(100).mean().iloc[19500:20500]
        sns.lineplot(data=rolling_acc.reset_index(), x='index', y='is_correct', label=res_path[1]['classifier'])
        if res_path[1]['classifier'] == 'ficsum':
            rows = 0
            for k, row in cd[cd['change_detected'] == 1].iterrows():
                plt.axvline(x=row['index'], color="red")
            print(rows)
    plt.title(f"Dataset: {data_name}, Seed: {seed}")
    plt.ylabel(f"Rolling Accuracy")
    plt.xlabel(f"Observation")
    ax = plt.gca()
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, map({"CC": "SELeCT", "ficsum": "FICSUM"}.get, labels))
    plt.savefig(fr"case_study\{data_name}-{seed}_rolling_acc_zoom.pdf")
    plt.show()
#%%
classifier_map = {"CC": "SELeCT", "ficsum": "FICSUM"}
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
for seed in [5]:
    found_paths = []
    for rp in results_paths:
        try:
            data = json.load(rp.open())
            if data['data_name'] == data_name and data['seed'] == seed and data['classifier'] in ['CC', 'ficsum']:
                found_paths.append((rp, data))
        except:
            print(rp)
    found_paths.sort(key = lambda x: x[1]['classifier'])
    res_path = found_paths[1]
    for i, res_path in enumerate(found_paths):
        res_name = res_path[0].stem
        run_name = res_path[0].parent / f"{res_name.split('results_')[1]}.csv"
        run_df = pd.read_csv(run_name)
        # run_df.head()
        # cd = run_df.iloc[17500:22500][['change_detected']].reset_index()
        # rolling_acc = run_df[['is_correct']].rolling(100).mean().iloc[17500:22500]
        view_range = [19500, 20500]
        cd = run_df.iloc[view_range[0]:view_range[1]][['change_detected']].reset_index()
        concept = run_df[['is_correct', 'active_model', 'ground_truth_concept']].iloc[view_range[0]:view_range[1]]
        line_y = len(found_paths)-i
        plt.text(view_range[0], line_y+0.1, classifier_map.get(res_path[1]['classifier']))
        line_start = view_range[0]
        line_end = view_range[0]
        c = None
        ci = 0
        for k, row in concept.iterrows():
            cc = row['active_model']
            if c is not None and c != cc:
                plt.hlines(y=line_y, xmin=line_start, xmax=k, colors=colors[int(ci)%len(colors)], lw=5)
                line_start = k
                ci += 1
            c = cc
        plt.hlines(y=line_y, xmin=line_start, xmax=view_range[1], colors=colors[int(ci)%len(colors)], lw=5)

        line_y = 0
        plt.text(view_range[0], line_y+0.1, f"Ground Truth Concept")
        line_start = view_range[0]
        line_end = view_range[0]
        c = None
        ci = 0
        for k, row in concept.fillna(0).iterrows():
            cc = row['ground_truth_concept']
            if c is not None and c != cc:
                plt.hlines(y=line_y, xmin=line_start, xmax=k, colors=colors[int(ci)%len(colors)], lw=5)
                line_start = k
                ci += 1
            c = cc
        plt.hlines(y=line_y, xmin=line_start, xmax=view_range[1], colors=colors[int(ci)%len(colors)], lw=5)
            
        # sns.lineplot(data=rolling_acc.reset_index(), x='index', y='is_correct', label=res_path[1]['classifier'])
        if res_path[1]['classifier'] == 'ficsum':
            rows = 0
            for k, row in cd[cd['change_detected'] == 1].iterrows():
                plt.axvline(x=row['index'], color="red")
            print(rows)
    # plt.title(f"Dataset: {data_name}, Seed: {seed}")
    plt.ylabel(f"Rolling Accuracy")
    plt.xlabel(f"Observation")
    ax = plt.gca()
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, map(classifier_map.get, labels))
    plt.ylim([-0.25, 2.3])
    fig = plt.gcf()
    fig.set_size_inches(5, 2)
    ax.get_yaxis().set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.savefig(fr"case_study\{data_name}-{seed}_rolling_acc_concepts.pdf")
    plt.show()

#%%
classifier_map = {"CC": "SELeCT", "ficsum": "FICSUM"}
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
for seed in [5]:
    found_paths = []
    for rp in results_paths:
        try:
            data = json.load(rp.open())
            if data['data_name'] == data_name and data['seed'] == seed and data['classifier'] in ['CC', 'ficsum']:
                found_paths.append((rp, data))
        except:
            print(rp)
    found_paths.sort(key = lambda x: x[1]['classifier'])
    res_path = found_paths[1]
    fig = plt.figure(constrained_layout=True)
    spec = fig.add_gridspec(ncols=1, nrows=2, height_ratios=[1, 0.3])
    ax1 = fig.add_subplot(spec[0])
    ax2 = fig.add_subplot(spec[1], sharex=ax1)
    for i, res_path in enumerate(found_paths):
        res_name = res_path[0].stem
        run_name = res_path[0].parent / f"{res_name.split('results_')[1]}.csv"
        run_df = pd.read_csv(run_name)
        # run_df.head()
        # cd = run_df.iloc[17500:22500][['change_detected']].reset_index()
        # rolling_acc = run_df[['is_correct']].rolling(100).mean().iloc[17500:22500]
        view_range = [19500, 20500]
        cd = run_df.iloc[view_range[0]:view_range[1]][['change_detected']].reset_index()
        concept = run_df[['is_correct', 'active_model', 'ground_truth_concept']].iloc[view_range[0]:view_range[1]]
        rolling_acc = run_df[['is_correct']].rolling(100).mean().iloc[view_range[0]:view_range[1]]
        sns.lineplot(data=rolling_acc.reset_index(), x='index', y='is_correct', label=res_path[1]['classifier'], ax=ax1)
        
        line_y = len(found_paths)-i
        ax2.text(view_range[0], line_y+0.1, classifier_map.get(res_path[1]['classifier']))
        line_start = view_range[0]
        line_end = view_range[0]
        c = None
        ci = 0
        for k, row in concept.iterrows():
            cc = row['active_model']
            if c is not None and c != cc:
                ax2.hlines(y=line_y, xmin=line_start, xmax=k, colors=colors[int(ci)%len(colors)], lw=5)
                line_start = k
                ci += 1
            c = cc
        ax2.hlines(y=line_y, xmin=line_start, xmax=view_range[1], colors=colors[int(ci)%len(colors)], lw=5)

        line_y = 0
        ax2.text(view_range[0], line_y+0.1, f"Ground Truth Concept")
        line_start = view_range[0]
        line_end = view_range[0]
        c = None
        ci = 0
        for k, row in concept.fillna(0).iterrows():
            cc = row['ground_truth_concept']
            if c is not None and c != cc:
                ax2.hlines(y=line_y, xmin=line_start, xmax=k, colors=colors[int(ci)%len(colors)], lw=5)
                line_start = k
                ci += 1
            c = cc
        ax2.hlines(y=line_y, xmin=line_start, xmax=view_range[1], colors=colors[int(ci)%len(colors)], lw=5)
            
        # sns.lineplot(data=rolling_acc.reset_index(), x='index', y='is_correct', label=res_path[1]['classifier'])
        if res_path[1]['classifier'] == 'ficsum':
            rows = 0
            for k, row in cd[cd['change_detected'] == 1].iterrows():
                ax1.axvline(x=row['index'], color="red")
                ax2.axvline(x=row['index'], color="red")
            print(rows)
    # plt.title(f"Dataset: {data_name}, Seed: {seed}")
    plt.ylabel(f"Rolling Accuracy")
    plt.xlabel(f"Observation")
    ax = plt.gca()
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, map(classifier_map.get, labels))
    plt.ylim([-0.25, 2.3])
    fig = plt.gcf()
    fig.set_size_inches(5, 5)
    ax.get_yaxis().set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.savefig(fr"case_study\{data_name}-{seed}_rolling_acc_combined.pdf")
    plt.show()
#%%
run_df.columns