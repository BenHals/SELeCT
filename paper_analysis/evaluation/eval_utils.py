#%%
import pandas as pd
import numpy as np
import pathlib
import seaborn as sns
import matplotlib.pyplot as plt
import os
import json
import seaborn as sns


def get_classifier_matched_results(base_path, experiment_name, parameters_of_interest, IVs_of_interest, seeds, restrictions):
    if base_path is None:
        base_path = pathlib.Path(__file__).absolute().parents[1] / 'output' / experiment_name

    print(base_path)
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
    print(df['classifier'].unique())
    if df.shape[0] == 0:
        raise ValueError("Data not loaded - maybe path is wrong")
    if 'classifier' not in df.columns:
        df['classifier'] = "CC"
    if 'merge_threshold' not in df.columns:
        df['merge_threshold'] = -1
    if 'zero_prob_minimum' not in df.columns:
        df['zero_prob_minimum'] = -1
    if 'TMdropoff' not in df.columns:
        df['TMdropoff'] = 1.0
    if 'TMforward' not in df.columns:
        df['TMforward'] = 1
    if 'TMnoise' not in df.columns:
        df['TMnoise'] = 0.0
    if 'prev_state_prior' not in df.columns:
        df['prev_state_prior'] = 0.0
    if 'noise' not in df.columns:
        df['noise'] = 0.0
    if 'conceptdifficulty' not in df.columns:
        df['conceptdifficulty'] = 0.0
    df['conceptdifficulty'] = df['conceptdifficulty'].fillna(0.0)
    if 'correlation_merge' not in df.columns:
        df['correlation_merge'] = False
    if 'FICSUM' not in df.columns:
        df['FICSUM'] = False
    if 'GT_mean_f1' not in df.columns:
        df['GT_mean_f1'] = df['m-GT_mean_f1']
    if 'background_state_prior_multiplier' not in df.columns:
        df['background_state_prior_multiplier'] = 0.5
    df = df[df['seed'].isin(seeds) ]
    for restricted_param, restricted_vals in restrictions:
        df = df[df[restricted_param].isin(restricted_vals)]
    df['classifier'] = df['classifier'].fillna("CC")
    # Parameters used to define a run
    # matching_parameters = ["data_name", "seed", "classifier", "isources", "ifeatures", "optdetect", "optselect", "opthalf", "shuffleconcepts", "similarity_option", "MI_calc", "window_size", "sensitivity", "min_window_ratio", "fingerprint_grace_period", "state_grace_period_window_multiplier", "bypass_grace_period_threshold", "state_estimator_risk", "state_estimator_swap_risk", "minimum_concept_likelihood", "min_drift_likelihood_threshold", "min_estimated_posterior_threshold", "similarity_gap", "fp_gap", "nonactive_fp_gap", "observation_gap", "similarity_stdev_min", "similarity_stdev_max", "buffer_ratio", "merge_threshold", "correlation_merge", "fs_method", "fingerprint_method", "fingerprint_bins"]
    matching_parameters = ["data_name", "seed", "classifier", "conceptdifficulty", "noise", "drift_width", "prev_state_prior", "TMdropoff", "TMforward", "TMnoise", "zero_prob_minimum", "shuffleconcepts", "FICSUM", "multihop_penalty", "similarity_option", "MI_calc", "window_size", "sensitivity", "min_window_ratio", "fingerprint_grace_period", "state_grace_period_window_multiplier", "bypass_grace_period_threshold", "state_estimator_risk", "state_estimator_swap_risk", "minimum_concept_likelihood", "min_drift_likelihood_threshold", "min_estimated_posterior_threshold", "similarity_gap", "fp_gap", "nonactive_fp_gap", "observation_gap", "similarity_stdev_min", "similarity_stdev_max", "buffer_ratio", "merge_threshold", "correlation_merge", "fs_method", "fingerprint_method", "fingerprint_bins", "background_state_prior_multiplier"]
    classifiers = df['conceptdifficulty'].unique()
    print(classifiers)
    runs_df = df.groupby(matching_parameters).mean()[IVs_of_interest]
    # return df
    # print(runs_df.reset_index()['classifier'].unique())
    runs_by_classifier = runs_df.unstack('classifier')
    for IV in IVs_of_interest:
        IV_df = runs_by_classifier[IV]
        for classifier in classifiers:
            try:
                runs_by_classifier.loc[:, (f"{IV}-bounded", classifier)] = (IV_df[classifier] - IV_df["lower_bound"]) / (IV_df["upper_bound"] - IV_df["lower_bound"])
                runs_by_classifier.loc[:, (f"{IV}-bounded", classifier)] = runs_by_classifier.loc[:, (f"{IV}-bounded", classifier)].replace([np.inf, -np.inf], np.nan).replace(np.nan, 1)
            except:
                pass
    runs_by_classifier = runs_by_classifier.stack('classifier').reset_index()
    if 'classifier' in parameters_of_interest:
        bounded_df = runs_by_classifier.reset_index()
    else:
        bounded_df = runs_by_classifier[runs_by_classifier['classifier'] == "CC"].reset_index()
    bounded_df = bounded_df.dropna()
    groups = bounded_df.groupby(parameters_of_interest)
    sizes = groups.size()
    grouped = groups.aggregate(('mean', 'std'))[[*IVs_of_interest, *[f"{IV}-bounded" for IV in IVs_of_interest if f"{IV}-bounded" in bounded_df.columns]]]
    grouped['size'] = sizes
    return bounded_df, groups, grouped


#%%
# get_ipython().run_line_magic('matplotlib', 'ipympl')

# experiment_name = "bounds_test"
# parameters_of_interest = ["data_name", "sensitivity", "classifier"]
# IVs_of_interest = ['overall_accuracy', 'GT_mean_f1']
# seeds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45]
# restrictions = []
# selected, groups, grouped = get_classifier_matched_results(experiment_name, parameters_of_interest, IVs_of_interest, seeds, restrictions)
# # selected['overall_accuracy-bounded'].mean()
# grouped
#%%

# %%