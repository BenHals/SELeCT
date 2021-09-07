import argparse
import importlib
import itertools
import json
import logging
import math
import multiprocessing as mp
import os
from re import L
import sys
import pathlib
import subprocess
import time
import warnings
from collections import Counter
from logging.handlers import RotatingFileHandler
from multiprocessing import RLock, freeze_support, Lock
from contextlib import contextmanager

import numpy as np
import pandas as pd
import psutil
import tqdm
from pyinstrument import Profiler
from pympler.classtracker import ClassTracker

from skmultiflow.meta.adaptive_random_forests import AdaptiveRandomForestClassifier
from skmultiflow.meta.dynamic_weighted_majority import DynamicWeightedMajorityClassifier

from SELeCT.Classifier.hoeffding_tree_shap import \
    HoeffdingTreeSHAPClassifier
from SELeCT.Classifier.select_classifier import SELeCTClassifier
from SELeCT.Classifier.FiCSUM import FiCSUMClassifier
from SELeCT.Classifier.lower_bound_classifier import BoundClassifier
from SELeCT.Classifier.wrapper_classifier import WrapperClassifier
from SELeCT.Data.load_data import (AbruptDriftStream,
                                               get_inorder_concept_ranges,
                                               load_real_concepts,
                                               load_synthetic_concepts)

warnings.filterwarnings('ignore')


class NpEncoder(json.JSONEncoder):
    """ Custom encoder for numpy data types """

    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):

            return int(obj)

        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)

        elif isinstance(obj, (np.complex_, np.complex64, np.complex128)):
            return {'real': obj.real, 'imag': obj.imag}

        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()

        elif isinstance(obj, (np.bool_)):
            return bool(obj)

        elif isinstance(obj, (np.void)):
            return None

        elif isinstance(obj, pathlib.Path):
            return str(obj)

        return json.JSONEncoder.default(self, obj)

# Load synthetic concepts (up to concept_max) and turn into a stream
# Concepts are placed in order (repeats) number of times.
# The idea is to use half the stream to learn concepts, then half
# to test for similarity, so 2 repeats to learn concepts then
# test on 2.


def make_stream(options):
    if options['data_type'] == "Synthetic":
        concepts = load_synthetic_concepts(options['data_name'],
                                           options['seed'],
                                           raw_data_path=options['raw_data_path'] / 'Synthetic',
                                           difficulty=options['conceptdifficulty'] if options['conceptdifficulty'] > 0 else None)
    else:
        concepts = load_real_concepts(options['data_name'],
                                      options['seed'],
                                      nrows=options['max_rows'],
                                      raw_data_path=options['raw_data_path'] / 'Real',
                                      sort_examples=True)
    
    stream_concepts, length = get_inorder_concept_ranges(concepts, concept_length=options['concept_length'], seed=options['seed'], repeats=options['repeats'], concept_max=options[
                                                         'concept_max'], repeat_proportion=options['repeatproportion'], shuffle=options['data_type'] != "Synthetic" or options['shuffleconcepts'],
                                                         dropoff=options['TMdropoff'], nforward=options['TMforward'], noise=options['TMnoise'])
    options['length'] = length
    try:
        stream = AbruptDriftStream(
            stream_concepts,  length, random_state=options['seed'], width=None if options['drift_width'] == 0 else options['drift_width'])
    except Exception as e:
        return None, None, None, None
    all_classes = stream._get_target_values()
    return stream, stream_concepts, length, list(all_classes)

@contextmanager
def aquire_lock(lock, timeout=-1):
    l = lock.acquire(timeout=timeout, blocking=True)
    try:
        yield l
    finally:
        if l:
            lock.release()

def get_package_status(force=False):
    data = []
    for package in ["SELeCT"]:
        try:
            loc = str(importlib.util.find_spec(
                package).submodule_search_locations[0])
        except Exception as e:
            try:
                loc = str(importlib.util.find_spec(
                    package).submodule_search_locations._path[0])
            except:
                namespace = importlib.util.find_spec(
                    package).submodule_search_locations
                loc = str(namespace).split('[')[1].split(']')[0]
                loc = loc.split(',')[0]
                loc = loc.replace("'", "")
        loc = str(pathlib.Path(loc).resolve())
        try:
            commit = subprocess.check_output(
                ["git", "describe", "--always"], cwd=loc).strip().decode()
            branch = subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=loc).strip().decode()
            changes = subprocess.call(["git", "diff", "--quiet"], cwd=loc)
            changes_cached = subprocess.call(
                ["git", "diff", "--cached", "--quiet"])
        except:
            commit = "NOGITFOUND"
            branch = "NA"
            changes = None
        if changes and not force:
            print(f"{package} has uncommitted files: {changes}")
            input(
                "Are you sure you want to run with uncommitted code? Press any button to continue...")
        package_data = f"{package}-{branch}-{commit}"
        data.append(package_data)
    return '_'.join(data)


def process_option(params):
    option, lock, queue = params
    if lock:
        tqdm.tqdm.set_lock(lock)
    np.seterr(all='ignore')
    warnings.filterwarnings('ignore')
    logging.info(pathlib.Path(sys.executable).as_posix().split('/')[-3])
    mp_process = mp.current_process()
    mp_id = 1
    try:
        mp_id = None
        if queue:
            mp_id = queue.get(timeout=10)
        if mp_id is None:
            mp_id = int(mp_process.name.split('-')[1]) % option['cpu']
    except:
        pass

    if mp_id is None:
        mp_id = 1

    proc = psutil.Process(os.getpid())
    if option['setaffinity']:
        # set cpu affinity for process
        possible_cpus = proc.cpu_affinity()
        possible_cpus = possible_cpus[:min(len(possible_cpus), option['pcpu'])]
        pcpu = possible_cpus[mp_id % len(possible_cpus)]
        try:
            proc.cpu_affinity([pcpu])
        except Exception as e:
            pass

    def get_drift_point_accuracy(log, follow_length=250):
        if not 'drift_occured' in log.columns or not 'is_correct' in log.columns:
            return 0, 0, 0, 0
        dpl = log.index[log['drift_occured'] == 1].tolist()
        dpl = dpl[1:]
        if len(dpl) == 0:
            return 0, 0, 0, 0

        following_drift = np.unique(np.concatenate(
            [np.arange(i, min(i+follow_length+1, len(log))) for i in dpl]))
        filtered = log.iloc[following_drift]
        num_close = filtered.shape[0]
        accuracy, kappa, kappa_m, kappa_t = get_performance(filtered)
        return accuracy, kappa, kappa_m, kappa_t

    def get_driftdetect_point_accuracy(log, follow_length=250):
        if not 'change_detected' in log.columns:
            return 0, 0, 0, 0
        if not 'drift_occured' in log.columns:
            return 0, 0, 0, 0
        dpl = log.index[log['change_detected'] == 1].tolist()
        drift_indexes = log.index[log['drift_occured'] == 1].tolist()
        if len(dpl) < 1:
            return 0, 0, 0, 0
        following_drift = np.unique(np.concatenate(
            [np.arange(i, min(i+1000+1, len(log))) for i in drift_indexes]))
        following_detect = np.unique(np.concatenate(
            [np.arange(i, min(i+follow_length+1, len(log))) for i in dpl]))
        following_both = np.intersect1d(
            following_detect, following_drift, assume_unique=True)
        filtered = log.iloc[following_both]
        num_close = filtered.shape[0]
        if num_close == 0:
            return 0, 0, 0, 0
        accuracy, kappa, kappa_m, kappa_t = get_performance(filtered)
        return accuracy, kappa, kappa_m, kappa_t

    def get_performance(log):
        sum_correct = log['is_correct'].sum()
        num_observations = log.shape[0]
        accuracy = sum_correct / num_observations
        values, counts = np.unique(log['y'], return_counts=True)
        majority_class = values[np.argmax(counts)]
        majority_correct = log.loc[log['y'] == majority_class]
        num_majority_correct = majority_correct.shape[0]
        majority_accuracy = num_majority_correct / num_observations
        if majority_accuracy < 1:
            kappa_m = (accuracy - majority_accuracy) / (1 - majority_accuracy)
        else:
            kappa_m = 0
        temporal_filtered = log['y'].shift(1, fill_value=0.0)
        temporal_correct = log['y'] == temporal_filtered
        temporal_accuracy = temporal_correct.sum() / num_observations
        kappa_t = (accuracy - temporal_accuracy) / (1 - temporal_accuracy)

        our_counts = Counter()
        gt_counts = Counter()
        for v in values:
            our_counts[v] = log.loc[log['p'] == v].shape[0]
            gt_counts[v] = log.loc[log['y'] == v].shape[0]

        expected_accuracy = 0
        for cat in values:
            expected_accuracy += (gt_counts[cat]
                                  * our_counts[cat]) / num_observations
        expected_accuracy /= num_observations
        if expected_accuracy < 1:
            kappa = (accuracy - expected_accuracy) / (1 - expected_accuracy)
        else:
            kappa = 0

        return accuracy, kappa, kappa_m, kappa_t

    def get_recall_precision(log, model_column="active_model"):
        ground_truth = log['ground_truth_concept'].fillna(
            method='ffill').astype(int).values
        system = log[model_column].fillna(method='ffill').astype(int).values
        gt_values, gt_total_counts = np.unique(
            ground_truth, return_counts=True)
        sys_values, sys_total_counts = np.unique(system, return_counts=True)
        matrix = np.array([ground_truth, system]).transpose()
        recall_values = {}
        precision_values = {}
        gt_results = {}
        sys_results = {}
        overall_results = {
            'Max Recall': 0,
            'Max Precision': 0,
            'Precision for Max Recall': 0,
            'Recall for Max Precision': 0,
            'GT_mean_f1': 0,
            'GT_mean_recall': 0,
            'GT_mean_precision': 0,
            'MR by System': 0,
            'MP by System': 0,
            'PMR by System': 0,
            'RMP by System': 0,
            'MODEL_mean_f1': 0,
            'MODEL_mean_recall': 0,
            'MODEL_mean_precision': 0,
            'Num Good System Concepts': 0,
            'GT_to_MODEL_ratio': 0,
        }
        gt_proportions = {}
        sys_proportions = {}

        for gt_i, gt in enumerate(gt_values):
            gt_total_count = gt_total_counts[gt_i]
            gt_mask = matrix[matrix[:, 0] == gt]
            sys_by_gt_values, sys_by_gt_counts = np.unique(
                gt_mask[:, 1], return_counts=True)
            gt_proportions[gt] = gt_mask.shape[0] / matrix.shape[0]
            max_recall = None
            max_recall_sys = None
            max_precision = None
            max_precision_sys = None
            max_f1 = None
            max_f1_sys = None
            max_f1_recall = None
            max_f1_precision = None
            for sys_i, sys in enumerate(sys_by_gt_values):
                sys_by_gt_count = sys_by_gt_counts[sys_i]
                sys_total_count = sys_total_counts[sys_values.tolist().index(
                    sys)]
                if gt_total_count != 0:
                    recall = sys_by_gt_count / gt_total_count
                else:
                    recall = 1

                recall_values[(gt, sys)] = recall

                sys_proportions[sys] = sys_total_count / matrix.shape[0]
                if sys_total_count != 0:
                    precision = sys_by_gt_count / sys_total_count
                else:
                    precision = 1
                precision_values[(gt, sys)] = precision

                f1 = 2 * ((recall * precision) / (recall + precision))

                if max_recall == None or recall > max_recall:
                    max_recall = recall
                    max_recall_sys = sys
                if max_precision == None or precision > max_precision:
                    max_precision = precision
                    max_precision_sys = sys
                if max_f1 == None or f1 > max_f1:
                    max_f1 = f1
                    max_f1_sys = sys
                    max_f1_recall = recall
                    max_f1_precision = precision
            precision_max_recall = precision_values[(gt, max_recall_sys)]
            recall_max_precision = recall_values[(gt, max_precision_sys)]
            gt_result = {
                'Max Recall': max_recall,
                'Max Precision': max_precision,
                'Precision for Max Recall': precision_max_recall,
                'Recall for Max Precision': recall_max_precision,
                'f1': max_f1,
                'recall': max_f1_recall,
                'precision': max_f1_precision,
            }
            gt_results[gt] = gt_result
            overall_results['Max Recall'] += max_recall
            overall_results['Max Precision'] += max_precision
            overall_results['Precision for Max Recall'] += precision_max_recall
            overall_results['Recall for Max Precision'] += recall_max_precision
            overall_results['GT_mean_f1'] += max_f1
            overall_results['GT_mean_recall'] += max_f1_recall
            overall_results['GT_mean_precision'] += max_f1_precision

        for sys in sys_values:
            max_recall = None
            max_recall_gt = None
            max_precision = None
            max_precision_gt = None
            max_f1 = None
            max_f1_sys = None
            max_f1_recall = None
            max_f1_precision = None
            for gt in gt_values:
                if (gt, sys) not in recall_values:
                    continue
                if (gt, sys) not in precision_values:
                    continue
                recall = recall_values[(gt, sys)]
                precision = precision_values[(gt, sys)]

                f1 = 2 * ((recall * precision) / (recall + precision))

                if max_recall == None or recall > max_recall:
                    max_recall = recall
                    max_recall_gt = gt
                if max_precision == None or precision > max_precision:
                    max_precision = precision
                    max_precision_gt = gt
                if max_f1 == None or f1 > max_f1:
                    max_f1 = f1
                    max_f1_sys = sys
                    max_f1_recall = recall
                    max_f1_precision = precision

            precision_max_recall = precision_values[(max_recall_gt, sys)]
            recall_max_precision = recall_values[(max_precision_gt, sys)]
            sys_result = {
                'Max Recall': max_recall,
                'Max Precision': max_precision,
                'Precision for Max Recall': precision_max_recall,
                'Recall for Max Precision': recall_max_precision,
                'f1': max_f1
            }
            sys_results[sys] = sys_result
            overall_results['MR by System'] += max_recall * \
                sys_proportions[sys]
            overall_results['MP by System'] += max_precision * \
                sys_proportions[sys]
            overall_results['PMR by System'] += precision_max_recall * \
                sys_proportions[sys]
            overall_results['RMP by System'] += recall_max_precision * \
                sys_proportions[sys]
            overall_results['MODEL_mean_f1'] += max_f1 * sys_proportions[sys]
            overall_results['MODEL_mean_recall'] += max_f1_recall * \
                sys_proportions[sys]
            overall_results['MODEL_mean_precision'] += max_f1_precision * \
                sys_proportions[sys]
            if max_recall > 0.75 and precision_max_recall > 0.75:
                overall_results['Num Good System Concepts'] += 1

        # Get average over concepts by dividing by number of concepts
        # Don't need to average over models as we already multiplied by proportion.
        overall_results['Max Recall'] /= gt_values.size
        overall_results['Max Precision'] /= gt_values.size
        overall_results['Precision for Max Recall'] /= gt_values.size
        overall_results['Recall for Max Precision'] /= gt_values.size
        overall_results['GT_mean_f1'] /= gt_values.size
        overall_results['GT_mean_recall'] /= gt_values.size
        overall_results['GT_mean_precision'] /= gt_values.size
        overall_results['GT_to_MODEL_ratio'] = overall_results['Num Good System Concepts'] / \
            len(gt_values)
        return overall_results

    def get_discrimination_results(log, model_column="active_model"):
        """ Calculate how many standard deviations the active state
        is from other states. 
        We first split the active state history into chunks representing 
        each segment.
        We then shrink this by 50 on each side to exclude transition periods.
        We then compare the distance from the active state to each non-active state
        in terms of stdev. We use the max of the active state stdev or comparison stdev
        for the given chunk, representing how much the active state could be discriminated
        from the comparison state.
        We return a set of all comparisons, a set of average per active state, and overall average.
        """
        models = log[model_column].unique()
        # Early similarity is unstable, so exclude first 250 obs
        all_state_active_similarity = log['all_state_active_similarity'].replace(
            '-1', np.nan).replace(-1, np.nan).dropna().astype(str).str.split(";", expand=True).astype(float)[250:]
        if len(all_state_active_similarity.columns) == 0:
            return -1, None, None
        # Scale to between 0 and 1, so invariant
        # to the size of the similarity function.

        values = np.concatenate([all_state_active_similarity[m].dropna(
        ).values for m in all_state_active_similarity.columns])
        try:
            max_similarity = np.percentile(values, 90)
        except:
            return None, None, 0
        min_similarity = min(values)

        # Split into chunks using the active model.
        # I.E. new chunk every time the active model changes.
        # We shrink chunks by 50 each side to discard transition.
        model_changes = log[model_column] != log[model_column].shift(
            1).fillna(method='bfill')
        chunk_masks = model_changes.cumsum()
        chunks = chunk_masks.unique()
        divergences = {}
        active_model_mean_divergences = {}
        mean_divergence = []

        # Find the number of observations we are interested in.
        # by combining chunk masks.
        all_chunks = None
        for chunk in chunks:
            chunk_mask = chunk_masks == chunk
            chunk_shift = chunk_mask.shift(50, fill_value=0)
            smaller_mask = chunk_mask & chunk_shift
            chunk_shift = chunk_mask.shift(-50, fill_value=0)
            smaller_mask = smaller_mask & chunk_shift
            all_state_active_similarity = log['all_state_active_similarity'].loc[smaller_mask].replace(
                '-1', np.nan).replace(-1, np.nan).dropna().astype(str).str.split(";", expand=True).astype(float)

            # We skip chunks with only an active state.
            if len(all_state_active_similarity.columns) < 2:
                continue
            if all_chunks is None:
                all_chunks = smaller_mask
            else:
                all_chunks = all_chunks | smaller_mask

        # If we only have one state, we don't
        # have any divergences
        if all_chunks is None:
            return None, None, 0

        for chunk in chunks:
            chunk_mask = chunk_masks == chunk
            chunk_shift = chunk_mask.shift(50, fill_value=0)
            smaller_mask = chunk_mask & chunk_shift
            chunk_shift = chunk_mask.shift(-50, fill_value=0)
            smaller_mask = smaller_mask & chunk_shift

            # state similarity is saved in the csv as a ; seperated list, where the index is the model ID.
            # This splits this column out into a column per model.
            all_state_active_similarity = log['all_state_active_similarity'].loc[smaller_mask].replace(
                '-1', np.nan).replace(-1, np.nan).dropna().astype(str).str.split(";", expand=True).astype(float)
            if all_state_active_similarity.shape[0] < 1:
                continue
            active_model = log[model_column].loc[smaller_mask].unique()[0]
            if active_model not in all_state_active_similarity:
                continue
            for m in all_state_active_similarity.columns:
                all_state_active_similarity[m] = (
                    all_state_active_similarity[m] - min_similarity) / (max_similarity - min_similarity)
                all_state_active_similarity[m] = np.clip(
                    all_state_active_similarity[m], 0, 1)
            # Find the proportion this chunk takes up of the total.
            # We use this to proportion the results.
            chunk_proportion = smaller_mask.sum() / all_chunks.sum()
            chunk_mean = []
            for m in all_state_active_similarity.columns:
                if m == active_model:
                    continue

                # If chunk is small, we may only see 0 or 1 observations.
                # We can't get a standard deviation from this, so we skip.
                if all_state_active_similarity[m].shape[0] < 2:
                    continue
                # Use the max of the active state, and comparison state as the Stdev.
                # You cannot distinguish if either is larger than difference.
                if active_model in all_state_active_similarity:
                    scale = np.mean([all_state_active_similarity[m].std(
                    ), all_state_active_similarity[active_model].std()])
                else:
                    scale = all_state_active_similarity[m].std()
                divergence = all_state_active_similarity[m] - \
                    all_state_active_similarity[active_model]
                avg_divergence = divergence.sum() / divergence.shape[0]

                scaled_avg_divergence = avg_divergence / scale if scale > 0 else 0

                # Mutiply by chunk proportion to average across data set.
                # Chunks are not the same size, so cannot just mean across chunks.
                scaled_avg_divergence *= chunk_proportion
                if active_model not in divergences:
                    divergences[active_model] = {}
                if m not in divergences[active_model]:
                    divergences[active_model][m] = scaled_avg_divergence
                if active_model not in active_model_mean_divergences:
                    active_model_mean_divergences[active_model] = []
                active_model_mean_divergences[active_model].append(
                    scaled_avg_divergence)
                chunk_mean.append(scaled_avg_divergence)

            if len(all_state_active_similarity.columns) > 1 and len(chunk_mean) > 0:
                mean_divergence.append(np.mean(chunk_mean))

        # Use sum because we multiplied by proportion already, so just need to add up.
        mean_divergence = np.sum(mean_divergence)
        for m in active_model_mean_divergences:
            active_model_mean_divergences[m] = np.sum(
                active_model_mean_divergences[m])

        return divergences, active_model_mean_divergences, mean_divergence

    def plot_feature_weights(log):
        feature_weights = log['feature_weights'].replace(
            '-1', np.nan).replace(-1, np.nan).dropna().astype(str).str.split(";", expand=True)

    def dump_results(option, log_path, result_path, merges, log=None):
        log_df = None
        if log is not None:
            log_df = log
        else:
            log_df = pd.read_csv(log_dump_path)
        
        # Find the final merged identities for each model ID
        df['merge_model'] = df['active_model'].copy()
        for m_init in merges:
            m_to = merges[m_init]
            while m_to in merges:
                m_from = m_to
                m_to = merges[m_from]
            df['merge_model'] = df['merge_model'].replace(m_init, m_to)
        
        # Fill in deleted models with the next model, as is done in AiRStream
        df['repair_model'] = df['merge_model'].copy()
        # Get deleted models from the progress log, some will have been deleted from merging
        # but others will just have been deleted
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
        
        # set deleted vals to nan
        for dm in deleted_models:
            df['repair_model'] = df['repair_model'].replace(dm, np.nan)
        df['repair_model'] = df['repair_model'].fillna(method="bfill")

            

        overall_accuracy = log_df['overall_accuracy'].values[-1]
        overall_time = log_df['cpu_time'].values[-1]
        overall_mem = log_df['ram_use'].values[-1]
        peak_fingerprint_mem = log_df['ram_use'].values.max()
        average_fingerprint_mem = log_df['ram_use'].values.mean()
        final_feature_weight = log_df['feature_weights'].values[-1]
        try:
            feature_weights_strs = final_feature_weight.split(';')
            feature_weights = {}
            for ftr_weight_str in feature_weights_strs:
                feature_name, feature_value = ftr_weight_str.split(':')
                feature_weights[feature_name] = float(feature_value)
        except:
            feature_weights = {"NoneRecorded": -1}

        acc, kappa, kappa_m, kappa_t = get_performance(log_df)
        result = {
            'overall_accuracy': overall_accuracy,
            'acc': acc,
            'kappa': kappa,
            'kappa_m': kappa_m,
            'kappa_t': kappa_t,
            'overall_time': overall_time,
            'overall_mem': overall_mem,
            'peak_fingerprint_mem': peak_fingerprint_mem,
            'average_fingerprint_mem': average_fingerprint_mem,
            'feature_weights': feature_weights,
            **option
        }
        for delta in [50, 250, 500]:
            acc, kappa, kappa_m, kappa_t = get_drift_point_accuracy(
                log_df, delta)
            result[f"drift_{delta}_accuracy"] = acc
            result[f"drift_{delta}_kappa"] = kappa
            result[f"drift_{delta}_kappa_m"] = kappa_m
            result[f"drift_{delta}_kappa_t"] = kappa_t
            acc, kappa, kappa_m, kappa_t = get_driftdetect_point_accuracy(
                log_df, delta)
            result[f"driftdetect_{delta}_accuracy"] = acc
            result[f"driftdetect_{delta}_kappa"] = kappa
            result[f"driftdetect_{delta}_kappa_m"] = kappa_m
            result[f"driftdetect_{delta}_kappa_t"] = kappa_t

        match_results = get_recall_precision(log_df, 'active_model')
        for k, v in match_results.items():
            result[f"nomerge-{k}"] = v
        match_results = get_recall_precision(log_df, 'merge_model')
        for k, v in match_results.items():
            result[f"m-{k}"] = v
        match_results = get_recall_precision(log_df, 'repair_model')
        for k, v in match_results.items():
            result[f"r-{k}"] = v
        for k, v in match_results.items():
            result[f"{k}"] = v

        discriminations, active_model_mean_discriminations, mean_discrimination = get_discrimination_results(
            log_df, 'active_model')
        result['nomerge_mean_discrimination'] = mean_discrimination
        discriminations, active_model_mean_discriminations, mean_discrimination = get_discrimination_results(
            log_df, 'merge_model')
        result['m_mean_discrimination'] = mean_discrimination
        discriminations, active_model_mean_discriminations, mean_discrimination = get_discrimination_results(
            log_df, 'repair_model')
        result['r_mean_discrimination'] = mean_discrimination
        result['mean_discrimination'] = mean_discrimination

        with result_path.open('w+') as f:
            json.dump(result, f, cls=NpEncoder)
        log_df.to_csv(log_dump_path, index=False)

    profiler = None
    if option['pyinstrument']:
        profiler = Profiler()
        profiler.start()
    stream, stream_concepts, length, classes = make_stream(option)
    if stream is None:
        return None
    UID = hash(tuple(f"{k}{str(v)}" for k, v in option.items()))
    window_size = option['window_size']

    learner = HoeffdingTreeSHAPClassifier

    # use an observation_gap of -1 as auto, take 1000 observations across the stream
    if option['observation_gap'] == -1:
        option['observation_gap'] = math.floor(length / 1000)
    
    # Find ground truth active concept
    stream_names = [c[3] for c in stream_concepts]
    ground_truth_concept_init = None
    for c in stream_concepts:
        concept_start = c[0]
        if concept_start <= 0 < c[1]:
            ground_truth_concept_init = stream_names.index(c[3])

    if option['classifier'].lower() == 'cc':
        classifier = SELeCTClassifier(
            learner=learner,
            window_size=option['window_size'],
            active_head_monitor_gap=option['similarity_gap'],
            fingerprint_update_gap=option['fp_gap'],
            non_active_fingerprint_update_gap=option['nonactive_fp_gap'],
            observation_gap=option['observation_gap'],
            sim_measure=option['similarity_option'],
            ignore_sources=option['isources'],
            ignore_features=option['ifeatures'],
            similarity_num_stdevs=option['similarity_stdev_thresh'],
            similarity_min_stdev=option['similarity_stdev_min'],
            similarity_max_stdev=option['similarity_stdev_max'],
            buffer_ratio=option['buffer_ratio'],
            feature_selection_method=option['fs_method'],
            fingerprint_method=option['fingerprint_method'],
            fingerprint_bins=option['fingerprint_bins'],
            MI_calc=option['MI_calc'],
            sensitivity = option['sensitivity'],
            min_window_ratio = option['min_window_ratio'],
            fingerprint_grace_period = option['fingerprint_grace_period'],
            state_grace_period_window_multiplier = option['state_grace_period_window_multiplier'],
            bypass_grace_period_threshold = option['bypass_grace_period_threshold'],
            state_estimator_risk = option['state_estimator_risk'],
            state_estimator_swap_risk = option['state_estimator_swap_risk'],
            minimum_concept_likelihood = option['minimum_concept_likelihood'],
            min_drift_likelihood_threshold = option['min_drift_likelihood_threshold'],
            min_estimated_posterior_threshold = option['min_estimated_posterior_threshold'],
            correlation_merge=option['correlation_merge'],
            merge_threshold=option['merge_threshold'],
            background_state_prior_multiplier=option['background_state_prior_multiplier'],
            zero_prob_minimum=option['zero_prob_minimum'],
            multihop_penalty=option['multihop_penalty'],
            prev_state_prior=option['prev_state_prior'],
            MAP_selection=option['MAP_selection']
            )
    
    elif option['classifier'].lower() == 'cc_basicprior':
        classifier = SELeCTClassifier(
            learner=learner,
            window_size=option['window_size'],
            active_head_monitor_gap=option['similarity_gap'],
            fingerprint_update_gap=option['fp_gap'],
            non_active_fingerprint_update_gap=option['nonactive_fp_gap'],
            observation_gap=option['observation_gap'],
            sim_measure=option['similarity_option'],
            ignore_sources=option['isources'],
            ignore_features=option['ifeatures'],
            similarity_num_stdevs=option['similarity_stdev_thresh'],
            similarity_min_stdev=option['similarity_stdev_min'],
            similarity_max_stdev=option['similarity_stdev_max'],
            buffer_ratio=option['buffer_ratio'],
            feature_selection_method=option['fs_method'],
            fingerprint_method=option['fingerprint_method'],
            fingerprint_bins=option['fingerprint_bins'],
            MI_calc=option['MI_calc'],
            sensitivity = option['sensitivity'],
            min_window_ratio = option['min_window_ratio'],
            fingerprint_grace_period = option['fingerprint_grace_period'],
            state_grace_period_window_multiplier = option['state_grace_period_window_multiplier'],
            bypass_grace_period_threshold = option['bypass_grace_period_threshold'],
            state_estimator_risk = option['state_estimator_risk'],
            state_estimator_swap_risk = option['state_estimator_swap_risk'],
            minimum_concept_likelihood = option['minimum_concept_likelihood'],
            min_drift_likelihood_threshold = option['min_drift_likelihood_threshold'],
            min_estimated_posterior_threshold = option['min_estimated_posterior_threshold'],
            correlation_merge=option['correlation_merge'],
            merge_threshold=option['merge_threshold'],
            background_state_prior_multiplier=1.0,
            zero_prob_minimum=0.9999,
            multihop_penalty=0.0,
            prev_state_prior=0.0,
            MAP_selection=option['MAP_selection']
            )
    elif option['classifier'].lower() == 'cc_map':
        classifier = SELeCTClassifier(
            learner=learner,
            window_size=option['window_size'],
            active_head_monitor_gap=option['similarity_gap'],
            fingerprint_update_gap=option['fp_gap'],
            non_active_fingerprint_update_gap=option['nonactive_fp_gap'],
            observation_gap=option['observation_gap'],
            sim_measure=option['similarity_option'],
            ignore_sources=option['isources'],
            ignore_features=option['ifeatures'],
            similarity_num_stdevs=option['similarity_stdev_thresh'],
            similarity_min_stdev=option['similarity_stdev_min'],
            similarity_max_stdev=option['similarity_stdev_max'],
            buffer_ratio=option['buffer_ratio'],
            feature_selection_method=option['fs_method'],
            fingerprint_method=option['fingerprint_method'],
            fingerprint_bins=option['fingerprint_bins'],
            MI_calc=option['MI_calc'],
            sensitivity = option['sensitivity'],
            min_window_ratio = option['min_window_ratio'],
            fingerprint_grace_period = option['fingerprint_grace_period'],
            state_grace_period_window_multiplier = option['state_grace_period_window_multiplier'],
            bypass_grace_period_threshold = option['bypass_grace_period_threshold'],
            state_estimator_risk = option['state_estimator_risk'],
            state_estimator_swap_risk = option['state_estimator_swap_risk'],
            minimum_concept_likelihood = option['minimum_concept_likelihood'],
            min_drift_likelihood_threshold = option['min_drift_likelihood_threshold'],
            min_estimated_posterior_threshold = option['min_estimated_posterior_threshold'],
            correlation_merge=option['correlation_merge'],
            merge_threshold=option['merge_threshold'],
            background_state_prior_multiplier=option['background_state_prior_multiplier'],
            zero_prob_minimum=option['zero_prob_minimum'],
            multihop_penalty=option['multihop_penalty'],
            prev_state_prior=option['prev_state_prior'],
            MAP_selection=True
            )
    elif option['classifier'].lower() == 'cc_nomerge':
        classifier = SELeCTClassifier(
            learner=learner,
            window_size=option['window_size'],
            active_head_monitor_gap=option['similarity_gap'],
            fingerprint_update_gap=option['fp_gap'],
            non_active_fingerprint_update_gap=option['nonactive_fp_gap'],
            observation_gap=option['observation_gap'],
            sim_measure=option['similarity_option'],
            ignore_sources=option['isources'],
            ignore_features=option['ifeatures'],
            similarity_num_stdevs=option['similarity_stdev_thresh'],
            similarity_min_stdev=option['similarity_stdev_min'],
            similarity_max_stdev=option['similarity_stdev_max'],
            buffer_ratio=option['buffer_ratio'],
            feature_selection_method=option['fs_method'],
            fingerprint_method=option['fingerprint_method'],
            fingerprint_bins=option['fingerprint_bins'],
            MI_calc=option['MI_calc'],
            sensitivity = option['sensitivity'],
            min_window_ratio = option['min_window_ratio'],
            fingerprint_grace_period = option['fingerprint_grace_period'],
            state_grace_period_window_multiplier = option['state_grace_period_window_multiplier'],
            bypass_grace_period_threshold = option['bypass_grace_period_threshold'],
            state_estimator_risk = option['state_estimator_risk'],
            state_estimator_swap_risk = option['state_estimator_swap_risk'],
            minimum_concept_likelihood = option['minimum_concept_likelihood'],
            min_drift_likelihood_threshold = option['min_drift_likelihood_threshold'],
            min_estimated_posterior_threshold = option['min_estimated_posterior_threshold'],
            correlation_merge=False,
            merge_threshold=1.0,
            background_state_prior_multiplier=option['background_state_prior_multiplier'],
            zero_prob_minimum=option['zero_prob_minimum'],
            multihop_penalty=option['multihop_penalty'],
            prev_state_prior=option['prev_state_prior'],
            MAP_selection=option['MAP_selection']
            )
    elif option['classifier'].lower() == 'ficsum':
        classifier = FiCSUMClassifier(
            learner=learner,
            window_size=option['window_size'],
            similarity_gap=option['similarity_gap'],
            fingerprint_update_gap=option['fp_gap'],
            non_active_fingerprint_update_gap=option['nonactive_fp_gap'],
            observation_gap=option['observation_gap'],
            sim_measure=option['similarity_option'],
            ignore_sources=option['isources'],
            ignore_features=option['ifeatures'],
            similarity_num_stdevs=option['similarity_stdev_thresh'],
            similarity_min_stdev=option['similarity_stdev_min'],
            similarity_max_stdev=option['similarity_stdev_max'],
            buffer_ratio=option['buffer_ratio'],
            feature_selection_method=option['fs_method'],
            fingerprint_method=option['fingerprint_method'],
            fingerprint_bins=option['fingerprint_bins'],
            MI_calc=option['MI_calc'],
            min_window_ratio = option['min_window_ratio'])
    # elif option['lower_bound']:
    elif option['classifier'].lower() == 'lower_bound':
        option['optselect'] = True
        option['optdetect'] = True

        classifier = BoundClassifier(
            learner=learner,
            window_size=option['window_size'],
            bounds="lower",
            init_concept_id=ground_truth_concept_init
        )
    # elif option['upper_bound']:
    elif option['classifier'].lower() == 'upper_bound':
        option['optselect'] = True
        option['optdetect'] = True
        classifier = BoundClassifier(
            learner=learner,
            window_size=option['window_size'],
            bounds="upper",
            init_concept_id=ground_truth_concept_init
        )
    elif option['classifier'].lower() == 'middle_bound':
        option['optselect'] = True
        option['optdetect'] = True
        classifier = BoundClassifier(
            learner=learner,
            window_size=option['window_size'],
            bounds="middle",
            init_concept_id=ground_truth_concept_init
        )
    elif option['classifier'].lower() == 'arf':
        classifier = WrapperClassifier(
        learner=AdaptiveRandomForestClassifier,
        init_concept_id=ground_truth_concept_init
        )
    elif option['classifier'].lower() == 'dwm':
        classifier = WrapperClassifier(
        learner=DynamicWeightedMajorityClassifier,
        init_concept_id=ground_truth_concept_init
        )
    else:
        raise ValueError(f"classifier option '{option['classifier']}' is not valid")

    # print(type(classifier))
    output_path = option['base_output_path'] / option['experiment_name'] / \
        option['package_status'] / option['data_name'] / str(option['seed'])
    output_path.mkdir(parents=True, exist_ok=True)
    run_index = 0
    run_name = f"run_{UID}_{run_index}"
    log_dump_path = output_path / f"{run_name}.csv"
    options_dump_path = output_path / f"{run_name}_options.txt"
    options_dump_path_partial = output_path / f"partial_{run_name}_options.txt"
    results_dump_path = output_path / f"results_{run_name}.txt"

    # Look for existing file using this name.
    # This will happen for all other option types, so
    # look for other runs with the same options.
    json_options = json.loads(json.dumps(option, cls=NpEncoder))

    def compare_options(A, B):
        for k in A:
            if k in ['log_name', 'package_status', 'base_output_path', 'cpu', 'pcpu']:
                continue
            if k not in A or k not in B:
                continue
            if A[k] != B[k]:
                return False
        return True
    other_runs = output_path.glob('*_options.txt')
    for other_run_path in other_runs:
        if 'partial' in other_run_path.stem:
            continue
        else:
            with other_run_path.open() as f:
                existing_options = json.load(f)
            if compare_options(existing_options, json_options):
                if option['pyinstrument']:
                    profiler.stop()
                return other_run_path

    while options_dump_path.exists() or options_dump_path_partial.exists():
        run_index += 1
        run_name = f"runother_{UID}_{run_index}"
        log_dump_path = output_path / f"{run_name}.csv"
        options_dump_path = output_path / f"{run_name}_options.txt"
        options_dump_path_partial = output_path / \
            f"partial_{run_name}_options.txt"
        results_dump_path = output_path / f"results_{run_name}.txt"

    partial_log_size = 2500
    partial_logs = []
    partial_log_index = 0

    with options_dump_path_partial.open('w+') as f:
        json.dump(option, f, cls=NpEncoder)

    right = 0
    wrong = 0
    stream_names = [c[3] for c in stream_concepts]

    monitoring_data = []
    monitoring_header = ('example', 'y', 'p', 'is_correct', 'right_sum', 'wrong_sum', 'overall_accuracy', 'active_model', 'ground_truth_concept', 'drift_occured', 'change_detected', 'model_evolution',
                         'active_state_active_similarity', 'active_state_buffered_similarity', 'all_state_buffered_similarity', 'all_state_active_similarity', 'feature_weights', 'concept_likelihoods', 'concept_priors', 'concept_priors_1h', 'concept_priors_2h', 'concept_posteriors', "adwin_likelihood_estimate", "adwin_posterior_estimate", "adwin_likelihood_estimate_background", "adwin_posterior_estimate_background", "merges", 'deletions', 'cpu_time', 'ram_use', 'fingerprint_ram')
    logging.info(option)
    logging.info(classifier.__dict__)
    start_time = time.process_time()

    def memory_usage_psutil():
        # return the memory usage in MB
        mem = proc.memory_info()[0] / float(2 ** 20)
        return mem
    start_mem = memory_usage_psutil()
    tracker = ClassTracker()
    if not hasattr(classifier, 'fingerprint_type'):
        classifier.fingerprint_type = {}
        classifier.active_state = -1
    tracker.track_class(classifier.fingerprint_type)
    try:
        tracker.create_snapshot()
    except:
        pass
    ram_use = 0
    aff = proc.cpu_affinity() if option['setaffinity'] else -1

    progress_bar = None
    if mp_id < 100:
        if lock:
            l = lock.acquire(timeout=10, blocking=True)
            if l:
                progress_bar = tqdm.tqdm(total=option['length'], position=min(mp_id+2, 11), desc=f"CPU {aff} - Worker {mp_id} {str(UID)[:4]}...", leave=False, mininterval=3.0, lock_args=(True, 0.01), ascii=True)
                lock.release()
        else:
            progress_bar = tqdm.tqdm(total=option['length'], position=min(mp_id+2, 11), desc=f"CPU {aff} - Worker {mp_id} {str(UID)[:4]}...", leave=False, mininterval=3.0, ascii=True)
    pbar_updates = 0

    # noise_rng = np.random.RandomState(option['seed'])
    noise_rng = np.random.default_rng()
    
    for i in range(option['length']):
        current_merges = None
        observation_monitoring = {}
        observation_monitoring['example'] = i
        X, y = stream.next_sample()
        if option['noise'] > 0:
            noise_roll = noise_rng.rand()
            if noise_roll < option['noise']:
                y = np.array([noise_rng.choice(classes)])

        observation_monitoring['y'] = int(y[0])
        p = classifier.predict(X)
        observation_monitoring['p'] = int(y[0])
        e = y[0] == p[0]
        observation_monitoring['is_correct'] = int(e)
        right += y[0] == p[0]
        observation_monitoring['right_sum'] = right
        wrong += y[0] != p[0]
        observation_monitoring['wrong_sum'] = wrong
        observation_monitoring['overall_accuracy'] = right / (right + wrong)
        if option['minimal_output']:
            observation_monitoring['overall_accuracy'] = round(observation_monitoring['overall_accuracy'] * 1000000) / 1000000

        # Find ground truth active concept
        ground_truth_concept_index = None
        for c in stream_concepts:
            concept_start = c[0]
            if concept_start <= i < c[1]:
                ground_truth_concept_index = stream_names.index(c[3])

        # Control parameters
        drift_occured = False
        concept_drift = False
        concept_drift_target = None
        concept_transition = False
        classifier.manual_control = False
        classifier.force_stop_learn_fingerprint = False
        classifier.force_transition = False
        classifier.force_transition_only = False
        num_concepts_to_lock = 6
        if option['optdetect'] or ((option['opthalf'] or option['opthalflock']) and i < option['concept_length'] * num_concepts_to_lock):
            classifier.force_transition_only = True

        # if option['opthalflock'] and i > option['concept_length'] * num_concepts_to_lock:
        # if i > 2000:
        #     classifier.force_lock_weights = True
        # else:
        #     classifier.force_lock_weights = False

        # For control, find if there was a ground truth
        # drift, with some delay.
        for c in stream_concepts[1:]:
            concept_start = c[0]
            if i == concept_start:
                drift_occured = True
        for c in stream_concepts[1:]:
            concept_start = c[0]
            if i == concept_start + window_size + 10:
                concept_drift = True
                concept_drift_target = stream_names.index(c[3])
        if concept_drift and (option['optdetect'] or ((option['opthalf'] or option['opthalflock']) and i < option['concept_length'] * num_concepts_to_lock)):
            classifier.manual_control = True
            classifier.force_transition = True
            classifier.force_stop_learn_fingerprint = True
            if option['optselect'] or ((option['opthalf'] or option['opthalflock']) and i < option['concept_length'] * num_concepts_to_lock):
                classifier.force_transition_to = concept_drift_target
        if concept_transition:
            classifier.force_stop_learn_fingerprint = True

        classifier.partial_fit(X, y, classes=classes)
        # Collect monitoring data for storage.
        current_active_model = classifier.active_state
        observation_monitoring['active_model'] = current_active_model
        observation_monitoring['ground_truth_concept'] = int(ground_truth_concept_index) if ground_truth_concept_index is not None else ground_truth_concept_index
        observation_monitoring['drift_occured'] = int(drift_occured)
        observation_monitoring['change_detected'] = int(classifier.detected_drift)
        observation_monitoring['model_evolution'] = classifier.get_active_state(
        ).current_evolution

        if option['classifier'] == 'CC':
            if classifier.monitor_active_state_active_similarity is not None and not option['minimal_output']:
                observation_monitoring['active_state_active_similarity'] = classifier.monitor_active_state_active_similarity
            else:
                observation_monitoring['active_state_active_similarity'] = -1 if not option['minimal_output'] else None

            if classifier.monitor_active_state_buffered_similarity is not None and not option['minimal_output']:
                observation_monitoring['active_state_buffered_similarity'] = classifier.monitor_active_state_buffered_similarity
            else:
                observation_monitoring['active_state_buffered_similarity'] = -1 if not option['minimal_output'] else None

            buffered_data = classifier.monitor_all_state_buffered_similarity
            if buffered_data is not None and not option['minimal_output']:
                buffered_accuracy, buffered_stats, buffered_window, buffered_similarities = buffered_data
                concept_similarities = [
                    (int(k), f"{v:.4f}") for k, v in buffered_similarities.items() if k != 'active']
                concept_similarities.sort(key=lambda x: x[0])
                observation_monitoring['all_state_buffered_similarity'] = ';'.join(
                    [str(x[1]) for x in concept_similarities])
            else:
                observation_monitoring['all_state_buffered_similarity'] = -1 if not option['minimal_output'] else None

            weights = classifier.monitor_feature_selection_weights
            if weights is not None and option['save_feature_weights'] and not option['minimal_output']:
                observation_monitoring['feature_weights'] = ';'.join(
                    [f"{s}{f}:{v}" for s, f, v in weights])
            else:
                observation_monitoring['feature_weights'] = -1 if not option['minimal_output'] else None

            # if not option['FICSUM']:
            concept_likelihoods = classifier.concept_likelihoods
            if concept_likelihoods is not None and not option['minimal_output']:
                observation_monitoring['concept_likelihoods'] = ';'.join(
                    [f"{cid}:{v:.4f}" for cid, v in concept_likelihoods.items()])
            else:
                observation_monitoring['concept_likelihoods'] = -1 if not option['minimal_output'] else None

            # concept_likelihoods_smoothed = classifier.concept_likelihoods_smoothed
            # if concept_likelihoods_smoothed is not None:
            #     observation_monitoring['concept_likelihoods_smoothed'] = ';'.join(
            #         [f"{cid}:{v}" for cid, v in concept_likelihoods_smoothed.items()])
            # else:
            #     observation_monitoring['concept_likelihoods_smoothed'] = -1
            # concept_posteriors_smoothed = classifier.concept_posteriors_smoothed
            # if concept_posteriors_smoothed is not None:
            #     observation_monitoring['concept_posteriors_smoothed'] = ';'.join(
            #         [f"{cid}:{v}" for cid, v in concept_posteriors_smoothed.items()])
            # else:
            #     observation_monitoring['concept_posteriors_smoothed'] = -1

            concept_priors = classifier.concept_priors
            if concept_priors is not None and not option['minimal_output']:
                observation_monitoring['concept_priors'] = ';'.join(
                    [f"{cid}:{v:.4f}" for cid, v in concept_priors.items()])
            else:
                observation_monitoring['concept_priors'] = -1 if not option['minimal_output'] else None

            concept_priors_2h = classifier.concept_priors_2h
            if concept_priors_2h is not None and not option['minimal_output']:
                observation_monitoring['concept_priors_2h'] = ';'.join(
                    [f"{cid}:{v:.4f}" for cid, v in concept_priors_2h.items()])
            else:
                observation_monitoring['concept_priors_2h'] = -1 if not option['minimal_output'] else None
            concept_priors_1h = classifier.concept_priors_1h
            if concept_priors_1h is not None and not option['minimal_output']:
                observation_monitoring['concept_priors_1h'] = ';'.join(
                    [f"{cid}:{v:.4f}" for cid, v in concept_priors_1h.items()])
            else:
                observation_monitoring['concept_priors_1h'] = -1 if not option['minimal_output'] else None
            concept_posteriors = classifier.concept_posteriors
            if concept_posteriors is not None and not option['minimal_output']:
                observation_monitoring['concept_posteriors'] = ';'.join(
                    [f"{cid}:{v:.4f}" for cid, v in concept_posteriors.items()])
            else:
                observation_monitoring['concept_posteriors'] = -1 if not option['minimal_output'] else None

            adwin_likelihood_estimate = {}
            # adwin_likelihood_estimate[-1] = classifier.background_state.get_estimated_likelihood() if classifier.background_state else -1
            adwin_likelihood_estimate.update({i:s.get_estimated_likelihood() for i,s in classifier.state_repository.items()})
            if adwin_likelihood_estimate is not None and not option['minimal_output']:
                observation_monitoring['adwin_likelihood_estimate'] = ';'.join(
                    [f"{cid}:{v:.4f}" for cid, v in adwin_likelihood_estimate.items()])
            else:
                observation_monitoring['adwin_likelihood_estimate'] = "0:0" if not option['minimal_output'] else None
            observation_monitoring['adwin_likelihood_estimate_background'] = classifier.background_state.get_estimated_likelihood() if classifier.background_state and not option['minimal_output'] else 0
            
            adwin_posterior_estimate = {}
            # adwin_posterior_estimate[-1] = classifier.background_state.get_estimated_posterior() if classifier.background_state else -1
            adwin_posterior_estimate.update({i:s.get_estimated_posterior() for i,s in classifier.state_repository.items()})
            if adwin_posterior_estimate is not None and not option['minimal_output']:
                observation_monitoring['adwin_posterior_estimate'] = ';'.join(
                    [f"{cid}:{v:.4f}" for cid, v in adwin_posterior_estimate.items()])
            else:
                observation_monitoring['adwin_posterior_estimate'] = "0:0" if not option['minimal_output'] else None
            observation_monitoring['adwin_posterior_estimate_background'] = classifier.background_state.get_estimated_posterior() if classifier.background_state and not option['minimal_output'] else 0

        
        # Save merges to progress on change
        merges = classifier.merges if hasattr(classifier, "merges") else {}
        if merges != current_merges:
            observation_monitoring['merges'] = ';'.join([f"{from_id}:{to_id}" for from_id, to_id in merges.items()])
            current_merges = merges

        deletions = classifier.deletions if hasattr(classifier, "deletions") else []
        observation_monitoring['deletions'] = ';'.join([str(v) for v in deletions]) if len(deletions) > 0 else ""
        


        all_state_active_similarity = classifier.monitor_all_state_active_similarity
        if all_state_active_similarity is not None and not option['minimal_output']:
            active_accuracy, active_stats, active_fingerprints, active_window, active_similarities = all_state_active_similarity
            concept_similarities = [
                (int(k), f"{v:.4f}") for k, v in active_similarities.items() if k != 'active']
            concept_similarities.sort(key=lambda x: x[0])
            observation_monitoring['all_state_active_similarity'] = ';'.join(
                [str(x[1]) for x in concept_similarities])
        else:
            observation_monitoring['all_state_active_similarity'] = -1 if not option['minimal_output'] else None

        observation_monitoring['detected_drift'] = classifier.detected_drift
        observation_monitoring['concept_drift'] = concept_drift

        observation_monitoring['cpu_time'] = time.process_time() - start_time
        observation_monitoring['ram_use'] = ram_use
        last_memory_snap = tracker.snapshots[-1]
        if hasattr(last_memory_snap, "classes"):
            observation_monitoring['fingerprint_ram'] = last_memory_snap.classes[-1]['avg']
        else:
            observation_monitoring['fingerprint_ram'] = 0.0

        monitoring_data.append(observation_monitoring)

        dump_start = time.process_time()
        if len(monitoring_data) >= partial_log_size:
            try:
                tracker.create_snapshot()
            except:
                pass
            ram_use = memory_usage_psutil() - start_mem
            log_dump_path_partial = output_path / \
                f"partial_{run_name}_{partial_log_index}.csv"
            df = pd.DataFrame(monitoring_data, columns=monitoring_header)
            df.to_csv(log_dump_path_partial, index=False)
            partial_log_index += 1
            partial_logs.append(log_dump_path_partial)
            monitoring_data = []
            df = None
        dump_end = time.process_time()
        dump_time = dump_end - dump_start
        start_time -= dump_time

        # try to aquire the lock to update progress bar.
        # We don't care too much so use a short timeout!
        pbar_updates += 1
        if progress_bar:
            if lock:
                if pbar_updates >= 250:
                    l = lock.acquire(blocking=False)
                    if l:
                        progress_bar.update(n=pbar_updates)
                        pbar_updates = 0
                        progress_bar.refresh(lock_args=(False))
                        lock.release()
            else:
                progress_bar.update(n=1)



    df = None
    for partial_log in partial_logs:
        if df is None:
            df = pd.read_csv(partial_log)
        else:
            next_log = pd.read_csv(partial_log)
            df = df.append(next_log)
    if df is None:
        df = pd.DataFrame(monitoring_data, columns=monitoring_header)
    else:
        df = df.append(pd.DataFrame(
            monitoring_data, columns=monitoring_header))
    df = df.reset_index(drop=True)
    df.to_csv(log_dump_path, index=False)
    with options_dump_path.open('w+') as f:
        json.dump(option, f, cls=NpEncoder)

    for partial_log in partial_logs:
        partial_log.unlink()
    options_dump_path_partial.unlink()

    dump_results(option, log_dump_path, results_dump_path, classifier.merges if hasattr(classifier, "merges") else {}, df)
    if option['pyinstrument']:
        profiler.stop()
        res = profiler.output_text(unicode=True, color=True)
        print(res)
        with open("profile.txt", 'w+') as f:
            f.write(res)
    
    # if progress_bar:
    #     if lock or True:
    #         l = lock.acquire(timeout=10, blocking=True)
    #         if l:
    #             progress_bar.close()
    #             lock.release()
    #     else:
            # progress_bar.close()
    
    if queue:
        queue.put(mp_id)

    return options_dump_path


if __name__ == "__main__":
    freeze_support()
    my_parser = argparse.ArgumentParser()
    my_parser.add_argument('--datasets', default='cmc', nargs="*", type=str)
    my_parser.add_argument('--classifier', default='CC', nargs="*", type=str)
    my_parser.add_argument('--seeds', default=1, nargs="*", type=int)
    my_parser.add_argument('--seedaction', default="list",
                           type=str, choices=['new', 'list', 'reuse'])
    my_parser.add_argument('--datalocation', default="RawData", type=str)
    my_parser.add_argument('--outputlocation', default="output", type=str)
    my_parser.add_argument('--loglocation', default="experimentlog", type=str)
    my_parser.add_argument('--experimentname', default="expDefault", type=str)
    my_parser.add_argument('--desc', default="", type=str)
    my_parser.add_argument('--optionslocation', default=None, type=str)
    my_parser.add_argument(
        '--fsmethod', default="fisher_overall", type=str, nargs="*")
    my_parser.add_argument('--fingerprintmethod',
                           default="descriptive", type=str, nargs="*")
    my_parser.add_argument('--fingerprintbins',
                           default=10, type=int, nargs="*")
    my_parser.add_argument('--logging', action='store_true')
    my_parser.add_argument('--minimal_output', action='store_true')
    my_parser.add_argument('--MAP_selection', action='store_true')
    my_parser.add_argument('--pyinstrument', action='store_true')
    my_parser.add_argument('--ficsum', action='store_true')
    my_parser.add_argument('--single', action='store_true')
    my_parser.add_argument('--lockpbar', action='store_false')
    my_parser.add_argument('--forcegitcheck', action='store_true')
    my_parser.add_argument('--experimentopts', action='store_true')
    my_parser.add_argument('--cpu', default=2, type=int)
    my_parser.add_argument('--setaffinity', action='store_true')
    my_parser.add_argument('--pcpu', default=-1, type=int)
    my_parser.add_argument('--repeats', default=3, type=int)
    my_parser.add_argument('--concept_length', default=5000, type=int)
    my_parser.add_argument('--concept_max', default=6, type=int)
    my_parser.add_argument('--repeatproportion', default=1.0, type=float)
    my_parser.add_argument('--TMdropoff', default=1.0, type=float)
    my_parser.add_argument('--TMforward', default=1, type=int)
    my_parser.add_argument('--TMnoise', default=0.0, type=float)
    my_parser.add_argument('--drift_width', default=0, type=float, nargs="*")
    my_parser.add_argument('--noise', default=0, type=float, nargs="*")
    my_parser.add_argument('--conceptdifficulty', default=0, type=float, nargs="*")
    my_parser.add_argument('--maxrows', default=75000, type=int)
    my_parser.add_argument('--sim', default='metainfo', nargs="*", type=str)
    my_parser.add_argument('--MIcalc', default='metainfo', nargs="*", type=str)
    my_parser.add_argument('--window_size', default=100, nargs="*", type=int)
    my_parser.add_argument('--sensitivity', default=0.05, nargs="*", type=float)
    my_parser.add_argument('--min_window_ratio', default=0.65, nargs="*", type=float)
    my_parser.add_argument('--fingerprint_grace_period', default=10, nargs="*", type=int)
    my_parser.add_argument('--state_grace_period_window_multiplier', default=10, nargs="*", type=int)
    my_parser.add_argument('--bypass_grace_period_threshold', default=0.2, nargs="*", type=float)
    my_parser.add_argument('--state_estimator_risk', default=0.5, nargs="*", type=float)
    my_parser.add_argument('--state_estimator_swap_risk', default=0.75, nargs="*", type=float)
    my_parser.add_argument('--minimum_concept_likelihood', default=0.005, nargs="*", type=float)
    my_parser.add_argument('--min_drift_likelihood_threshold', default=0.175, nargs="*", type=float)
    my_parser.add_argument('--min_estimated_posterior_threshold', default=0.2, nargs="*", type=float)
    my_parser.add_argument('--sim_gap', default=5, nargs="*", type=int)
    my_parser.add_argument('--fp_gap', default=15, nargs="*", type=int)
    # my_parser.add_argument('--fp_gap', default=6, nargs="*", type=int)
    my_parser.add_argument('--na_fp_gap', default=50, nargs="*", type=int)
    my_parser.add_argument('--ob_gap', default=5, nargs="*", type=int)
    my_parser.add_argument('--sim_stdevs', default=3, nargs="*", type=float)
    my_parser.add_argument(
        '--min_sim_stdev', default=0.015, nargs="*", type=float)
    my_parser.add_argument(
        '--max_sim_stdev', default=0.175, nargs="*", type=float)
    my_parser.add_argument(
        '--buffer_ratio', default=0.20, nargs="*", type=float)
    my_parser.add_argument(
        '--merge_threshold', default=0.95, nargs="*", type=float)
    my_parser.add_argument(
        '--background_state_prior_multiplier', default=0.4, nargs="*", type=float)
    my_parser.add_argument('--zero_prob_minimum', default=0.7, nargs="*", type=float)
    my_parser.add_argument('--multihop_penalty', default=0.7, nargs="*", type=float)
    my_parser.add_argument('--prev_state_prior', default=50, nargs="*", type=float)
    my_parser.add_argument('--no_merge', action='store_true')
    my_parser.add_argument('--optdetect', action='store_true')
    my_parser.add_argument('--optselect', action='store_true')
    my_parser.add_argument('--opthalf', action='store_true')
    my_parser.add_argument('--opthalflock', action='store_true')
    my_parser.add_argument('--shuffleconcepts', action='store_true')
    my_parser.add_argument('--save_feature_weights', action='store_true')
    my_parser.add_argument('--isources', nargs="*",
                           help="set sources to be ignored, from feature, f{i}, labels, predictions, errors, error_distances")
    my_parser.add_argument('--ifeatures', default=['IMF', 'MI', 'pacf'], nargs="*",
                           help="set features to be ignored, any meta-information feature")
    my_parser.add_argument('--classes', nargs='*', help='We try to detect classes automatically\
                                when the normalizer is set up, but sometimes this does not find\
                                rare classes. In this case, manually pass all clases in the dataset.')
    args = my_parser.parse_args()

    real_drift_datasets = ['AQSex', 'AQTemp',
                           'Arabic', 'cmc', 'UCI-Wine', 'qg']
    real_unknown_datasets = ['Airlines', 'Arrowtown', 'AWS', 'Beijing', 'covtype', 'Elec', 'gassensor', 'INSECTS-abn', 'INSECTS-irbn', 'INSECTS-oocn',
                             'KDDCup', 'Luxembourg', 'NOAA', 'outdoor', 'ozone', 'Poker', 'PowerSupply', 'Rangiora', 'rialto', 'Sensor', 'Spam', 'SpamAssassin']
    synthetic_MI_datasets = ['RTREESAMPLE', 'HPLANESAMPLE']
    synthetic_perf_only_datasets = ['STAGGER', 'RTREEMedSAMPLE', 'RBFMed']
    synthetic_unused = ['STAGGERS', 'RTREE', 'HPLANE', 'RTREEEasy', 'RTREEEasySAMPLE', 'RBFEasy', 'RTREEEasyF',
                        'RTREEEasyA', 'SynEasyD', 'SynEasyA', 'SynEasyF', 'SynEasyDA', 'SynEasyDF', 'SynEasyAF', 'SynEasyDAF']
    synthetic_dist = ["RTREESAMPLE_Diff", "RTREESAMPLE_HARD", "WINDSIM", 'SigNoiseGenerator-1', 'SigNoiseGenerator-2', 'SigNoiseGenerator-3', 'SigNoiseGenerator-4', 'SigNoiseGenerator-5', 'SigNoiseGenerator-6', 'SigNoiseGenerator-7', 'SigNoiseGenerator-8', 'SigNoiseGenerator-9', 'SigNoiseGenerator-10', 'FeatureWeightExpGenerator',
                      'RTREESAMPLEHP-23', 'RTREESAMPLEHP-14', 'RTREESAMPLEHP-A', 'RTREESAMPLE-UB', 'RTREESAMPLE-NB', 'RTREESAMPLE-DB', 'RTREESAMPLE-UU', 'RTREESAMPLE-UN', 'RTREESAMPLE-UD', 'RTREESAMPLE-NU', 'RTREESAMPLE-NN', 'RTREESAMPLE-ND', 'RTREESAMPLE-DU', 'RTREESAMPLE-DN', 'RTREESAMPLE-DD']
    datasets = set()
    for ds in (args.datasets if type(args.datasets) is list else [args.datasets]):
        if ds == 'all_exp':
            for x in [*real_drift_datasets]:
                datasets.add((x, 'Real'))
            for x in [*synthetic_MI_datasets, *synthetic_perf_only_datasets]:
                datasets.add((x, 'Synthetic'))
        elif ds == 'real':
            for x in [*real_drift_datasets]:
                datasets.add((x, 'Real'))
        elif ds == 'synthetic':
            for x in [*synthetic_MI_datasets, *synthetic_perf_only_datasets, *synthetic_dist]:
                datasets.add((x, 'Synthetic'))
        elif ds in real_drift_datasets:
            datasets.add((ds, 'Real'))
        elif ds in real_unknown_datasets:
            datasets.add((ds, 'Real'))
        elif ds in synthetic_MI_datasets:
            datasets.add((ds, 'Synthetic'))
        elif ds in synthetic_perf_only_datasets:
            datasets.add((ds, 'Synthetic'))
        elif ds in synthetic_unused:
            datasets.add((ds, 'Synthetic'))
        elif ds in synthetic_dist:
            datasets.add((ds, 'Synthetic'))
        else:
            raise ValueError("Dataset not found")

    seeds = []
    num_seeds = 0
    base_seeds = []
    if args.seedaction == 'reuse':
        num_seeds = args.seeds if type(
            args.seeds) is not list else args.seeds[0]
        base_seeds = np.random.randint(0, 9999, size=num_seeds)
    if args.seedaction == 'new':
        num_seeds = args.seeds if type(
            args.seeds) is not list else args.seeds[0]
        seeds = np.random.randint(0, 9999, size=num_seeds)
    if args.seedaction == 'list':
        seeds = args.seeds if type(args.seeds) is list else [args.seeds]
        num_seeds = len(seeds)

    raw_data_path = pathlib.Path(args.datalocation).resolve()
    if not raw_data_path.exists():
        raise ValueError(f"Data location {raw_data_path} does not exist")
    for ds, ds_type in datasets:
        data_file_location = raw_data_path / ds_type / ds
        if not data_file_location.exists():
            if ds_type == 'Synthetic':
                data_file_location.mkdir(parents=True, exist_ok=True)
            else:
                raise ValueError(
                    f"Data file {data_file_location} does not exist")
    base_output_path = pathlib.Path(args.outputlocation).resolve()
    if not base_output_path.exists():
        base_output_path.mkdir(parents=True)
    log_path = pathlib.Path(args.loglocation).resolve()
    if not log_path.exists():
        log_path.mkdir(parents=True)
    given_options = None
    if args.optionslocation is not None:
        options_path = pathlib.Path(args.optionslocation).resolve()
        if options_path.exists():
            with options_path.open() as f:
                given_options = json.load(f)
    

    desc_path = base_output_path / args.experimentname / "desc.txt"
    desc_ver = 1
    while desc_path.exists():
        desc_path = base_output_path / args.experimentname / f"desc_{desc_ver}.txt"
        desc_ver += 1

    desc_path.parent.mkdir(parents=True, exist_ok=True)
    with open(desc_path, 'w+') as f:
        f.write(args.desc)

    log_name = f"{args.experimentname}-{time.time()}"
    if args.logging:
        logging.basicConfig(handlers=[RotatingFileHandler(
            f"{log_path}/{log_name}.log", maxBytes=500000000, backupCount=100)], level=logging.INFO)
    with (log_path / f"e{log_name}.txt").open('w+') as f:
        json.dump(args.__dict__, f)

    package_status = get_package_status(force=args.forcegitcheck)

    if given_options:
        option_set = given_options
    else:
        option_set = []
        dataset_options = list(datasets)

        classifier_options = args.classifier if type(args.classifier) is list else [args.classifier]
        similarity_options = args.sim if type(args.sim) is list else [args.sim]
        MIcalc = args.MIcalc if type(args.MIcalc) is list else [args.MIcalc]
        fs_options = args.fsmethod if type(
            args.fsmethod) is list else [args.fsmethod]
        fingerprint_options = args.fingerprintmethod if type(
            args.fingerprintmethod) is list else [args.fingerprintmethod]
        fingerprint_bins_options = args.fingerprintbins if type(
            args.fingerprintbins) is list else [args.fingerprintbins]
        window_size_options = args.window_size if type(
            args.window_size) is list else [args.window_size]
        sensitivity_options = args.sensitivity if type(
            args.sensitivity) is list else [args.sensitivity]
        min_window_ratio_options = args.min_window_ratio if type(
            args.min_window_ratio) is list else [args.min_window_ratio]
        fingerprint_grace_period_options = args.fingerprint_grace_period if type(
            args.fingerprint_grace_period) is list else [args.fingerprint_grace_period]
        state_grace_period_window_multiplier_options = args.state_grace_period_window_multiplier if type(
            args.state_grace_period_window_multiplier) is list else [args.state_grace_period_window_multiplier]
        bypass_grace_period_threshold_options = args.bypass_grace_period_threshold if type(
            args.bypass_grace_period_threshold) is list else [args.bypass_grace_period_threshold]
        state_estimator_risk_options = args.state_estimator_risk if type(
            args.state_estimator_risk) is list else [args.state_estimator_risk]
        state_estimator_swap_risk_options = args.state_estimator_swap_risk if type(
            args.state_estimator_swap_risk) is list else [args.state_estimator_swap_risk]
        minimum_concept_likelihood_options = args.minimum_concept_likelihood if type(
            args.minimum_concept_likelihood) is list else [args.minimum_concept_likelihood]
        min_drift_likelihood_threshold_options = args.min_drift_likelihood_threshold if type(
            args.min_drift_likelihood_threshold) is list else [args.min_drift_likelihood_threshold]
        min_estimated_posterior_threshold_options = args.min_estimated_posterior_threshold if type(
            args.min_estimated_posterior_threshold) is list else [args.min_estimated_posterior_threshold]
        sim_gap_options = args.sim_gap if type(
            args.sim_gap) is list else [args.sim_gap]
        fp_gap_options = args.fp_gap if type(
            args.fp_gap) is list else [args.fp_gap]
        na_fp_gap_options = args.na_fp_gap if type(
            args.na_fp_gap) is list else [args.na_fp_gap]
        ob_gap_options = args.ob_gap if type(
            args.ob_gap) is list else [args.ob_gap]
        sim_stdevs_options = args.sim_stdevs if type(
            args.sim_stdevs) is list else [args.sim_stdevs]
        min_sim_stdev_options = args.min_sim_stdev if type(
            args.min_sim_stdev) is list else [args.min_sim_stdev]
        max_sim_stdev_options = args.max_sim_stdev if type(
            args.max_sim_stdev) is list else [args.max_sim_stdev]
        buffer_ratio_options = args.buffer_ratio if type(
            args.buffer_ratio) is list else [args.buffer_ratio]
        merge_threshold_options = args.merge_threshold if type(
            args.merge_threshold) is list else [args.merge_threshold]
        background_state_prior_multiplier_options = args.background_state_prior_multiplier if type(
            args.background_state_prior_multiplier) is list else [args.background_state_prior_multiplier]
        zero_prob_minimum_options = args.zero_prob_minimum if type(
            args.zero_prob_minimum) is list else [args.zero_prob_minimum]
        multihop_penalty_options = args.multihop_penalty if type(
            args.multihop_penalty) is list else [args.multihop_penalty]
        prev_state_prior_options = args.prev_state_prior if type(
            args.prev_state_prior) is list else [args.prev_state_prior]
        drift_width_options = args.drift_width if type(
            args.drift_width) is list else [args.drift_width]
        noise_options = args.noise if type(
            args.noise) is list else [args.noise]
        conceptdifficulty_options = args.conceptdifficulty if type(
            args.conceptdifficulty) is list else [args.conceptdifficulty]
        
        if args.pcpu == -1:
            args.pcpu = args.cpu

        if not args.experimentopts:
            classifier_options = list(itertools.product(classifier_options,
                                                        similarity_options,
                                                        MIcalc,
                                                        fs_options,
                                                        fingerprint_options,
                                                        fingerprint_bins_options,
                                                        window_size_options,
                                                        sensitivity_options,
                                                        min_window_ratio_options,
                                                        fingerprint_grace_period_options,
                                                        state_grace_period_window_multiplier_options,
                                                        bypass_grace_period_threshold_options,
                                                        state_estimator_risk_options,
                                                        state_estimator_swap_risk_options,
                                                        minimum_concept_likelihood_options,
                                                        min_drift_likelihood_threshold_options,
                                                        min_estimated_posterior_threshold_options,
                                                        sim_gap_options,
                                                        fp_gap_options,
                                                        na_fp_gap_options,
                                                        ob_gap_options,
                                                        sim_stdevs_options,
                                                        min_sim_stdev_options,
                                                        max_sim_stdev_options,
                                                        buffer_ratio_options,
                                                        merge_threshold_options,
                                                        background_state_prior_multiplier_options,
                                                        zero_prob_minimum_options,
                                                        multihop_penalty_options,
                                                        prev_state_prior_options,
                                                        drift_width_options,
                                                        noise_options,
                                                        conceptdifficulty_options
                                                        ))
        else:
            classifier_options = list(itertools.product(classifier_options,
                                                        fingerprint_bins_options,
                                                        window_size_options,
                                                        sensitivity_options,
                                                        min_window_ratio_options,
                                                        fingerprint_grace_period_options,
                                                        state_grace_period_window_multiplier_options,
                                                        bypass_grace_period_threshold_options,
                                                        state_estimator_risk_options,
                                                        state_estimator_swap_risk_options,
                                                        minimum_concept_likelihood_options,
                                                        min_drift_likelihood_threshold_options,
                                                        min_estimated_posterior_threshold_options,
                                                        sim_gap_options,
                                                        fp_gap_options,
                                                        na_fp_gap_options,
                                                        ob_gap_options,
                                                        sim_stdevs_options,
                                                        min_sim_stdev_options,
                                                        max_sim_stdev_options,
                                                        buffer_ratio_options,
                                                        merge_threshold_options,
                                                        background_state_prior_multiplier_options,
                                                        zero_prob_minimum_options,
                                                        multihop_penalty_options,
                                                        prev_state_prior_options,
                                                        drift_width_options,
                                                        noise_options,
                                                        conceptdifficulty_options
                                                        ))

        for ds_name, ds_type in dataset_options:
            # If we are reusing, find out what seeds are already in use
            # otherwise, make a new one.
            if args.seedaction == 'reuse':
                seed_location = raw_data_path / ds_type / ds_name / "seeds"
                if not seed_location.exists():
                    seeds = []
                else:
                    seeds = [int(str(f.stem))
                             for f in seed_location.iterdir() if f.is_dir()]
                for i in range(num_seeds):
                    if i < len(seeds):
                        continue
                    seeds.append(base_seeds[i])
                if len(seeds) < 1:
                    raise ValueError(
                        f"Reuse seeds selected by no seeds exist for data set {ds_name}")

            for seed in seeds:
                if not args.experimentopts:
                    for (classifier_option, sim_opt, MIcalc, fs_opt, fingerprint_opt, fingerprint_bins_opt, ws_opt,                                                         sensitivity_options,
                                                        min_window_ratio_options,
                                                        fingerprint_grace_period_options,
                                                        state_grace_period_window_multiplier_options,
                                                        bypass_grace_period_threshold_options,
                                                        state_estimator_risk_options,
                                                        state_estimator_swap_risk_options,
                                                        minimum_concept_likelihood_options,
                                                        min_drift_likelihood_threshold_options,
                                                        min_estimated_posterior_threshold_options, sim_gap_opt, fp_gap_opt, na_fp_gap_opt, ob_gap_opt, sim_std_opt, min_sim_opt, max_sim_opt, br_opt, merge_threshold_options, background_state_prior_multiplier_options, zero_prob_minimum_options, multihop_penalty_options, prev_state_prior_options, drift_width_opt, noise_opt, conceptdifficulty_opt) in classifier_options:
                        option = {
                            'classifier': classifier_option,
                            'base_output_path': base_output_path,
                            'raw_data_path': raw_data_path,
                            'data_name': ds_name,
                            'data_type': ds_type,
                            'max_rows': args.maxrows,
                            'seed': seed,
                            'seed_action': args.seedaction,
                            'package_status': package_status,
                            'log_name': log_name,
                            'pyinstrument': args.pyinstrument,
                            'FICSUM': args.ficsum,
                            'minimal_output': args.minimal_output,
                            'MAP_selection': args.MAP_selection,
                            'setaffinity': args.setaffinity,
                            'pcpu': args.pcpu,
                            'cpu': args.cpu,
                            'experiment_name': args.experimentname,
                            'repeats': args.repeats,
                            'concept_max': args.concept_max,
                            'concept_length': args.concept_length,
                            'repeatproportion': args.repeatproportion,
                            'TMdropoff': args.TMdropoff,
                            'TMforward': args.TMforward,
                            'TMnoise': args.TMnoise,
                            'drift_width': drift_width_opt,
                            'noise': noise_opt,
                            'conceptdifficulty': conceptdifficulty_opt,
                            'framework': "system",
                            'isources': args.isources,
                            'ifeatures': args.ifeatures,
                            'optdetect': args.optdetect,
                            'optselect': args.optselect,
                            'opthalf': args.opthalf,
                            'opthalflock': args.opthalflock,
                            'save_feature_weights': args.save_feature_weights,
                            'shuffleconcepts': args.shuffleconcepts,
                            'similarity_option': sim_opt,
                            'MI_calc': MIcalc,
                            'window_size': ws_opt,
                            'sensitivity': sensitivity_options,
                            'min_window_ratio': min_window_ratio_options,
                            'fingerprint_grace_period': fingerprint_grace_period_options,
                            'state_grace_period_window_multiplier': state_grace_period_window_multiplier_options,
                            'bypass_grace_period_threshold': bypass_grace_period_threshold_options,
                            'state_estimator_risk': state_estimator_risk_options,
                            'state_estimator_swap_risk': state_estimator_swap_risk_options,
                            'minimum_concept_likelihood': minimum_concept_likelihood_options,
                            'min_drift_likelihood_threshold': min_drift_likelihood_threshold_options,
                            'min_estimated_posterior_threshold': min_estimated_posterior_threshold_options,
                            'similarity_gap': sim_gap_opt,
                            'fp_gap': fp_gap_opt,
                            'nonactive_fp_gap': na_fp_gap_opt,
                            'observation_gap': ob_gap_opt,
                            'take_observations': ob_gap_opt != 0,
                            'similarity_stdev_thresh': sim_std_opt,
                            'similarity_stdev_min': min_sim_opt,
                            'similarity_stdev_max': max_sim_opt,
                            'buffer_ratio': br_opt,
                            "merge_threshold": merge_threshold_options,
                            "background_state_prior_multiplier": background_state_prior_multiplier_options,
                            "zero_prob_minimum": zero_prob_minimum_options,
                            "multihop_penalty": multihop_penalty_options,
                            "prev_state_prior": prev_state_prior_options,
                            "correlation_merge": not args.no_merge,
                            'fs_method': fs_opt,
                            'fingerprint_method': fingerprint_opt,
                            'fingerprint_bins': fingerprint_bins_opt,
                        }
                        stream, stream_concepts, length, classes = make_stream(
                            option)
                        option_set.append(option)
                else:
                    for (classifier_option, fingerprint_bins_opt, ws_opt, sensitivity_options,
                            min_window_ratio_options,
                            fingerprint_grace_period_options,
                            state_grace_period_window_multiplier_options,
                            bypass_grace_period_threshold_options,
                            state_estimator_risk_options,
                            state_estimator_swap_risk_options,
                            minimum_concept_likelihood_options,
                            min_drift_likelihood_threshold_options,
                            min_estimated_posterior_threshold_options, sim_gap_opt, fp_gap_opt, na_fp_gap_opt, ob_gap_opt, sim_std_opt, min_sim_opt, max_sim_opt, br_opt, merge_threshold_options, background_state_prior_multiplier_options, zero_prob_minimum_options, multihop_penalty_options, prev_state_prior_options, drift_width_opt, noise_opt, conceptdifficulty_opt) in classifier_options:
                        for exp_fingerprint, exp_fsmethod, sim_opt in [('cache', 'fisher_overall', 'metainfo'), ('cache', 'fisher', 'metainfo'), ('cache', 'CacheMIHy', 'metainfo'), ('cachehistogram', 'Cachehistogram_MI', 'metainfo'), ('cachesketch', 'sketch_MI', 'metainfo'), ('cachesketch', 'sketch_covMI', 'metainfo'), ('cachesketch', 'sketch_MI', 'sketch'), ('cachesketch', 'sketch_covMI', 'sketch')]:
                            # Only need to run default and fisher on one bin size, as it doesn't do anything
                            if exp_fsmethod in ['default', 'fisher', 'fisher_overall'] and fingerprint_bins_opt != fingerprint_bins_options[0]:
                                continue
                            option = {
                                'classifier_options': classifier_option,
                                'base_output_path': base_output_path,
                                'raw_data_path': raw_data_path,
                                'data_name': ds_name,
                                'data_type': ds_type,
                                'max_rows': args.maxrows,
                                'seed': seed,
                                'seed_action': args.seedaction,
                                'package_status': package_status,
                                'log_name': log_name,
                                'pyinstrument': args.pyinstrument,
                                'FICSUM': args.ficsum,
                                'minimal_output': args.minimal_output,
                                'MAP_selection': args.MAP_selection,
                                'setaffinity': args.setaffinity,
                                'pcpu': args.pcpu,
                                'cpu': args.cpu,
                                'experiment_name': args.experimentname,
                                'repeats': args.repeats,
                                'concept_max': args.concept_max,
                                'concept_length': args.concept_length,
                                'repeatproportion': args.repeatproportion,
                                'TMdropoff': args.TMdropoff,
                                'TMforward': args.TMforward,
                                'TMnoise': args.TMnoise,
                                'drift_width': drift_width_opt,
                                'noise': noise_opt,
                                'conceptdifficulty': conceptdifficulty_opt,
                                'framework': "system",
                                'isources': args.isources,
                                'ifeatures': args.ifeatures,
                                'optdetect': args.optdetect,
                                'optselect': args.optselect,
                                'opthalf': args.opthalf,
                                'opthalflock': args.opthalflock,
                                'save_feature_weights': args.save_feature_weights,
                                'shuffleconcepts': args.shuffleconcepts,
                                'similarity_option': sim_opt,
                                'MI_calc': MIcalc[0],
                                'window_size': ws_opt,
                                'sensitivity': sensitivity_options,
                                'min_window_ratio': min_window_ratio_options,
                                'fingerprint_grace_period': fingerprint_grace_period_options,
                                'state_grace_period_window_multiplier': state_grace_period_window_multiplier_options,
                                'bypass_grace_period_threshold': bypass_grace_period_threshold_options,
                                'state_estimator_risk': state_estimator_risk_options,
                                'state_estimator_swap_risk': state_estimator_swap_risk_options,
                                'minimum_concept_likelihood': minimum_concept_likelihood_options,
                                'min_drift_likelihood_threshold': min_drift_likelihood_threshold_options,
                                'min_estimated_posterior_threshold': min_estimated_posterior_threshold_options,
                                'similarity_gap': sim_gap_opt,
                                'fp_gap': fp_gap_opt,
                                'nonactive_fp_gap': na_fp_gap_opt,
                                'observation_gap': ob_gap_opt,
                                'take_observations': ob_gap_opt != 0,
                                'similarity_stdev_thresh': sim_std_opt,
                                'similarity_stdev_min': min_sim_opt,
                                'similarity_stdev_max': max_sim_opt,
                                'buffer_ratio': br_opt,
                                "merge_threshold": merge_threshold_options,
                                "background_state_prior_multiplier": background_state_prior_multiplier_options,
                                "zero_prob_minimum": zero_prob_minimum_options,
                                "multihop_penalty": multihop_penalty_options,
                                "prev_state_prior": prev_state_prior_options,
                                "correlation_merge": not args.no_merge,
                                'fs_method': exp_fsmethod,
                                'fingerprint_method': exp_fingerprint,
                                'fingerprint_bins': fingerprint_bins_opt,
                            }
                            stream, stream_concepts, length, classes = make_stream(
                                option)
                            option_set.append(option)
    with (log_path / f"e{log_name}_option_set.txt").open('w+') as f:
        json.dump(option_set, f, cls=NpEncoder)
    if args.single:
        run_files = []
        for o in tqdm.tqdm(option_set, total=len(option_set), position=1, desc="Experiment", leave=True):
            # print(o)
            run_files.append(process_option((o, None, None)))
    else:
        manager = mp.Manager()

        lock = manager.RLock() if args.lockpbar else None
        tqdm.tqdm.set_lock(lock)
        print(lock)
        pool = mp.Pool(processes=args.cpu, maxtasksperchild=1)
        run_files = []
        queue = manager.Queue()
        for c_index in range(args.cpu):
            queue.put(c_index)
        overall_prog_bar = tqdm.tqdm(total=len(
            option_set), position=1, desc="Experiment", leave=True, miniters=1, lock_args=(True, 0.01), ascii=True)
        for result in pool.imap_unordered(func=process_option, iterable=((o, lock, queue) for o in option_set), chunksize=1):
            run_files.append(result)
            if lock or True:
                l = lock.acquire(blocking=False)
                if l:
                    overall_prog_bar.update(n=1)
                    overall_prog_bar.refresh(lock_args=(False))
                    lock.release()
            else:
                overall_prog_bar.update(n=1)
        # print(run_files)
        pool.close()
    with (log_path / f"e{log_name}_run_files.txt").open('w+') as f:
        json.dump(run_files, f, cls=NpEncoder)
