import itertools
import logging
import math
import warnings
from collections import deque
from copy import copy, deepcopy

import numpy as np
import scipy.stats
from runstats import Regression
from scipy.sparse import base
from scipy.spatial.distance import correlation, cosine, euclidean, jaccard
from scipy.stats.stats import pearsonr
from SELeCT.Classifier.feature_selection.fisher_score import fisher_score
from SELeCT.Classifier.feature_selection.mutual_information import *
from SELeCT.Classifier.feature_selection.online_feature_selection import (
    feature_selection_cached_MI, feature_selection_fisher,
    feature_selection_fisher_overall, feature_selection_histogram_covredMI,
    feature_selection_histogramMI, feature_selection_None,
    feature_selection_original, mi_cov_from_fingerprint_sketch,
    mi_from_cached_fingerprint_bins, mi_from_fingerprint_histogram_cache,
    mi_from_fingerprint_sketch)
from SELeCT.Classifier.fingerprint import (FingerprintBinningCache,
                                           FingerprintCache,
                                           FingerprintSketchCache,
                                           MinimalFingerprint)
from SELeCT.Classifier.metafeature_extraction import (
    get_concept_stats, get_concept_stats_from_base, observations_to_timeseries,
    update_timeseries, window_to_timeseries)
from SELeCT.Classifier.normalizer import Normalizer
from SELeCT.Classifier.rolling_stats import RollingTimeseries
from skmultiflow.drift_detection.adwin import ADWIN
from skmultiflow.utils import check_random_state, get_dimensions

warnings.filterwarnings('ignore')


def get_dimension_weights(fingerprints, state_active_non_active_fingerprints, normalizer, state_id=None, feature_selection_method="default"):
    """ Use feature selection methods to weight
    each meta-information feature.
    """
    feature_selection_func = None

    if feature_selection_method in ["uniform", "None"]:
        feature_selection_func = feature_selection_None
    if feature_selection_method in ["fisher_overall", "default"]:
        feature_selection_func = feature_selection_fisher_overall
    if feature_selection_method == "fisher":
        feature_selection_func = feature_selection_fisher
    if feature_selection_method in ["original"]:
        feature_selection_func = feature_selection_original

    if feature_selection_method in ["gaussian_approx", "CacheMIHy"]:
        def feature_selection_func(f, naf, n, sid): return feature_selection_cached_MI(
            f, mi_from_cached_fingerprint_bins, naf, n, sid)
    if feature_selection_method in ["sketch", "sketch_MI"]:
        def feature_selection_func(f, naf, n, sid): return feature_selection_histogramMI(
            f, mi_from_fingerprint_sketch, naf, n, sid)
    if feature_selection_method in ["sketch_covariance", "sketch_covMI"]:
        def feature_selection_func(f, naf, n, sid): return feature_selection_histogramMI(
            f, mi_cov_from_fingerprint_sketch, naf, n, sid)
    if feature_selection_method in ["sketch_covariance_weighted", "sketch_covMI_weighted"]:
        def feature_selection_func(f, naf, n, sid): return feature_selection_histogramMI(
            f, mi_cov_from_fingerprint_sketch, naf, n, sid, weighted=True)
    if feature_selection_method in ["sketch_covariance_redundancy", "sketch_covredMI"]:
        def feature_selection_func(f, naf, n, sid): return feature_selection_histogram_covredMI(
            f, mi_from_fingerprint_sketch, naf, n, sid)
    if feature_selection_method in ["histogram", "Cachehistogram_MI"]:
        def feature_selection_func(f, naf, n, sid): return feature_selection_histogramMI(
            f, mi_from_fingerprint_histogram_cache, naf, n, sid)
    if feature_selection_func is None:
        raise ValueError(
            f"no falid feature selection method {feature_selection_method}")
    return feature_selection_func(fingerprints, state_active_non_active_fingerprints, normalizer, state_id)


def get_cosine_distance(A, B, weighted, weights):
    """ Get cosine distance between vectors A and B.
    Weight vectors first if weighted is set.
    Note that if weighted, it is really the mahalanobis cosine distance,
    as we weight each dimension by 1 / stdev.
    """
    try:
        if not weighted:
            c = cosine(A, B)
        else:
            c = cosine(A, B, w=weights)

    except:
        c = np.nan
    if np.isnan(c):
        c = 0 if ((not np.any(A)) and (not np.any(B))) else 1
    return c

def get_euclidean_distance(A, B, weighted, weights):
    """ Get euclidean distance between vectors A and B.
    Weight vectors first if weighted is set.
    Note that if weighted, it is really the mahalanobis distance,
    as we weight each dimension by 1 / stdev.
    """
    try:
        if not weighted:
            c = euclidean(A, B)
        else:
            c = euclidean(A, B, w=weights)

    except:
        c = np.nan
    if np.isnan(c):
        c = 0 if ((not np.any(A)) and (not np.any(B))) else 1
    return c

def get_jaccard_distance(A, B, weighted, weights):
    """ Get jaccard distance between vectors A and B.
    Weight vectors first if weighted is set.
    """
    try:
        if not weighted or True:
            c = jaccard(A, B)
        else:
            c = jaccard(A, B, w=weights)

    except:
        c = np.nan
    if np.isnan(c):
        c = 0 if ((not np.any(A)) and (not np.any(B))) else 1
    return c

def get_jmahal_distance(A, B, weighted, weights):
    """ Get jaccard distance between vectors A and B.
    Weight vectors first if weighted is set.
    """
    try:
        if not weighted or True:
            j = jaccard(A, B)
        else:
            j = jaccard(A, B, w=weights)
        if not weighted:
            m = cosine(A, B)
        else:
            m = cosine(A, B, w=weights)

        c = j + math.exp(m)

    except:
        c = np.nan
    if np.isnan(c):
        c = 0 if ((not np.any(A)) and (not np.any(B))) else 1
    return c



def get_pearson_distance(A, B, weighted, weights):
    """ Get pearson distance between vectors A and B.
    Weight vectors first if weighted is set.
    """
    try:
        if not weighted:
            p = pearsonr(A, B)
        else:
            p = pearsonr(A*weights, B*weights)
    except:
        p = [np.nan]
    if np.isnan(p[0]):
        A_is_constant = np.all(A == A[0])
        B_is_constant = np.all(B == B[0])
        p = [0] if (A_is_constant and B_is_constant) else [1]
    return 1 - p[0]


def get_histogram_probability(A, fingerprint, weighted, weights):
    """ Get the probability of vector A being drawn from fingerprint.
    Assumes independance between features, so multiplies probabilities.
    Used as a distance measure.
    """
    total_probability = 0
    for s in fingerprint.normalizer.sources:
        for f in fingerprint.normalizer.features:
            vector_index = fingerprint.normalizer.get_ignore_index(s, f)
            if vector_index is None:
                continue
            histogram = fingerprint.fingerprint[s][f]["Histogram"]
            histogram_total = fingerprint.fingerprint[s][f]["seen"]
            bin_index = fingerprint.get_bin(
                value=A[vector_index], source=s, feature=f)
            bin_count = histogram[bin_index]
            probability = bin_count / histogram_total
            likelihood = probability * weights[vector_index]
            if likelihood > 0:
                total_probability += math.log(probability *
                                              weights[vector_index])
    return total_probability


def make_detector(warn=False, s=1e-5, update_gap=4):
    """ Create a drift detector. If warn, create a warning detector with higher sensitivity, to trigger prior to the main detector.
    """
    sensitivity = s * 2 if warn else s
    detector = ADWIN(delta=sensitivity)
    detector.set_clock(update_gap-1)
    return detector


def set_fingerprint_bins(make_fingerprint_func, num_bins):
    def make_fingerprint(*args, **kwargs):
        return make_fingerprint_func(*args, num_bins=num_bins, **kwargs)
    return make_fingerprint


def get_fingerprint_constructor(fingerprint_method, num_bins):
    """ Get the appropriate constructor for the passed options.
    """
    if fingerprint_method in ['descriptive', 'cache']:
        return set_fingerprint_bins(FingerprintCache, num_bins)
    if fingerprint_method in ['histogram', 'cachehistogram']:
        return set_fingerprint_bins(FingerprintBinningCache, num_bins)
    if fingerprint_method in ['sketch', 'cachesketch']:
        return set_fingerprint_bins(FingerprintSketchCache, num_bins)
    raise ValueError("Not a valid fingerprint method")


def get_fingerprint_type(fingerprint_method):
    """ Get the appropriate class type for the passed options.
    """
    if fingerprint_method in ['descriptive', 'cache']:
        return FingerprintCache
    if fingerprint_method in ['histogram', 'cachehistogram']:
        return FingerprintBinningCache
    if fingerprint_method in ['sketch', 'cachesketch']:
        return FingerprintSketchCache
    raise ValueError("Not a valid fingerprint method")


class Observation:
    """ Represents an observation held by a state. May be labelled by the state, or not.
    """
    def __init__(self, X, y, seen_at, ob_id):
        self.X = X
        self.y = y
        self.seen_at = seen_at
        self.ob_id = ob_id
        self.labelled = False
        self.p = None
        self.ev = None
        self.correctly_classified = None
        self.last_labeled_at = None
        self.fit_on = False
    
    def label(self, p, evolution, ts):
        self.p = p
        self.ev = evolution
        self.correctly_classified = self.y == self.p
        self.labelled = True
        self.last_labeled_at = ts
    
    def __repr__(self):
        return f"<{self.X}, {self.y}, {self.p}| {self.ev} @ {self.seen_at}>"
    
    def serialize(self):
        # return str(self.__dict__)
        return str(f"{self.p}{self.ob_id}{self.seen_at}{self.last_labeled_at}{self.ev}{self.fit_on}")

class ObservationTimeseries:
    """ Represents a window of observations as a set of behavior timeseries.
    """
    def __init__(self, window_size):
        self.window_size = window_size
        self.features = []
        self.labels = RollingTimeseries(window_size=self.window_size)
        self.predictions = RollingTimeseries(window_size=self.window_size)
        self.errors = RollingTimeseries(window_size=self.window_size)
        self.error_distances = RollingTimeseries(window_size=self.window_size)
        self.n_features = None
        self.last_distance = None
        self.initialized = False
        self.seen = 0
    
    def initialize(self, observation: Observation):
        self.n_features = len(observation.X)
        for i in range(self.n_features):
            self.features.append(RollingTimeseries(window_size=self.window_size))
        self.initialized = True

    def add_observation(self, observation: Observation):
        self.seen += 1
        if observation is None:
            return
        if not self.initialized:
            self.initialize(observation)
        
        for fi, f in enumerate(observation.X):
            self.features[fi].update(f)
        self.labels.update(observation.y)
        self.predictions.update(observation.p)
        self.errors.update(observation.correctly_classified)
        if not observation.correctly_classified:
            if self.last_distance is None:
                self.last_distance = self.seen
            else:
                distance = self.seen - self.last_distance
                self.error_distances.update(distance)
                self.last_distance = self.seen

        if self.last_distance is not None:
            # We don't want to retain distances above the window size.
            distance = min(self.seen - self.last_distance, self.window_size - 1)
            # We want to keep the timeseries estimators bounded within data from the
            # last window_size elements.
            # Since error_distances does not store elements directly, but rather distances,
            # we need to treat it differently.
            # E.g., if window size is 50, and error_distances stores 50 elements each with value
            # 10, this actually is data on the last 500 observations!
            # This is bad, because it means our sliding window may not only include recent data.
            # We can instead count the number of elements observed as the current sum + distance to last error.
            # Limiting this limits the number of observations we can look back, which is what we want.
            while (self.error_distances._sum + distance) >= self.window_size:
                if self.error_distances._nobs > 1:
                    self.error_distances._remove_old()
                else:
                    self.error_distances = RollingTimeseries(window_size=self.window_size)
                
    
    def get_concept_stats(self, model, can_use_feature_base, feature_base=None, feature_base_flat=None, stored_shap=None, ignore_sources=None, ignore_features=None, normalizer=None):
        """ Wrapper for calculating the stats of a set
        of timeseries generated by a given model.
        """
        if ignore_sources is None:
            ignore_sources = []
        if ignore_features is None:
            ignore_features = []
        concept_stats = {}

        if not can_use_feature_base:
            feature_base = None
            feature_base_flat = None

        normalizer_initialized = False
        flat_ignore_vec = None
        if normalizer.ignore_indexes is not None:
            flat_ignore_vec = np.empty(normalizer.ignore_num_signals)
            normalizer_initialized = True

        if 'FI' not in ignore_features or stored_shap is None:
            shap_model = stored_shap
            X = np.column_stack([f.get_np_timeseries() for f in self.features])
            shaps = shap_model.shap_values(
                X, check_additivity=False, approximate=True)
            # If there is only 1 label, shaps just returns the matrix, otherwise it returns
            # a list of matricies. This converts the single case into a list.
            if not isinstance(shaps, list):
                shaps = [shaps]
            mean_shaps = np.sum(np.abs(shaps[0]), axis=0)
            SHAP_vals = [abs(x) for x in mean_shaps]
        else:
            SHAP_vals = [0 for x in range(len(self.features))]

        if 'features' not in ignore_sources:
            for f1, f in enumerate(self.features):
                feature_name = f"f{f1}"
                if feature_name not in ignore_sources:
                    if not feature_base:
                        stats = f.get_stats(SHAP_vals[f1], ignore_features=ignore_features)
                        if normalizer_initialized:
                            for stats_feature, stat_value in stats.items():
                                ignore_index = normalizer.ignore_indexes[feature_name][stats_feature]
                                flat_ignore_vec[ignore_index] = stat_value
                    else:
                        # Need to copy so we don't overwrite FI
                        stats = copy(feature_base[feature_name])
                        if 'FI' not in ignore_features:
                            stats["FI"] = SHAP_vals[f1]
                        if normalizer_initialized:
                            source_start, source_end = normalizer.ignore_source_ranges[feature_name]
                            flat_ignore_vec[source_start:source_end] = feature_base_flat[source_start:source_end]
                            if 'FI' not in ignore_features:
                                flat_ignore_vec[normalizer.ignore_indexes[feature_name]["FI"]] = SHAP_vals[f1]

                    concept_stats[feature_name] = stats

        if 'labels' not in ignore_sources:
            if not feature_base:
                stats = self.labels.get_stats(ignore_features=ignore_features)
                if normalizer_initialized:
                    for stats_feature, stat_value in stats.items():
                        ignore_index = normalizer.ignore_indexes['labels'][stats_feature]
                        flat_ignore_vec[ignore_index] = stat_value
            else:
                stats = feature_base["labels"]
                if normalizer_initialized:
                    source_start, source_end = normalizer.ignore_source_ranges['labels']
                    flat_ignore_vec[source_start:source_end] = feature_base_flat[source_start:source_end]
            concept_stats["labels"] = stats
        if 'predictions' not in ignore_sources:
            stats = self.predictions.get_stats(ignore_features=ignore_features)
            concept_stats["predictions"] = stats
            if normalizer_initialized:
                for stats_feature, stat_value in stats.items():
                    ignore_index = normalizer.ignore_indexes['predictions'][stats_feature]
                    flat_ignore_vec[ignore_index] = stat_value
        if 'errors' not in ignore_sources:
            stats = self.errors.get_stats(ignore_features=ignore_features)
            concept_stats["errors"] = stats
            if normalizer_initialized:
                for stats_feature, stat_value in stats.items():
                    ignore_index = normalizer.ignore_indexes['errors'][stats_feature]
                    flat_ignore_vec[ignore_index] = stat_value
        if 'error_distances' not in ignore_sources:
            stats = self.error_distances.get_stats(ignore_features=ignore_features)
            concept_stats["error_distances"] = stats
            if normalizer_initialized:
                for stats_feature, stat_value in stats.items():
                    ignore_index = normalizer.ignore_indexes['error_distances'][stats_feature]
                    flat_ignore_vec[ignore_index] = stat_value
        if not normalizer_initialized:
            normalizer.init_signals(concept_stats)
            return self.get_concept_stats(model, can_use_feature_base, feature_base=feature_base, feature_base_flat=feature_base_flat, stored_shap=stored_shap, ignore_sources=ignore_sources, ignore_features=ignore_features, normalizer=normalizer)
        return concept_stats, flat_ignore_vec



class ConceptState:
    """ Represents a data stream concept.
    Maintains current descriptive information, including a classifier, fingerprint, evolution state and recent performance in stationary conditions.
    """

    def __init__(self, id, learner, fingerprint_update_gap, fingerprint_method, fingerprint_bins, window_size, min_window_size, dirty_data_length, normalizer, fingerprint_grace_period, estimator_risk_level, buffer_ratio, metainfo_method="metainfo"):
        self.id = id
        self.classifier = learner
        self.seen = 0
        self.recurrences = 0
        self.seen_since_transition = 0
        self.fingerprint = None
        self.fingerprint_cache = []
        self.non_active_fingerprints = {}
        self.non_active_fingerprint_buffer = deque()
        self.fingerprint_update_gap = fingerprint_update_gap
        self.last_fingerprint_update = self.fingerprint_update_gap * -1
        self.active_similarity_record = None

        self.similarity_vector_buffer_length = 50
        self.similarity_vector_buffer = deque()

        self.dirty_data_length = dirty_data_length
        self.fingerprint_dirty_data = False
        self.fingerprint_dirty_performance = False
        self.num_dirty_performance = None
        self.trigger_dirty_performance_end = False

        self.current_evolution = self.classifier.evolution
        self.fingerprint_method = fingerprint_method
        self.fingerprint_bins = fingerprint_bins
        self.fingerprint_constructor = get_fingerprint_constructor(
            self.fingerprint_method, self.fingerprint_bins)

        self.stable_likelihood = False

        # What risk level for culling old data do we want our estimator to use.
        # A higher risk level has a high chance of removing old data.
        # A lower risk level has a lower chance of removing old data.
        self.estimator_risk_level = estimator_risk_level
        self.adwin_likelihood_estimator = ADWIN(delta=self.estimator_risk_level)
        self.adwin_posterior_estimator = ADWIN(delta=self.estimator_risk_level)

        self.buffer_ratio = buffer_ratio

        self.normalizer = normalizer

        # Max size of the stable and head windows
        self.window_size = window_size

        # Min size for the windows in order to calculate statistics
        # The closer to window size this is the more accurate, but
        # the higher delay in collecting windows before data can be
        # calculated
        self.min_window_size = min_window_size
        self.recent_data = deque()
        self.head_window = deque(maxlen=self.window_size)

        self.head_timeseries = ObservationTimeseries(self.window_size)
        self.stable_timeseries = ObservationTimeseries(self.window_size)

        # Maximum amount of data to store. The buffer grows with state age,
        # So this is just a cap. Shouldn't have much impact, just to stop infinite memory growth.
        self.max_recent_data = 2000
        self.buffer_length = 0

        # We cache the metainfo for ranges of observations.
        # Since we may calculate it once for the head_window
        # and again later for the stable window.
        self.metainfo_cache = {}
        self.metainfo_cache_age = deque()

        self.metainfo_method = metainfo_method

        #How many fingerprints are needed to be seen to 
        # use a fingerprint.
        # Mainly for calculating weights, we don't 
        # want to calculate a weight with a fingerprint which
        # has just been reset as standard deviation will not be accurate
        self.fingerprint_grace_period = fingerprint_grace_period

        self.last_seen_ob_id = None
        self.stable_window_cache = None
        self.head_window_cache = None

        self.last_stable_observation = None
        self.new_stable_observation = False

        self.correlations = {}

    def get_buffer_len(self):
        state_age = np.mean((self.seen, self.seen_since_transition)) + 1
        buffer_len = math.floor(state_age * self.buffer_ratio)
        return buffer_len

    def add_correlations(self, other_states):
        for state in other_states:
            if state.id not in self.correlations:
                self.correlations[state.id] = [Regression(), 0]
            self.correlations[state.id][0].push(float(self.get_estimated_likelihood()), float(state.get_estimated_likelihood()))
            self.correlations[state.id][1] += 1

    def get_correlations(self):
        correlations = []
        for i,d in self.correlations.items():
            no_corr = len(d[0]) < 2 or d[0]._xstats.variance() == 0 or d[0]._ystats.variance() == 0
            corr = d[0].correlation() if not no_corr else 0
            correlations.append((i, corr, d[1]))
        return correlations

    def get_clean_fingerprint(self):
        """ Get the most recent fingerprint with proper statistics.
        This should usually be the current one, unless it has just been
        reset to handle an evolution
        """
        ret_fp = None
        for fp in [self.fingerprint, *self.fingerprint_cache[::-1]]:
            if fp and fp.seen_since_performance_reset > self.fingerprint_grace_period:
                ret_fp = fp
                break
        if not ret_fp:
            ret_fp = self.fingerprint
        return ret_fp

    def partial_fit(self, X, y, sample_weight, classes):
        self.classifier.partial_fit(
            np.asarray([X]),
            np.asarray([y]),
            sample_weight=np.asarray([sample_weight]),
            classes=classes
        )
        classifier_evolved = self.classifier.evolution > self.current_evolution

        # If we evolved, we reclassify items in our buffer
        # This is purely for training purposes, as we have
        # already emitted our prediction for these values.
        # These have not been used to fit yet, so this is not biased.
        # This captures the current behaviour of the model.
        # We evolve our state, or reset plasticity of performance
        # features so they can be molded to the new behaviour.
        if classifier_evolved:
            self.handle_evolution()
        return classifier_evolved
    
    def handle_evolution(self):
        """ Handle an evolution where the fit changed the model in a way which
        might significantly change behaviour.
        We need to account for this in terms
        of our fingerprint.
        """
        self.start_evolution_transition(self.dirty_data_length)

        # Force a refresh of stored observations so that predictions
        # and error behaviors match new behavior
        self.refresh_recent_data(hard=True)
    
    def observe_likelihood(self, likelihood):
        self.adwin_likelihood_estimator.add_element(likelihood)
        # need to call detected change to get the detector to cut old windows
        self.adwin_likelihood_estimator.detected_change()

    def observe_posterior(self, posterior):
        self.adwin_posterior_estimator.add_element(posterior)
        # need to call detected change to get the detector to cut old windows
        self.adwin_posterior_estimator.detected_change()
    
    def get_estimated_likelihood(self):
        """ Get an estimate of current likelihood from 
        the ADWIN estimator.
        Uses the Hoeffding bound to forget old observations which are
        significantly different
        """
        return self.adwin_likelihood_estimator.estimation

    def get_estimated_posterior(self):
        """ Get an estimate of current posterior probability from 
        the ADWIN estimator.
        Uses the Hoeffding bound to forget old observations which are
        significantly different
        """
        return self.adwin_posterior_estimator.estimation

    def get_sim_observation_probability(self, sim_value, similarity_calc, similarity_min_stdev, similarity_max_stdev):
        """ Return probability of observing the passed similarity, assuming similarity is distributed according to recently seen similarity values.
        """

        # First, recalculate similarity on a record of recently seen observations.
        # Recalculating accounts for changes in normalization etc which may change how similarity behaves.
        if self.active_similarity_record is not None:
            def similarity_calc_func(x, y, z): return similarity_calc(
                x, state=self, fingerprint_to_compare=y, flat_nonorm_current_metainfo=z)
            recent_similarity, normal_stdev = self.get_state_recent_similarity(
                similarity_calc_func)
        else:
            # State has no recent data, so we cannot compare to the normal distribution.
            return 0

        #logging.info(f"recent similarity: {recent_similarity}, recent stdev: {normal_stdev}, sim value: {sim_value}, evolution: {self.current_evolution}")
        recent_stdev = max(normal_stdev, similarity_min_stdev)
        recent_stdev = min(recent_stdev, similarity_max_stdev)

        # Returns probability of observing a sim_value at least as far from the mean.
        p_val = scipy.stats.norm.sf(
            np.abs(sim_value-recent_similarity), loc=0, scale=recent_stdev) * 2
        
        #logging.info(f"Likelihood: {p_val}")

        return p_val

    def update_recent_data(self, new_observation):
        """ Maintains a stable window (s) which is |b| observations behind current observations.
        The head window (h) holds the most recent observations from the stream.
        The stable window holds observations older than |b|.
        We set |b| to be a ratio of observations, up to some max.
        """
        # we set maxlen for the head_window, so don't need to manually handle size
        # of head_window
        self.head_window.append(new_observation)

        self.recent_data.append(new_observation)
        # We need to retain the buffer, as well as the stable window of window size
        # after the buffer.
        total_recent_data = min(self.buffer_length + self.window_size, self.max_recent_data)
        while len(self.recent_data) > total_recent_data:
            self.recent_data.popleft()
        
        self.head_timeseries.add_observation(new_observation)

        # We want to add a possible stable observation to the timeseries, but only if it is new
        most_recent_stable_observation = self.get_last_stable_observation()
        self.new_stable_observation = most_recent_stable_observation != self.last_stable_observation
        if self.new_stable_observation:
            self.stable_timeseries.add_observation(most_recent_stable_observation)
            self.last_stable_observation = most_recent_stable_observation
        
        if self.stable_timeseries.seen < self.get_stable_window_n():
            self.refresh_recent_data()
        if min(self.stable_timeseries.seen, self.stable_timeseries.window_size) > self.get_stable_window_n():
            self.refresh_recent_data()

    
    def get_stable_window_n(self):
        """ Get the number of items in the stable window
        """
        stable_window_n = min(max(len(self.recent_data) - self.buffer_length, 0), self.window_size)
        return stable_window_n
    
    def get_last_stable_observation(self):
        """ Return the most recent stable observation
        """
        stable_window = self.get_stable_window()
        return stable_window[-1] if len(stable_window) > 0 else None

    def has_valid_stable_window(self, min_window_size):
        """ Check if the state has a stable window above
        the passed minimum window size.
        """
        return self.get_stable_window_n() >= min_window_size

    def has_valid_head_window(self, min_window_size):
        """ Check if the state has a head window above
        the passed minimum window size.
        """
        return len(self.get_head_window()) >= min_window_size

    def get_stable_window(self):
        """ Get a window of at most self.window_size
        stable observations. An observation is stable if it 
        has passed a buffer, and the state has not transitioned.
        """
        # Cache the window based on the id of the last seen observation.
        # If we are asked for the window again without seeing a new observation, we can just
        # return from the cache
        if self.stable_window_cache and self.stable_window_cache[-1] == self.last_seen_ob_id:
            return self.stable_window_cache[0]
        else:
            stable_window_n = self.get_stable_window_n()
            stable_window = tuple(itertools.islice(self.recent_data, stable_window_n))
            self.stable_window_cache = (stable_window, self.last_seen_ob_id)
            return stable_window
    
    def get_buffer(self):
        """ Get the observations currently being buffered.
        """
        stable_window_n = self.get_stable_window_n()
        return tuple(itertools.islice(self.recent_data, stable_window_n, len(self.recent_data)))
    
    def get_head_window(self):
        """ Get the at most self.window_size most recently seen observations
        """
        if self.head_window_cache and self.head_window_cache[-1] == self.last_seen_ob_id:
            return self.head_window_cache[0]
        else:
            head_window = tuple(self.head_window)
            self.head_window_cache = (head_window, self.last_seen_ob_id)
            return head_window
    
    def refresh_recent_data(self, hard=False):
        """ Re-call predict on all recent data.
        We do this if the classifier changes, and we might have 
        different behaviour.
        We don't need to update head_window separately, as observations are shared.
        """
        for ob in self.recent_data:
            if not ob.labelled or hard:
                self.label_observation(ob)
        
        # We need to refresh the stable timeseries, since we have
        # relabeled the observations.
        # Note this the stable window is a view of recent data, so should
        # be updated as well.
        self.stable_timeseries = ObservationTimeseries(self.window_size)
        self.head_timeseries = ObservationTimeseries(self.window_size)
        for ob in self.get_stable_window():
            self.stable_timeseries.add_observation(ob)
        for ob in self.get_head_window():
            self.head_timeseries.add_observation(ob)

        self.stable_window_cache = None
        self.head_window_cache = None

    def label_observation(self, ob):
        """ Label an observation based on the current behavior of the classifier.
        """
        p = self.classifier.predict([ob.X])[0]
        ob.label(p, self.current_evolution, self.seen)
        return p

    def background_observe(self, X, y, ob_id, buffer_len, label=False):
        """ Observe an observation from the background. Add to buffers and windows, but
        to not increment any counters etc.
        Must set the desired buffer len to buffer the item.
        """
        self.buffer_length = min(buffer_len, len(self.recent_data))
        new_observation = Observation(X, y, self.seen, ob_id)
        p = None
        if label:
            p = self.label_observation(new_observation)
        
        self.last_seen_ob_id = ob_id
        self.update_recent_data(new_observation)
        return p

    def observe(self, X, y, ob_id, buffer_len, label=False, fit=False, sample_weight=None, classes=None):
        """ Update counters refering to observation count, and add observation to windows.
        Call fit to also fit the on the most recent stable observation.
        """
        self.seen += 1
        self.seen_since_transition += 1
        self.buffer_length = min(buffer_len, len(self.recent_data))
        new_observation = Observation(X, y, self.seen, ob_id)
        p = None
        if label:
            p = self.label_observation(new_observation)
        
        self.last_seen_ob_id = ob_id
        self.update_recent_data(new_observation)
        
        if fit:

            recent_stable_ob = self.get_last_stable_observation()
            if recent_stable_ob and (not recent_stable_ob.fit_on):
                #logging.info(f"Fitting state {self.id} on {recent_stable_ob}")
                self.partial_fit(recent_stable_ob.X, recent_stable_ob.y, sample_weight, classes)
                recent_stable_ob.fit_on = True

        # Decide if fingerprint needs to be updated
        if (self.seen - self.last_fingerprint_update) >= self.fingerprint_update_gap:
            self.should_update_fingerprint = True
        
        # Handle possibly dirty information after an evolution.
        self.trigger_dirty_performance_end = False
        if self.num_dirty_performance is not None:
            self.num_dirty_performance -= 1
            if self.num_dirty_performance < 0:
                self.trigger_dirty_performance_end = True

        # When we end the evolution trainsition state, which occurs
        # after window_size observations have passed through the
        # buffer so we have entirely observations of new behaviour,
        # we start a new, clean concept.
        if self.trigger_dirty_performance_end and self.has_valid_stable_window(self.min_window_size):
            evolved_model_stats, concept_model_flat = self.get_stable_metainfo(self.metainfo_method, False)
            self.end_evolution_transition(evolved_model_stats)
            if self.fingerprint is not None:
                self.fingerprint_changed_since_last_weight_calc = True

        return p

    def incorp_non_active_fingerprint(self, stats, active_state, normalizer):
        """ Update our fingerprint of the current state on observations drawn from a different concept
        """
        if active_state not in self.non_active_fingerprints:
            self.non_active_fingerprints[active_state] = self.fingerprint_constructor(
                stats, normalizer=normalizer)
        else:
            self.non_active_fingerprints[active_state].incorperate(stats, 1)

    def update_non_active_fingerprint(self, stats, active_state, normalizer):
        """ Update our fingerprint of the current state on observations drawn from a different concept
        """
        self.incorp_non_active_fingerprint(
            stats, active_state, normalizer=normalizer)

    def incorp_fingerprint(self, stats, normalizer):
        """ Update our fingerprint representing observations drawn from this concept
        """
        if self.fingerprint is None:
            self.fingerprint = self.fingerprint_constructor(
                stats, normalizer=normalizer)
        else:
            self.fingerprint.incorperate(stats, 1)

    def update_fingerprint(self, stats, ex, normalizer):
        """ Update our fingerprint representing observations drawn from this concept
        """
        self.incorp_fingerprint(stats, normalizer)
        self.last_fingerprint_update = self.seen
        self.should_update_fingerprint = False

    def incorp_similarity(self, similarity, similarity_vector, flat_similarity_vec):
        """ Update current statistics in an online manner.
        """
        if self.active_similarity_record is None:
            self.active_similarity_record = {
                "value": similarity, "var": 0, "seen": 1, "M": similarity, "S": 0}
        else:
            # Uses online mean and stdev algorithm
            value = similarity
            current_value = self.active_similarity_record["value"]
            current_weight = self.active_similarity_record["seen"]
            # Take the weighted average of current and new
            new_value = ((current_value * current_weight) +
                         value) / (current_weight + 1)

            # Formula for online stdev
            k = self.active_similarity_record["seen"] + 1
            last_M = self.active_similarity_record["M"]
            last_S = self.active_similarity_record["S"]

            new_M = last_M + (value - last_M)/k
            new_S = last_S + (value - last_M)*(value - new_M)

            variance = new_S / (k - 1)

            self.active_similarity_record["value"] = new_value
            self.active_similarity_record["var"] = variance
            self.active_similarity_record["seen"] = k
            self.active_similarity_record["M"] = new_M
            self.active_similarity_record["S"] = new_S
            self.active_similarity_record.pop("stdev", 0)
        fingerprint_copy = MinimalFingerprint(copy(self.fingerprint.flat_ignore_vec), self.fingerprint.id)
        self.similarity_vector_buffer.append(
            (similarity, similarity_vector, flat_similarity_vec, fingerprint_copy))
        if len(self.similarity_vector_buffer) >= self.similarity_vector_buffer_length:
            old_sim, old_vec, old_flat_vec, old_fingerprint = self.similarity_vector_buffer.popleft()
            self.remove_similarity(old_sim)

    def remove_similarity(self, similarity):
        """ Update current statistics in an online manner.
        """
        current_value = self.active_similarity_record["value"]
        current_weight = self.active_similarity_record["seen"]
        # Take the weighted average of current and new
        new_value = ((current_value * current_weight) -
                     similarity) / (current_weight - 1)

        # Formula for online stdev
        k = self.active_similarity_record["seen"] - 1
        last_M = self.active_similarity_record["M"]
        last_S = self.active_similarity_record["S"]

        new_M = last_M - (similarity - last_M)/k
        new_S = last_S - (similarity - new_M)*(similarity - last_M)

        variance = new_S / (k - 1)

        self.active_similarity_record["value"] = new_value
        self.active_similarity_record["var"] = variance
        self.active_similarity_record["seen"] = k
        self.active_similarity_record["M"] = new_M
        self.active_similarity_record["S"] = new_S
        self.active_similarity_record.pop("stdev", 0)
    
    def get_similarity_stdev(self):
        if "stdev" not in self.active_similarity_record:
            stdev = math.sqrt(self.active_similarity_record["var"]) if self.active_similarity_record["var"] > 0 else 0
            self.active_similarity_record["stdev"] = stdev
        return self.active_similarity_record["stdev"]

    def add_similarity_record(self, similarity, similarity_vector, flat_similarity_vec, ex):
        self.incorp_similarity(
            similarity, similarity_vector, flat_similarity_vec)

    def get_state_recent_similarity(self, similarity_calc_function):
        """ Want to get an idea of the 'normal' similarity
        to this state. We store this, however there is an issue
        when normalization etc changes, the similarity we have
        stored may be different to the similarity we would calculate
        now. So we also store the 100 <sim_vec_buffer_len> most recent
        vectors for which similarity was calculated. We calculate 
        similarity again, and adjust our similarity record based on the difference.
        """
        similarity_record = self.active_similarity_record["value"]
        similarity_stdev_record = self.get_similarity_stdev()
        #logging.info(f"recorded sim: {similarity_record}, stdev: {similarity_stdev_record}")
        adjust_ratios = []
        # Testing all would be slow, so just every n
        for old_similarity_reading, similarity_vec, flat_similarity_vec, similarity_fingerprint in list(self.similarity_vector_buffer)[::5]:
            if old_similarity_reading == 0:
                continue
            new_similarity_reading = similarity_calc_function(
                similarity_vec, similarity_fingerprint, flat_similarity_vec)
            adjust_ratios.append(new_similarity_reading /
                                 old_similarity_reading)
        if len(adjust_ratios) > 0:
            average_ratio = np.mean(adjust_ratios)
        else:
            average_ratio = 1
        #logging.info(f"ratio: {average_ratio}")
        return similarity_record * average_ratio, similarity_stdev_record * average_ratio



    def start_evolution_transition(self, dirty_length):
        """ Set state to an evolution phase. In this phase, we regard fingerprint meta-info as potentially dirty, as we know that statistics regarding classifier performance must have changed.
        """
        self.current_evolution = self.classifier.evolution

        # We add the current fingerprint, which contains the previous
        # bevaviour to the cache so we can use it while we recreate
        # a fingerprint of new behaviour.
        # For the next window_size, our buffered_window will contain
        # elements with the old behaviour and new, so will not give
        # accurate meta-info. We create a temporary fingerprint to capture
        # this, then when window is clean we start a new fingerprint.
        if self.fingerprint is not None:
            cache_fingerprint = deepcopy(self.fingerprint)
            cache_fingerprint.dirty_performance = self.fingerprint_dirty_performance
            cache_fingerprint.dirty_data = self.fingerprint_dirty_data
            self.fingerprint_cache.append(cache_fingerprint)
            self.fingerprint_cache = self.fingerprint_cache[-6:]
            self.fingerprint.initiate_evolution_plasticity()
            self.fingerprint.id += 1

        self.num_dirty_performance = dirty_length
        self.fingerprint_dirty_performance = True

    def end_evolution_transition(self, clean_base_stats):
        """ End state evolution phase. Called when enough observations have passed to regard data as clean.
        """

        # We add the current fingerprint, a transition period,
        # to the cache.
        # Using a starting set of clean performance stats,
        # from a window now fully containing new behaviour,
        # we again reset plasticity so we can learn.
        if self.fingerprint is not None:
            cache_fingerprint = deepcopy(self.fingerprint)
            cache_fingerprint.dirty_performance = self.fingerprint_dirty_performance
            cache_fingerprint.dirty_data = self.fingerprint_dirty_data
            self.fingerprint_cache.append(cache_fingerprint)
            self.fingerprint_cache = self.fingerprint_cache[-5:]
            self.fingerprint.initiate_clean_evolution_plasticity(
                clean_base_stats)
            self.fingerprint.id += 1

        self.num_dirty_performance = None
        self.fingerprint_dirty_performance = False
        self.trigger_dirty_performance_end = False


    
    def get_head_metainfo(self, method, add_to_normalizer, base_window=None, feature_base=None, feature_base_flat=None, stored_shap=None):
        window = self.get_head_window()
        original_metainfo, original_flat = self.select_metainfo_method(method, window, self.head_timeseries, add_to_normalizer, base_window, feature_base, feature_base_flat, stored_shap)

        return original_metainfo, original_flat

    def get_stable_metainfo(self, method, add_to_normalizer, base_window=None, feature_base=None, feature_base_flat=None, stored_shap=None):
        window = self.get_stable_window()
        original_metainfo, original_flat = self.select_metainfo_method(method, window, self.stable_timeseries, add_to_normalizer, base_window, feature_base, feature_base_flat, stored_shap)
        return original_metainfo, original_flat
    
    def select_metainfo_method(self, method, window, timeseries, add_to_normalizer, base_window, feature_base, feature_base_flat, stored_shap):
        if method == "metainfo":
            return self._metainfo_calc(window, timeseries, add_to_normalizer, base_window, feature_base, feature_base_flat, stored_shap)
        elif method == "accuracy":
            return self._accuracy_calc(window, add_to_normalizer)
        elif method == "feature":
            return self._feature_calc(window, add_to_normalizer)

    def _metainfo_calc(self, window, timeseries, add_to_normalizer=True, base_window=None, feature_base=None, feature_base_flat=None, stored_shap=None):
        """ Calculate meta-information from a given window.
        We use window_name to cache timeseries calculations, so they can be updated with
        less processing. ONLY USE THE SAME NAME IF THE WINDOW IS GOING TO BE THE SAME!
        buffered is used to calculate how many observations to update.
        
        Can handle passing a base set of metainfo, calculated on the SAME observations, using a different classifier.
        In this case, we can avoid recalculating metainfo which does not change between classifier, such as the features.

        If no stored_shap is passed, we use the shap of the active model. So pass this if not using this
        model as a base.
        """
        # Convert the window into a hashable representation to cache
        # If the hash is already in the cache, we can just return its metainfo
        # Since we store the evolution data in the hash, we know the window in the cache
        # must be the same
        window_serialization = (window[0].serialize(), window[-1].serialize())
        repeat_calc = False
        if window_serialization in self.metainfo_cache:
            return self.metainfo_cache[window_serialization]

        if stored_shap is None:
            stored_shap = self.classifier.shap_model

        # Check that we can use the feature base
        # to avoid recalculating feature metainfo, which are the same
        # for all classifiers. 
        # We ensure that the window the feature base is from is the exact window
        # we are calculating on.
        # This may not be true, if the feature base is from a larger window, i.e., background state
        # In this case we just recalculate fully
        can_use_feature_base = False
        if feature_base:
            tests = [base_window is not None,
                    feature_base is not None,
                    feature_base_flat is not None,
                    window[0].ob_id == base_window[0].ob_id,
                    window[-1].ob_id == base_window[-1].ob_id]
            can_use_feature_base = all(tests)
                                    
        
        current_metainfo, flat_vec = timeseries.get_concept_stats(self.classifier, can_use_feature_base, feature_base, feature_base_flat, stored_shap, ignore_sources=self.normalizer.ignore_sources,
                                                                     ignore_features=self.normalizer.ignore_features,
                                                                     normalizer=self.normalizer)

        if add_to_normalizer and not repeat_calc:
            self.normalizer.add_stats(current_metainfo)
        
        self.metainfo_cache[window_serialization] = (current_metainfo, flat_vec)
        self.metainfo_cache_age.append((window_serialization, window[0].ob_id))

        # Pop off all cached ranges which are not stored in recent data any more
        # We calculate this as when the start of the range is older than the oldest
        # stored element.
        while self.metainfo_cache_age[0][1] < self.recent_data[0].ob_id:
            rem_window, rem_age = self.metainfo_cache_age.popleft()
            if rem_window in self.metainfo_cache:
                self.metainfo_cache.pop(rem_window)

        return current_metainfo, flat_vec

    def _accuracy_calc(self, window, add_to_normalizer=True):
        accuracy = sum([x.correctly_classified for x in window]) / len(window)
        current_metainfo = {"Overall": {"Accuracy": accuracy}}
        if add_to_normalizer:
            self.normalizer.add_stats(current_metainfo)

        return current_metainfo, np.array([accuracy])

    def _feature_calc(self, window, add_to_normalizer=True):
        recent_X = window[-1].X
        x_features = {}
        flat_vec = np.empty(len(recent_X))
        for i, x in enumerate(recent_X):
            feature_name = f"f{i}"
            x_features[feature_name] = x
            flat_vec[i] = x
        current_metainfo = {"Features": x_features}

        if add_to_normalizer:
            self.normalizer.add_stats(current_metainfo)

        return current_metainfo, flat_vec

    def cull_recent_data(self):
        """ cull old data, possibly from the last concept.
        We remove the oldest elements, down to min_window_size
        """

        while len(self.recent_data) > self.min_window_size:
            self.recent_data.popleft()
            if len(self.recent_data) < len(self.head_window):
                self.head_window.popleft()
        self.buffer_length = self.get_buffer_len()
        self.refresh_recent_data()
    
    def transition(self, from_id, to_id):
        self.cull_recent_data()
        self.seen_since_transition = 0
        if self.id == from_id:
            self.transition_from()
        if self.id == to_id:
            self.transition_to()

    def transition_to(self):
        # Priors will have changed a lot, so we reset
        self.adwin_posterior_estimator.reset()
        self.recurrences += 1

    def transition_from(self):
        # Priors will have changed a lot, so we reset
        self.adwin_posterior_estimator.reset()


    def __str__(self):
        return f"<State {self.id}>"

    def __repr__(self):
        return self.__str__()


def check_fs_method(feature_selection_method):
    """ Check that the feature selection method is valid,
    and return the possible fingerprint formats for the method.
    """
    possible_fingerprints = None
    if feature_selection_method in ["uniform", "None"]:
        possible_fingerprints = [
            "cache", "descriptive", "sketch", "cachesketch", "histogram", "cachehistogram"]
    if feature_selection_method in ["fisher_overall", "default"]:
        possible_fingerprints = [
            "cache", "descriptive", "sketch", "cachesketch", "histogram", "cachehistogram"]
    if feature_selection_method == "fisher":
        possible_fingerprints = [
            "cache", "descriptive", "sketch", "cachesketch", "histogram", "cachehistogram"]
    if feature_selection_method in ["original"]:
        possible_fingerprints = [
            "cache", "descriptive", "sketch", "cachesketch", "histogram", "cachehistogram"]

    if feature_selection_method in ["gaussian_approx", "CacheMIHy"]:
        possible_fingerprints = ["cache", "descriptive"]
    if feature_selection_method in ["sketch", "sketch_MI"]:
        possible_fingerprints = ["sketch", "cachesketch"]
    if feature_selection_method in ["sketch_covariance", "sketch_covMI"]:
        possible_fingerprints = ["sketch", "cachesketch"]
    if feature_selection_method in ["sketch_covariance_weighted", "sketch_covMI_weighted"]:
        possible_fingerprints = ["sketch", "cachesketch"]
    if feature_selection_method in ["sketch_covariance_redundancy", "sketch_covredMI"]:
        possible_fingerprints = ["sketch", "cachesketch"]
    if feature_selection_method in ["histogram", "Cachehistogram_MI"]:
        possible_fingerprints = ["histogram", "cachehistogram"]

    return possible_fingerprints


class SELeCTClassifier:
    def __init__(self,
                 suppress=False,
                 learner=None,
                 sensitivity=0.1,
                 poisson=6,
                 normalizer=None,
                 window_size=100,
                 active_head_monitor_gap=5,
                 fingerprint_update_gap=15,
                 non_active_fingerprint_update_gap=50,
                 observation_gap=5,
                 sim_measure="metainfo",
                 MI_calc="metainfo",
                 ignore_sources=None,
                 ignore_features=['IMF', 'MI', 'pacf'],
                 similarity_num_stdevs=3,
                 similarity_min_stdev=0.015,
                 similarity_max_stdev=0.15,
                 buffer_ratio=0.2,
                 feature_selection_method="default",
                 fingerprint_method="auto",
                 fingerprint_bins=10,
                 min_window_ratio=0.65,
                 fingerprint_grace_period=10,
                 state_grace_period_window_multiplier=10,
                 bypass_grace_period_threshold=0.2,
                 state_estimator_risk=0.5,
                 state_estimator_swap_risk=0.75,
                 minimum_concept_likelihood=0.005,
                 min_drift_likelihood_threshold=0.175,
                 min_estimated_posterior_threshold=0.2,
                 correlation_merge=True,
                 merge_threshold=0.9,
                 background_state_prior_multiplier=0.3,
                 zero_prob_minimum = 0.7,
                 multihop_penalty=0.75,
                 prev_state_prior=0,
                 MAP_selection=False):

        if learner is None:
            raise ValueError('Need a learner')

        self.feature_selection_method = feature_selection_method
        possible_fingerprints = check_fs_method(self.feature_selection_method)
        if possible_fingerprints is None:
            raise ValueError(
                "No matching feature selection method, check fsmethod option")
        if fingerprint_method == "auto":
            self.fingerprint_method = possible_fingerprints[0]
        else:
            if fingerprint_method not in possible_fingerprints:
                raise ValueError(
                    "Fingerprint constructor does not match feature selection method, check fsmethod and fingerprintmethod option")
            self.fingerprint_method = fingerprint_method
        self.fingerprint_bins = fingerprint_bins
        if self.fingerprint_method in [
            "cache", "sketch", "cachesketch", "histogram", "cachehistogram"] and (self.fingerprint_bins is None or self.fingerprint_bins <= 0):
            raise ValueError("Need to specify fingerprint bins with this fingerprint method")
        self.fingerprint_constructor = get_fingerprint_constructor(
            self.fingerprint_method, self.fingerprint_bins)
        self.fingerprint_type = get_fingerprint_type(self.fingerprint_method)
        # learner is the classifier used by each state.
        # papers use HoeffdingTree from scikit-multiflow
        self.learner = learner

        # sensitivity is the sensitivity of the concept
        # drift detector
        self.sensitivity = sensitivity
        self.base_sensitivity = sensitivity
        self.current_sensitivity = sensitivity

        # suppress debug info
        self.suppress = suppress

        # rand_weights is if a strategy is setting sample
        # weights for training
        self.rand_weights = poisson > 1

        # poisson is the strength of sample weighting
        # based on leverage bagging
        self.poisson = poisson

        # Size of the window used to calculate feature vector.
        # Smaller can be used quicker and less autocorrelation, but may be unstable
        self.window_size = window_size

        # Minimum window size to calculate feature vector.
        # Smaller lets us make calculations earlier, but again may be unstable
        self.min_window_size = window_size*min_window_ratio

        # The gap between monitoring the likelihood of the active state on the head window.
        # Used for drift detection, so should be relatively small
        self.active_head_monitor_gap = active_head_monitor_gap

        # The gap between calculating new fingerprints for the active state on the stable
        # window. 
        # Smaller is more frequent updates, but possible autocorrelation issues if smaller than window size
        self.fingerprint_update_gap = fingerprint_update_gap

        # Gap between updating the non-active state fingerprints
        self.non_active_fingerprint_update_gap = non_active_fingerprint_update_gap

        # Gap between calculating the likelihood of alternative states
        self.observation_gap = observation_gap

        # What ratio of currently seen data should be held in the buffer?
        # We use a ratio rather than a raw value so that at the start, we still
        # can use some data to train.
        # There is a max defined in ConceptState
        self.buffer_ratio = buffer_ratio

        # How many stats does a fingerprint need to see before it is considered valid?
        # Mainly used when considering weights.
        # A small number, 1 or 2, may cause issues with low stdev when calculating weights.
        # A large number will not be using up to date info for weights
        self.fingerprint_grace_period = fingerprint_grace_period

        # How many observations should we give a state before we properly consider transitioning away?
        # I.E., how many observations untill a new state should be stable?
        # Pass in as a multiplier of window size, as this is how many 'full windows' of stats we want to see
        self.state_grace_period = self.window_size * state_grace_period_window_multiplier

        # What likelihood threshold is needed to bypass the grace period?
        # I.E., if we see this likelihood in an alternative state, we transition to that state.
        # # Too low will have many false positive transitions, but too low may miss a good alternative for a bit 
        self.bypass_grace_period_threshold = bypass_grace_period_threshold

        # What is the risk level we want for culling old probability records?
        # Higher = more aggressive dropping of old data
        self.state_estimator_risk = state_estimator_risk

        # What is the risk level we want when considering swapping to an alternative state?
        # Higher = will swap with a lower likelihood difference
        self.state_estimator_swap_risk = state_estimator_swap_risk
        
        # What is the minimum likelihood we consider?
        # I.E., any values below this are treated the same
        self.minimum_concept_likelihood = minimum_concept_likelihood

        # What is the minimum allowed likelihood for the current state at a drift?
        # Below this, we always swap away from the current state
        self.min_drift_likelihood_threshold = min_drift_likelihood_threshold

        # What is the minimum allowed posterior for the current state?
        # Below this, we always swap away from the current state
        self.min_estimated_posterior_threshold = min_estimated_posterior_threshold

        # True or false depending on if models should be merged
        self.correlation_merge = correlation_merge
        self.merge_threshold = merge_threshold

        # How many sets of statistics must the normalizer see before
        # we can calculate realistic similarities.
        # NOTE: these are calculated periodically, not every observation
        self.normalizer_grace_period = 5

        # The multipler for the background state prior
        self.background_state_prior_multiplier = background_state_prior_multiplier

        # Smoothing for unseen priors
        # self.smoothing_factor = smoothing_factor
        self.zero_prob_minimum = zero_prob_minimum

        # Penalty for considering future transitions.
        # A lower value indicates less consideration,
        # e.g, 0.75 means the probability of transitioning
        # in 2 steps is work 75% of the probability of transitioning
        # in one step.
        self.multihop_penalty = multihop_penalty

        self.prev_state_prior = prev_state_prior

        # Whether or not to use simple Maximum A Posteriori selection
        # for the next active state.
        # If False, we use a more robust statistical test (Hoeffding bound)
        # rather than just pick the most probable state.
        self.MAP_selection = MAP_selection


        self.in_warning = False
        self.last_warning_point = 0
        self.warning_detected = False

        # initialize waiting state. If we don't have enough
        # data to select the next concept, we wait until we do.
        self.waiting_for_concept_data = False

        # init the current number of states
        self.max_state_id = 0

        # init randomness
        self.random_state = None
        self._random_state = check_random_state(self.random_state)

        self.ex = -1
        self.buffered_ex = -1
        self.buffered_ex_bg = -1
        self.classes = None
        self._train_weight_seen_by_model = 0

        self.active_state_is_new = True

        # init data which is exposed to evaluators
        self.found_change = False
        self.num_states = 1
        self.active_state = self.max_state_id
        self.states = []



        self.non_active_fingerprint_last_observation = self.non_active_fingerprint_update_gap * -1
        self.last_active_head_monitor = self.active_head_monitor_gap * -1
        self.ignore_sources = ignore_sources
        self.ignore_features = ignore_features
        self.normalizer = normalizer or Normalizer(
            ignore_sources=self.ignore_sources, ignore_features=self.ignore_features, fingerprint_constructor=self.fingerprint_constructor)
        self.fingerprint_window = deque(maxlen=self.window_size)
        self.fingerprint_window_bg = deque(maxlen=self.window_size)
        self.fingerprint_similarity_detector = make_detector(s=sensitivity, update_gap=self.active_head_monitor_gap)
        self.fingerprint_similarity_detector_warn = make_detector(
            s=get_warning_sensitivity(sensitivity), update_gap=self.active_head_monitor_gap)

        # Observation gap controls how often all states are compared to the 
        # active stream in order to calculate likelihood.
        self.take_observations = observation_gap > 0
        self.last_observation = 0
        self.monitor_all_state_active_similarity = None
        self.monitor_all_state_buffered_similarity = None
        self.monitor_active_state_active_similarity = None
        self.monitor_active_state_buffered_similarity = None
        self.detected_drift = False
        self.deletions = []

        self.sim_measure = sim_measure
        self.MI_calc = MI_calc

        self.similarity_num_stdevs = similarity_num_stdevs
        self.similarity_min_stdev = similarity_min_stdev
        self.similarity_max_stdev = similarity_max_stdev

        # track the last predicted label
        self.last_label = 0

        # set up repository
        self.state_repository = {}
        init_id = self.max_state_id
        self.max_state_id += 1
        init_state = ConceptState(init_id, self.learner(), self.fingerprint_update_gap,
                                  fingerprint_method=self.fingerprint_method, fingerprint_bins=self.fingerprint_bins,
                                  window_size=window_size, min_window_size = self.min_window_size, dirty_data_length=self.window_size, normalizer=self.normalizer, fingerprint_grace_period=self.fingerprint_grace_period, estimator_risk_level=self.state_estimator_risk, buffer_ratio=self.buffer_ratio, metainfo_method=self.sim_measure)
        self.state_repository[init_id] = init_state
        self.occurences = {}
        self.occurences[init_id] = 1
        init_state.transition_to()
        self.active_state_id = init_id

        self.manual_control = False
        self.force_transition = False
        self.force_transition_only = False
        self.force_learn_fingerprint = False
        self.force_stop_learn_fingerprint = False
        self.force_transition_to = None
        self.force_transition_to = None
        self.force_lock_weights = False
        self.force_locked_weights = None
        self.force_stop_fingerprint_age = None
        self.force_stop_add_to_normalizer = False

        self.buffer = deque()
        self.buffered_window = deque(maxlen=self.window_size)
        self.buffer_bg = deque()
        self.buffered_window_bg = deque(maxlen=self.window_size)

        self.observations_since_last_transition = 0
        self.trigger_point = self.window_size + 1

        self.trigger_transition_check = False
        self.trigger_attempt_transition_check = False
        self.last_trigger_made_shadow = True
        self.last_transition = None

        self.window_MI_cache = {}
        self.timeseries_cache = {}
        self.all_states_buffered_cache = {}
        self.all_states_active_cache = {}
        self.weights_cache = None
        self.fingerprint_changed_since_last_weight_calc = True

        self.buffered_metainfo = None
        self.buffered_normed_flat = None
        self.buffered_nonormed_flat = None
        self.active_metainfo = None
        self.active_normed_flat = None
        self.active_nonormed_flat = None

        self.monitor_feature_selection_weights = []
        self.concept_priors = {}
        self.concept_likelihoods = {}
        self.concept_posteriors = {}

        #TESTING
        self.concept_priors_2h = {}
        self.concept_priors_1h = {}
        self.concept_priors_combined = {}
        self.concept_posterior_mh = {}

        self.concept_transitions_standard = {}
        self.concept_transitions_warning = {}
        self.concept_transitions_drift = {}
        self.concept_transitions_change = {}


        # Background model for considering a new state
        self.background_state = None
        self.background_detector = make_detector(
            s=get_warning_sensitivity(sensitivity), update_gap=self.active_head_monitor_gap)

        
        self.merges = {}
        self.deactivated_states = {}
        





    def get_active_state(self):
        return self.state_repository[self.active_state_id]

    def construct_state_object(self, id=-1):
        return ConceptState(id, self.learner(), self.fingerprint_update_gap,
                                  fingerprint_method=self.fingerprint_method, fingerprint_bins=self.fingerprint_bins,
                                  window_size=self.window_size, min_window_size = self.min_window_size, dirty_data_length=self.window_size, normalizer=self.normalizer, fingerprint_grace_period=self.fingerprint_grace_period, estimator_risk_level=self.state_estimator_risk, buffer_ratio=self.buffer_ratio, metainfo_method=self.sim_measure)

    def make_state(self):
        new_id = self.max_state_id
        self.max_state_id += 1
        state = self.construct_state_object(new_id)
        return new_id, state

    def reset(self):
        pass

    def get_temporal_x(self, X):
        return np.concatenate([X], axis=None)

    def predict(self, X):
        """
        Predict using the model of the currently active state.
        """
        temporal_X = self.get_temporal_x(X)

        return self.get_active_state().classifier.predict([temporal_X])

    def partial_fit(self, X, y, classes=None, sample_weight=None, masked=False):
        """
        Fit an array of observations.
        Splits input into individual observations and
        passes to a helper function _partial_fit.
        Randomly weights observations depending on 
        Config.
        """

        if masked:
            return
        if self.classes is None and classes is not None:
            self.classes = classes
        if y is not None:
            row_cnt, _ = get_dimensions(X)
            if sample_weight is None:
                sample_weight = np.ones(row_cnt)
            if row_cnt != len(sample_weight):
                raise ValueError(
                    'Inconsistent number of instances ({}) and weights ({}).'
                    .format(row_cnt, len(sample_weight)))
            for i in range(row_cnt):
                if sample_weight[i] != 0.0:
                    self._train_weight_seen_by_model += sample_weight[i]
                    self.ex += 1
                    if self.rand_weights and self.poisson >= 1:
                        k = self.poisson
                        sample_weight[i] = k
                    self._partial_fit(X[i], y[i], sample_weight[i], masked)

    def get_imputed_label(self, X, prediction, last_label):
        """ Get a label.
        Imputes when the true label is masked
        """

        return prediction


    def _cosine_similarity(self, current_metainfo, state=None, fingerprint_to_compare=None, flat_norm_current_metainfo=None, flat_nonorm_current_metainfo=None, distance_metric="cosine"):
        if state is None:
            state = self.get_active_state()

        # allow outside locking of weights (for evaluation, not in normal activity)
        if not self.force_lock_weights:

            # Can reuse weights if they exist and were updated this observation, or they haven't changed since last time.
            if self.weights_cache and (not self.normalizer.changed_since_last_weight_calc) and (not self.fingerprint_changed_since_last_weight_calc):
                weights, flat_weight_vector, ignore_flat_weight_vector, sorted_feature_weights = self.weights_cache[
                    0]

            else:
                state_non_active_fingerprints = {k: (v.get_clean_fingerprint(), v.non_active_fingerprints)
                                                 for k, v in self.state_repository.items() if v.get_clean_fingerprint() is not None}
                weights, flat_weight_vector, ignore_flat_weight_vector, sorted_feature_weights = get_dimension_weights(list([state.get_clean_fingerprint() for state in self.state_repository.values(
                ) if state.get_clean_fingerprint() is not None]), state_non_active_fingerprints,  self.normalizer, feature_selection_method=self.feature_selection_method)
                self.weights_cache = (
                    (weights, flat_weight_vector, ignore_flat_weight_vector, sorted_feature_weights), self.ex)
                self.fingerprint_changed_since_last_weight_calc = False
            self.force_locked_weights = (
                weights, flat_weight_vector, ignore_flat_weight_vector, sorted_feature_weights)
        else:
            weights, flat_weight_vector, ignore_flat_weight_vector, sorted_feature_weights = self.force_locked_weights

        # Make weights availiable for logging
        self.monitor_feature_selection_weights = sorted_feature_weights

        # Set weights to be on non-ignored features only, and normalize
        weights_vec = ignore_flat_weight_vector
        normed_weights = (weights_vec) / (np.max(weights_vec))

        # Normalize the meta-info vector
        if flat_nonorm_current_metainfo is not None:
            stat_vec, stat_nonorm_vec = self.normalizer.norm_flat_vector(
                flat_nonorm_current_metainfo), flat_nonorm_current_metainfo
        else:
            stat_vec, stat_nonorm_vec = self.normalizer.get_flat_vector(
                current_metainfo)

        # Want to check the current fingerprint,
        # and last clean fingerprint
        # and most recent dirty fingerprint (incase there isn't another clean one and recent is new)
        # This covers the cases where we have just evolved, so the most recent fingerprint is not representative of normal behavior
        fingerprints_to_check = []
        if fingerprint_to_compare is None:
            fingerprints_to_check.append(
                state.fingerprint)
            for cached_fingerprint in state.fingerprint_cache[::-1]:
                if cached_fingerprint is None:
                    continue
                fingerprints_to_check.append(cached_fingerprint)
                if len(fingerprints_to_check) > 5:
                    break
        else:
            fingerprints_to_check = [fingerprint_to_compare]
        similarities = []
        # NOTE: We actually calculate cosine distance, so we return the MINIMUM DISTANCE
        # This is confusing, as you would think if we were really working with similarity it
        # would be maximum!
        # TODO: rename to distance
        for fpi, fp in enumerate(fingerprints_to_check):
            if fp is None:
                continue
            fingerprint_nonorm_vec = fp.flat_ignore_vec
            fingerprint_vec = self.normalizer.norm_flat_vector(
                fingerprint_nonorm_vec)
            
            if distance_metric == "cosine":
                similarity = get_cosine_distance(
                    stat_vec, fingerprint_vec, True, normed_weights)
            if distance_metric == "euclid":
                similarity = get_euclidean_distance(
                    stat_vec, fingerprint_vec, True, normed_weights)
            elif distance_metric == "jaccard":
                similarity = get_jaccard_distance(
                    stat_vec, fingerprint_vec, True, normed_weights)
            elif distance_metric == "jmahal":
                similarity = get_jmahal_distance(
                    stat_vec, fingerprint_vec, True, normed_weights)

            similarities.append(similarity)
        # We take the minimum similarity to recent fingerprints
        # (similarity is really distance, so minimum is the best!)
        min_similarity = min(similarities)

        return min_similarity

    def _sketch_cosine_similarity(self, current_metainfo, state=None, fingerprint_to_compare=None, flat_norm_current_metainfo=None, flat_nonorm_current_metainfo=None):
        if state is None:
            state = self.get_active_state()
        if not self.force_lock_weights:

            # Can reuse weights if they exist and were updated this observation, or they haven't changed since last time.
            if self.weights_cache and (not self.normalizer.changed_since_last_weight_calc) and (not self.fingerprint_changed_since_last_weight_calc):
                weights, flat_weight_vector, ignore_flat_weight_vector, sorted_feature_weights = self.weights_cache[
                    0]

            else:
                state_non_active_fingerprints = {k: (v.get_clean_fingerprint(), v.non_active_fingerprints)
                                                 for k, v in self.state_repository.items() if v.get_clean_fingerprint() is not None}
                weights, flat_weight_vector, ignore_flat_weight_vector, sorted_feature_weights = get_dimension_weights(list([state.get_clean_fingerprint() for state in self.state_repository.values(
                ) if state.get_clean_fingerprint() is not None]), state_non_active_fingerprints,  self.normalizer, feature_selection_method=self.feature_selection_method)
                self.weights_cache = (
                    (weights, flat_weight_vector, ignore_flat_weight_vector, sorted_feature_weights), self.ex)
                self.fingerprint_changed_since_last_weight_calc = False
            self.force_locked_weights = (
                weights, flat_weight_vector, ignore_flat_weight_vector, sorted_feature_weights)
        else:
            weights, flat_weight_vector, ignore_flat_weight_vector, sorted_feature_weights = self.force_locked_weights

        self.monitor_feature_selection_weights = sorted_feature_weights

        weights_vec = ignore_flat_weight_vector
        normed_weights = (weights_vec) / (np.max(weights_vec))

        if flat_nonorm_current_metainfo is not None:
            stat_vec, stat_nonorm_vec = self.normalizer.norm_flat_vector(
                flat_nonorm_current_metainfo), flat_nonorm_current_metainfo
        else:
            stat_vec, stat_nonorm_vec = self.normalizer.get_flat_vector(
                current_metainfo)

        # Want to check the current fingerprint,
        # and last clean fingerprint
        # and most recent dirty fingerprint (incase there isn't another clean one and recent is new)
        fingerprints_to_check = []
        if fingerprint_to_compare is None:
            fingerprints_to_check.append(
                state.fingerprint)
            for cached_fingerprint in state.fingerprint_cache[::-1]:
                fingerprints_to_check.append(cached_fingerprint)
                if len(fingerprints_to_check) > 5:
                    break
        else:
            fingerprints_to_check = [fingerprint_to_compare]
        similarities = []
        # NOTE: We actually calculate cosine distance, so we return the MINIMUM DISTANCE
        # This is confusing, as you would think if we were really working with similarity it
        # would be maximum!
        # TODO: rename to distance
        for fp in fingerprints_to_check:
            sketch_similarities = []
            fp_sketch = fp.sketch.get_observation_matrix()
            for sketch_row in range(fp_sketch.shape[0]):
                fingerprint_nonorm_vec = fp_sketch[sketch_row, :]
                fingerprint_vec = self.normalizer.norm_flat_vector(
                    fingerprint_nonorm_vec)
                similarity = get_cosine_distance(
                    stat_vec, fingerprint_vec, True, normed_weights)
                sketch_similarities.append(similarity)
            fingerprint_vec, fingerprint_nonorm_vec = self.normalizer.get_flat_vector(
                fp.fingerprint_values)
            fingerprint_vec = self.normalizer.norm_flat_vector(
                fingerprint_nonorm_vec)
            similarity = get_cosine_distance(
                stat_vec, fingerprint_vec, True, normed_weights)
            similarities.append(np.min(sketch_similarities))
            similarities.append(similarity)
        min_similarity = min(similarities)
        return min_similarity

    def _histogram_similarity(self, current_metainfo, state=None, fingerprint_to_compare=None, flat_norm_current_metainfo=None, flat_nonorm_current_metainfo=None):
        if state is None:
            state = self.get_active_state()
        if not self.force_lock_weights:

            # Can reuse weights if they exist and were updated this observation, or they haven't changed since last time.
            if self.weights_cache and (not self.normalizer.changed_since_last_weight_calc) and (not self.fingerprint_changed_since_last_weight_calc):
                weights, flat_weight_vector, ignore_flat_weight_vector, sorted_feature_weights = self.weights_cache[
                    0]

            else:
                state_non_active_fingerprints = {k: (v.get_clean_fingerprint(), v.non_active_fingerprints)
                                                 for k, v in self.state_repository.items() if v.get_clean_fingerprint() is not None}
                weights, flat_weight_vector, ignore_flat_weight_vector, sorted_feature_weights = get_dimension_weights(list([state.get_clean_fingerprint() for state in self.state_repository.values(
                ) if state.get_clean_fingerprint() is not None]), state_non_active_fingerprints,  self.normalizer, feature_selection_method=self.feature_selection_method)
                self.weights_cache = (
                    (weights, flat_weight_vector, ignore_flat_weight_vector, sorted_feature_weights), self.ex)
                self.fingerprint_changed_since_last_weight_calc = False
            self.force_locked_weights = (
                weights, flat_weight_vector, ignore_flat_weight_vector, sorted_feature_weights)
        else:
            weights, flat_weight_vector, ignore_flat_weight_vector, sorted_feature_weights = self.force_locked_weights

        self.monitor_feature_selection_weights = sorted_feature_weights

        weights_vec = ignore_flat_weight_vector
        normed_weights = (weights_vec) / (np.max(weights_vec))

        if flat_nonorm_current_metainfo is not None:
            stat_vec, stat_nonorm_vec = self.normalizer.norm_flat_vector(
                flat_nonorm_current_metainfo), flat_nonorm_current_metainfo
        else:
            stat_vec, stat_nonorm_vec = self.normalizer.get_flat_vector(
                current_metainfo)

        # Want to check the current fingerprint,
        # and last clean fingerprint
        # and most recent dirty fingerprint (incase there isn't another clean one and recent is new)
        fingerprints_to_check = []
        if fingerprint_to_compare is None:
            fingerprints_to_check.append(
                state.fingerprint)
            for cached_fingerprint in state.fingerprint_cache[::-1]:
                fingerprints_to_check.append(cached_fingerprint)
                if len(fingerprints_to_check) > 5:
                    break
        else:
            fingerprints_to_check = [fingerprint_to_compare]
        similarities = []
        for fp in fingerprints_to_check:
            similarity = get_histogram_probability(
                stat_vec, fp, True, normed_weights)
            similarities.append(similarity)
        max_similarity = max(similarities)
        return max_similarity

    def _accuracy_similarity(self, current_metainfo, state=None, fingerprint_to_compare=None):
        if state is None:
            state = self.get_active_state()
        state_accuracy = current_metainfo["Overall"]["Accuracy"]
        if fingerprint_to_compare is None:
            fingerprint_accuracy = state.fingerprint.fingerprint_values["Overall"]["Accuracy"]
        else:
            fingerprint_accuracy = fingerprint_to_compare.fingerprint_values[
                "Overall"]["Accuracy"]
        similarity = abs(state_accuracy - fingerprint_accuracy)
        return similarity

    def get_similarity_to_active_state(self, current_metainfo, flat_norm_current_metainfo=None, flat_nonorm_current_metainfo=None):
        return self.get_similarity(current_metainfo, state=self.get_active_state(), flat_norm_current_metainfo=flat_norm_current_metainfo, flat_nonorm_current_metainfo=flat_nonorm_current_metainfo)
    
    def get_similarity_to_background_state(self, current_metainfo, flat_norm_current_metainfo=None, flat_nonorm_current_metainfo=None):
        return self.get_similarity(current_metainfo, state=self.background_state, flat_norm_current_metainfo=flat_norm_current_metainfo, flat_nonorm_current_metainfo=flat_nonorm_current_metainfo)

    def get_similarity(self, current_metainfo, state=None, fingerprint_to_compare=None, flat_norm_current_metainfo=None, flat_nonorm_current_metainfo=None):
        """ Get similarity, selecting the appropriate similarity function given self.sim_measure
        """
        if self.sim_measure == "metainfo":
            return self._cosine_similarity(current_metainfo, state, fingerprint_to_compare=fingerprint_to_compare, flat_norm_current_metainfo=flat_norm_current_metainfo, flat_nonorm_current_metainfo=flat_nonorm_current_metainfo, distance_metric="cosine")
        if self.sim_measure == "jaccard":
            return self._cosine_similarity(current_metainfo, state, fingerprint_to_compare=fingerprint_to_compare, flat_norm_current_metainfo=flat_norm_current_metainfo, flat_nonorm_current_metainfo=flat_nonorm_current_metainfo, distance_metric="jaccard")
        if self.sim_measure == "euclid":
            return self._cosine_similarity(current_metainfo, state, fingerprint_to_compare=fingerprint_to_compare, flat_norm_current_metainfo=flat_norm_current_metainfo, flat_nonorm_current_metainfo=flat_nonorm_current_metainfo, distance_metric="euclid")
        if self.sim_measure == "jmahal":
            return self._cosine_similarity(current_metainfo, state, fingerprint_to_compare=fingerprint_to_compare, flat_norm_current_metainfo=flat_norm_current_metainfo, flat_nonorm_current_metainfo=flat_nonorm_current_metainfo, distance_metric="jmahal")
        if self.sim_measure == "sketch":
            return self._sketch_cosine_similarity(current_metainfo, state, fingerprint_to_compare=fingerprint_to_compare, flat_norm_current_metainfo=flat_norm_current_metainfo, flat_nonorm_current_metainfo=flat_nonorm_current_metainfo)
        if self.sim_measure == "histogram":
            return self._histogram_similarity(current_metainfo, state, fingerprint_to_compare=fingerprint_to_compare, flat_norm_current_metainfo=flat_norm_current_metainfo, flat_nonorm_current_metainfo=flat_nonorm_current_metainfo)
        if self.sim_measure == "accuracy":
            return self._accuracy_similarity(current_metainfo, state, fingerprint_to_compare=fingerprint_to_compare)
        raise ValueError("similarity method not set")




    def check_background_state_initialized(self):
        if not self.background_state:
            self.build_background_state()

    def build_background_state(self):
        #logging.info("building background state")
        self.background_state = self.construct_state_object()

    def check_take_measurement(self):
        """
        Check if we need to take a measurement for drift detection.
        This uses the head window, as we are testing if the head of the stream is
        different to the current fingerprint. This should fire relatively often,
        so as to trigger drift detection warnings with minimal delay.

        We check:
            If there has been the correct gap since the last check
            If the active state has a valid (min length) window
            If we have some manual control (for evaluation and experimentation)
        """
        take_measurement = (
            self.ex - self.last_active_head_monitor) >= self.active_head_monitor_gap
        take_measurement = take_measurement and self.get_active_state().has_valid_head_window(self.min_window_size)
        if self.manual_control:
            take_measurement = self.force_learn_fingerprint and not self.force_stop_learn_fingerprint
        return take_measurement

    def reset_partial_fit_settings(self):
        """ Set class level handlers for the current partial fit
        """
        # Reset monitoring variables
        self.monitor_active_state_active_similarity = None
        self.monitor_active_state_buffered_similarity = None
        self.monitor_all_state_buffered_similarity = None
        self.monitor_all_state_active_similarity = None
        self.deletions = []

    def update_similarity_change_detection(self, similarity, likelihood):
        """ Update change detectors with active state active similarity deviation.
        """
        # We add similarity to detections directly.
        # This might not be the best method, as it is directional
        # and is not independed, very autocorrelated if this updates more
        # than window_size.
        # Using standard deviations away from normal might help directionality,
        # but not auto correlation.
        # Tested vs likelihood - they are similar measures but the change at different rates
        # and have different ranges. I.e., likelihood is stdevs mapped onto a normal distribution
        # so ~3 -> 0.03. (also increase in stdevs -> decrease in likelihood)
        # Found that stdevs is a bit more stable - possibly because it changes linearly
        # unlike likelihood?
        if self.get_active_state().active_similarity_record is not None:
            self.fingerprint_similarity_detector.add_element(
                likelihood)
            self.fingerprint_similarity_detector_warn.add_element(
                likelihood)

    def update_bg_similarity_change_detection(self, similarity, likelihood):
        """ Update change detectors with active state active similarity deviation.
        """
        # We add similarity to detections directly.
        # This might not be the best method, as it is directional
        # and is not independed, very autocorrelated if this updates more
        # than window_size.
        # Using standard deviations away from normal might help directionality,
        # but not auto correlation.
        # Tested vs likelihood - they are similar measures but the change at different rates
        # and have different ranges. I.e., likelihood is stdevs mapped onto a normal distribution
        # so ~3 -> 0.03. (also increase in stdevs -> decrease in likelihood)
        # Found that stdevs is a bit more stable - possibly because it changes linearly
        # unlike likelihood?
        if self.background_state.active_similarity_record is not None:
            self.background_detector.add_element(
                likelihood)

    def check_do_update_fingerprint(self):
        """ Check if active fingerprint should be updated
        """
        # Update fingerprint one state has not been updated for x observations
        # and we have stored enough data to build a window.
        do_update_fingerprint = self.get_active_state().should_update_fingerprint
        do_update_fingerprint &= self.get_active_state().has_valid_stable_window(self.min_window_size)
        
        # Don't update the fingerpritn if the normalizer is too new.
        # This is because our normalization will not work well, and give poor results.
        do_update_fingerprint &= self.normalizer.seen_stats > self.normalizer_grace_period

        # Extra triggers for control from calling program.
        if self.manual_control:
            do_update_fingerprint = self.force_learn_fingerprint
        do_update_fingerprint &= not self.force_stop_learn_fingerprint
        no_fingerprint_age_limit = self.force_stop_fingerprint_age is None or (
            not (self.get_active_state().seen > self.force_stop_fingerprint_age))
        do_update_fingerprint &= no_fingerprint_age_limit
        return do_update_fingerprint
    
    def record_state_stable_similarity(self, state):
        """ Calculate state similarity to stable window, and record it.
        Requires a valid fingerprint to be active, i.e., one which has seen enough data.
        If invalid, we return the metainfo but no similarity
        """
        valid_window = state.has_valid_stable_window(self.min_window_size)
        metainfo, metainfo_flat = state.get_stable_metainfo(method=self.sim_measure, add_to_normalizer=valid_window)
        if is_valid_fingerprint(state.fingerprint) and valid_window:
            similarity = self.get_similarity(
                metainfo,
                state=state, 
                flat_nonorm_current_metainfo=metainfo_flat)

            state.add_similarity_record(similarity, deepcopy(
                metainfo), metainfo_flat, self.ex)
            #logging.info(f"State similarity after: {state.active_similarity_record}")
            return similarity, metainfo, metainfo_flat
        return None, metainfo, metainfo_flat


    def check_do_update_non_active_fingerprint(self):
        """ Returns if non active fingerprints should be updated.
        """
        # Update non-active fingerprints every x observations
        # and we have stored enough data to build a window.
        do_update_non_active_fingerprint = self.ex - \
            self.non_active_fingerprint_last_observation >= self.non_active_fingerprint_update_gap
        do_update_non_active_fingerprint &= self.get_active_state().has_valid_stable_window(self.min_window_size)

        # Extra triggers for control from calling program.
        if self.manual_control:
            do_update_non_active_fingerprint = self.force_learn_fingerprint
        do_update_non_active_fingerprint = do_update_non_active_fingerprint and not self.force_stop_learn_fingerprint
        no_fingerprint_age_limit = self.force_stop_fingerprint_age is None or (
            not (self.get_active_state().seen > self.force_stop_fingerprint_age))
        do_update_non_active_fingerprint = do_update_non_active_fingerprint and no_fingerprint_age_limit

        return do_update_non_active_fingerprint
    
    def init_normalizer(self):
        """ Calculate and add head window metainfo to the normalizer
        if it has seen less than grace_period observations.
        We call this to add a few metainfo windows to the normalizer to give it a boost starting up
        """
        if self.normalizer.seen_stats < self.normalizer_grace_period and self.get_active_state().has_valid_head_window(self.min_window_size):
            self.get_active_state().get_head_metainfo(method=self.sim_measure, add_to_normalizer=True)



    def evaluate_all_fingerprint_similarity(self, metainfo, metainfo_flat, use_stable, add_to_normalizer, update_na_record, record_likelihood):

        # Debug data, for printing graph
        observation_stats = {}
        observation_similarities = {}

        # Save model-agnostic features so don't need to recalculate
        fingerprint_feature_base = metainfo

        observation_stats["active"] = metainfo

        active_state_window = self.get_active_state().get_stable_window() if use_stable else self.get_active_state().get_head_window()
        # Get the stats generated by each stored model
        # To do this, we need to reclassify using that model
        # to get correct meta-information such as error rate.
        # However we can reuse the feature data.
        # -1 is the id for background state
        for concept_id in [*self.state_repository.keys(), -1]:
            if concept_id != -1:
                state = self.state_repository[concept_id]
            else:
                state = self.background_state
            if state is None:
                continue
            if state.fingerprint is None:
                continue
            valid_window = state.has_valid_stable_window(self.min_window_size) if use_stable else state.has_valid_head_window(self.min_window_size)
            if not valid_window:
                continue
            if concept_id != self.active_state_id:
                concept_classifier = state.classifier
                window_metainfo_func = state.get_stable_metainfo if use_stable else state.get_head_metainfo
                concept_metainfo, concept_metainfo_flat = window_metainfo_func(method=self.sim_measure, add_to_normalizer=add_to_normalizer, base_window = active_state_window, feature_base=fingerprint_feature_base,feature_base_flat=metainfo_flat,
                                                                                        stored_shap=concept_classifier.shap_model)

                if update_na_record:
                    state.update_non_active_fingerprint(
                        concept_metainfo, self.active_state_id, normalizer=self.normalizer)
            else:
                concept_metainfo = fingerprint_feature_base
                concept_metainfo_flat = metainfo_flat

            observation_stats[concept_id] = concept_metainfo
            similarity = self.get_similarity(
                concept_metainfo, state=state, flat_nonorm_current_metainfo=concept_metainfo_flat)

            # We should only record likelihood when looking at the active fingerprint window, as this contains the most recent elements.
            if record_likelihood:
                self.concept_likelihoods[concept_id] = state.get_sim_observation_probability(
                    similarity, self.get_similarity, self.similarity_min_stdev, self.similarity_max_stdev)
                #logging.info(f"Updating likelihood for state {concept_id}: {self.concept_likelihoods[concept_id]}")

            observation_similarities[concept_id] = similarity
            if concept_id == self.active_state_id:
                observation_similarities["active"] = similarity

        if use_stable:
            self.monitor_all_state_buffered_similarity = [
                {}, observation_stats, self.buffered_window, observation_similarities]
            self.non_active_fingerprint_last_observation = self.ex
            self.fingerprint_changed_since_last_weight_calc = True
        else:
            self.monitor_all_state_active_similarity = [
                {}, observation_stats, {}, self.fingerprint_window, observation_similarities]
            self.last_observation = self.ex

    def update_all_fingerprints_stable(self):
        """ Calculate similarity between ALL states (even non active) on the buffered window.
        This is required for calculating feature weights, so we know which features differ between concepts.
        """
        active_stable_metainfo, active_stable_metainfo_flat = self.get_active_state().get_stable_metainfo(method=self.sim_measure, add_to_normalizer=True)
        self.evaluate_all_fingerprint_similarity(metainfo=active_stable_metainfo, metainfo_flat=active_stable_metainfo_flat,
                                                 use_stable=True, add_to_normalizer=True, update_na_record=True, record_likelihood=False)
    

    
    

    def check_warning_detector(self):
        """ Check if the warning detector has fired. Updates relevant statistics and triggers.
        If we have fired, recreates the detector in order to reset detection from the current position.
        """
        # If the warning detector fires we record the position
        # and reset. We take the most recent warning as the
        # start of out window, assuming this warning period
        # contains elements of the new state.
        # the change detector monitors likelihood, so we care about a DECREASE
        # in likelihood only
        if get_ADWIN_decrease(self.fingerprint_similarity_detector_warn):
        # When we use stdevs, look for increase

            self.warning_detected = False
            self.last_warning_point = self.ex
            self.fingerprint_similarity_detector_warn = make_detector(
                s=get_warning_sensitivity(self.get_current_sensitivity()), update_gap=self.active_head_monitor_gap)
        if not self.in_warning:
            self.last_warning_point = max(0, self.ex - 100)

    def check_found_change(self, detected_drift):
        """ Checks if we have found change. This either comes from the drift detector triggering, or if we are propagating a past detection forward.
        IMPORTANT: We DONT trigger on good change! If the similarity has increased!
        Propagation occurs if there was not enough data to make a decision at the previous step.
        We also check for manual triggering, for experimentation purposes (evaluating perfect drift.)
        """
        found_change = detected_drift or self.waiting_for_concept_data

        # Just for monitoring
        self.detected_drift = detected_drift or self.force_transition

        # Don't trigger on good changes
        # Or if we haven't tried this model enough to make a fingerprint
        if found_change:
            if self.get_active_state().fingerprint is None:
                found_change = False
            else:
                pass

        if self.manual_control or self.force_transition_only:
            found_change = self.force_transition
            self.trigger_transition_check = False
            self.trigger_attempt_transition_check = False

        return found_change

    def check_take_active_observation(self, found_change):
        take_active_observation = self.take_observations
        take_active_observation = take_active_observation and self.get_active_state().has_valid_head_window(self.min_window_size) and self.get_active_state().fingerprint is not None
        take_active_observation = take_active_observation and (
            self.ex - self.last_observation >= self.observation_gap or found_change)

        return take_active_observation

    def monitor_all_fingerprints_head(self):
        """ Calculate the similarity of ALL states to the current head window (the most recent data).
        This is just for monitoring purposes at the moment.
        """
        active_head_metainfo, active_head_metainfo_flat = self.get_active_state().get_head_metainfo(method=self.sim_measure, add_to_normalizer=True)

        self.evaluate_all_fingerprint_similarity(metainfo=active_head_metainfo, metainfo_flat=active_head_metainfo_flat,
                                                 use_stable=False, add_to_normalizer=False, update_na_record=False, record_likelihood=True)

    def attempt_transition(self, concept_probabilities):
        """ Attempt to transition states. Evaluates performance, and selects the best performing state to transition too.
        Can transition to the current state.
        Fails if not enough data to calculate best state, in which case the transition attempt is propagated to the next observation.
        """
        init_state = self.active_state_id
        self.in_warning = False
        # Find the inactive models most suitable for the current stream. Also return a shadow model
        # trained on the warning period.
        # If none of these have high accuracy, hold off the adaptation until we have the data.
        ranked_alternatives, use_shadow, shadow_model, can_find_good_model = self.rank_inactive_models_at_drift(concept_probabilities)


        # Handle manual transitions, for experimentation.
        if self.manual_control and self.force_transition_to is not None:
            if self.force_transition_to in self.state_repository:
                ranked_alternatives = [self.force_transition_to]
                use_shadow = False
                shadow_model = None
                can_find_good_model = True
            else:
                ranked_alternatives = []
                use_shadow = True
                shadow_state = ConceptState(self.force_transition_to, self.learner(
                ), self.fingerprint_update_gap,
                                  fingerprint_method=self.fingerprint_method, fingerprint_bins=self.fingerprint_bins,
                                  window_size=self.window_size, min_window_size = self.min_window_size, dirty_data_length=self.window_size, normalizer=self.normalizer, fingerprint_grace_period=self.fingerprint_grace_period, estimator_risk_level=self.state_estimator_risk, buffer_ratio=self.buffer_ratio, metainfo_method=self.sim_measure)
                shadow_model = shadow_state.classifier
                can_find_good_model = True

        # can_find_good_model is True if at least one model has a close similarity to normal performance.
        if not can_find_good_model:
            # If we did not have enough data to find any good concepts to
            # transition to, wait until we do.
            self.waiting_for_concept_data = True
            return init_state

        self.last_trigger_made_shadow = use_shadow

        return self.parse_apply_transition_from_alternatives(ranked_alternatives, use_shadow, shadow_model, can_find_good_model)

    def parse_apply_transition_from_alternatives(self, ranked_alternatives, use_shadow, shadow_model, can_find_good_model):
        """ Parse transition properties from alternatives and shadow. Perform the transition.
        """
        init_state = self.active_state_id
        # we need handle a proper transition if this is a transition to a different model.
        proper_transition = use_shadow or (
            ranked_alternatives[-1] != self.active_state_id)

        # If we determined the shadow is the best model, we mark it as a new state,
        # copy the model trained on the warning period across and set as the active state
        if use_shadow:
            shadow_id, shadow_state = self.make_state()
            shadow_state.classifier = shadow_model
            self.state_repository[shadow_id] = shadow_state
            transition_target_id = shadow_id
        else:
            transition_target_id = ranked_alternatives[-1]
            
        # If we transition to a different model, we need to perform setup.
        if proper_transition:
            init_state = self.perform_transition(transition_target_id, use_shadow)
        return init_state

    def check_state_probabilities(self, concept_probabilities):
        init_state = self.active_state_id
        ranked_alternatives, use_shadow, shadow_model, can_find_good_model = self.rank_inactive_models_posterior(concept_probabilities)
        if can_find_good_model:
            init_state = self.parse_apply_transition_from_alternatives(ranked_alternatives, use_shadow, shadow_model, can_find_good_model)
        return init_state
    def reset_background_model(self):
        current_sensitivity = self.get_current_sensitivity()
        self.background_state = None
        self.background_detector = make_detector(
            s=get_warning_sensitivity(current_sensitivity), update_gap=self.active_head_monitor_gap)
        
        # Reset the cache for the shadow state
        self.all_states_active_cache.pop(-1, 0)
        self.all_states_buffered_cache.pop(-1, 0)

    def reset_states_at_transition(self, from_id, to_id):
        for sid, state in self.state_repository.items():
            state.transition(from_id, to_id)

    def delete_state(self, delete_id, decrement_total=True):
        """ Delete a state from all transition matricies.
        Can either remove from total, or retain.
        Retaining implicitly reduces probability of other states.
        Returns what the old delete_id was transformed into, None.
        """
        self.state_repository.pop(delete_id, 0)
        self.deletions.append(delete_id)
        matricies = [
            self.concept_transitions_standard, 
            self.concept_transitions_warning, 
            self.concept_transitions_drift, 
            self.concept_transitions_change, 
        ]
        # Need to delete from transition matrix
        # First Delete row
        for matrix in matricies:
            matrix.pop(delete_id, 0)

        # Then transitions into this state
        # We delete the entry and subtract from total
        for matrix in matricies:
            for from_state in list(matrix.keys()):
                if from_state == delete_id or from_state == "total":
                    continue
                n_transitions = matrix[from_state].pop(delete_id, 0)
               
                # By not deleting, we are assigning some probablility to a 'deleted' state implicitly.
                if decrement_total:
                    matrix[from_state]['total'] -= n_transitions
        
        return None

    def get_prev_state_from_transitions(self, state_id):
        possible_previous_states = [(0, 'T')]
        for prev_id in self.state_repository.keys():
            if prev_id == state_id or prev_id not in self.concept_transitions_standard:
                continue
            if state_id in self.concept_transitions_standard[prev_id]:
                n_trans = self.concept_transitions_standard[prev_id][state_id]
                possible_previous_states.append((n_trans, prev_id))
        prev_state_count, prev_state = max(possible_previous_states, key=lambda x: x[0])
        prev_state_transition = f"T-{prev_state}"
        return (prev_state_count, prev_state, prev_state_transition)


    def delete_into_transition_state(self, delete_id):
        """ Merge a state into a generic transition state following the previous state.
        In many cases, i.e., gradual drift between state 0 -> state 1, there will be some
        transition state between them which has no meaning. We don't want to store all of these
        states, as they have no individual meaning, but we want to store the idea that there is
        some generic transition following state 0 and going to state 1. S0 -> T0 -> S1.

        So if we see S0->S2->S1 and decide S2 was only a transition, we delete it and merge its
        transitions into T-0.

        Returns what the delete_id turned into so we can update transitions.
        (e.g., returns T-0 rather than 2)
        """

        # First we need to work out what the previous state was, as we don't store this
        # (maybe we should).
        # We can find this as an entry in the transition matrix.
        # Should only be one, but we handle if there are multiple. We take the max num transitions
        # to be the previous state
        prev_state_count, prev_state, prev_state_transition = self.get_prev_state_from_transitions(delete_id)

        init_state = prev_state_transition
        self.delete_merge_state(delete_id, init_state)

        # We create a transition state for all states, so need to also
        # merge the one for the delete state.
        self.delete_merge_state(f"T-{delete_id}", init_state)

        return init_state
    def perform_transition(self, transition_target_id, is_new_state):
        """ Sets the active state and associated meta-info
        Resets detectors, buffer and both observation windows so clean data from the new concept can be collected.
        """
        # We reset drift detection as the new performance will not
        # be the same, and reset history for the new state.
        from_id = self.active_state_id
        to_id = transition_target_id
        #logging.info(f"Transition to {transition_target_id}")
        self.active_state_is_new = is_new_state
        self.active_state_id = transition_target_id
        self.occurences[to_id] = self.occurences.get(to_id, 0) + 1
        current_sensitivity = self.get_current_sensitivity()
        self.waiting_for_concept_data = False
        self.fingerprint_similarity_detector = make_detector(
            s=current_sensitivity, update_gap=self.active_head_monitor_gap)
        self.fingerprint_similarity_detector_warn = make_detector(
            s=get_warning_sensitivity(current_sensitivity), update_gap=self.active_head_monitor_gap)
        self.reset_background_model()
        try:
            self.get_active_state().stable_likelihood = False
        except Exception as e:
            print(f"{from_id}->{to_id}")
            print(self.merges)
            print(self.deactivated_states)
            raise e


        # Delete a state from the repository if it got swapped out before
        # it reached stability, and this is the first time we have used it.
        # We merge its transition data into the state which replaces it.
        # (could just delete instead, but since these are short states by
        # definition, we assume the next state should just replace it.)
        from_state = self.state_repository[from_id]
        init_state = from_id
        if from_state.recurrences < 2 and not from_state.stable_likelihood:
            if from_id in self.state_repository:
                self.deactivated_states[from_id] = self.state_repository[from_id]
            init_state = self.delete_into_transition_state(from_id)

        self.reset_states_at_transition(from_id, to_id)
        # Set triggers to future evaluation of this attempt
        self.observations_since_last_transition = 0
        return init_state

    def add_to_transition_matrix(self, init_s, curr_s, matrix, weight=1):
        matrix[init_s][curr_s] = matrix[init_s].get(curr_s, 0) + weight
        matrix[init_s]['total'] += weight
        return matrix

    def record_transition(self, detected_drift, initial_state):
        """ Record an observation to observation level transition from the initial_state to the current state.
        Depends on if a drift was detected, or if in warning which records are updated.
        """
        if initial_state in self.deletions or initial_state is None:
            if initial_state in self.merges:
                while initial_state in self.merges:
                    initial_state = self.merges[initial_state]
            else:
                raise ValueError("Recording transition from deleted state")
        current_state = self.active_state_id
        assert current_state != 'T'

        created_new_state = False
        for matrix in [self.concept_transitions_standard, self.concept_transitions_warning, self.concept_transitions_drift, self.concept_transitions_change]:
            if initial_state not in matrix:
                matrix[initial_state] = {}
                matrix[initial_state]['total'] = 0
            if current_state not in matrix:
                matrix[current_state] = {}
                matrix[current_state]['total'] = 0
                created_new_state = True
            assert len(self.concept_transitions_standard.keys()) == len(matrix.keys())

        # Add a prior going the other way, to simulate unconfidence in new state
        # Only do this once, when we are transitioning into a new state 
        # i.e., current_state is new
        # We don't want to add the prior to the generic 'transition' state.
        # created_new_state = self.concept_transitions_standard[current_state]['total'] == 0
        if 'T' not in str(initial_state):
            if created_new_state:
                self.add_to_transition_matrix(current_state, initial_state, self.concept_transitions_standard, weight=self.prev_state_prior)

        # We always add to the standard matrix
        self.add_to_transition_matrix(initial_state, current_state, self.concept_transitions_standard)

        # If we are in a warning period we add to the warning, and if we have detected drift we add to the warning and drift matricies
        if self.in_warning:
            self.add_to_transition_matrix(initial_state, current_state, self.concept_transitions_warning)
        elif detected_drift:
            self.add_to_transition_matrix(initial_state, current_state, self.concept_transitions_warning)
            self.add_to_transition_matrix(initial_state, current_state, self.concept_transitions_drift)

        if initial_state != current_state:
            self.add_to_transition_matrix(initial_state, current_state, self.concept_transitions_change)


    def get_transition_probabilities_from_state(self, initial_state, merge_with=None, change_matrix=False, empty_as_zero=False):
        transition_matrix = self.concept_transitions_standard if not change_matrix else self.concept_transitions_change
        if len(transition_matrix) == 0:
            return {initial_state: 1 if not empty_as_zero else 0}
        empty_state = {'total': 1, initial_state: 1} if not empty_as_zero else {'total': 0, initial_state: 0}
        trans_from_init = transition_matrix.get(initial_state, empty_state)
        if trans_from_init['total'] == 0:
            trans_from_init = empty_state
        trans_from_init_merge = transition_matrix.get(merge_with, {'total': 0})

        transition_probabilities = {}
        for next_state in transition_matrix:
            if next_state == 'total':
                continue
            trans = trans_from_init.get(next_state, 0)
            trans_merge = trans_from_init_merge.get(next_state, 0)
            n_trans = trans + trans_merge
            total = (trans_from_init['total']) + (trans_from_init_merge['total']) - (trans_from_init_merge.get(merge_with, 0))
            prob = (n_trans / total) if total > 0 else 0
            transition_probabilities[next_state] = prob
        return transition_probabilities
    
    def adjust_transition_probabilities(self, probabilities, min_prob, penalty_multiplier=1.0):
        """ Adjust a set of transition probabilities from a state so that 0 probability corresponds to min_prob,
        and the original [0, 1] region is between [min_prop, 1].
        Returned probabilities DO NOT sum to 1, and must be normalized if this is required (e.g., calculating multihop)
        """
        variable_region = 1 - min_prob
        scale_ratio = 1 / variable_region
        adjusted_probabilities = {}
        for k in probabilities:
            adjusted_probabilities[k] = ((probabilities[k]*penalty_multiplier) / scale_ratio) + min_prob
        return adjusted_probabilities, scale_ratio

    def normalize_transition_probabilities(self, probabilities):
        probability_sum = sum(probabilities.values())
        normalized_probabilities = {}
        for k in probabilities:
            normalized_probabilities[k] = probabilities[k] / probability_sum
        return normalized_probabilities

    def get_adjusted_probabilities_from_state(self, initial_state, min_prob, merge_with=None, change_matrix=False, normalize=False):
        probs = self.get_transition_probabilities_from_state(initial_state, merge_with, change_matrix)
        adjusted_probs, scale_ratio = self.adjust_transition_probabilities(probs, min_prob)
        if normalize:
            normalized_probs = self.normalize_transition_probabilities(adjusted_probs)
            return normalized_probs, scale_ratio
        return adjusted_probs, scale_ratio
    
    def get_multihop_transition_probs(self, initial_state, n_hops, penalty_multiplier):
        """ Calculate probability of being in a state after n_hops.
        Essentially a matrix exponentiation, so could speed up if this is an issue.
        Would require storing the matrix first.
        """
        current_hop_state_probs = {initial_state: 1}
        next_hop_state_probs = {}

        # We only use the change matrix for multihop, as we are interested in what state we would be in if we 
        # transitioned twice.
        normalized_cache = {}

        for i in range(n_hops):
            for next_si in [*self.state_repository.keys(), *[f"T-{k}" for k in self.state_repository.keys()]]:
                prob_transition_to = []
                for curr_si in current_hop_state_probs:
                    if curr_si in normalized_cache:
                        probs_from_current = normalized_cache[curr_si].get(next_si, 0)
                    else:
                        probs = self.get_transition_probabilities_from_state(curr_si, None, change_matrix=True, empty_as_zero=True)
                        normalized_cache[curr_si] = probs
                        probs_from_current = probs.get(next_si, 0)
                    prob_this_hop = current_hop_state_probs[curr_si]
                    prob_transition_to.append(prob_this_hop*probs_from_current)

                #update prob of next state to be sum of the probability of being in current state * prob of transitioning to the next state, over all current states
                next_hop_state_probs[next_si] = sum(prob_transition_to)
            current_hop_state_probs = self.resolve_transition_probabilities(next_hop_state_probs)
            next_hop_state_probs = {}
        return self.adjust_transition_probabilities(current_hop_state_probs, self.zero_prob_minimum, penalty_multiplier)[0]

    def resolve_transition_probabilities(self, probabilities):
        resolved_probs = {}
        for s in probabilities:
            state_prob = probabilities[s]
            if state_prob== 0 or 'T' not in str(s):
                resolved_probs[s] = resolved_probs.get(s, 0) + state_prob
            else:
                next_state_probs = self.get_transition_probabilities_from_state(s, None, change_matrix=True)
                for n in next_state_probs:
                    resolved_probs[n] = resolved_probs.get(n, 0) + (next_state_probs[n] * state_prob)
        return resolved_probs

    def get_observation_priors(self, initial_state, detected_drift, merge_from_with=None):
        standard_prob, scale_ratio = self.get_adjusted_probabilities_from_state(initial_state, self.zero_prob_minimum, merge_with=merge_from_with, change_matrix=False)
        change_prob, scale_ratio = self.get_adjusted_probabilities_from_state(initial_state, self.zero_prob_minimum, merge_with=merge_from_with, change_matrix=True)
        
        # Normalize priors by the maximum possible prior given smoothing settings.
        # This seems to make (shorter?) data sets better, but some worse.
        # TODO: Try to work out why? Possible a parameters setting:
        # Make make the system more sensitive, so reduce swap risk?
        # Or maybe makes priors change weirdly? Though normalization should make this better
        # not worse.
        if detected_drift:
            return change_prob
        return standard_prob
    
    def get_transition_priors(self, initial_state, n_hops, penalty_multiplier):
        standard_prob = self.get_multihop_transition_probs(initial_state, n_hops=n_hops, penalty_multiplier=penalty_multiplier)
        return standard_prob
    
    def get_prior_max(self, initial_state, in_warning, detected_drift, merge_state_with = None):
        initial_state_recurrences = self.occurences[initial_state]
        merge_state_recurrences = max(self.occurences.get(merge_state_with, 1), 1)
        matrix = self.concept_transitions_standard
        if detected_drift:
            matrix = self.concept_transitions_change
        elif in_warning:
            matrix = self.concept_transitions_warning
        
        num_concepts = len(matrix.keys())
        k = len(self.state_repository)
        assert num_concepts == len(self.concept_transitions_standard.keys())
        if num_concepts == 0:
            return 1.0
        if initial_state in matrix:
            total_m = max([c['total'] / max(self.occurences[k], 1) for k,c in matrix.items()])
            merge_transitions = matrix.get(merge_state_with, {'total': 0})
            total_merge = (merge_transitions['total'] / merge_state_recurrences) - (merge_transitions.get(merge_state_with, 0) / merge_state_recurrences)
            total_c = (matrix[initial_state]['total'] / initial_state_recurrences) + total_merge
            scaling_factor = (self.smoothing_factor + total_m) / ((self.smoothing_factor * num_concepts) + total_c)
            adjustment_factor = ((self.smoothing_factor * k) + total_c) / ((self.smoothing_factor * k) + total_m)
            return scaling_factor * adjustment_factor
        else:
            total_m = max([c['total'] for c in matrix.values()])
            merge_transitions = matrix.get(merge_state_with, {'total': 0})
            total_merge = merge_transitions['total'] - merge_transitions.get(merge_state_with, 0)
            total_c = 0 + total_merge
            scaling_factor = (self.smoothing_factor + total_m) / ((self.smoothing_factor * num_concepts) + total_c)
            adjustment_factor = ((self.smoothing_factor * k) + total_c) / ((self.smoothing_factor * k) + total_m)
            return scaling_factor * adjustment_factor


    def get_state_posteriors(self, initial_state, detected_drift):
        np.seterr(all='ignore')
        state_posteriors = {}
        state_priors = {}
        self.concept_priors_2h = {}

        merge_from_with = None
        init_state = self.state_repository[initial_state]
        if init_state.recurrences < 2 and not init_state.stable_likelihood:
            prev_state_count, prev_state, prev_state_transition = self.get_prev_state_from_transitions(initial_state)
            # Require that the previous state was a real state, not another transition state.
            if prev_state != 'T':
                merge_from_with = prev_state_transition
        adjusted_priors = self.get_observation_priors(initial_state, detected_drift, merge_from_with=merge_from_with)
        single_hop_prior = self.get_transition_priors(initial_state if not merge_from_with else merge_from_with,n_hops=1,penalty_multiplier=self.multihop_penalty)
        two_hop_prior = self.get_transition_priors(initial_state if not merge_from_with else merge_from_with,n_hops=2,penalty_multiplier=self.multihop_penalty**2)
        for state_id in self.state_repository:

            state_prior = adjusted_priors[state_id]
            state_prior_1h = single_hop_prior[state_id]
            state_prior_2h = two_hop_prior[state_id]
            self.concept_priors_1h[state_id] =state_prior_1h
            self.concept_priors_2h[state_id] =state_prior_2h

            state_prior_mh = max(state_prior, state_prior_1h, state_prior_2h)
            state_prior_normalized = state_prior_mh

            state_likelihood = max(self.concept_likelihoods.get(state_id, 0), self.minimum_concept_likelihood)

            if not state_prior_normalized or state_prior_normalized <= 0:
                print(f"Bad state prior: {state_prior}, {state_prior_1h}, {state_prior_2h}")
                state_prior_normalized = self.zero_prob_minimum
            state_posteriors[state_id] = math.exp(math.log(state_likelihood) + math.log(state_prior_normalized))
            state_priors[state_id] = state_prior_normalized

        self.concept_posteriors.update(state_posteriors)
        self.concept_priors.update(state_priors)

        # Add current estimate to history
        for concept_id, concept_posterior in state_posteriors.items():
            state = self.state_repository[concept_id] if concept_id >= 0 else self.background_state
            state.observe_posterior(concept_posterior)

            state_likelihood = max(self.concept_likelihoods.get(concept_id, 0), self.minimum_concept_likelihood)
            state.observe_likelihood(state_likelihood)

            if not state.stable_likelihood:
                if state.get_estimated_likelihood() > self.bypass_grace_period_threshold or state.seen_since_transition >= self.state_grace_period:
                    state.stable_likelihood = True
        if -1 in self.concept_likelihoods and self.background_state:
            if self.background_state.has_valid_stable_window(self.min_window_size):
                background_likelihood = self.concept_likelihoods[-1]
                self.background_state.observe_likelihood(background_likelihood)
                background_posterior = self.get_background_posterior()
                self.background_state.observe_posterior(background_posterior)

        return state_posteriors


    def apply_state_correlations(self):
        if not self.correlation_merge:
            return None
        merges = {}
        for i, (sid, s) in enumerate(self.state_repository.items()):
            compare_states = list(self.state_repository.values())[i+1:]
            s.add_correlations(compare_states)

            # We consider merging if both states are stable, with more than 1 ocurrence.
            state_correlations = s.get_correlations()
            for state_id, correlation, seen in state_correlations:
                if state_id not in self.state_repository:
                    # remove the old state from the correlations list
                    s.correlations.pop(state_id)
                    continue
                compare_state = self.state_repository[state_id]
                if correlation > self.merge_threshold and max(compare_state.recurrences, s.recurrences) > 1 and seen > self.state_grace_period:
                    merged_states = merges.setdefault(sid, [])
                    merged_states.append(state_id)
        
        deactivated_states = set()
        for sid in merges:
            keep_candidates = [sid, *merges[sid]]
            # We keep the candiate with the max seen value. If there is a tie, we keep the model with the lower ID.
            selected_candidate = max([(self.state_repository[i].seen, i) for i in keep_candidates], key=lambda x: (x[0], -1 * x[1]))
            for i in keep_candidates:
                if i != selected_candidate[1]:
                    deactivated_states.add((i, selected_candidate[1]))
                    self.deactivated_states[i] = self.state_repository[i]
                    self.merges[i] = selected_candidate[1]
        
        merge_active_state = None
        for i, j in deactivated_states:
            if i == self.active_state_id:
                merge_active_state = i
            else:
                self.delete_merge_state(i, j)
                self.delete_merge_state(f"T-{i}", f"T-{j}")
        
        return merge_active_state
        
        

        
    def delete_merge_state(self, merge_from, merge_into):
        # Handle merging transition states, which are
        # not in the repository
        if 'T' not in str(merge_from):
            self.state_repository.pop(merge_from, 0)
            self.deletions.append(merge_from)

        matricies = [
            self.concept_transitions_standard, 
            self.concept_transitions_warning, 
            self.concept_transitions_drift, 
            self.concept_transitions_change, 
        ]

        # Need to merge from transition matrix
        # First merge transitions from merge_from into those from
        for matrix in matricies:
            transitions_from = matrix.pop(merge_from, {})
            new_transitions = matrix.setdefault(merge_into, {'total': 0})
            for to_id in transitions_from:
                if to_id == "total":
                    continue
                n_trans_merge = transitions_from[to_id]
                n_trans_into = new_transitions.setdefault(to_id, 0)
                new_transitions[to_id] = n_trans_merge + n_trans_into
                new_transitions['total'] += n_trans_merge

        # Then merge transitions into this state
        # We delete the entry and add to entry for merge_into
        for matrix in matricies:
            for from_state in list(matrix.keys()):
                n_trans_merge = matrix[from_state].pop(merge_from, 0)
                n_trans_to = matrix[from_state].setdefault(merge_into, 0)
                matrix[from_state][merge_into] = n_trans_merge + n_trans_to
        
        # Fix change transition matrix
        # The merge may add in a self-transition, which we don't 
        # want in this matrix.
        self_transitions = self.concept_transitions_change[merge_into].pop(merge_into, 0)
        self.concept_transitions_change[merge_into]['total'] = self.concept_transitions_change[merge_into].get('total', 0) - self_transitions

        merge_from_occurences = self.occurences.pop(merge_from, 0)
        self.occurences[merge_into] = self.occurences.get(merge_into, 0) + merge_from_occurences



    def _partial_fit(self, X, y, sample_weight, masked=False):

        # init defaults for trackers
        found_change = False
        self.detected_drift = False
        current_sensitivity = self.get_current_sensitivity()
        self.warning_detected = False
        self.observations_since_last_transition += 1
        initial_state = self.active_state_id
        self.check_background_state_initialized()

        # get_temporal_x, and get_imputed_label are to deal with masked
        # values where we don't see true label.
        # As the functions are now, they don't do anything extra.
        # But could be extended to reuse last made prediction as
        # the label for example.
        temporal_X = self.get_temporal_x(X)

        # Older states can wait for longer before receiving new data
        # Since we assume stationary concepts, they should work well
        # even using data from only the previous concept.
        # But we also want to start incorporating data sooner that just raw age
        # so take mean of age and time since transition to incorporate both.
        state_age = np.mean((self.get_active_state().seen, self.get_active_state().seen_since_transition)) + 1
        buffer_len = math.floor(state_age * self.buffer_ratio)
        # buffer_len = math.floor(self.observations_since_last_transition * self.buffer_ratio)
        #logging.info(f"buffer age at {self.ex}: {buffer_len}")
        # Show observation to foreground and background states, to fit on and label
        foreground_p = self.get_active_state().observe(temporal_X, y, self.ex, buffer_len, label=True, fit=True, sample_weight=sample_weight, classes=self.classes)
        
        # The background state is periodically reset, so needs its own buffer.
        # This means that it must maintain different head and stable windows
        background_state_age = np.mean((self.background_state.seen, self.background_state.seen_since_transition)) + 1
        # background_state_age = np.mean((self.background_state.seen))
        background_buffer_len = math.floor(background_state_age * self.buffer_ratio)
        #logging.info(f"bg buffer age at {self.ex}: {background_buffer_len}")
        background_p = self.background_state.observe(temporal_X, y, self.ex, background_buffer_len, label=True, fit=True, sample_weight=sample_weight, classes=self.classes)
        
        # Show observation to inactive states to label
        # This is just to measure behaviour, so we do NOT fit
        for state in [s for sid, s in self.state_repository.items() if sid != self.active_state_id]:
            state.background_observe(temporal_X, y, self.ex, buffer_len, label=True)

        # Setup global vars handling the current partial fit
        self.reset_partial_fit_settings()
        
        # We add stats to the normalizer to initialize it
        self.init_normalizer()
        
        take_measurement = self.check_take_measurement()
        if take_measurement:
            # Take measurement of foreground state
            if is_valid_fingerprint(self.get_active_state().fingerprint):
                active_head_metainfo, active_head_metainfo_flat = self.get_active_state().get_head_metainfo(method=self.sim_measure, add_to_normalizer=True)
                
                similarity = self.get_similarity_to_active_state(
                    current_metainfo=active_head_metainfo, flat_nonorm_current_metainfo=active_head_metainfo_flat)
                
                active_state_likelihood = self.get_active_state().get_sim_observation_probability(
                    similarity, self.get_similarity, self.similarity_min_stdev, self.similarity_max_stdev)

                self.update_similarity_change_detection(similarity, active_state_likelihood)
                # For # logging
                self.monitor_active_state_active_similarity = similarity
            
            #Take measurement of background state
            if is_valid_fingerprint(self.background_state.fingerprint):
                background_head_metainfo, background_head_metainfo_flat = self.background_state.get_head_metainfo(method=self.sim_measure, add_to_normalizer=True)
                bg_similarity = self.get_similarity_to_background_state(
                    current_metainfo=background_head_metainfo, flat_nonorm_current_metainfo=background_head_metainfo_flat)
                bg_state_likelihood = self.background_state.get_sim_observation_probability(
                    bg_similarity, self.get_similarity, self.similarity_min_stdev, self.similarity_max_stdev)
                self.update_bg_similarity_change_detection(bg_similarity, bg_state_likelihood)
                #logging.info(f"bg state active sim: {bg_similarity}")
                #logging.info(f"Active state active likelihood: {bg_state_likelihood}")
            self.last_active_head_monitor = self.ex

        do_update_fingerprint = self.check_do_update_fingerprint()
        if do_update_fingerprint:
            # We update fingerprints using the stable window
            # to make it less likely we are incorperating
            # a fingerprint from a transition region or a new
            # concept.
            # Downside is this is slightly behind, but with the
            # assumption that a concept is stationairy, we
            # should be able to handle slightly out of date
            # fingerprints.
            #logging.info(f"updating active state {self.get_active_state().id} at {self.ex}")
            similarity, metainfo, metainfo_flat = self.record_state_stable_similarity(self.get_active_state())
            if is_valid_fingerprint(self.get_active_state().fingerprint):
                if similarity is not None:
                    self.monitor_active_state_buffered_similarity = similarity
            
            # Only update fingerprint with metainfo from a valid stable window
            if self.get_active_state().has_valid_stable_window(self.min_window_size):
                self.get_active_state().update_fingerprint(
                    metainfo, self.ex, normalizer=self.normalizer)
            
            if self.background_state.has_valid_stable_window(self.min_window_size):
                similarity, metainfo, metainfo_flat = self.record_state_stable_similarity(self.background_state)
                self.background_state.update_fingerprint(metainfo, self.ex, normalizer=self.normalizer)

            self.fingerprint_changed_since_last_weight_calc = True
            # If we have seen a few fingerprints, so as to reduce variability,
            # we record this as a running 'normal' similarity.

        do_update_non_active_fingerprint = self.check_do_update_non_active_fingerprint()
        if do_update_non_active_fingerprint:
            self.update_all_fingerprints_stable()

        # Test similarity of each state to the head of the stream. This calculates likelihoods for each state.
        take_active_observation = self.check_take_active_observation(
            found_change)
        if take_active_observation:
            self.monitor_all_fingerprints_head()
            
        # Check background detector, if triggered reset
        try:
            # the background detector looks at likelihood, so we want to 
            # reset on a DECREASE in likelihood (as this signals a bad change)
            if get_ADWIN_decrease(self.background_detector):
                self.reset_background_model()
        except:
            self.reset_background_model()

        self.check_warning_detector()

        # Check main detector
        # The drift detector monitors likelihood, so we care about a DECREASE in likelihood only.
        detected_drift = get_ADWIN_decrease(self.fingerprint_similarity_detector)
        # WHen we use stdevs want to look for increase

        # If the main state trigers, or a previous detection is propagating due to lack of data,
        # trigger a change
        found_change = self.check_found_change(detected_drift)

        # If we detected change and didn't monitor likelihoods this step,
        # monitor now so we have the most up to date info
        if found_change and not take_active_observation:
            assert self.get_active_state().has_valid_head_window(self.min_window_size)
            self.monitor_all_fingerprints_head()

        concept_probabilities = self.get_state_posteriors(initial_state, detected_drift)

        # add state correlations to monitoring.
        # Merges correlated states.
        # If we try to merge the active state, we need to handle it.
        merge_active_state_id = self.apply_state_correlations()

        # We have three reasons for attempting a check: We detected a drift, we want to evaluate the current state, or we have a propagating transition attempt.
        # If we have detected a drift, we consider state similarity directly.
        # Otherwise, we behave probabalistically.
        #<TODO> combine these approaches cleanly
        if found_change or self.trigger_transition_check or self.trigger_attempt_transition_check:
            initial_state = self.attempt_transition(concept_probabilities)
        else:
            initial_state = self.check_state_probabilities(concept_probabilities)
        
        # Check if we still need to handle merging the active state
        # We transition to the merge target then pop the active state.
        # If we did handle it, with some other drift etc, we can just pop it off.
        if self.active_state_id == merge_active_state_id:
            merge_target = self.merges[merge_active_state_id]
            while merge_target in self.merges:
                merge_target = self.merges[merge_target]
            initial_state = self.perform_transition(merge_target, is_new_state=False)
            # Need to check, the merge_state may have been deleted 
            # if it was too small
            if merge_active_state_id in self.state_repository:
                self.delete_merge_state(merge_active_state_id, merge_target)
                self.delete_merge_state(f"T-{merge_active_state_id}", f"T-{merge_target}")
        elif merge_active_state_id is not None and merge_active_state_id in self.state_repository:
            self.delete_merge_state(merge_active_state_id, self.active_state_id)
            self.delete_merge_state(f"T-{merge_active_state_id}", f"T-{self.active_state_id}")


        self.record_transition(detected_drift, initial_state)

        # Set exposed info for evaluation.
        self.active_state = self.active_state_id
        self.found_change = found_change
        self.states = self.state_repository
        self.current_sensitivity = current_sensitivity


    def rank_inactive_models_at_drift(self, concept_log_posteriors):
        #logging.info(
        #    f"Ranking inactive models on at drift at {self.ex}")
        if self.get_active_state().fingerprint is None:
            return None, None, None, False


        recent_shadow_model = self.learner()

        state_probabilities = []
        filter_states = set()
        concept_likelihoods = {k:v for k,v in self.concept_likelihoods.items()}
        for concept_id in [*self.state_repository.keys()]:
            state = self.state_repository[concept_id]

            if state.fingerprint is None:
                continue
            
            probability = concept_likelihoods[concept_id]

            #logging.info(
            #    f"State {concept_id} performance: likelihood {probability}")
            #logging.info(
            #    f"State {concept_id} performance: posterior {concept_posteriors[concept_id]}")
            #logging.info(
            #    f"State {concept_id} performance: likelihood history {self.concept_likelihoods_history.get(concept_id, [])}")
            #logging.info(
            #    f"State {concept_id} performance: posterior history {self.concept_posteriors_history.get(concept_id, [])}")
            
            #logging.info(
            #    f"State {concept_id} performance adwin: posterior {state.get_estimated_posterior()}")
            #logging.info(
            #    f"State {concept_id} performance adwin: likelihood {state.get_estimated_likelihood()}")
                
            accept_state = True
            # A likelihood of 0.03 once it is stable corresponds to being at least ~3 st.dev from the mean
            if probability < self.min_drift_likelihood_threshold and state.stable_likelihood:
                accept_state = False

            # logging.info(f"{state.id}, {probability}, {accept_state}")
            if not accept_state:
                filter_states.add(concept_id)
                #logging.info(f"State {concept_id} filtered")
            state_probabilities.append(
                (concept_id, probability))

        # State suitability in sorted (ascending) order
        state_probabilities.sort(key=lambda x: x[1])
        use_shadow = False
        # If we have filtered all current states, we need to create a new one
        if len(filter_states) == len(state_probabilities):
            use_shadow = True
            #logging.info("No accepted states")
        return [[x[0] for x in state_probabilities if x[0] not in filter_states], use_shadow, recent_shadow_model, True]

    def calc_background_prior(self, active_state_prior=None):
        if active_state_prior is None:
            active_state_prior = self.concept_priors.get(self.active_state_id, 1.0)

        # Here we decide our prior for the background state, our idea of how likely it should be.
        # We start with a prior proportionate to the current active state. By default this is 0.3,
        # So the chance of shifting to a new state is 30% of the chance of staying in the current state.
        # Then we multiply by the inverse number of concepts. 
        # This adds the idea that new concepts get rarer as the stream progresses.
        # Intuitively, this is because if we have many concepts stored, it is more likely we reuse
        # one rather than shift to a totally new concept.
        # Does this limit our ability to handle many new concepts? This depends on the parameters, and 
        # manamegement of the size of the repository. 
        # The falloff is determined by the background_state_prior_multipler so this is adjustable.
        #  If old concepts are deleted from the repository, then it is also easier to add new concepts.
        #  Thus the prior of adding a new concept never falls too low.
        num_concepts = len(self.state_repository.keys())
        new_state_falloff = 1 / num_concepts
        return (self.background_state_prior_multiplier * active_state_prior) * new_state_falloff

    def get_background_posterior(self, active_state_prior=None):
        """ Get the posterior probability of the background state
        We can set the prior based on our prior knowledge of a new state being created, e.g., using volatility
        """
        return self.concept_likelihoods[-1] * self.calc_background_prior(active_state_prior)

    # For monitoring
    # def get_background_posterior_history(self):
    #     """ Get the posterior probability history of the background state
    #     We can set the prior based on our prior knowledge of a new state being created, e.g., using volatility
    #     """
    #     return [p*self.calc_background_prior() for p in self.concept_likelihoods_history.get(-1, [0])]

    # def get_background_likelihood_history(self):
    #     """ Get the posterior probability history of the background state
    #     We can set the prior based on our prior knowledge of a new state being created, e.g., using volatility
    #     """
    #     return [p for p in self.concept_likelihoods_history.get(-1, [0])]

    def hoeffding_bound_and_above(self, W0, W1, delta=0.05):
        """ Check that W1 is sufficiently different to W0 for its size,
        AND that its mean is greater than W0
        """
        n0 = len(W0)
        n1 = len(W1)
        u0 = np.mean(W0)
        u1 = np.mean(W1)
        m = 1/ ((1/n0) + (1/n1))
        delta_p = delta/ (n0 + n1)
        e_cut_squared = (1/(2*m))* math.log(4/delta_p)

        diff = math.pow(u0 - u1, 2)
        #logging.info(f"diff squared: {diff}, e_cut_squared: {e_cut_squared}, u0: {u0} u1: {u1}, above: {u1 > u0}")
        return diff >= e_cut_squared and u1 > u0

    def hoeffding_bound2_and_above(self, W0, W1, delta=0.05):
        """ Check that W1 is sufficiently different to W0 for its size,
        AND that its mean is greater than W0
        """
        n0 = len(W0) * 25
        n1 = len(W1) * 25
        u0 = np.mean(W0)
        u1 = np.mean(W1)
        diff = abs(u0 - u1)
        try:
            sdtev = np.var(W0)
            m = 1/ ((1/n0) + (1/n1))
            delta_p = delta/ math.log((n0 + n1))
            e_cut = math.sqrt((2/(m))*sdtev*math.log(2/delta_p)) + (2/(3*m))*math.log(2/delta_p)

            #logging.info(f"diff squared: {diff}, e_cut: {e_cut}, u0: {u0} u1: {u1}, above: {u1 > u0}")
            return diff >= e_cut and u1 > u0
        except Exception as e:
            return False

    def hoeffding_bound_2(self, n0, n1, u0, u1, v0, v1, delta=0.05):
        diff = abs(u0 - u1)
        try:
            m = 1/ ((1/n0) + (1/n1))
            delta_p = delta/ math.log((n0 + n1))
            e_cut = math.sqrt((2/(m))*v0*math.log(2/delta_p)) + (2/(3*m))*math.log(2/delta_p)

            #logging.info(f"diff vals squared: {diff}, e_cut: {e_cut}, u0: {u0} u1: {u1}, above: {u1 > u0}")
            return diff >= e_cut
        except Exception as e:
            return False
    def hoeffding_bound2_and_above_vals(self, n0, n1, u0, u1, v0, v1, delta=0.05):
        """ Check that W1 is sufficiently different to W0 for its size,
        AND that its mean is greater than W0
        """
        return self.hoeffding_bound_2(n0, n1, u0, u1, v0, v1, delta) and u1 > u0
        diff = abs(u0 - u1)
        try:
            m = 1/ ((1/n0) + (1/n1))
            delta_p = delta/ math.log((n0 + n1))
            e_cut = math.sqrt((2/(m))*v0*math.log(2/delta_p)) + (2/(3*m))*math.log(2/delta_p)

            #logging.info(f"diff vals squared: {diff}, e_cut: {e_cut}, u0: {u0} u1: {u1}, above: {u1 > u0}")
            return diff >= e_cut and u1 > u0
        except Exception as e:
            return False
    
    def passes_any_hbound(self, W0, W1):
        """ Checks if any set of the most recent elements of W0 and W1
        pass the hoeffding bound checking if W1 is different and above W0
        """
        passes = False
        for n in range(0, min(len(W0), len(W1)), 10):
            passes |= self.hoeffding_bound2_and_above(W0[n:], W1[n:])
        return passes

    def rank_inactive_models_posterior(self, concept_log_posteriors):
        #logging.info(
        #    f"Ranking inactive models bsed on posterior probability on current stream at {self.ex}, seen {self.get_active_state().seen_since_transition}")

        state_probabilities = []
        filter_states = set()
        for concept_id in [*self.state_repository.keys(), -1]:
            if concept_id != -1:
                state = self.state_repository[concept_id]
            else:
                state = self.background_state
            if not state:
                continue
            if state.fingerprint is None:
                #logging.info(f"state {concept_id} fingerprint is None, skipped")
                continue
            
            # Logging
            # if concept_id != -1:
            #     probability = self.get_state_posterior(concept_id)
            #     probability_history = [v for v in self.concept_posteriors_history.get(concept_id, [])]
            # else:
            #     probability = self.get_background_posterior()
            #     probability_history = [v for v in self.get_background_posterior_history()]
            #logging.info(
            #    f"State {concept_id} performance: posterior {probability}")
            #logging.info(
            #    f"State {concept_id} performance history: posterior {probability_history}")
            #logging.info(
            #    f"State {concept_id} performance smoothed: posterior {self.concept_posteriors_smoothed.get(concept_id, -1)}")
            #logging.info(
            #    f"State {concept_id} performance adwin: posterior {state.get_estimated_posterior()}")
            # End Logging

            accept_state = True

            # # We want to compare posterior probabilities, unless
            # compare_posteriors = concept_id != -1

            # We always accept the current state
            # Otherwise, we check that the state probability is sufficiently
            # different to the probability of the active state, and is also above
            # We use the hoeffding bound, to give a threshold for 'sufficient' difference
            # given the current sample size. We test all possible cut points.
            if concept_id != self.active_state_id:
                
                u0 = self.get_active_state().get_estimated_posterior()
                u1 = state.get_estimated_posterior()
                n0 = self.get_active_state().adwin_posterior_estimator._width
                n1 = state.adwin_posterior_estimator._width
                # Don't bother checking if either state has seen 0, or if the tested state is worse
                if n1 == 0 or n0 == 0 or u0 >= u1:
                    accept_state = False
                else:
                    v0 = self.get_active_state().adwin_posterior_estimator.variance
                    v1 = state.adwin_posterior_estimator.variance
                    accept_state = self.hoeffding_bound2_and_above_vals(n0=n0, n1=n1, u0=u0, u1=u1, v0=v0, v1=v1, delta=self.state_estimator_swap_risk)
                #logging.info(f"State {state.id} accepted with hbound check {accept_state}")
            else:
                # If the active state has reached a stable performance at least once,
                # And has dropped below a likelihood threshold, we consider a transision to have occured.
                # min_likelihood = self.min_drift_likelihood_threshold
                min_likelihood = self.min_estimated_posterior_threshold
                if all([
                state.get_estimated_likelihood() < min_likelihood,
                state.stable_likelihood]):
                    accept_state = False
                #logging.info(f"State {state.id} accepted with min check {accept_state}")
            # logging.info(f"{state.id}, {state.get_estimated_posterior()}, {state.get_estimated_likelihood()}, {accept_state}")
            
            # If using MAP, we just take the most probable state
            # so don't filter with a statistical test
            if self.MAP_selection:
                accept_state = True

            if not accept_state:
                filter_states.add(concept_id)
                #logging.info(f"State {concept_id} filtered")
            state_probabilities.append(
                (concept_id, state.get_estimated_posterior(), state.get_estimated_likelihood()))

        # State suitability in sorted (ascending) order
        state_probabilities.sort(key=lambda x: x[1])
        filtered_state_probabilities = [x for x in state_probabilities if x[0] not in filter_states]

        use_shadow = False
        use_background = False
        if len(filter_states) == len(state_probabilities):
            use_shadow = True
            #logging.info("No accepted states")
        else:
            if filtered_state_probabilities[-1][0] == -1 and (-1 not in filter_states):
                use_background = True

        shadow_model = None
        if use_background:
            shadow_model = self.background_state.classifier

        if use_shadow:
            shadow_model = self.learner()

        # Grace period, we don't want to immediately replace a new state
        # Unless we have a very probable alternative
        if not self.get_active_state().stable_likelihood:
            max_likelihood = 0
            if not (use_shadow or use_background):
                max_id, max_prob, max_likelihood = filtered_state_probabilities[-1]
            elif use_background:
                max_id, max_prob, max_likelihood = -1, self.background_state.get_estimated_posterior(), self.background_state.get_estimated_likelihood()
            if max_likelihood < self.bypass_grace_period_threshold:
                if self.get_active_state().fingerprint is None or not self.get_active_state().stable_likelihood:
                    return None, None, None, False
                if self.get_active_state().seen_since_transition < self.window_size * 3:
                    return None, None, None, False

        ret_val = [[x[0] for x in filtered_state_probabilities], use_shadow or use_background, shadow_model, True]
        #logging.info(f"{ret_val}")
        #logging.info(f"{filter_states}")
        return ret_val

    def get_current_sensitivity(self):
        return self.base_sensitivity


def is_valid_fingerprint(fp):
    return fp is not None and fp.seen > 5

def get_warning_sensitivity(s):
    return s * 2

def get_ADWIN_change_direction(adwin_detector):
    # Sometimes throws an error 'underflow in variance'
    # TODO: work out a fix
    try:
        e_before = adwin_detector.estimation
        drift_detected = adwin_detector.detected_change()
        e_after = adwin_detector.estimation
    except:
        drift_detected = False

    if not drift_detected or e_before == e_after:
        direction = 0
    elif e_before < e_after:
        direction = 1
    else:
        direction = -1
    
    return direction

def get_ADWIN_increase(adwin_detector):
    """ Returns true if an adwin detector has detected change, and the estimate
    after dropping the old data is higher.
    Calls detected_change(), so gets the detector to process changes.
    """
    return get_ADWIN_change_direction(adwin_detector) == 1

def get_ADWIN_decrease(adwin_detector):
    """ Returns true if an adwin detector has detected change, and the estimate
    after dropping the old data is lower.
    Calls detected_change(), so gets the detector to process changes.
    """
    return get_ADWIN_change_direction(adwin_detector) == -1
