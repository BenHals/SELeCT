import warnings
import numpy as np
from skmultiflow.utils import check_random_state, get_dimensions



warnings.filterwarnings('ignore')


class ConceptState:
    """ Represents a data stream concept.
    Maintains current descriptive information, including a classifier, fingerprint, evolution state and recent performance in stationary conditions.
    """

    def __init__(self, id, learner):
        self.id = id
        self.classifier = learner
        self.current_evolution = self.classifier.evolution
    
    def predict(self, X):
        return self.classifier.predict(X)
    
    def partial_fit(self, X, y, sample_weight, classes):
        self.classifier.partial_fit(
            np.asarray([X]),
            np.asarray([y]),
            sample_weight=np.asarray([sample_weight]),
            classes=classes
        )
        classifier_evolved = self.classifier.evolution > self.current_evolution
        self.current_evolution = self.classifier.evolution

class BoundClassifier:
    def __init__(self,
                 suppress=False,
                 learner=None,
                 poisson=6,
                 bounds="lower",
                 window_size=100,
                 init_concept_id=0):

        if learner is None:
            raise ValueError('Need a learner')

        self.learner = learner

        # suppress debug info
        self.suppress = suppress

        # rand_weights is if a strategy is setting sample
        # weights for training
        self.rand_weights = poisson > 1

        # poisson is the strength of sample weighting
        # based on leverage bagging
        self.poisson = poisson

        self.lower_bound = bounds == "lower"
        self.upper_bound = bounds == "upper"
        self.middle_bound = bounds == "middle"

        if not (self.lower_bound or self.upper_bound or self.middle_bound):
            raise ValueError("Need to set bounds to lower or upper")
        
        self.window_size = window_size

        self.known_drift_point = -1 * (window_size + 1)

        # init the current number of states
        self.max_state_id = init_concept_id

        # init randomness
        self.random_state = None
        self._random_state = check_random_state(self.random_state)

        self.ex = -1
        self.classes = None
        self._train_weight_seen_by_model = 0

        self.active_state_is_new = True

        # init data which is exposed to evaluators
        self.found_change = False
        self.num_states = 1
        self.active_state = self.max_state_id
        self.states = []

        self.detected_drift = False
        self.deletions = []

        # track the last predicted label
        self.last_label = 0

        self.concept_transitions_standard = {}
        self.concept_transitions_warning = {}
        self.concept_transitions_drift = {}

        # set up repository
        self.state_repository = {}
        # init_id = self.max_state_id
        # self.max_state_id += 1
        init_state_id, init_state = self.make_state(init_concept_id)       
        self.state_repository[init_state_id] = init_state
        self.active_state_id = init_state_id

        self.observations_since_last_transition = 0

        self.force_transition_to = None
        self.force_transition = False
        
        self.merges = {}
        self.deactivated_states = {}

        self.fingerprint_type = ConceptState

        self.in_warning = False

        self.current_evolution = init_state.current_evolution

        self.monitor_all_state_active_similarity = None
        self.monitor_all_state_buffered_similarity = None
        self.monitor_active_state_active_similarity = None
        self.monitor_active_state_buffered_similarity = None
        self.monitor_feature_selection_weights = None
        self.concept_likelihoods = {}

        

    def construct_state_object(self, id=-1):
        return ConceptState(id, self.learner())

    def make_state(self, new_id):
        state = self.construct_state_object(new_id)
        self.max_state_id = max(self.max_state_id, new_id + 1)
        return new_id, state



    def get_active_state(self):
        return self.state_repository[self.active_state_id]

    def reset(self):
        pass

    def get_temporal_x(self, X):
        return np.concatenate([X], axis=None)

    def predict(self, X):
        """
        Predict using the model of the currently active state.
        """
        temporal_X = self.get_temporal_x(X)

        return self.get_active_state().predict([temporal_X])

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

    def perform_drift_detection_for_bounds(self):
        if self.lower_bound or not self.force_transition:
            return None
        
        if self.force_transition_to is None:
            raise ValueError("Upper bounds, but no force_transition_to for perfect re-identification")
        return self.force_transition_to

    def perform_transition(self, transition_target_id):
        """ Sets the active state and associated meta-info
        Resets detectors, buffer and both observation windows so clean data from the new concept can be collected.
        """
        # We reset drift detection as the new performance will not
        # be the same, and reset history for the new state.
        from_id = self.active_state_id
        to_id = transition_target_id
        #logging.info(f"Transition to {transition_target_id}")
        # print(f"Transition to {to_id}")
        is_new_state = to_id not in self.state_repository
        self.active_state_is_new = is_new_state
        self.active_state_id = to_id
        if is_new_state:
            # print(f"Making new state {to_id}")
            n_id, n_state = self.make_state(to_id)
            self.state_repository[to_id] = n_state

        # Set triggers to future evaluation of this attempt
        self.observations_since_last_transition = 0
        self.known_drift_point = -1 * (self.window_size + 1)

    def _partial_fit(self, X, y, sample_weight, masked=False):

        # init defaults for trackers
        found_change = False
        self.detected_drift = False
        self.warning_detected = False
        self.observations_since_last_transition += 1
        initial_state = self.active_state_id

        # get_temporal_x, and get_imputed_label are to deal with masked
        # values where we don't see true label.
        # As the functions are now, they don't do anything extra.
        # But could be extended to reuse last made prediction as
        # the label for example.
        temporal_X = self.get_temporal_x(X)

        foreground_p = self.get_active_state().predict([temporal_X])[0]
        self.get_active_state().partial_fit(temporal_X, y, sample_weight=sample_weight, classes=self.classes)

        detected_drift = self.perform_drift_detection_for_bounds()

        # We have three reasons for attempting a check: We detected a drift, we want to evaluate the current state, or we have a propagating transition attempt.
        # If we have detected a drift, we consider state similarity directly.
        # Otherwise, we behave probabalistically.
        #<TODO> combine these approaches cleanly
        if detected_drift is not None:
            if self.middle_bound:
                detected_drift = self.max_state_id
                self.max_state_id += 1
            self.perform_transition(detected_drift)


        self.record_transition(detected_drift, initial_state)

        # Set exposed info for evaluation.
        self.active_state = self.active_state_id
        self.found_change = detected_drift
        self.states = self.state_repository
        self.current_sensitivity = 0.05


    def add_to_transition_matrix(self, init_s, curr_s, matrix):
        if init_s not in matrix:
            matrix[init_s] = {}
            matrix[init_s]['total'] = 0
        matrix[init_s][curr_s] = matrix[init_s].get(curr_s, 0) + 1
        matrix[init_s]['total'] += 1
        return matrix

    def record_transition(self, detected_drift, initial_state):
        """ Record an observation to observation level transition from the initial_state to the current state.
        Depends on if a drift was detected, or if in warning which records are updated.
        """
        current_state = self.active_state_id
        # We always add to the standard matrix
        self.add_to_transition_matrix(initial_state, current_state, self.concept_transitions_standard)

        # If we are in a warning period we add to the warning, and if we have detected drift we add to the warning and drift matricies
        if self.in_warning:
            self.add_to_transition_matrix(initial_state, current_state, self.concept_transitions_warning)
        elif detected_drift is not None:
            self.add_to_transition_matrix(initial_state, current_state, self.concept_transitions_warning)
            self.add_to_transition_matrix(initial_state, current_state, self.concept_transitions_drift)


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