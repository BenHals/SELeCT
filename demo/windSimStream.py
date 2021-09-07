import matplotlib
import logging
import matplotlib.pyplot as plt
# matplotlib.use('qt5agg')
from matplotlib.collections import LineCollection
import matplotlib.animation as animation
import matplotlib.colors
import numpy as np
import pandas as pd
import argparse
from typing import List
from collections import deque
from skmultiflow.meta import AdaptiveRandomForestClassifier
from skmultiflow.trees import HoeffdingAdaptiveTreeClassifier
import networkx as nx


from river.metrics import Accuracy, Rolling, KappaM

from SELeCT.Classifier.select_classifier import SELeCTClassifier
from SELeCT.Classifier.hoeffding_tree_shap import HoeffdingTreeSHAP, HoeffdingTreeSHAPClassifier
from SELeCT.Data.windsim_generator import WindSimGenerator
Vector = List[float]

logging.basicConfig(filename = "demo.log", level=logging.INFO)

def pandas_fill(arr):
    df = pd.Series(arr)
    df.fillna(method='bfill', inplace=True)
    out = df.to_numpy()
    return out

def numpy_fill(arr):
    '''Solution provided by Divakar.'''
    mask = np.isnan(arr)
    idx = np.where(~mask,np.arange(mask.shape[0])-20,0)
    np.minimum.accumulate(idx[::-1],axis=0, out=idx[::-1])
    idx = idx + 20
    out = arr[idx]
    return out

def handle_merges_and_deletion(history, merges, deletions):
    merge_history = np.copy(history)
    for m_init in merges:
        m_to = merges[m_init]
        while m_to in merges:
            m_from = m_to
            m_to = merges[m_from]
        merge_history = np.where(merge_history == m_init, m_to, merge_history)
    repair_history = np.copy(merge_history)
    for dm in deletions:
        repair_history = np.where(repair_history == dm, np.nan, repair_history)
    try:
        repair_history = pandas_fill(repair_history)
    except Exception as e:
        print(repair_history)
        raise e

    return history, merge_history.astype(int), repair_history.astype(int)

def segment_history(history, ex):
    diff = np.insert(history[1:] == history[:-1], 0, False)
    idx = (np.arange(history.shape[0]) - history.shape[0]) + ex
    starts = np.vstack((history[~diff], idx[~diff])).T
    return starts

def plot_TM(ax, classifier, concept_colors, c_init):
    T = classifier.concept_transitions_standard
    G = nx.DiGraph()
    for ID in classifier.state_repository:
        G.add_node(ID)
        G.add_edge(ID, ID)
        G.add_node(f"T-{ID}")
        G.add_edge(f"T-{ID}", f"T-{ID}")
    for from_id, from_T in T.items():
        total_T = from_T['total']
        for to_id, n_T in [(i,t) for i,t in from_T.items() if i != 'total']:
            if to_id != from_id and n_T > 0:
                G.add_edge(from_id, to_id, weight=n_T, label=n_T)
    
    node_colors = []
    node_edges = []
    for n in G.nodes:
        try:
            ID = int(float(str(n).split('-')[-1]))
        except:
            ID = 5
        node_colors.append(concept_colors[(ID + c_init) % len(concept_colors)])
        node_edges.append(concept_colors[(ID + c_init) % len(concept_colors)] if ID!=classifier.active_state_id else "black")

    emit_edge_labels = {(n1,n2):f"{d['label']:.0f}" for n1,n2,d in G.edges(data=True) if n1!=n2}
    pos = nx.circular_layout(G)
    nx.draw(G, pos=pos, with_labels=True, ax=ax, node_color=node_colors, edgecolors=node_edges)
    nx.draw_networkx_edge_labels(G , pos, edge_labels=emit_edge_labels, ax=ax)

def plot(frame, stream, classifier, classifier_baseline, history_len, im, acc_line, acc_c_line, baseline_line, gt_lc, sys_lc, sys_nomr_lc, likelihood_lc, text_obs, flush=False):
    global count
    global concept_colors
    global current_concept
    global concept_cm
    global merges
    global deletions
    global right
    global wrong
    global x_history
    global acc_history
    global baseline_history
    global acc_c_history
    global gt_segments
    global gt_history
    global gt_colors
    global sys_segments
    global sys_history
    global sys_colors
    global sys_nomr_segments
    global sys_nomr_colors
    global likelihood_history
    global likelihood_segments
    global likelihood_segment_colors
    global recall
    global precision
    global F1
    global adwin_likelihood_estimate
    global acc
    global acc_baseline
    global rolling_acc
    global kappa
    global last_state

    X,y = stream.next_sample()
    if X is not None:
        p = classifier.predict(X)
        try:
            with np.errstate(all='ignore'):
                p_base = classifier_baseline.predict(X)
        except Exception as e:
            print(f"error {e}")
            p_base = [0]
        adwin_likelihood_estimate = {}
        adwin_likelihood_estimate.update({i:s.get_estimated_likelihood() for i,s in classifier.state_repository.items()})
        acc.update(p[0], y[0])
        acc_baseline.update(p_base[0], y[0])
        rolling_acc.update(p[0], y[0])
        kappa.update(p[0], y[0])
        x_history.append(classifier.ex)
        acc_history.append(acc.get())
        baseline_history.append(acc_baseline.get())
        acc_c_history.append(rolling_acc.get())

        # Update concept history
        for concept_id in adwin_likelihood_estimate:
            concept_hist = likelihood_history.setdefault(concept_id, deque(maxlen=history_len))
            concept_hist.append((classifier.ex, adwin_likelihood_estimate[concept_id]))
        
        # remove any deleted concepts
        for concept_id in list(likelihood_history.keys()):
            if concept_id not in adwin_likelihood_estimate:
                likelihood_history.pop(concept_id)

        likelihood_segments = []
        likelihood_segment_colors = []
        for concept_id, concept_hist in likelihood_history.items():
            seg = []
            seg_len = len(concept_hist)
            for px, py in concept_hist:
                seg.append((px, py))
            likelihood_segments.append(seg)
            likelihood_segment_colors.append(concept_colors[(concept_id + stream.init_concept) % len(concept_colors)])
            

        curr_gt_concept = stream.concept
        curr_sys_concept = classifier.active_state_id
        gt_history.append(curr_gt_concept)
        sys_history.append(curr_sys_concept)
        if curr_gt_concept not in concept_cm:
            concept_cm[curr_gt_concept] = {}
            gt_totals[curr_gt_concept] = 0
        if curr_sys_concept not in concept_cm[curr_gt_concept]:
            concept_cm[curr_gt_concept][curr_sys_concept] = 0
        if curr_sys_concept not in sys_totals:
            sys_totals[curr_sys_concept] = 0
        concept_cm[curr_gt_concept][curr_sys_concept] += 1
        gt_totals[curr_gt_concept] += 1
        sys_totals[curr_sys_concept] += 1

        recall = concept_cm[curr_gt_concept][curr_sys_concept] / gt_totals[curr_gt_concept]
        precision = concept_cm[curr_gt_concept][curr_sys_concept] / sys_totals[curr_sys_concept]
        F1 = 2 * ((precision * recall) / (precision + recall))

        np_sys_history = np.array(sys_history)
        merges.update(classifier.merges if hasattr(classifier, "merges") else {})
        deletions += classifier.deletions if hasattr(classifier, "deletions") else []
        sys_h, sys_merge, sys_repair = handle_merges_and_deletion(np_sys_history, merges, deletions)


        gt_seg_starts = segment_history(np.array(gt_history), classifier.ex)
        gt_segments = []
        gt_colors = []
        seg_end = classifier.ex
        for line in gt_seg_starts[::-1]:
            gt_segments.append([[line[1], 0], [seg_end, 0]])
            gt_colors.append(concept_colors[line[0] % len(concept_colors)])
            seg_end = line[1]

        sys_seg_starts = segment_history(sys_repair, classifier.ex)
        sys_segments = []
        sys_colors = []
        seg_end = classifier.ex
        for line in sys_seg_starts[::-1]:
            sys_segments.append([[line[1], 0.75], [seg_end, 0.75]])
            sys_colors.append(concept_colors[(line[0] + stream.init_concept) % len(concept_colors)])
            seg_end = line[1]

        sys_nomr_seg_starts = segment_history(sys_h, classifier.ex)

        seg_end = classifier.ex
        for line in sys_nomr_seg_starts[::-1]:
            sys_nomr_segments.append([[line[1], 0.25], [seg_end, 0.25]])
            sys_nomr_colors.append(concept_colors[(line[0] + stream.init_concept) % len(concept_colors)])
            seg_end = line[1]

        
        classifier.partial_fit(X, y, classes=list(range(0, 21)))
        classifier_baseline.partial_fit(X, y, classes=list(range(0, 21)))
    z = stream.get_last_image()
    artists = []
    if count % 50 == 0:
        sample_text.set_text(f"Sample: {classifier.ex}")
        next_drift_text.set_text(f"Next Drift in: {1000 - (classifier.ex % 1000)}")
        acc_text.set_text(f"Accuracy: {acc.get():.2%}")
        r_acc_text.set_text(f"Rolling: {rolling_acc.get():.2%}")
        baseline_text.set_text(f"Baseline: {acc_baseline.get():.2%}")
        kappaM_text.set_text(f"KappaM: {kappa.get():.2%}")

        if len(sys_history) > 1:
            sys_text.set_text(f"System State: {sys_history[-1]}")
            gt_text.set_text(f"GT Concept: {gt_history[-1]}")
        recall_text.set_text(f"Recall: {recall:.2f}")
        precision_text.set_text(f"precision: {precision:.2f}")
        F1_text.set_text(f"F1: {F1:.2f}")

        concept_likelihood_text.set_text(f"Concept Likelihoods: {', '.join('{0}: {1:.2%}'.format(k, v) for k,v in adwin_likelihood_estimate.items())}")
        merge_text.set_text(f"Merges: {' '.join('{0} -> {1}'.format(k, v) for k,v in merges.items())}")
        deletion_text.set_text(f"Deletions: {str(deletions)}")

    if last_state != classifier.active_state_id:
        ax8.clear()
        plot_TM(ax8, classifier, concept_colors, stream.init_concept)
        ax8.relim()
        ax8.autoscale_view(False,True,True)
        x_lim = ax8.get_xlim()
        x_lim_range = x_lim[1] - x_lim[0]
        ax8.set_xlim([x_lim[0] - x_lim_range*0.2, x_lim[1] + x_lim_range*0.2])
        y_lim = ax8.get_ylim()
        y_lim_range = y_lim[1] - y_lim[0]
        ax8.set_ylim([y_lim[0] - y_lim_range*0.2, y_lim[1] + y_lim_range*0.2])
        last_state = classifier.active_state_id
        fig.canvas.resize_event()

    if count % 12 == 0:
    # if count % 1 == 0:
        # ax.clear()
        # plt.clf()

        # plt.imshow(z, norm = matplotlib.colors.Normalize(0, 255))
        im.set_data(z)
    if count % 2 == 0:
        acc_line.set_data(list(x_history), list(acc_history))
        acc_c_line.set_data(list(x_history), list(acc_c_history))
        baseline_line.set_data(list(x_history), list(baseline_history))
        ax2.set_xlim([max(0, classifier.ex - (history_len - 1)), max(history_len, classifier.ex+1)])
        ax3.set_xlim([max(0, classifier.ex - (history_len - 1)), max(history_len, classifier.ex+1)])
        ax6.set_xlim([max(0, classifier.ex - (history_len - 1)), max(history_len, classifier.ex+1)])
        gt_lc.set_segments(gt_segments)
        gt_lc.set_color(gt_colors)
        sys_lc.set_segments(sys_segments)
        sys_lc.set_color(sys_colors)
        sys_nomr_lc.set_segments(sys_nomr_segments)
        sys_nomr_lc.set_color(sys_nomr_colors)
        likelihood_lc.set_segments(likelihood_segments)
        likelihood_lc.set_color(likelihood_segment_colors)
    artists = [im, acc_line, acc_c_line, baseline_line, gt_lc, sys_lc, sys_nomr_lc, likelihood_lc, *text_obs.values()]

    count += 1
    if count >= drift_count:
        current_concept += 1
        stream.set_concept((current_concept % n_concepts) + 1)
        count = 0
    return artists
    
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-ns", "--nsensor", type=int,
        help="Number of sensors", default=8)
    ap.add_argument("-st", "--sensortype",
        help="How sensors are arranged", default="circle", choices=["circle", "grid"])
    args = vars(ap.parse_args())

    n_concepts = 4
    concepts = []
    for c in range(n_concepts):
        concepts.append(np.random.randint(0, 1000))
    current_concept = 1
    stream = WindSimGenerator(concept=current_concept+1, produce_image=True, num_sensors= args['nsensor'], sensor_pattern=args['sensortype'])
    stream.prepare_for_use()

    classifier = SELeCTClassifier(
        learner=HoeffdingTreeSHAPClassifier)

    # classifier = SELeCTClassifier(
    #     learner=HoeffdingTreeSHAPClassifier,
    #     window_size=25,
    #     active_head_monitor_gap=5,
    #     fingerprint_update_gap=5,
    #     non_active_fingerprint_update_gap=20,
    #     observation_gap=5,
    #     ignore_features=["IMF", "MI", "pacf"],
    #     similarity_min_stdev=0.015,
    #     similarity_max_stdev=0.15,
    #     buffer_ratio=0.2,
    #     sensitivity = 0.1,
    #     min_window_ratio = 0.65,
    #     fingerprint_grace_period = 10,
    #     state_grace_period_window_multiplier = 10,
    #     bypass_grace_period_threshold = 0.2,
    #     state_estimator_risk = 0.5,
    #     state_estimator_swap_risk = 0.75,
    #     minimum_concept_likelihood = 0.01,
    #     min_drift_likelihood_threshold = 0.175,
    #     min_estimated_posterior_threshold = 0.2,
    #     correlation_merge=True,
    #     merge_threshold=0.9,
    #     multihop_penalty=1.0)

    # classifier_baseline = HoeffdingTreeSHAPClassifier()
    classifier_baseline = AdaptiveRandomForestClassifier()
    # classifier_baseline = HoeffdingAdaptiveTreeClassifier()
    
    history_len = 500
    count = 0
    concept_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    merges = {}
    deletions = []
    print(concept_colors)
    acc = Accuracy()
    acc_baseline = Accuracy()
    x_history = deque(maxlen=history_len)
    acc_history = deque(maxlen=history_len)
    acc_c_history = deque(maxlen=history_len)
    baseline_history = deque(maxlen=history_len)
    likelihood_history = {}
    likelihood_segments = []
    likelihood_segment_colors = []
    gt_segments = deque(maxlen=history_len)
    gt_history = deque(maxlen=history_len)
    gt_colors = deque(maxlen=history_len)
    sys_segments = deque(maxlen=history_len)
    sys_history = deque(maxlen=history_len)
    sys_colors = deque(maxlen=history_len)
    sys_nomr_segments = deque(maxlen=history_len)
    sys_nomr_colors = deque(maxlen=history_len)
    recall = 0
    precision = 0
    F1 = 0
    adwin_likelihood_estimate = {}
    last_state = -1

    concept_cm = {}
    gt_totals = {}
    sys_totals = {}

    rolling_acc = Rolling(Accuracy(), window_size =100)
    kappa = KappaM()
    drift_count = 1000

    fig = plt.figure(figsize=(10, 5))
    gs = fig.add_gridspec(7, 5, height_ratios=[0.5, 0.5, 0.5, 1, 1, 1, 1], width_ratios = [1, 1, 1, 1, 1.5])
    gs.update(wspace=0.025, hspace=0.005)
    ax1 = fig.add_subplot(gs[0:3, :3])
    ax1.axis('off')
    

    ax2 = fig.add_subplot(gs[3, :4])
    ax2.set_xlim([0, history_len])
    ax2.set_ylim([0, 1])
    ax2.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False)
    ax2.set_title("Performance")

    ax3 = fig.add_subplot(gs[4, :4])
    ax3.set_xlim([0, history_len])
    ax3.set_ylim([-0.1, 1])
    ax3.set_yticks([0.0, 0.25, 0.5, 0.75])
    ax3.set_yticklabels(["Ground Truth Concepts", "System State", "Sys State w Merging", "Sys State w Repair"])
    ax3.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False)
    ax3.set_title("Active Concept")

    ax6 = fig.add_subplot(gs[5, :4])
    ax6.set_xlim([0, history_len])
    ax6.set_ylim([0, 1])
    ax6.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False)
    ax6.set_title("Concept Likelihoods")

    ax4 = fig.add_subplot(gs[0:3, 3])
    ax4.set_frame_on(False)
    ax4.axis('off')

    ax5 = fig.add_subplot(gs[3:5, 4])
    ax5.axis('off')
    ax5.set_frame_on(True)
    ax5.set_xlim([0, 1])
    ax5.set_ylim([0, 1])

    ax7 = fig.add_subplot(gs[6, :4])
    ax7.set_xlim([0, history_len])
    ax7.axis('off')
    ax7.set_frame_on(True)

    ax8 = fig.add_subplot(gs[5:7, 4])
    for spine in ax8.spines.values():
        spine.set_visible(False)
    ax8.tick_params(top='off', bottom='off', left='off', right='off', labelleft='off', labelbottom='on')
    ax8.set_xlim([-2, 2])
    ax8.set_ylim([-2, 2])
    ax8.set_title("System FSM")

    IMAGE_SIZE = 100
    array = np.zeros(shape=(IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
    array[0, 0] = 99 # this value allow imshow to initialise it's color scale
    im = ax1.imshow(array, cmap="gray", norm = matplotlib.colors.Normalize(0, 255))
    acc_line, = ax2.plot([], [], 'r-')
    acc_c_line, = ax2.plot([], [], 'g-')
    baseline_line, = ax2.plot([], [], 'b-')
    gt_lc = LineCollection(gt_segments)
    gt_lc.set_linewidth(2)
    ax3.add_collection(gt_lc)
    sys_lc = LineCollection(sys_segments)
    sys_lc.set_linewidth(2)
    ax3.add_collection(sys_lc)
    sys_nomr_lc = LineCollection(sys_segments)
    sys_nomr_lc.set_linewidth(2)
    ax3.add_collection(sys_nomr_lc)

    likelihood_lc = LineCollection(gt_segments)
    likelihood_lc.set_linewidth(2)
    ax6.add_collection(likelihood_lc)

    sample_text = ax4.text(1, 0.9, "Sample:      ", clip_on=False,transform = ax4.transAxes,horizontalalignment='right')
    next_drift_text = ax4.text(1, 0.7, "Next Drift in:      ",horizontalalignment='right')

    acc_text = ax5.text(0.1, 0.91, "Accuracy:      ",horizontalalignment='left')
    ax5.plot([0.01, 0.05], [0.95, 0.95], 'r-')
    r_acc_text = ax5.text(0.1, 0.82, "Rolling:      ",horizontalalignment='left')
    ax5.plot([0.01, 0.05], [0.85, 0.85], 'g-')
    kappaM_text = ax5.text(0.1, 0.73, "KappaM:      ",horizontalalignment='left')
    ax5.plot([0.01, 0.05], [0.69, 0.69], 'b-')
    baseline_text = ax5.text(0.1, 0.65, "Baseline",horizontalalignment='left')

    sys_text = ax5.text(0.1, 0.5, "System State:      ",horizontalalignment='left')
    gt_text = ax5.text(0.1, 0.4, "GT Concept:      ",horizontalalignment='left')
    recall_text = ax5.text(0.1, 0.3, "Recall:      ",horizontalalignment='left')
    precision_text = ax5.text(0.1, 0.2, "precision:      ",horizontalalignment='left')
    F1_text = ax5.text(0.1, 0.1, "F1:      ",horizontalalignment='left')
    concept_likelihood_text = ax7.text(0, 0.8, "Concept Likelihoods: ",horizontalalignment='left')
    merge_text = ax7.text(0, 0.5, "Merges: ",horizontalalignment='left')
    deletion_text = ax7.text(0, 0.2, "Deletions: ",horizontalalignment='left')
    text_obs = {"sample_text": sample_text,
        "next_drift_text": next_drift_text,
        "acc_text": acc_text,
        "r_acc_text": r_acc_text,
        "baseline_text": baseline_text,
        "kappaM_text": kappaM_text,
        "sys_text": sys_text,
        "gt_text": gt_text,
        "recall_text": recall_text,
        "precision_text": precision_text,
        "F1_text": F1_text,
        "concept_likelihood_text": concept_likelihood_text,
        "merge_text": merge_text,
        "deletion_text": deletion_text,
    }
    gs.tight_layout(fig, rect=[0, 0, 1, 1], w_pad = 0.05, h_pad = 0.05)
    ani = animation.FuncAnimation(fig, plot, fargs=(stream, classifier, classifier_baseline, history_len, im, acc_line,  acc_c_line, baseline_line, gt_lc, sys_lc, sys_nomr_lc, likelihood_lc, text_obs, False), interval=0.001, blit=True)
    plt.show()
