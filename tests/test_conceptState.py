import itertools
from SELeCT.Classifier.select_classifier import ConceptState
from SELeCT.Classifier.normalizer import Normalizer
from SELeCT.Classifier.hoeffding_tree_shap import HoeffdingTreeSHAP
from skmultiflow.data.stagger_generator import STAGGERGenerator


def test_observe():
    cs = ConceptState(0, HoeffdingTreeSHAP(), 10,  "cache", 10, 50, 0.25, dirty_data_length=50, normalizer=Normalizer(), fingerprint_grace_period=10, estimator_risk_level=0.05, buffer_ratio=0.2, metainfo_method="metainfo")
    for i in range(5):
        cs.observe([i, i], i, i, label=True, buffer_len=1)
    assert str(list(cs.get_stable_window())) == '[<[0, 0], 0, 0| 0 @ 1>, <[1, 1], 1, 0| 0 @ 2>, <[2, 2], 2, 0| 0 @ 3>, <[3, 3], 3, 0| 0 @ 4>]'
    assert str(list(cs.get_buffer())) == '[<[4, 4], 4, 0| 0 @ 5>]'
    assert str(list(cs.get_head_window())) == '[<[0, 0], 0, 0| 0 @ 1>, <[1, 1], 1, 0| 0 @ 2>, <[2, 2], 2, 0| 0 @ 3>, <[3, 3], 3, 0| 0 @ 4>, <[4, 4], 4, 0| 0 @ 5>]'
    for i in range(2500):
        cs.observe([i, i], i, i, label=True, buffer_len=2000)
    assert len(list(cs.get_stable_window())) <= 50
    assert len(list(cs.get_head_window())) <= 50
    assert len(list(cs.get_buffer())) <= 2000

def test_refresh():
    cs = ConceptState(0, HoeffdingTreeSHAP(), 10,  "cache", 10, 50, 0.25, dirty_data_length=50, normalizer=Normalizer(), fingerprint_grace_period=10, estimator_risk_level=0.05, buffer_ratio=0.2, metainfo_method="metainfo")
    for i in range(5):
        cs.observe([i, i], i, i, label=True, buffer_len=1)
    assert str(list(cs.get_stable_window())) == '[<[0, 0], 0, 0| 0 @ 1>, <[1, 1], 1, 0| 0 @ 2>, <[2, 2], 2, 0| 0 @ 3>, <[3, 3], 3, 0| 0 @ 4>]'
    assert str(list(cs.get_buffer())) == '[<[4, 4], 4, 0| 0 @ 5>]'
    assert str(list(cs.get_head_window())) == '[<[0, 0], 0, 0| 0 @ 1>, <[1, 1], 1, 0| 0 @ 2>, <[2, 2], 2, 0| 0 @ 3>, <[3, 3], 3, 0| 0 @ 4>, <[4, 4], 4, 0| 0 @ 5>]'

    stream = STAGGERGenerator()
    for i in range(5000):
        X,y = stream.next_sample()
        cs.partial_fit(X[0], y[0], sample_weight = 1, classes=[0, 1])
    cs.refresh_recent_data(hard=True)
    evolution_num = cs.classifier.evolution
    print(str(list(cs.get_stable_window())))
    print(f'[<[0, 0], 0, 0| {evolution_num} @ 1>, <[1, 1], 1, 0| {evolution_num} @ 2>, <[2, 2], 2, 0| {evolution_num} @ 3>, <[3, 3], 3, 0| {evolution_num} @ 4>]')
    print(evolution_num)
    for ob in itertools.chain(cs.get_stable_window(), cs.get_buffer(), cs.get_head_window()):
        p = cs.classifier.predict([ob.X])
        assert p == ob.p
        assert evolution_num == ob.ev
test_observe()
test_refresh()