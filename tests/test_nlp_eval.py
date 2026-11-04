import pytest

from sdpo.nlp.evaluate import classification_report_dict
from sdpo.nlp.finetune import PHRASEBANK_MAP, TWITTER_MAP


def test_twitter_label_mapping():
    # Twitter: 0 Bearish, 1 Bullish, 2 Neutral -> unified 0 neg, 1 neu, 2 pos
    assert [TWITTER_MAP[i] for i in (0, 1, 2)] == [0, 2, 1]


def test_phrasebank_label_mapping():
    assert [PHRASEBANK_MAP[i] for i in (0, 1, 2)] == [0, 1, 2]


def test_report_known_predictions():
    r = classification_report_dict([0, 1, 2, 2], [0, 1, 2, 1])
    assert r["accuracy"] == 0.75 and r["confusion"][2] == [0, 1, 1]
    # per-class F1: neg 1.0, neu 2/3, pos 2/3
    assert r["macro_f1"] == pytest.approx((1 + 2 / 3 + 2 / 3) / 3)
