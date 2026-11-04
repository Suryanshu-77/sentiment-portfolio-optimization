import numpy as np
import torch

from sdpo.metrics import rank_ic
from sdpo.models.forecaster import LSTMForecaster, RidgeForecaster, predict, train_lstm


def test_lstm_forward_shape():
    assert LSTMForecaster()(torch.randn(8, 12, 7)).shape == (8,)


def test_lstm_recovers_planted_signal(synthetic_splits):
    X_tr, y_tr, X_va, y_va, X_te, y_te = synthetic_splits
    m = train_lstm(X_tr, y_tr, X_va, y_va, max_epochs=15, device="cpu")
    assert np.nanmean(rank_ic(predict(m, X_te), y_te)) > 0.1


def test_ridge_recovers_planted_signal(synthetic_splits):
    X_tr, y_tr, _, _, X_te, y_te = synthetic_splits
    m = RidgeForecaster().fit(X_tr, y_tr)
    assert np.nanmean(rank_ic(predict(m, X_te), y_te)) > 0.1


def test_predictions_are_in_return_units(synthetic_splits):
    # Black-Litterman needs Q in weekly-return units, not standardized units
    X_tr, y_tr, X_va, y_va, X_te, _ = synthetic_splits
    p = predict(train_lstm(X_tr, y_tr, X_va, y_va, max_epochs=3, device="cpu"), X_te)
    assert p.shape == X_te.shape[:2] and 1e-4 < np.std(p) < 0.05


def test_lstm_is_deterministic_with_seed(synthetic_splits):
    X_tr, y_tr, X_va, y_va, X_te, _ = synthetic_splits
    a = predict(train_lstm(X_tr, y_tr, X_va, y_va, max_epochs=2, device="cpu"), X_te)
    b = predict(train_lstm(X_tr, y_tr, X_va, y_va, max_epochs=2, device="cpu"), X_te)
    np.testing.assert_array_equal(a, b)


def test_nan_targets_are_ignored(synthetic_splits):
    X_tr, y_tr, X_va, y_va, X_te, _ = synthetic_splits
    y_tr = y_tr.copy()
    y_tr[::7, 3] = np.nan
    assert np.isfinite(predict(train_lstm(X_tr, y_tr, X_va, y_va, max_epochs=1,
                                          device="cpu"), X_te)).all()
    assert np.isfinite(predict(RidgeForecaster().fit(X_tr, y_tr), X_te)).all()
