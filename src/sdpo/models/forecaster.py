"""Strategy A forecasters: next-week cross-sectional excess return from a 12-week feature
sequence. One model is shared across tickers (pooled), so it learns a common mapping."""

import copy

import numpy as np
import torch
from sklearn.linear_model import Ridge
from torch import nn

from sdpo import config


class LSTMForecaster(nn.Module):
    def __init__(self, n_features: int = 7, hidden: int = 32, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, 1)
        # Targets are trained standardized; this rescales outputs back to return units.
        self.register_buffer("y_scale", torch.tensor(1.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [B,L,F] -> [B]
        out, _ = self.lstm(x)
        return self.head(self.drop(out[:, -1])).squeeze(-1) * self.y_scale


def _flatten(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """[T,N,L,F], [T,N] -> rows with a finite target."""
    Xf = X.reshape(-1, *X.shape[-2:])
    yf = y.reshape(-1)
    ok = np.isfinite(yf)
    return Xf[ok].astype(np.float32), yf[ok].astype(np.float32)


def train_lstm(X_tr, y_tr, X_va, y_va, seed: int = config.SEED, max_epochs: int = 50,
               patience: int = 5, lr: float = 1e-3, batch: int = 256,
               device: str | None = None) -> LSTMForecaster:
    """Adam + MSE on standardized targets, early stopping on validation loss; returns the
    best-validation weights, on CPU."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    Xt, yt = _flatten(X_tr, y_tr)
    Xv, yv = _flatten(X_va, y_va)
    scale = float(yt.std()) or 1.0
    Xt, yt = torch.from_numpy(Xt).to(device), torch.from_numpy(yt / scale).to(device)
    Xv, yv = torch.from_numpy(Xv).to(device), torch.from_numpy(yv / scale).to(device)

    model = LSTMForecaster(n_features=Xt.shape[-1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    gen = torch.Generator().manual_seed(seed)
    best, best_state, bad = float("inf"), copy.deepcopy(model.state_dict()), 0
    for _ in range(max_epochs):
        model.train()
        for idx in torch.randperm(len(Xt), generator=gen).split(batch):
            idx = idx.to(device)
            opt.zero_grad()
            loss = nn.functional.mse_loss(model(Xt[idx]), yt[idx])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val = nn.functional.mse_loss(model(Xv), yv).item()
        if val < best - 1e-6:
            best, best_state, bad = val, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    model.y_scale.fill_(scale)
    return model.cpu().eval()


class RidgeForecaster:
    """Linear ablation: Ridge(alpha=1) on the flattened 12x7 sequence."""

    def __init__(self, alpha: float = 1.0):
        self.model = Ridge(alpha=alpha)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RidgeForecaster":
        Xf, yf = _flatten(X, y)
        self.model.fit(Xf.reshape(len(Xf), -1), yf)
        return self

    def predict_flat(self, Xf: np.ndarray) -> np.ndarray:
        return self.model.predict(Xf.reshape(len(Xf), -1))


def predict(model: LSTMForecaster | RidgeForecaster, X: np.ndarray) -> np.ndarray:
    """Forecasts for X [..., L, F] -> [...] in weekly-return units."""
    lead = X.shape[:-2]
    Xf = X.reshape(-1, *X.shape[-2:]).astype(np.float32)
    if isinstance(model, RidgeForecaster):
        out = model.predict_flat(Xf)
    else:
        with torch.no_grad():
            out = model(torch.from_numpy(Xf)).numpy()
    return out.reshape(lead)
