"""Black-Litterman: blend an equilibrium prior with forecast-based views.

All quantities are in weekly return units. Shapes: N assets, K views.
"""

import numpy as np

from sdpo import config


def implied_prior(sigma: np.ndarray, w: np.ndarray, delta: float = config.BL_DELTA) -> np.ndarray:
    """Equilibrium excess returns implied by holding w: pi = delta * Sigma @ w."""
    return delta * np.asarray(sigma) @ np.asarray(w)


def bl_posterior(sigma: np.ndarray, pi: np.ndarray, P: np.ndarray, Q: np.ndarray,
                 omega: np.ndarray, tau: float = config.BL_TAU) -> np.ndarray:
    """Posterior expected returns mu_BL, shape [N].

    sigma [N,N] asset covariance, pi [N] prior, P [K,N] view picking matrix, Q [K] view
    returns, omega [K,K] view uncertainty (diagonal: bigger = less trusted view).

        mu_BL = [(tau*Sigma)^-1 + P' Omega^-1 P]^-1 [(tau*Sigma)^-1 pi + P' Omega^-1 Q]
    """
    pi = np.asarray(pi).ravel()
    Q = np.asarray(Q).ravel()
    P = np.asarray(P)
    
    ts_inv = np.linalg.inv(tau * np.asarray(sigma))
    omega_inv = np.linalg.inv(np.asarray(omega))
    
    A = ts_inv + P.T @ omega_inv @ P
    b = ts_inv @ pi + P.T @ omega_inv @ Q
    
    return np.linalg.solve(A, b)
