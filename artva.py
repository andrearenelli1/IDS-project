"""
artva.py
========
Sorgente ARTVA modellata come dipolo magnetico verticale.

    S(r) = moment · sqrt(1 + 3·cos²θ) / r³

dove θ è l'angolo tra il vettore r (drone − sorgente) e l'asse z del dipolo.
"""

from __future__ import annotations

import numpy as np

from config import ARTVA_MOMENT, ARTVA_NOISE_STD


class ARTVASource:
    """
    Sorgente ARTVA — dipolo magnetico verticale.

    Parameters
    ----------
    position : (3,) [x, y, z] in coordinate UTM [m]
    moment   : momento magnetico normalizzato [A·m²]
    rng_seed : seme per il generatore di rumore
    """

    def __init__(
        self,
        position: np.ndarray,
        moment: float = ARTVA_MOMENT,
        rng_seed: int = 1,
    ) -> None:
        self.position = np.asarray(position, dtype=float)
        self.moment   = moment
        self._rng     = np.random.default_rng(rng_seed)

    def signal(self, pos: np.ndarray, noisy: bool = True) -> float:
        """
        Intensità segnale ARTVA in 'pos' (3,).

        Parameters
        ----------
        pos   : posizione del sensore [x, y, z]
        noisy : se True aggiunge rumore gaussiano additivo

        Returns
        -------
        S ≥ 0
        """
        r_vec     = np.asarray(pos) - self.position
        r         = np.linalg.norm(r_vec)
        if r < 1e-3:
            r = 1e-3          # evita singolarità
        cos_theta = r_vec[2] / r
        S = self.moment * np.sqrt(1.0 + 3.0 * cos_theta**2) / r**3
        if noisy:
            S += self._rng.normal(0, ARTVA_NOISE_STD)
        return max(0.0, S)
