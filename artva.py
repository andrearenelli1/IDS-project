"""
artva.py
Single ARTVA Dipole Source Model
"""
# In case to allow for forward references in type hints
from __future__ import annotations 

# Imports
import numpy as np
from config import ARTVA_MOMENT, ARTVA_NOISE_STD

class ARTVASource:
    """
    Parameters:
    θ      = position of the ARTVA source (victim) in world frame [x, y, z]
    moment = normalized magnetic moment [A·m²]
    seed   = seed for noise generation
    """

    def __init__(self, theta: np.ndarray, moment: float = ARTVA_MOMENT, seed: int = 1) -> None:
        self._theta  = np.asarray(theta, dtype=float)
        self._moment = moment
        self._seed   = np.random.default_rng(seed)

    def signal(self, x: np.ndarray, noisy: bool = True) -> float:
        """
        Returns the magnetic field strength S at position x due to the ARTVA source:
        r_vec = x - θ
        r_norm = ||x - θ||
        S = m * sqrt(1 + 3*cos²(ψ)) / r_norm³
        m = magnetic moment
        cos(ψ) = r_vec_z / r_norm 

        Parameters:
        x     = true position where the signal is measured, in world frame [x, y, z]
        noisy = whether to add Gaussian noise to the signal
        """
        # Vector r
        r_vec  = np.asarray(x) - self._theta
        r_norm = np.linalg.norm(r_vec)

        # To avoid singularities
        if r_norm < 1e-3:
            r_norm = 1e-3
        
        # Angle psi and signal strength S
        cos_psi = r_vec[2] / r_norm
        S = self._moment * np.sqrt(1.0 + 3.0 * cos_psi**2) / r_norm**3

        # Add Gaussian noise if requested
        if noisy:
            S += self._seed.normal(0, ARTVA_NOISE_STD)

        return max(0.0, S)