import numpy as np
import matplotlib.pyplot as plt


def signal_model(p, theta, m, sigma, noisy=True):
    r_vec  = np.asarray(p) - theta
    r_norm = np.linalg.norm(r_vec)

    # to avoid singularities
    if r_norm < 1e-3:
        r_norm = 1e-3

    cos_psi = -r_vec[2] / r_norm
    S = m * np.sqrt(1.0 + 3.0 * cos_psi**2) / r_norm**3
    if noisy:
        S += np.random.normal(0, sigma)
    return max(0.0, S)

def measurement_model(S, sinpsi, m=1.0):
    S = np.maximum(S, 1e-12)
    r = (m * np.sqrt(1 + 3*(1 - sinpsi**2)) / S) ** (1/3)
    return r


class ParticleFilter:
    def __init__(self, n_particles, state_dim, measurement_dim):
        self.n_particles = n_particles
        self.state_dim = state_dim
        self.measurement_dim = measurement_dim
        self.particles = np.random.rand(n_particles, state_dim)  # Initialize particles randomly
        self.weights = np.ones(n_particles) / n_particles  # Initialize weights uniformly

    def _sinpsi_max(self, p, S, z_ground, grid=512):
        # LiDAR-imposed ceiling on sinpsi.
        # The victim is buried, so each particle must stay below the
        # surface: ξz ≤ z_ground. In polar coordinates this constraint translates
        # into a minimum vertical depth below the drone d = r·cosψ ≥ d_min,
        # with d_min = p_z − z_ground (≈ AGL altitude read from LiDAR).
        d_min = p[2] - z_ground
        if d_min <= 0.0:
            return 1.0
        sp    = np.linspace(0.0, 1.0, grid)
        depth = measurement_model(S, sp) * np.sqrt(1.0 - sp ** 2)  # r·cosψ, decreasing in sinpsi
        valid = depth >= d_min
        if not valid[0]:
            # even straight-down (sinpsi=0) doesn't reach the ground: the drone
            # is practically above the source. Sample nearly vertical; z
            # is still clamped below z_ground by the final clamp.
            return 0.0
        return float(sp[valid][-1])

    def initialize_particles(self, p, S, z_ground=None):
        phi_min, phi_max = 0.0, 2*np.pi
        sinpsi_min, sinpsi_max = 0.0, 1.0
        if z_ground is not None:
            sinpsi_max = self._sinpsi_max(p, S, z_ground)
        self.particles = np.column_stack([
            np.ones(self.n_particles),
            np.random.uniform(phi_min, phi_max, self.n_particles),
            np.random.uniform(sinpsi_min, sinpsi_max, self.n_particles),
        ])
        self.weights = np.ones(self.n_particles) / self.n_particles
        self.particles[:, 0] = measurement_model(S, self.particles[:, 2])
        self.particles = self.polar_to_world(p, self.particles)
        if z_ground is not None:
            # safety clamp for the degenerate case (sinpsi_max=0)
            np.minimum(self.particles[:, 2], z_ground, out=self.particles[:, 2])

    def update_weights(self, p, S, m, sigma):
        r_vecs  = np.asarray(p) - self.particles
        r_norms = np.maximum(np.linalg.norm(r_vecs, axis=1), 1e-3)
        cos_psi = -r_vecs[:, 2] / r_norms
        S_pred  = m * np.sqrt(1.0 + 3.0 * cos_psi**2) / r_norms**3

        # Adaptive sigma: combines calibrated additive noise and multiplicative noise (5% of signal).
        # Essential for cooperative updates: drones close to the source have signals
        # orders of magnitude higher, which would drive all weights to zero with fixed sigma.
        effective_sigma = np.sqrt(sigma**2 + (0.20 * S)**2)

        log_like = -0.5 * ((S - S_pred) / effective_sigma) ** 2
        log_like -= log_like.max()   # shift to prevent underflow

        self.weights *= np.exp(log_like)
        total = np.sum(self.weights)
        if total > 0:
            self.weights /= total
        else:
            self.weights[:] = 1.0 / self.n_particles

    def resample_particles(self, z_ground=None, jitter_std=None):
        # Resample only when the filter is degenerate (N_eff < N/2).
        # When weights are still diverse it is not needed: avoids premature loss
        # of diversity that causes depth estimate collapse.
        N_eff = 1.0 / np.sum(self.weights ** 2)
        if N_eff > self.n_particles / 2:
            return

        # Systematic resampling
        cumsum    = np.cumsum(self.weights)
        positions = (np.arange(self.n_particles) + np.random.uniform()) / self.n_particles
        indexes   = np.searchsorted(cumsum, positions)

        self.particles = self.particles[indexes]
        self.weights   = np.ones(self.n_particles) / self.n_particles

        # Jitter with absolute minimum floor: prevents total particle collapse
        # near the source, where std would be ~0 and depth becomes stuck.
        if jitter_std is not None:
            std = jitter_std
        else:
            std = np.maximum(
                np.std(self.particles, axis=0) * 0.1,
                np.array([1.0, 1.0, 0.5]),   # [m] floor in x, y, z
            )
        self.particles += np.random.normal(0, std, self.particles.shape)

        # Keep particles within the LiDAR-imposed bound: jitter
        # may push them above the surface (impossible for a buried victim).
        # Those above z_ground are reflected below, preserving
        # density near the ground.
        if z_ground is not None:
            above = self.particles[:, 2] > z_ground
            self.particles[above, 2] = 2.0 * z_ground - self.particles[above, 2]

    def polar_to_world(self, p, polar_coords):
        # Convert polar coordinates to world coordinates
        r = polar_coords[:, 0]
        phi = polar_coords[:, 1]
        sinpsi = polar_coords[:, 2]
        x = r * np.cos(phi)*sinpsi
        y = r * np.sin(phi)*sinpsi
        z = -r * np.sqrt(1 - sinpsi**2)
        res = p + np.column_stack((x, y, z))
        return res

    def world_to_polar(self, world_coords):
        # Convert world coordinates to polar coordinates
        x = world_coords[:, 0]
        y = world_coords[:, 1]
        z = world_coords[:, 2]
        r = np.sqrt(x**2 + y**2 + z**2)
        phi = np.arctan2(y, x)
        sinpsi = np.sqrt(x**2 + y**2) / r
        return np.column_stack((r, phi, sinpsi))



if __name__ == "__main__":
    from matplotlib.animation import FuncAnimation

    n_particles = 1000
    m, sigma = 1.0, 1e-4
    pf = ParticleFilter(n_particles, state_dim=3, measurement_dim=2)

    theta = np.array([1.0, 1.0, -1.0])

    positions = np.array([
        [10, 10, 1.5],
        [8,  8,  1.5],
        [6,  10, 1.5],
        [8,  12, 1.5],
        [5,  5,  1.5],
        [3,  8,  1.5],
        [1,  10, 1.5],
        [0,  8,  1.5],
        [2,  5,  1.5],
        [4,  3,  1.5],
        [6,  2,  1.5],
        [8,  3,  1.5],
        [10, 5,  1.5],
    ])

    # Initialize with first measurement
    p0 = positions[0]
    S0 = signal_model(p0, theta, m=m, sigma=sigma)
    pf.initialize_particles(p0, S0)

    # Run filter and store snapshots
    snapshots = [(pf.particles.copy(), pf.weights.copy(), p0)]
    for p in positions[1:]:
        S = signal_model(p, theta, m=m, sigma=sigma)
        pf.update_weights(p, S, m=m, sigma=sigma)
        pf.resample_particles()
        snapshots.append((pf.particles.copy(), pf.weights.copy(), p))

    # Animation
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    def update(frame):
        ax.cla()
        particles, weights, p = snapshots[frame]
        ax.scatter(particles[:, 0], particles[:, 1], particles[:, 2],
                   c=weights, cmap='viridis', s=5)
        ax.scatter(*theta, color='red', s=80, marker='*', label='True source')
        ax.scatter(*p, color='blue', s=60, marker='^', label='Observer')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_title(f'Step {frame}')
        ax.legend()

    ani = FuncAnimation(fig, update, frames=len(snapshots), interval=800, repeat=True)
    plt.show()

# ════════════════════════════════════════════════════════════════════════════
# Uncertainty metrics based on the 2D confidence ellipse
# ════════════════════════════════════════════════════════════════════════════
#
# Shared geometry used consistently by run_experiments.py (sweep), by
# visualization.py (main plots) and by plot_results.py (sweep analysis):
#
#   - confidence ellipse at level `conf` (default 95%) of a drone's PF estimate,
#     derived from the 2×2 weighted covariance of the particles;
#   - ellipse area = π · k² · sqrt(det Σ)   with k² = χ²(2 d.o.f.) quantile;
#   - IoU (intersection-over-union) between the ellipses of two drones, computed
#     exactly on convex polygons (Sutherland–Hodgman + shoelace).
#
# For 2 degrees of freedom the χ² quantile has closed form: k² = -2·ln(1 - conf)
# (e.g. conf = 0.95  →  k² ≈ 5.991).


# ─────────────────────────── Covariance / ellipse ───────────────────────────

def chi2_quantile_2dof(conf=0.95):
    """χ² quantile for 2 degrees of freedom (closed form)."""
    return -2.0 * np.log(1.0 - conf)


def weighted_mean_cov_xy(particles, weights):
    """
    Weighted mean and 2×2 covariance of particles in the xy plane.

    particles : (M, >=2)   particle cloud
    weights   : (M,)       weights (not necessarily normalised)
    returns   : (mean_xy (2,), cov_xy (2, 2))
    """
    xy = np.asarray(particles)[:, :2]
    w  = np.asarray(weights, dtype=float)
    s  = w.sum()
    w  = w / s if s > 0 else np.full(len(w), 1.0 / len(w))
    mean = np.average(xy, weights=w, axis=0)
    d    = xy - mean
    cov  = (w[:, None] * d).T @ d            # Σ = Σ_i w_i (x_i-μ)(x_i-μ)^T
    return mean, cov


def ellipse_area(cov, conf=0.95):
    """Confidence ellipse area: π · k² · sqrt(det Σ) [m²]."""
    det = float(np.linalg.det(cov))
    if det <= 0.0:
        return 0.0
    return float(np.pi * chi2_quantile_2dof(conf) * np.sqrt(det))


def ellipse_contains(point, mean, cov, conf=0.95):
    """
    True if `point` falls inside the confidence ellipse at level `conf`
    centred at `mean` with covariance `cov` (Mahalanobis test):
        (x-μ)^T Σ^{-1} (x-μ) ≤ k²,   k² = χ²₂(conf).
    """
    d = np.asarray(point, dtype=float)[:2] - np.asarray(mean, dtype=float)[:2]
    try:
        m2 = float(d @ np.linalg.solve(cov, d))
    except np.linalg.LinAlgError:
        return False
    return m2 <= chi2_quantile_2dof(conf)


def ellipse_axes_angle(cov, conf=0.95):
    """
    Semi-axes (a, b) and angle [rad] of the confidence ellipse, for drawing.
    a, b are the semi-axes along the eigenvectors of Σ.
    """
    k2 = chi2_quantile_2dof(conf)
    vals, vecs = np.linalg.eigh(cov)
    vals = np.clip(vals, 0.0, None)
    a, b = np.sqrt(k2 * vals)                # semi-axes
    major = vecs[:, int(np.argmax(vals))]    # eigenvector of largest eigenvalue
    angle = float(np.arctan2(major[1], major[0]))
    return float(a), float(b), angle


def ellipse_polygon(mean, cov, conf=0.95, n=72):
    """Approximates the confidence ellipse with a convex polygon of n vertices."""
    k2 = chi2_quantile_2dof(conf)
    vals, vecs = np.linalg.eigh(cov)
    vals = np.clip(vals, 0.0, None)
    semi = np.sqrt(k2 * vals)                # semi-axes in eigenvector space
    t = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    unit = np.stack([np.cos(t), np.sin(t)], axis=1)   # unit circle
    pts  = (unit * semi) @ vecs.T            # scale and rotate into data space
    return pts + np.asarray(mean)[:2]


# ─────────────────────────── Polygon geometry ───────────────────────────────

def _signed_area(poly):
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * (np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _polygon_area(poly):
    return abs(_signed_area(poly))


def _ensure_ccw(poly):
    return poly if _signed_area(poly) > 0 else poly[::-1]


def _convex_intersection(subject, clip):
    """
    Intersection of two convex polygons (Sutherland–Hodgman).
    Both must be CCW-oriented. Returns (K, 2) or None if empty.
    """
    def inside(p, a, b):
        return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]) >= 0.0

    def seg_intersect(p1, p2, a, b):
        x1, y1 = p1; x2, y2 = p2; x3, y3 = a; x4, y4 = b
        den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(den) < 1e-12:
            return p2
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den
        return np.array([x1 + t * (x2 - x1), y1 + t * (y2 - y1)])

    output = [np.asarray(p, dtype=float) for p in subject]
    cl     = [np.asarray(p, dtype=float) for p in clip]
    for i in range(len(cl)):
        a = cl[i]
        b = cl[(i + 1) % len(cl)]
        if not output:
            return None
        inp, output = output, []
        s = inp[-1]
        for e in inp:
            if inside(e, a, b):
                if not inside(s, a, b):
                    output.append(seg_intersect(s, e, a, b))
                output.append(e)
            elif inside(s, a, b):
                output.append(seg_intersect(s, e, a, b))
            s = e
    if len(output) < 3:
        return None
    return np.array(output)


def ellipse_iou(mean_i, cov_i, mean_j, cov_j, conf=0.95, n=72):
    """IoU (∈ [0, 1]) between the confidence ellipses of two drones."""
    pi = _ensure_ccw(ellipse_polygon(mean_i, cov_i, conf, n))
    pj = _ensure_ccw(ellipse_polygon(mean_j, cov_j, conf, n))
    inter = _convex_intersection(pi, pj)
    a_inter = _polygon_area(inter) if inter is not None else 0.0
    union   = _polygon_area(pi) + _polygon_area(pj) - a_inter
    return float(a_inter / union) if union > 0 else 0.0


# ─────────────────────────── Per-run aggregate metrics ──────────────────────

def run_ellipse_metrics(means, covs, conf=0.95):
    """
    Per-run metrics on the PF ellipses of drones with active PF.

    means : list of (2,) arrays     PF estimate centres
    covs  : list of (2, 2) arrays   weighted xy covariances
    returns:
        mean_area_m2 : mean area of the confidence ellipse [m²]
        mean_iou     : mean pairwise IoU between drones (inter-drone consensus)
    """
    if not covs:
        return float("nan"), float("nan")

    areas = [ellipse_area(c, conf) for c in covs]
    mean_area = float(np.mean(areas))

    ious = [
        ellipse_iou(means[a], covs[a], means[b], covs[b], conf)
        for a in range(len(means)) for b in range(a + 1, len(means))
    ]
    mean_iou = float(np.mean(ious)) if ious else float("nan")
    return mean_area, mean_iou
