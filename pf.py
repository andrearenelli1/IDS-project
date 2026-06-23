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
    
    def initialize_particles(self, p, S):
        phi_min, phi_max = 0.0, 2*np.pi
        sinpsi_min, sinpsi_max = 0, 1
        self.particles = np.column_stack([
            np.ones(self.n_particles),
            np.random.uniform(phi_min, phi_max, self.n_particles),
            np.random.uniform(sinpsi_min, sinpsi_max, self.n_particles),
        ])
        self.weights = np.ones(self.n_particles) / self.n_particles
        self.particles[:, 0] = measurement_model(S, self.particles[:, 2])
        self.particles = self.polar_to_world(p, self.particles)

    def update_weights(self, p, S, m, sigma):
        r_vecs  = np.asarray(p) - self.particles
        r_norms = np.maximum(np.linalg.norm(r_vecs, axis=1), 1e-3)
        cos_psi = -r_vecs[:, 2] / r_norms
        S_pred  = m * np.sqrt(1.0 + 3.0 * cos_psi**2) / r_norms**3

        # Sigma adattiva: combina rumore additivo calibrato e rumore moltiplicativo (5% del segnale).
        # Indispensabile per gli update cooperativi: droni vicini alla sorgente hanno segnali
        # ordini di grandezza più alti e renderebbero tutti i pesi zero con sigma fisso.
        effective_sigma = np.sqrt(sigma**2 + (0.20 * S)**2)

        log_like = -0.5 * ((S - S_pred) / effective_sigma) ** 2
        log_like -= log_like.max()   # shift per prevenire underflow

        self.weights *= np.exp(log_like)
        total = np.sum(self.weights)
        if total > 0:
            self.weights /= total
        else:
            self.weights[:] = 1.0 / self.n_particles

    def resample_particles(self, jitter_std=None):
        # Resample solo quando il filtro è degenerato (N_eff < N/2).
        # Quando i pesi sono ancora diversificati non serve: evita la perdita prematura
        # di diversità che causa il collasso della stima di profondità.
        N_eff = 1.0 / np.sum(self.weights ** 2)
        if N_eff > self.n_particles / 2:
            return

        # Systematic resampling
        cumsum    = np.cumsum(self.weights)
        positions = (np.arange(self.n_particles) + np.random.uniform()) / self.n_particles
        indexes   = np.searchsorted(cumsum, positions)

        self.particles = self.particles[indexes]
        self.weights   = np.ones(self.n_particles) / self.n_particles

        # Jitter con floor minimo assoluto: impedisce il collasso totale delle particelle
        # vicino alla sorgente, dove std sarebbe ~0 e la profondità risulta bloccata.
        if jitter_std is not None:
            std = jitter_std
        else:
            std = np.maximum(
                np.std(self.particles, axis=0) * 0.1,
                np.array([1.0, 1.0, 0.5]),   # [m] floor in x, y, z
            )
        self.particles += np.random.normal(0, std, self.particles.shape)

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