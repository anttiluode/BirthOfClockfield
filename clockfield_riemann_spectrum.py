# -*- coding: utf-8 -*-
import numpy as np
import networkx as nx
from scipy.linalg import eigh
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

# ==========================================
# Clockfield Graph Simulator
# ==========================================
class ClockfieldGraphSim:
    def __init__(self, N=500, k=8, p=0.15, tau=2.737, dt=0.045):
        self.N = N
        self.tau = tau
        self.dt = dt

        self.G = nx.watts_strogatz_graph(N, k, p)
        self.A = nx.to_numpy_array(self.G)

        self.theta = np.random.uniform(0, 2*np.pi, N)
        self.theta_old = np.copy(self.theta)

    def step(self):
        diffs = self.theta[:, None] - self.theta[None, :]
        diffs = np.arctan2(np.sin(diffs), np.cos(diffs))

        beta = np.sum(self.A * (diffs**2), axis=1)
        gamma = 1.0 / (1.0 + self.tau * beta + 1e-12)**2

        force = np.sum(self.A * np.sin(diffs), axis=1)
        accel = (gamma**2) * force

        velocity = self.theta - self.theta_old
        new_theta = self.theta + velocity + (self.dt**2) * accel

        self.theta_old = self.theta.copy()
        self.theta = np.mod(new_theta, 2*np.pi)

    def get_effective_dynamical_matrix(self):
        diffs = self.theta[:, None] - self.theta[None, :]
        cos_diffs = np.cos(diffs)

        M = -self.A * cos_diffs
        np.fill_diagonal(M, np.sum(self.A * cos_diffs, axis=1))

        return M


# ==========================================
# 1. RUN SIMULATION
# ==========================================
N_NODES = 500
STEPS = 5000

sim = ClockfieldGraphSim(N=N_NODES)

print(f"=== Clockfield Graph N={N_NODES} ===")
print(f"Equilibrating {STEPS} steps...")

for _ in range(STEPS):
    sim.step()

print("Equilibration complete.")

# ==========================================
# 2. BUILD MATRIX + EIGENVALUES
# ==========================================
print("Building weighted dynamical matrix...")

M = sim.get_effective_dynamical_matrix()

evals = eigh(M, eigvals_only=True)
evals = np.sort(np.real(evals))

# Remove zero / near-zero modes
evals = evals[evals > 1e-6]

# Trim edges (very important for clean stats)
cut = len(evals) // 10
evals = evals[cut:-cut]

# ==========================================
# 3. PROPER SPECTRAL UNFOLDING
# ==========================================
print("Performing proper unfolding...")

# Cumulative spectral index
N_E = np.arange(len(evals), dtype=float)

# Smooth it (this estimates the mean spectral trend)
smooth_N_E = gaussian_filter1d(N_E, sigma=20)

# Compute local density
dN = np.diff(smooth_N_E)
dE = np.diff(evals)

local_density = dN / (dE + 1e-12)

# Unfolded spacings
normalized_spacings = dE * local_density

# ==========================================
# 4. STATISTICS
# ==========================================
var_s = np.var(normalized_spacings)
small_s_fraction = np.mean(normalized_spacings < 0.5)
mean_s = np.mean(normalized_spacings)

print("\n=== CLOCKFIELD SPECTRAL RESULTS ===")
print(f"Mean spacing:           {mean_s:.4f}  (target ~1.0)")
print(f"Variance:               {var_s:.4f}  (GUE ~0.178)")
print(f"s < 0.5 fraction:       {small_s_fraction:.4f}  (GUE ~0.07, Poisson ~0.39)")

if var_s < 0.3 and small_s_fraction < 0.2:
    print("VERDICT: Strong GUE-like behavior (Riemann-like)")
elif var_s < 0.6:
    print("VERDICT: Moderate repulsion (approaching GUE)")
else:
    print("VERDICT: Still noisy (Poisson-like)")

# ==========================================
# 5. PLOT
# ==========================================
s_vals = np.linspace(0, 4, 200)

# GUE (Wigner surmise)
wigner_gue = (32 / (np.pi**2)) * (s_vals**2) * np.exp(-(4 / np.pi) * s_vals**2)

# Poisson
poisson = np.exp(-s_vals)

plt.figure(figsize=(11, 6))

plt.hist(
    normalized_spacings,
    bins=60,
    density=True,
    alpha=0.75,
    color='darkviolet',
    label=f'Clockfield\nVar={var_s:.3f}'
)

plt.plot(s_vals, wigner_gue, 'k--', linewidth=2.5, label='GUE (Wigner)')
plt.plot(s_vals, poisson, 'r:', linewidth=2, label='Poisson')

plt.title('Clockfield Spectral Fidelity (Graph, N=500)', fontsize=14)
plt.xlabel('Normalized Spacing (s)', fontsize=12)
plt.ylabel('P(s)', fontsize=12)

plt.xlim(0, 3.5)
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()