import numpy as np
import networkx as nx
from scipy.linalg import eigh
import matplotlib.pyplot as plt

class ClockfieldGraphSim:
    def __init__(self, N=500, k=8, p=0.15, tau=2.737, dt=0.045):
        self.N = N
        self.tau = tau
        self.dt = dt
        
        # Build small-world graph and get adjacency matrix
        self.G = nx.watts_strogatz_graph(N, k, p)
        self.A = nx.to_numpy_array(self.G)
        
        # Initialize random phases
        self.theta = np.random.uniform(0, 2*np.pi, N)
        self.theta_old = np.copy(self.theta)

    def step(self):
        # 1. Vectorized phase differences: Δθ_ij = θ_i - θ_j
        diffs = self.theta[:, None] - self.theta[None, :]
        diffs = np.arctan2(np.sin(diffs), np.cos(diffs)) # Wrap angles
        
        # 2. Compute Frustration (beta)
        beta = np.sum(self.A * (diffs**2), axis=1)
        
        # 3. Compute Metric (gamma)
        gamma = 1.0 / (1.0 + self.tau * beta + 1e-12)**2
        
        # 4. Compute Forces
        force = np.sum(self.A * np.sin(diffs), axis=1) # Note: sin(θ_j - θ_i) is handled by the matrix orientation
        
        # 5. Apply Clockfield Acceleration (scaled by local time metric Γ^2)
        accel = (gamma**2) * force
        
        # 6. Update kinematics
        velocity = self.theta - self.theta_old
        new_theta = self.theta + velocity + (self.dt**2) * accel
        
        self.theta_old = self.theta.copy()
        self.theta = np.mod(new_theta, 2*np.pi)

# ==========================================
# 1. RUN SIMULATION
# ==========================================
N_NODES = 500
STEPS = 5000

print(f"=== Initializing Clockfield Graph (N={N_NODES}) ===")
sim = ClockfieldGraphSim(N=N_NODES)

print(f"Equilibrating {STEPS} steps (Vectorized)...")
for _ in range(STEPS):
    sim.step()
print("Equilibration complete.")

# ==========================================
# 2. EXTRACT SPECTRUM - NOW USING PLAIN LAPLACIAN
# ==========================================
print("Extracting the Plain Graph Laplacian...")

L = nx.laplacian_matrix(sim.G).toarray().astype(float)
evals = eigh(L, eigvals_only=True)
evals = np.sort(np.real(evals))
evals = evals[evals > 1e-6]  # Drop the zero-mode

# Unfold the spectrum
diffs = np.diff(evals)
mean_spacing = np.mean(diffs)
normalized_spacings = diffs / mean_spacing

# ==========================================
# 3. COMPUTE STATISTICS
# ==========================================
var_s = np.var(normalized_spacings)
small_s_fraction = np.mean(normalized_spacings < 0.5)

print("\n=== CLOCKFIELD PLAIN LAPLACIAN SPECTRUM RESULTS (N=500) ===")
print(f"  Variance of spacings: {var_s:.4f} (Target GUE: ~0.178, Poisson: 1.0)")
print(f"  s < 0.5 fraction:     {small_s_fraction:.4f} (Target GUE: ~0.068, Poisson: ~0.39)")

if var_s < 0.5:
    print("  VERDICT: Strong Level Repulsion Detected! (Riemann / GUE behavior)")
else:
    print("  VERDICT: Weak Repulsion (Poisson behavior)")

# ==========================================
# 4. PLOT DISTRIBUTION VS THEORY
# ==========================================
s_vals = np.linspace(0, 4, 200)
# Wigner Surmise for GUE (Quantum Chaos / Riemann Zeros)
wigner_gue = (32 / (np.pi**2)) * (s_vals**2) * np.exp(-(4 / np.pi) * (s_vals**2))
# Poisson (Random/Uncoupled)
poisson = np.exp(-s_vals)

plt.figure(figsize=(10, 6))
plt.hist(normalized_spacings, bins=40, density=True, alpha=0.6, color='indigo', label=f'Clockfield Plain Laplacian\n(Var={var_s:.3f})')
plt.plot(s_vals, wigner_gue, 'k--', linewidth=2, label='Wigner Surmise (GUE / Riemann)')
plt.plot(s_vals, poisson, 'r:', linewidth=2, label='Poisson (Random Noise)')

plt.title(f"Clockfield Spectral Fidelity - Plain Laplacian (N={N_NODES})\nRigid Backbone Emerges from Relational Graph", fontsize=14)
plt.xlabel("Normalized Spacing (s)", fontsize=12)
plt.ylabel("Probability Density P(s)", fontsize=12)
plt.xlim(0, 3.5)
plt.legend(fontsize=12)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# Also save the plot and key stats
np.save('normalized_spacings_plain_laplacian.npy', normalized_spacings)
print("Plot generated and spacings saved.")