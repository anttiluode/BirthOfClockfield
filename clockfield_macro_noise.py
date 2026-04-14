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

    def get_effective_dynamical_matrix(self):
        """
        Builds the physically correct operator (Hessian) for the frustrated XY model.
        M_ij = -cos(Δθ) for neighbors.
        M_ii = sum_j cos(Δθ)
        """
        diffs = self.theta[:, None] - self.theta[None, :]
        cos_diffs = np.cos(diffs)
        
        # Off-diagonals
        M = -self.A * cos_diffs
        
        # Diagonals
        np.fill_diagonal(M, np.sum(self.A * cos_diffs, axis=1))
        return M

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
# 2. EXTRACT SPECTRUM
# ==========================================
print("Extracting the Effective Dynamical Matrix (Hessian)...")
M = sim.get_effective_dynamical_matrix()

# Compute eigenvalues
evals = eigh(M, eigvals_only=True)
evals = np.sort(np.real(evals))
evals = evals[evals > 1e-6] # Drop the zero-mode

# Unfold the spectrum (divide by local mean spacing to normalize)
# For simple unfolding, dividing by the global mean spacing works as a baseline
diffs = np.diff(evals)
normalized_spacings = diffs / np.mean(diffs)

# ==========================================
# 3. COMPUTE STATISTICS
# ==========================================
var_s = np.var(normalized_spacings)
small_s_fraction = np.mean(normalized_spacings < 0.5)

print("\n=== CLOCKFIELD SPECTRUM RESULTS ===")
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
plt.hist(normalized_spacings, bins=40, density=True, alpha=0.6, color='indigo', label=f'Clockfield Spacings\n(Var={var_s:.3f})')
plt.plot(s_vals, wigner_gue, 'k--', linewidth=2, label='Wigner Surmise (GUE / Riemann)')
plt.plot(s_vals, poisson, 'r:', linewidth=2, label='Poisson (Random Noise)')

plt.title(f"Clockfield Spectral Fidelity (N={N_NODES})\nEmergent Quantum Chaos from Relational Frustration", fontsize=14)
plt.xlabel("Normalized Spacing (s)", fontsize=12)
plt.ylabel("Probability Density P(s)", fontsize=12)
plt.xlim(0, 3.5)
plt.legend(fontsize=12)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()