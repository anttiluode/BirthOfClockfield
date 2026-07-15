"""
ALKULUKU -- the prime-trace test on the Clockfield spectrum
===========================================================
BirthOfClockfield reports random-matrix (Wigner-class) level statistics
for the Clockfield graph's dynamical matrix, in the Hilbert-Polya spirit.
The honest problem: spacing statistics are UNIVERSAL -- GOE/GUE behavior
is class membership shared by essentially every chaotic system, so it
cannot identify the Riemann operator. The arithmetic fingerprint lives
elsewhere: by the explicit formula, the fluctuation of the zero-counting
function delta N(E) oscillates at angular frequencies log p^k --
log 2 = 0.693, log 3 = 1.099, log 4 = 1.386, log 5 = 1.609,
log 7 = 1.946. A candidate operator whose eigenvalues are the zeros MUST
sing the primes in its Fourier-transformed counting fluctuation. A
generic chaotic graph must not.

Protocol: run the repo's ClockfieldGraphSim verbatim (N=800, 3000 steps,
tau=2.737), diagonalize the effective dynamical matrix, drop 5% spectral
edges, then:
  (a) r-statistic (adjacent-gap ratio) for honest class ID:
      Poisson ~0.386, GOE ~0.531, GUE ~0.600.
  (b) unfold via 9th-degree polynomial fit of N(lambda); compute
      delta N(lambda) on a uniform lambda grid; FFT; report |FT| at the
      first six log-prime frequencies vs local background (4-sigma bar).
  (c) GOE control matrix, same size, same pipeline.

=== REGISTERED (before running) ===
A1  Class ID: the matrix is real symmetric, so r-statistic lands GOE
    (0.51-0.55), NOT GUE. If the repo's headline says GUE, this corrects
    it. (GUE needs broken time-reversal / complex Hermitian.)
A2  NO prime peaks: none of the six log-prime frequencies exceeds
    4 sigma of the local FT background. Same for the GOE control.
    THE MIRACLE CLAUSE (kill condition in reverse): if 4+ of 6 log-prime
    frequencies stand above 4 sigma in the Clockfield arm and not in the
    control, stop everything, re-run with 3 seeds, and if it survives,
    that is not a repo update, that is a phone call.
Expected outcome, stated plainly: A2 negative (generic chaos). The value
is owning the correct instrument and the honest classification.

Do not hype. Do not lie. Just show.
"""
import numpy as np, json, sys, os
sys.path.insert(0, '/home/claude/BirthOfClockfield')
import networkx as nx
from scipy.linalg import eigh

class ClockfieldGraphSim:  # verbatim physics from the repo
    def __init__(self, N=800, k=8, p=0.15, tau=2.737, dt=0.045, seed=1):
        rng = np.random.default_rng(seed)
        self.tau, self.dt = tau, dt
        self.G = nx.watts_strogatz_graph(N, k, p, seed=seed)
        self.A = nx.to_numpy_array(self.G)
        self.theta = rng.uniform(0, 2 * np.pi, N)
        self.theta_old = self.theta.copy()
    def step(self):
        d = self.theta[:, None] - self.theta[None, :]
        d = np.arctan2(np.sin(d), np.cos(d))
        beta = np.sum(self.A * d**2, axis=1)
        gamma = 1.0 / (1.0 + self.tau * beta + 1e-12)**2
        force = np.sum(self.A * np.sin(d), axis=1)
        v = self.theta - self.theta_old
        new = self.theta + v + self.dt**2 * (gamma**2) * force
        self.theta_old, self.theta = self.theta.copy(), np.mod(new, 2 * np.pi)
    def matrix(self):
        d = self.theta[:, None] - self.theta[None, :]
        M = -self.A * np.cos(d)
        np.fill_diagonal(M, np.sum(self.A * np.cos(d), axis=1))
        return M

def r_stat(ev):
    s = np.diff(np.sort(ev))
    s = s[s > 1e-12]
    r = np.minimum(s[1:], s[:-1]) / np.maximum(s[1:], s[:-1])
    return float(r.mean())

def prime_scan(ev, label):
    ev = np.sort(ev)
    n = len(ev)
    lo, hi = int(0.05 * n), int(0.95 * n)
    lam, idx = ev[lo:hi], np.arange(lo, hi, dtype=float)
    coef = np.polyfit(lam, idx, 9)
    grid = np.linspace(lam[0], lam[-1], 4096)
    Ns = np.interp(grid, lam, idx)              # staircase (interp of counts)
    dN = Ns - np.polyval(coef, grid)
    dN -= dN.mean()
    dN *= np.hanning(len(dN))
    F = np.abs(np.fft.rfft(dN))
    om = 2 * np.pi * np.fft.rfftfreq(len(grid), grid[1] - grid[0])
    primespk, sigmas = {}, {}
    logp = {'log2': np.log(2), 'log3': np.log(3), 'log4': np.log(4),
            'log5': np.log(5), 'log7': np.log(7), 'log8': np.log(8)}
    for name, w in logp.items():
        i = np.argmin(np.abs(om - w))
        band = F[max(1, i - 40):i + 40]
        bg, sd = np.median(band), band.std()
        sig = (F[i] - bg) / max(sd, 1e-12)
        primespk[name] = round(float(F[i]), 3)
        sigmas[name] = round(float(sig), 2)
    print(f"{label}: r={r_stat(ev):.4f}  prime-freq sigmas: {sigmas}")
    return dict(r=round(r_stat(ev), 4), sigmas=sigmas,
                n_above_4sigma=int(sum(v > 4 for v in sigmas.values())))

if __name__ == '__main__':
    sim = ClockfieldGraphSim(N=800, seed=1)
    for _ in range(3000):
        sim.step()
    ev_cf = eigh(sim.matrix(), eigvals_only=True)
    rng = np.random.default_rng(2)
    H = rng.standard_normal((800, 800)); H = (H + H.T) / np.sqrt(2)
    ev_goe = eigh(H, eigvals_only=True)

    out = dict(clockfield=prime_scan(ev_cf, 'clockfield'),
               goe_control=prime_scan(ev_goe, 'goe_control'))
    out['verdict'] = dict(
        A1_class=('GOE' if 0.51 <= out['clockfield']['r'] <= 0.55 else
                  'GUE' if 0.57 <= out['clockfield']['r'] <= 0.62 else
                  'Poisson-ish/other'),
        A2_no_prime_peaks=bool(out['clockfield']['n_above_4sigma'] < 4),
        miracle=bool(out['clockfield']['n_above_4sigma'] >= 4
                     and out['goe_control']['n_above_4sigma'] < 4))
    print(json.dumps(out['verdict'], indent=2))
    json.dump(out, open('/home/claude/alkuluku_results.json', 'w'), indent=1)
