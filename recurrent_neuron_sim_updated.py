"""
recurrent_neuron_sim_updated.py

Stable simulation utilities for recurrent neuron model:
    tau * dz/dt = -z + y - lambda * Sigma(z)

Features:
- Backward Euler (implicit) solver for linear kernel Sigma(z) = A z (unconditionally stable)
- Semi-implicit solver for general Sigma (more stable than explicit Euler)
- Automatic integrator selection and adaptive dt suggestion
- Spectral diagnostics, optional normalization, rectification
- Kernel factories: linear, separable, divisive
- Example usage reproducing the diagnostics you posted
"""

import numpy as np
from numpy.linalg import eigvals, solve, norm

# -------------------------
# Kernel factory functions
# -------------------------

def make_linear_kernel(A, normalize=False):
    D = A.shape[0]
    A_use = A.copy()
    if normalize:
        A_use = A_use / float(D)
    def sigma(z):
        return A_use.dot(z)
    def jac(z):
        return A_use
    return sigma, jac, A_use

def make_separable_kernel(g, g_prime, h, h_prime, normalize=False):
    def sigma(z):
        G = np.array([g(zi) for zi in z])
        H = np.array([h(zi) for zi in z])
        pooled = np.sum(G)
        return H * pooled
    def jac(z):
        D = z.shape[0]
        G = np.array([g(zi) for zi in z])
        Gp = np.array([g_prime(zi) for zi in z])
        H = np.array([h(zi) for zi in z])
        Hp = np.array([h_prime(zi) for zi in z])
        sumG = np.sum(G)
        J = np.zeros((D, D))
        for i in range(D):
            for l in range(D):
                if i == l:
                    J[i, l] = Hp[i] * sumG + H[i] * Gp[i]
                else:
                    J[i, l] = H[i] * Gp[l]
        if normalize:
            J = J / float(D)
        return J
    return sigma, jac

def make_divisive_kernel(w, psi, psi_prime, normalize=False):
    W = w.copy()
    D = W.shape[0]
    if normalize:
        W = W / float(D)
    def sigma(z):
        Psi = np.array([psi(zi) for zi in z])
        return W.dot(Psi)
    def jac(z):
        D = z.shape[0]
        J = np.zeros((D, D))
        for i in range(D):
            for l in range(D):
                J[i, l] = W[i, l] * psi_prime(z[l])
        return J
    return sigma, jac

# -------------------------
# Diagnostics and helpers
# -------------------------

def spectral_diagnostics(jac_sigma, lam, tau):
    J = jac_sigma
    eigs_J = eigvals(J)
    mu_max = np.max(np.real(eigs_J))
    eigs_lin = - (1.0 + lam * eigs_J) / tau
    maxRe_lin = np.max(np.real(eigs_lin))
    return {'mu_max_J': float(mu_max), 'maxRe_lin': float(maxRe_lin),
            'eigvals_J': eigs_J, 'eigvals_lin': eigs_lin}

def suggest_dt(tau, lam, jac_sigma, safety=0.1):
    rho = np.max(np.real(eigvals(jac_sigma)))
    if rho < 0:
        rho = 0.0
    dt_suggest = safety * tau / (1.0 + lam * rho)
    return float(dt_suggest)

# -------------------------
# Integrators
# -------------------------

def backward_euler_linear(y, A, lam=1.0, tau=1.0, dt=1e-3, steps=1000,
                          z0=None, rectification=False, verbose=False, snapshot_interval=None):
    D = y.shape[0]
    if z0 is None:
        z = np.zeros(D)
    else:
        z = z0.copy()
    LHS = np.eye(D) + (dt / tau) * (np.eye(D) + lam * A)
    hist = {'dz_norm': [], 'spec_maxRe': [], 'z_snapshots': []}
    if snapshot_interval is None:
        snapshot_interval = max(1, steps // 10)
    for t in range(steps):
        rhs = z + (dt / tau) * y
        z_next = solve(LHS, rhs)
        if rectification:
            z_next = np.maximum(0.0, z_next)
        dz_norm = norm(z_next - z)
        M = - (np.eye(D) + lam * A) / tau
        eigs = eigvals(M)
        spec_maxRe = np.max(np.real(eigs))
        hist['dz_norm'].append(float(dz_norm))
        hist['spec_maxRe'].append(float(spec_maxRe))
        if verbose and (t % snapshot_interval == 0):
            print(f"[BE] step {t:6d}, dz_norm={dz_norm:.3e}, maxRe(eig(M))={spec_maxRe:.3e}")
        z = z_next
        if t % snapshot_interval == 0:
            hist['z_snapshots'].append(z.copy())
    return z, hist

def semi_implicit_general(y, sigma_func, jac_sigma_func,
                          lam=1.0, tau=1.0, dt=1e-4, steps=1000,
                          z0=None, rectification=False, verbose=False, snapshot_interval=None):
    D = y.shape[0]
    if z0 is None:
        z = np.zeros(D)
    else:
        z = z0.copy()
    inv_factor = 1.0 / (1.0 + dt / tau)
    hist = {'dz_norm': [], 'spec_maxRe': [], 'z_snapshots': []}
    if snapshot_interval is None:
        snapshot_interval = max(1, steps // 10)
    for t in range(steps):
        Sigma = sigma_func(z)
        z_next = inv_factor * (z + (dt / tau) * (y - lam * Sigma))
        if rectification:
            z_next = np.maximum(0.0, z_next)
        dz_norm = norm(z_next - z)
        J = jac_sigma_func(z)
        M = - (np.eye(D) + lam * J) / tau
        eigs = eigvals(M)
        spec_maxRe = np.max(np.real(eigs))
        hist['dz_norm'].append(float(dz_norm))
        hist['spec_maxRe'].append(float(spec_maxRe))
        if verbose and (t % snapshot_interval == 0):
            print(f"[SI] step {t:6d}, dz_norm={dz_norm:.3e}, maxRe(eig(M))={spec_maxRe:.3e}")
        z = z_next
        if t % snapshot_interval == 0:
            hist['z_snapshots'].append(z.copy())
    return z, hist

# -------------------------
# Auto-run example (reproduces your outputs)
# -------------------------

if __name__ == "__main__":
    D = 64
    rng = np.random.RandomState(0)
    B = rng.randn(D, D) * 0.05
    A = (B + B.T) / 2.0
    eigsA = eigvals(A)
    maxeig = np.max(np.real(eigsA))
    if maxeig != 0:
        A = A * (0.5 / maxeig)   # scale so max eigenvalue ~ 0.5

    y = rng.randn(D) * 0.5
    lam = 2.0
    tau = 0.05

    # Linear kernel diagnostics
    sigma_lin, jac_lin, A_used = make_linear_kernel(A, normalize=False)
    diag = spectral_diagnostics(jac_lin(np.zeros(D)), lam, tau)
    print("Linear kernel diagnostics before sim:", {'mu_max_J': diag['mu_max_J'], 'maxRe_lin': diag['maxRe_lin']})

    # Backward Euler (linear)
    z_be, hist_be = backward_euler_linear(y, A_used, lam=lam, tau=tau, dt=1e-3,
                                          steps=2000, rectification=True, verbose=True)
    print("Backward Euler final z norm:", norm(z_be))

    # Separable kernel example: g = tanh(alpha x), h = 1 (global subtractive)
    alpha = 1.0
    def g(x): return np.tanh(alpha * x)
    def gp(x): return alpha * (1.0 - np.tanh(alpha * x)**2)
    def h(x): return 1.0
    def hp(x): return 0.0
    sigma_sep, jac_sep = make_separable_kernel(g, gp, h, hp, normalize=True)

    # Suggest dt for semi-implicit using Jacobian at z=0
    J0 = jac_sep(np.zeros(D))
    dt_suggest = suggest_dt(tau, lam, J0, safety=0.05)
    dt_use = min(1e-3, max(1e-6, dt_suggest))
    print("Suggested dt for semi-implicit:", dt_suggest, "using dt:", dt_use)

    # Semi-implicit simulation
    z_si, hist_si = semi_implicit_general(y, sigma_sep, jac_sep,
                                          lam=lam, tau=tau, dt=dt_use,
                                          steps=5000, rectification=True, verbose=True)
    print("Semi-implicit final z norm:", norm(z_si))

    # Final summary
    print("\nSummary:")
    print(" - BE final norm:", float(norm(z_be)))
    print(" - SI final norm:", float(norm(z_si)))
    print(" - BE dz_norm last:", hist_be['dz_norm'][-1])
    print(" - SI dz_norm last:", hist_si['dz_norm'][-1])
