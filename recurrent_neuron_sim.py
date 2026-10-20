"""
recurrent_neuron_sim.py

Stable simulation utilities for recurrent neuron model:
    tau * dz/dt = -z + y - lambda * Sigma(z)

Features:
- Backward Euler (implicit) solver for linear kernel Sigma(z) = A z (unconditionally stable)
- Semi-implicit solver for general Sigma (more stable than explicit Euler)
- Spectral diagnostics (max real part of linearization)
- Optional normalization by D and rectification (z = max(0, z))
- Helper functions to build common kernels (linear, separable, divisive)
- Example usage at bottom
"""

import numpy as np
from numpy.linalg import eigvals, solve, norm

# -------------------------
# Kernel factory functions
# -------------------------

def make_linear_kernel(A, normalize=False):
    """Return sigma_func and jacobian for linear kernel Sigma(z) = A z.
    If normalize=True, uses A_normalized = A / D (D = A.shape[0]).
    """
    D = A.shape[0]
    if normalize:
        A = A / float(D)
    def sigma(z):
        return A.dot(z)
    def jac(z):
        return A
    return sigma, jac

def make_separable_kernel(g, g_prime, h, h_prime, normalize=False):
    """Separable kernel sigma(z_k, z_i) = g(z_k) * h(z_i).
    g, g_prime, h, h_prime are callables.
    """
    def sigma(z):
        G = np.array([g(zi) for zi in z])
        H = np.array([h(zi) for zi in z])
        return H * np.sum(G)
    def jac(z):
        D = z.shape[0]
        G = np.array([g(zi) for zi in z])
        Gp = np.array([g_prime(zi) for zi in z])
        H = np.array([h(zi) for zi in z])
        Hp = np.array([h_prime(zi) for zi in z])
        # J = h'(z_i) * sum_k g(z_k)  on diagonal + h(z_i) * g'(z_l) off-diagonal
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
    """Divisive normalization kernel used inside denominator:
       z_i = y_i / (1 + lambda * sum_k w_ik * psi(z_k))
       Here we return Sigma(z) = sum_k w_ik * psi(z_k) (the pooled term).
    """
    D = w.shape[0]
    W = w.copy()
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
# Integrators
# -------------------------

def backward_euler_linear(y, A, lam=1.0, tau=1.0, dt=1e-3, steps=1000,
                          z0=None, rectification=False, verbose=False):
    """
    Backward Euler for linear kernel Sigma(z) = A z.
    Solves: (I + (dt/tau)*(I + lam*A)) z_{n+1} = z_n + (dt/tau) * y
    Unconditionally stable for linear case.
    Returns final z and diagnostics history.
    """
    D = y.shape[0]
    if z0 is None:
        z = np.zeros(D)
    else:
        z = z0.copy()
    # Precompute left-hand matrix and factorize
    LHS = np.eye(D) + (dt / tau) * (np.eye(D) + lam * A)
    # For moderate D, direct solve is fine; for large D, use sparse solvers
    hist = {'dz_norm': [], 'spec_maxRe': [], 'z_snapshots': []}
    for t in range(steps):
        rhs = z + (dt / tau) * y
        z_next = solve(LHS, rhs)
        if rectification:
            z_next = np.maximum(0.0, z_next)
        dz_norm = norm(z_next - z)
        # Linearization matrix M = - (I + lam * A) / tau
        M = - (np.eye(D) + lam * A) / tau
        eigs = eigvals(M)
        spec_maxRe = np.max(np.real(eigs))
        hist['dz_norm'].append(dz_norm)
        hist['spec_maxRe'].append(spec_maxRe)
        if verbose and (t % max(1, steps//10) == 0):
            print(f"[BE] step {t:5d}, dz_norm={dz_norm:.3e}, maxRe(eig(M))={spec_maxRe:.3e}")
        z = z_next
        if t % max(1, steps//10) == 0:
            hist['z_snapshots'].append(z.copy())
    return z, hist

def semi_implicit_general(y, sigma_func, jac_sigma_func,
                          lam=1.0, tau=1.0, dt=1e-4, steps=1000,
                          z0=None, rectification=False, verbose=False):
    """
    Semi-implicit update:
      z_{n+1} = (I + (dt/tau) I)^{-1} [ z_n + (dt/tau) ( y - lam * Sigma(z_n) ) ]
    This is more stable than explicit Euler and cheap to compute.
    Returns final z and diagnostics history.
    """
    D = y.shape[0]
    if z0 is None:
        z = np.zeros(D)
    else:
        z = z0.copy()
    inv_factor = 1.0 / (1.0 + dt / tau)
    hist = {'dz_norm': [], 'spec_maxRe': [], 'z_snapshots': []}
    for t in range(steps):
        Sigma = sigma_func(z)
        z_next = inv_factor * (z + (dt / tau) * (y - lam * Sigma))
        if rectification:
            z_next = np.maximum(0.0, z_next)
        dz_norm = norm(z_next - z)
        # Linearization matrix M = - (I + lam * J_Sigma(z)) / tau
        J = jac_sigma_func(z)
        M = - (np.eye(D) + lam * J) / tau
        eigs = eigvals(M)
        spec_maxRe = np.max(np.real(eigs))
        hist['dz_norm'].append(dz_norm)
        hist['spec_maxRe'].append(spec_maxRe)
        if verbose and (t % max(1, steps//10) == 0):
            print(f"[SI] step {t:5d}, dz_norm={dz_norm:.3e}, maxRe(eig(M))={spec_maxRe:.3e}")
        z = z_next
        if t % max(1, steps//10) == 0:
            hist['z_snapshots'].append(z.copy())
    return z, hist

# -------------------------
# Utilities
# -------------------------

def spectral_diagnostics(jac_sigma, lam, tau):
    """Return spectral diagnostics for given Jacobian J_Sigma."""
    J = jac_sigma
    eigs_J = eigvals(J)
    mu_max = np.max(np.real(eigs_J))
    # Linearization matrix M = - (I + lam * J) / tau
    eigs_lin = - (1.0 + lam * eigs_J) / tau
    maxRe_lin = np.max(np.real(eigs_lin))
    return {'mu_max_J': mu_max, 'maxRe_lin': maxRe_lin, 'eigvals_J': eigs_J, 'eigvals_lin': eigs_lin}

def suggest_dt(tau, lam, jac_sigma, safety=0.1):
    """Suggest a conservative dt for explicit Euler stability:
       dt <= safety * tau / (1 + lam * rho(J))
    """
    rho = np.max(np.real(eigvals(jac_sigma)))
    if rho < 0:
        rho = 0.0
    dt_suggest = safety * tau / (1.0 + lam * rho)
    return dt_suggest

# -------------------------
# Example usage
# -------------------------

if __name__ == "__main__":
    # Example parameters (modify as needed)
    D = 64
    rng = np.random.RandomState(0)
    # Build a small random symmetric A with controlled spectral radius
    B = rng.randn(D, D) * 0.05
    A = (B + B.T) / 2.0
    # Optionally scale A to have max eigenvalue ~ 0.5
    eigsA = eigvals(A)
    maxeig = np.max(np.real(eigsA))
    if maxeig != 0:
        A = A * (0.5 / maxeig)

    y = rng.randn(D) * 0.5
    lam = 2.0
    tau = 0.05

    # Linear kernel simulation (backward Euler)
    sigma_lin, jac_lin = make_linear_kernel(A, normalize=False)
    # Spectral diagnostics before simulation
    diag = spectral_diagnostics(jac_lin(np.zeros(D)), lam, tau)
    print("Linear kernel diagnostics before sim:", {
        'mu_max_J': diag['mu_max_J'],
        'maxRe_lin': diag['maxRe_lin']
    })
    # Backward Euler is stable even if maxRe_lin > 0 (continuous system stable if maxRe_lin < 0)
    z_be, hist_be = backward_euler_linear(y, A, lam=lam, tau=tau, dt=1e-3,
                                          steps=2000, rectification=True, verbose=True)
    print("Backward Euler final z norm:", norm(z_be))

    # Separable kernel example: g = tanh(alpha x), h = 1 (global subtractive)
    alpha = 1.0
    def g(x): return np.tanh(alpha * x)
    def gp(x): return alpha * (1.0 - np.tanh(alpha * x)**2)
    def h(x): return 1.0
    def hp(x): return 0.0
    sigma_sep, jac_sep = make_separable_kernel(g, gp, h, hp, normalize=True)

    # Semi-implicit simulation for separable kernel
    # Suggest dt based on Jacobian at initial z=0
    J0 = jac_sep(np.zeros(D))
    dt_suggest = suggest_dt(tau, lam, J0, safety=0.05)
    dt_use = min(1e-3, max(1e-6, dt_suggest))
    print("Suggested dt for semi-implicit:", dt_suggest, "using dt:", dt_use)
    z_si, hist_si = semi_implicit_general(y, sigma_sep, jac_sep,
                                          lam=lam, tau=tau, dt=dt_use,
                                          steps=5000, rectification=True, verbose=True)
    print("Semi-implicit final z norm:", norm(z_si))
