import numpy as np
from numpy.linalg import eigvals, norm

def simulate_euler(y, sigma_func, jac_sigma_func, z0=None,
                   lam=1.0, tau=1.0, dt=0.01, steps=1000,
                   rectification=False, verbose=True):
    """
    Simulate tau * dz/dt = -z + y - lam * Sigma(z) with explicit Euler.
    - y: (D,) feedforward drive
    - sigma_func(z): returns Sigma(z) as (D,)
    - jac_sigma_func(z): returns Jacobian J_Sigma(z) as (D,D)
    - z0: initial state (D,)
    - lam, tau, dt, steps: simulation params
    - rectification: if True, apply z = max(0, z) after each step
    Returns z, history dict
    """
    D = y.shape[0]
    if z0 is None:
        z = np.zeros(D)
    else:
        z = z0.copy()
    hist = {'z': [], 'dz_norm': [], 'spec_radius': []}
    for t in range(steps):
        Sigma = sigma_func(z)
        dz = (-z + y - lam * Sigma) * (dt / tau)
        z = z + dz
        if rectification:
            z = np.maximum(0.0, z)
        # diagnostics
        dz_norm = norm(dz)
        J = jac_sigma_func(z)
        # linearization matrix eigenvalues
        M = - (np.eye(D) + lam * J) / tau
        eigs = eigvals(M)
        spec_rad = max(np.real(eigs))
        hist['z'].append(z.copy())
        hist['dz_norm'].append(dz_norm)
        hist['spec_radius'].append(spec_rad)
        if verbose and (t % max(1, steps//10) == 0):
            print(f"step {t:5d}, dz_norm={dz_norm:.3e}, maxRe(eig(M))={spec_rad:.3e}")
    return z, hist

# Example: linear kernel A
def make_linear_sigma(A):
    def sigma(z):
        return A.dot(z)
    def jac(z):
        return A
    return sigma, jac

# Example usage
if __name__ == "__main__":
    D = 50
    rng = np.random.RandomState(0)
    # random symmetric positive semidefinite A
    B = rng.randn(D, D) * 0.1
    A = (B + B.T) / 2.0
    y = rng.randn(D)
    sigma_func, jac_func = make_linear_sigma(A)
    z_final, history = simulate_euler(y, sigma_func, jac_func,
                                      lam=2.0, tau=0.05, dt=0.001,
                                      steps=5000, rectification=True)
    print("Final z norm:", np.linalg.norm(z_final))
