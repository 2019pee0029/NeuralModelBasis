# save as neuron_regression.py and run: python neuron_regression.py
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import shapiro, ttest_rel, wilcoxon
import math
import random

# -------------------------
# Synthetic regression data
# -------------------------
def make_synthetic_data(n_samples=2000, input_dim=8, noise_std=0.1, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.randn(n_samples, input_dim).astype(np.float32)
    # ground truth: nonlinear mapping
    W_true = rng.randn(input_dim, 1).astype(np.float32)
    y = np.tanh(X.dot(W_true)).squeeze() + 0.3 * np.sin(X[:, 0]*2.0)  # scalar target
    y += noise_std * rng.randn(n_samples).astype(np.float32)
    return X, y.reshape(-1, 1)

# -------------------------
# Model A: simple MLP
# -------------------------
class MLPPointNeuron(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 1)
    def forward(self, x):
        h = self.act(self.fc1(x))
        out = self.fc2(h)
        return out

# -------------------------
# Model B: lateral interaction layer (linear kernel)
# -------------------------
class LateralLayerLinear(nn.Module):
    """
    Implements z = (I + lambda * A)^{-1} y, where y = W x + b.
    For stability and parameter efficiency we parametrize A = S^T S (PSD).
    """
    def __init__(self, input_dim, hidden_dim, lam=1.0, eps=1e-6, normalize_A=False):
        super().__init__()
        self.W = nn.Linear(input_dim, hidden_dim)
        # param S to build PSD A = S^T S (hidden_dim x hidden_dim)
        self.S = nn.Parameter(0.1 * torch.randn(hidden_dim, hidden_dim))
        self.lam = lam
        self.normalize_A = normalize_A
        self.eps = eps  # small diag for numerical stability
        self.act = nn.ReLU()
        # final readout
        self.readout = nn.Linear(hidden_dim, 1)
    def forward(self, x):
        y = self.W(x)  # (B, hidden_dim)
        # build A = S^T S (PSD)
        S = self.S
        A = S.t().mm(S)  # (hidden_dim, hidden_dim)
        if self.normalize_A:
            A = A / A.shape[0]
        # Build matrix M = I + lam * A  (hidden_dim x hidden_dim)
        # Solve z = M^{-1} y for each batch element
        # For efficiency, factorize M once per forward (same for all batch)
        D = A.shape[0]
        M = torch.eye(D, device=A.device, dtype=A.dtype) + self.lam * A
        # add tiny diag for numerical stability
        M = M + self.eps * torch.eye(D, device=A.device, dtype=A.dtype)
        # solve linear system: z = M^{-1} y^T  (we solve for columns)
        # torch.solve is deprecated; use torch.linalg.solve
        # y has shape (B, D) -> transpose to (D, B) to solve M Z = Y^T
        Yt = y.t()
        Zt = torch.linalg.solve(M, Yt)  # (D, B)
        z = Zt.t()  # (B, D)
        v = self.act(z)
        out = self.readout(v)
        return out


class LateralLayerTanh(nn.Module):
    """Use damped recurrent dynamics with Sigma(z) = (A / D) tanh(z)."""
    def __init__(self, input_dim, hidden_dim, lam=1.0, iterations=20, eps=1e-6):
        super().__init__()
        self.W = nn.Linear(input_dim, hidden_dim)
        self.S = nn.Parameter(0.1 * torch.randn(hidden_dim, hidden_dim))
        self.lam = lam
        self.iterations = iterations
        self.eps = eps
        self.act = nn.ReLU()
        self.readout = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        y = self.W(x)
        A = self.S.t().mm(self.S) / self.S.shape[0]
        bound = torch.linalg.matrix_norm(A.detach(), ord='fro')
        step_size = 1.0 / (1.0 + self.lam * bound + self.eps)
        z = y
        for _ in range(self.iterations):
            sigma = torch.tanh(z).mm(A.t())
            z = z + step_size * (-z + y - self.lam * sigma)
        return self.readout(self.act(z))

# -------------------------
# Training utilities
# -------------------------
def train_model(model, X_train, y_train, X_val, y_val, epochs=200, batch_size=64,
                lr=1e-3, verbose=True, l1_s=0.0):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    N = X_train.shape[0]
    history = {'train_loss': [], 'val_loss': []}
    for ep in range(1, epochs+1):
        model.train()
        perm = np.random.permutation(N)
        losses = []
        for i in range(0, N, batch_size):
            idx = perm[i:i+batch_size]
            xb = torch.from_numpy(X_train[idx]).to(device)
            yb = torch.from_numpy(y_train[idx]).to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            if l1_s:
                if not hasattr(model, 'S'):
                    raise ValueError("l1_s requires a model with a lateral parameter S")
                loss = loss + l1_s * model.S.abs().sum() / model.S.shape[0]
            loss.backward()
            opt.step()
            losses.append(loss.item())
        epoch_train_loss = float(np.mean(losses))
        model.eval()
        with torch.no_grad():
            val_pred = model(torch.from_numpy(X_val).to(device))
            val_loss = loss_fn(val_pred, torch.from_numpy(y_val).to(device)).item()
        history['train_loss'].append(epoch_train_loss)
        history['val_loss'].append(val_loss)
        if ep % max(1, epochs//10) == 0 or ep == 1:
            if verbose:
                print(f"Epoch {ep:4d} train_loss={epoch_train_loss:.6f} val_loss={val_loss:.6f}")
    return model, history


def plot_results(history_a, history_b, mse_a, mse_b, lateral_s, lateral_a):
    fig, (loss_ax, mse_ax, s_ax, a_ax) = plt.subplots(1, 4, figsize=(20, 5))
    epochs_a = range(1, len(history_a['train_loss']) + 1)
    epochs_b = range(1, len(history_b['train_loss']) + 1)

    loss_ax.plot(epochs_a, history_a['train_loss'], label='Model A train')
    loss_ax.plot(epochs_a, history_a['val_loss'], linestyle='--', label='Model A validation')
    loss_ax.plot(epochs_b, history_b['train_loss'], label='Tanh model train')
    loss_ax.plot(epochs_b, history_b['val_loss'], linestyle='--', label='Tanh model validation')
    loss_ax.set(title='Training and validation loss', xlabel='Epoch', ylabel='MSE loss')
    loss_ax.legend()
    loss_ax.grid(True, alpha=0.3)

    bars = mse_ax.bar(['Model A (MLP)', 'Tanh lateral (λ=1)'], [mse_a, mse_b],
                      color=['tab:blue', 'tab:orange'])
    mse_ax.bar_label(bars, fmt='%.4f', padding=3)
    mse_ax.set(title='Final validation MSE', ylabel='MSE')
    mse_ax.grid(axis='y', alpha=0.3)

    s_image = s_ax.imshow(lateral_s, cmap='coolwarm', aspect='auto')
    s_ax.set(title='Learned S', xlabel='Column', ylabel='Row')
    fig.colorbar(s_image, ax=s_ax, fraction=0.046, pad=0.04)

    a_image = a_ax.imshow(lateral_a, cmap='viridis', aspect='auto')
    a_ax.set(title='A = SᵀS', xlabel='Column', ylabel='Row')
    fig.colorbar(a_image, ax=a_ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig('experiment_final_results.png', dpi=200, bbox_inches='tight')
    plt.show()


def inspect_lateral_matrix(model):
    with torch.no_grad():
        lateral_s = model.S.detach().cpu().numpy().copy()
    lateral_a = lateral_s.T @ lateral_s
    effective_a = lateral_a / lateral_a.shape[0]
    eigenvalues = np.linalg.eigvalsh(lateral_a)
    identity = np.eye(lateral_a.shape[0])
    raw_condition = np.linalg.cond(identity + model.lam * lateral_a)
    effective_condition = np.linalg.cond(identity + model.lam * effective_a)

    print("\nLearned lateral matrix diagnostics:")
    print(" Eigenvalues of A=S^T S:",
          np.array2string(eigenvalues, precision=5, separator=', '))
    print(f" Condition number of I + λA: {raw_condition:.6g}")
    print(" Condition number of I + λ(A/D) used by the model: "
          f"{effective_condition:.6g}")
    print(" Note: these condition numbers are linear-system proxies; the tanh model "
          "uses iterative nonlinear dynamics, not a direct solve.")
    return lateral_s, lateral_a

# -------------------------
# Run experiment
# -------------------------
def run_experiment(seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # data
    X, y = make_synthetic_data(n_samples=3000, input_dim=8, noise_std=0.08, seed=seed)
    # split
    n_train = 2400
    X_train, y_train = X[:n_train], y[:n_train]
    X_val, y_val = X[n_train:], y[n_train:]

    input_dim = X.shape[1]
    hidden_dim = 32

    lam = 1.0

    # Model A baseline
    modelA = MLPPointNeuron(input_dim, hidden_dim)
    print("Training Model A (MLP point neuron)...")
    modelA, historyA = train_model(modelA, X_train, y_train, X_val, y_val,
                                   epochs=200, batch_size=64, lr=1e-3)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    modelA.eval()
    with torch.no_grad():
        predA = modelA(torch.from_numpy(X_val).to(device)).cpu().numpy()
    squared_errors_a = np.square(predA.reshape(-1) - y_val.reshape(-1))
    mseA = float(np.mean(squared_errors_a))

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    modelB = LateralLayerTanh(input_dim, hidden_dim, lam=lam, eps=1e-5)
    print(f"\nTraining tanh lateral model (λ={lam:g}, Σ(z)=(A/D)tanh(z))...")
    modelB, historyB = train_model(
        modelB, X_train, y_train, X_val, y_val,
        epochs=200, batch_size=64, lr=1e-3)
    modelB.eval()
    with torch.no_grad():
        predB = modelB(torch.from_numpy(X_val).to(device)).cpu().numpy()
    squared_errors_b = np.square(predB.reshape(-1) - y_val.reshape(-1))
    mseB = float(np.mean(squared_errors_b))

    paired_differences = squared_errors_a - squared_errors_b
    normality_p = float(shapiro(paired_differences).pvalue)
    if np.allclose(paired_differences, 0.0):
        test_name = 'identical errors'
        test_statistic = 0.0
        test_p = 1.0
    elif normality_p >= 0.05:
        test_result = ttest_rel(squared_errors_a, squared_errors_b)
        test_name = 'paired t-test'
        test_statistic = float(test_result.statistic)
        test_p = float(test_result.pvalue)
    else:
        test_result = wilcoxon(squared_errors_a, squared_errors_b)
        test_name = 'Wilcoxon signed-rank test'
        test_statistic = float(test_result.statistic)
        test_p = float(test_result.pvalue)

    print("\nFinal validation MSEs:")
    print(f" Model A (MLP): {mseA:.6f}")
    print(f" Tanh lateral (λ={lam:g}): {mseB:.6f}")
    print("\nPaired per-sample squared-error comparison:")
    print(f" Test: {test_name} (two-sided), statistic={test_statistic:.4f}, p={test_p:.6g}")
    print(f" Shapiro-Wilk p for paired differences: {normality_p:.6g}")
    print(f" Mean squared-error difference (MLP - tanh): "
          f"{np.mean(paired_differences):.6f}")
    if test_p < 0.05:
        better_model = 'tanh lateral model' if np.mean(paired_differences) > 0 else 'MLP'
        print(f" Difference is statistically significant at α=0.05; lower errors: {better_model}.")
    else:
        print(" Difference is not statistically significant at α=0.05.")

    if mseB < mseA:
        print("Interpretation: The tanh lateral model improved validation MSE over the MLP.")
    elif mseA < mseB:
        print("Interpretation: The MLP achieved lower validation MSE than the tanh lateral model.")
    else:
        print("Interpretation: Both models achieved the same validation MSE.")

    lateral_s, lateral_a = inspect_lateral_matrix(modelB)
    plot_results(historyA, historyB, mseA, mseB, lateral_s, lateral_a)

    return modelA, modelB, (X_train, y_train, X_val, y_val)

if __name__ == "__main__":
    run_experiment(seed=42)
