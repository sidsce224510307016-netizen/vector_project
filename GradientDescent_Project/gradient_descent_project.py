"""
====================================================================
 SEM-4 Vector Calculus Project
 Topic 1: Gradient Descent for Finding Minimum Error in AI Models
 Student: [Your Name] | Enrollment No: [Your Enrollment No]
 4th Semester – Practical Work
====================================================================
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D

# ─────────────────────────────────────────────────────────────────
# SECTION 1: Linear Regression using Gradient Descent
# Problem: Predict House Prices based on Area (sq ft)
# ─────────────────────────────────────────────────────────────────

np.random.seed(42)

# ── 1.1  Dataset ──────────────────────────────────────────────────
X_raw = np.array([650, 800, 1000, 1200, 1500, 1800, 2100, 2500,
                  3000, 3500, 4000, 4500], dtype=float)
Y     = np.array([45,  55,  65,   80,   95,   115,  135,  160,
                  195, 230, 265,  300], dtype=float)   # price in ₹ lakhs

# Feature normalisation  (z-score)
X_mean, X_std = X_raw.mean(), X_raw.std()
X = (X_raw - X_mean) / X_std
m = len(X)

# ── 1.2  Gradient Descent ─────────────────────────────────────────

def compute_loss(w, b, X, Y):
    """Mean Squared Error  =  (1/2m) Σ (ŷ - y)²"""
    predictions = w * X + b
    return (1 / (2 * m)) * np.sum((predictions - Y) ** 2)

def gradient_descent(X, Y, w_init=0.0, b_init=0.0,
                     lr=0.1, epochs=1000):
    w, b      = w_init, b_init
    loss_hist = []
    w_hist    = [w]
    b_hist    = [b]

    for epoch in range(epochs):
        preds = w * X + b
        error = preds - Y

        # ∂L/∂w  =  (1/m) Σ error · x     (gradient w.r.t. weight)
        # ∂L/∂b  =  (1/m) Σ error          (gradient w.r.t. bias)
        dw = (1 / m) * np.dot(error, X)
        db = (1 / m) * np.sum(error)

        w -= lr * dw
        b -= lr * db

        loss = compute_loss(w, b, X, Y)
        loss_hist.append(loss)
        w_hist.append(w)
        b_hist.append(b)

    return w, b, loss_hist, w_hist, b_hist

w_final, b_final, loss_history, w_hist, b_hist = gradient_descent(
    X, Y, lr=0.1, epochs=500)

print("=" * 60)
print("  Gradient Descent – Linear Regression (House Price)")
print("=" * 60)
print(f"  Final weight  (w): {w_final:.4f}")
print(f"  Final bias    (b): {b_final:.4f}")
print(f"  Initial Loss     : {loss_history[0]:.4f}")
print(f"  Final   Loss     : {loss_history[-1]:.4f}")
print(f"  Loss Reduction   : {(loss_history[0]-loss_history[-1])/loss_history[0]*100:.2f}%")
print()

# ── 1.3  Predictions ──────────────────────────────────────────────
test_areas = [900, 1600, 2800, 3800]
print("  Sample Predictions:")
print(f"  {'Area (sq ft)':<15} {'Predicted Price (₹ L)':<25} {'Formula'}")
print("  " + "-" * 55)
for area in test_areas:
    x_norm = (area - X_mean) / X_std
    price  = w_final * x_norm + b_final
    print(f"  {area:<15} {price:<25.2f} w={w_final:.3f}, b={b_final:.3f}")
print()

# ─────────────────────────────────────────────────────────────────
# SECTION 2: Gradient Visualization (3-D Loss Surface)
# ─────────────────────────────────────────────────────────────────

w_range = np.linspace(w_final - 80, w_final + 80, 100)
b_range = np.linspace(b_final - 80, b_final + 80, 100)
W_grid, B_grid = np.meshgrid(w_range, b_range)

Loss_grid = np.zeros_like(W_grid)
for i in range(W_grid.shape[0]):
    for j in range(W_grid.shape[1]):
        Loss_grid[i, j] = compute_loss(W_grid[i, j], B_grid[i, j], X, Y)

# ─────────────────────────────────────────────────────────────────
# SECTION 3: Plots
# ─────────────────────────────────────────────────────────────────

fig = plt.figure(figsize=(18, 14))
fig.patch.set_facecolor('#0f1117')
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

ACCENT = '#00e5ff'
GOLD   = '#ffd700'
RED    = '#ff4d6d'
GRID_C = '#2a2a3e'

# ── Plot 1: Loss vs Epochs ────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
ax1.set_facecolor('#1a1a2e')
ax1.plot(loss_history, color=ACCENT, lw=2)
ax1.fill_between(range(len(loss_history)), loss_history,
                 alpha=0.15, color=ACCENT)
ax1.set_title('Loss Convergence', color='white', fontsize=12, fontweight='bold')
ax1.set_xlabel('Epoch', color='#aaaaaa', fontsize=9)
ax1.set_ylabel('MSE Loss', color='#aaaaaa', fontsize=9)
ax1.tick_params(colors='#888888')
ax1.grid(color=GRID_C, linestyle='--', alpha=0.5)
for spine in ax1.spines.values():
    spine.set_edgecolor('#333355')

# ── Plot 2: Regression Line ───────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
ax2.set_facecolor('#1a1a2e')
ax2.scatter(X_raw, Y, color=GOLD, s=60, zorder=5, label='Actual Data')
X_plot = np.linspace(X_raw.min(), X_raw.max(), 200)
X_plot_norm = (X_plot - X_mean) / X_std
Y_plot = w_final * X_plot_norm + b_final
ax2.plot(X_plot, Y_plot, color=RED, lw=2.5, label='Regression Line')
ax2.set_title('Regression Fit', color='white', fontsize=12, fontweight='bold')
ax2.set_xlabel('Area (sq ft)', color='#aaaaaa', fontsize=9)
ax2.set_ylabel('Price (₹ Lakhs)', color='#aaaaaa', fontsize=9)
ax2.tick_params(colors='#888888')
ax2.legend(facecolor='#2a2a3e', edgecolor='#555577', labelcolor='white', fontsize=8)
ax2.grid(color=GRID_C, linestyle='--', alpha=0.5)
for spine in ax2.spines.values():
    spine.set_edgecolor('#333355')

# ── Plot 3: 3-D Loss Surface ──────────────────────────────────────
ax3 = fig.add_subplot(gs[0, 2], projection='3d')
ax3.set_facecolor('#1a1a2e')
surf = ax3.plot_surface(W_grid, B_grid, Loss_grid,
                        cmap='plasma', alpha=0.8, linewidth=0)
ax3.scatter([w_final], [b_final], [loss_history[-1]],
            color=ACCENT, s=120, zorder=10, label='Minimum')
ax3.set_title('Loss Surface (3D)', color='white', fontsize=12, fontweight='bold')
ax3.set_xlabel('w', color='#aaaaaa', fontsize=8)
ax3.set_ylabel('b', color='#aaaaaa', fontsize=8)
ax3.set_zlabel('Loss', color='#aaaaaa', fontsize=8)
ax3.tick_params(colors='#777777', labelsize=7)

# ── Plot 4: Weight update trace ───────────────────────────────────
ax4 = fig.add_subplot(gs[1, 0])
ax4.set_facecolor('#1a1a2e')
ax4.plot(w_hist, color='#a78bfa', lw=2, label='w (weight)')
ax4.plot(b_hist, color='#34d399', lw=2, label='b (bias)')
ax4.set_title('Parameter Updates', color='white', fontsize=12, fontweight='bold')
ax4.set_xlabel('Epoch', color='#aaaaaa', fontsize=9)
ax4.set_ylabel('Parameter Value', color='#aaaaaa', fontsize=9)
ax4.tick_params(colors='#888888')
ax4.legend(facecolor='#2a2a3e', edgecolor='#555577', labelcolor='white', fontsize=8)
ax4.grid(color=GRID_C, linestyle='--', alpha=0.5)
for spine in ax4.spines.values():
    spine.set_edgecolor('#333355')

# ── Plot 5: Gradient magnitude ────────────────────────────────────
ax5 = fig.add_subplot(gs[1, 1])
ax5.set_facecolor('#1a1a2e')
grad_mag = [abs(loss_history[i] - loss_history[i-1])
            for i in range(1, len(loss_history))]
ax5.semilogy(grad_mag, color='#fb7185', lw=2)
ax5.set_title('Gradient Magnitude (log)', color='white', fontsize=12, fontweight='bold')
ax5.set_xlabel('Epoch', color='#aaaaaa', fontsize=9)
ax5.set_ylabel('|ΔLoss|', color='#aaaaaa', fontsize=9)
ax5.tick_params(colors='#888888')
ax5.grid(color=GRID_C, linestyle='--', alpha=0.5)
for spine in ax5.spines.values():
    spine.set_edgecolor('#333355')

# ── Plot 6: Contour map of loss surface ───────────────────────────
ax6 = fig.add_subplot(gs[1, 2])
ax6.set_facecolor('#1a1a2e')
contour = ax6.contourf(W_grid, B_grid, Loss_grid, levels=30, cmap='inferno')
plt.colorbar(contour, ax=ax6, label='Loss')
ax6.plot([w_final], [b_final], 'c*', markersize=14, label='Optimum')
# gradient path (sampled)
ax6.plot(w_hist[::10], b_hist[::10], 'w--o', markersize=3, lw=1, alpha=0.7, label='GD Path')
ax6.set_title('Loss Contour + GD Path', color='white', fontsize=12, fontweight='bold')
ax6.set_xlabel('w', color='#aaaaaa', fontsize=9)
ax6.set_ylabel('b', color='#aaaaaa', fontsize=9)
ax6.tick_params(colors='#888888')
ax6.legend(facecolor='#2a2a3e', edgecolor='#555577', labelcolor='white', fontsize=8)
for spine in ax6.spines.values():
    spine.set_edgecolor('#333355')

# ── Title ─────────────────────────────────────────────────────────
fig.suptitle('Gradient Descent – Vector Calculus Project  |  SEM-4',
             color='white', fontsize=16, fontweight='bold', y=0.98)

plt.savefig('/home/claude/gradient_descent_plots.png',
            dpi=150, bbox_inches='tight', facecolor='#0f1117')
plt.close()
print("  Plots saved → gradient_descent_plots.png")

# ─────────────────────────────────────────────────────────────────
# SECTION 4: Summary Statistics
# ─────────────────────────────────────────────────────────────────

# R² score
Y_pred = w_final * X + b_final
ss_res = np.sum((Y - Y_pred) ** 2)
ss_tot = np.sum((Y - Y.mean()) ** 2)
r2     = 1 - ss_res / ss_tot

print("=" * 60)
print("  Model Performance Metrics")
print("=" * 60)
print(f"  R² Score  : {r2:.4f}  ({r2*100:.2f}% variance explained)")
print(f"  Final MSE : {loss_history[-1]:.4f}")
print(f"  RMSE      : {np.sqrt(2 * loss_history[-1]):.4f}")
print()
print("  Vector Calculus Concepts Applied:")
print("  • Gradient  ∇L = [∂L/∂w, ∂L/∂b]  → direction of steepest ascent")
print("  • Update rule: θ ← θ − α·∇L      → steepest descent")
print("  • Convergence: ||∇L|| → 0         → minimum reached")
print("=" * 60)
