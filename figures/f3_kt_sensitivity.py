import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 9,
    'axes.titlesize': 10,
    'axes.labelsize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'lines.linewidth': 1.8,
})

C_BLUE = '#0072B2'
C_STAR = '#D55E00'

# Data
K_vals  = [4, 8, 12, 16, 20]
K_delta = [-1.47, -0.98, +4.90, +1.47, +1.96]

T_vals  = [0.3, 0.5, 0.7, 1.0]
T_delta = [-0.49, -0.49, +4.90, +0.49]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3))

# Left panel: K sensitivity
ax1.axhline(0, color='gray', linestyle='--', linewidth=0.7, alpha=0.5, zorder=1)
ax1.plot(K_vals, K_delta, 'o-', color=C_BLUE, markersize=6, zorder=2)
# Highlight K=12
idx_k12 = K_vals.index(12)
ax1.plot(K_vals[idx_k12], K_delta[idx_k12], '*', color=C_STAR,
         markersize=14, zorder=3, markeredgecolor='black', markeredgewidth=0.5)
ax1.set_xlabel('K (number of samples)')
ax1.set_ylabel('SICA $\\Delta$ (pp)')
ax1.set_title('(a) K Sensitivity', fontsize=10, pad=6)
ax1.set_ylim(-3, 6)
ax1.set_xticks(K_vals)
ax1.grid(True, alpha=0.15, zorder=0)

# Right panel: T sensitivity
ax2.axhline(0, color='gray', linestyle='--', linewidth=0.7, alpha=0.5, zorder=1)
ax2.plot(T_vals, T_delta, 'o-', color=C_BLUE, markersize=6, zorder=2)
# Highlight T=0.7
idx_t07 = T_vals.index(0.7)
ax2.plot(T_vals[idx_t07], T_delta[idx_t07], '*', color=C_STAR,
         markersize=14, zorder=3, markeredgecolor='black', markeredgewidth=0.5)
ax2.set_xlabel('T (temperature)')
ax2.set_ylabel('SICA $\\Delta$ (pp)')
ax2.set_title('(b) T Sensitivity', fontsize=10, pad=6)
ax2.set_ylim(-3, 6)
ax2.set_xticks(T_vals)
ax2.grid(True, alpha=0.15, zorder=0)

plt.tight_layout()

outdir = './figures'
fig.savefig(f'{outdir}/f3_kt_sensitivity.pdf')
fig.savefig(f'{outdir}/f3_kt_sensitivity.png')
plt.close()
print('Saved f3_kt_sensitivity.pdf and .png')
