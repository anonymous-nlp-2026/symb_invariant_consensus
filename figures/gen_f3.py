import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({'font.size': 9, 'axes.labelsize': 10, 'figure.dpi': 300})

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3))

# Left: K sensitivity
K_vals = [4, 8, 12, 16, 20]
delta_k = [-1.47, -0.98, 4.90, 1.47, 1.96]

ax1.axhline(y=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
ax1.plot(K_vals, delta_k, 'o-', color='#1f77b4', linewidth=1.5, markersize=6, zorder=2)
ax1.plot(12, 4.90, '*', color='#d62728', markersize=14, zorder=3, markeredgecolor='black', markeredgewidth=0.5)
ax1.set_xlabel('Number of Traces ($K$)')
ax1.set_ylabel('SICA $\\Delta$ (pp)')
ax1.set_title('(a) K Sensitivity', fontsize=10)
ax1.set_ylim(-3, 6)
ax1.set_xticks(K_vals)
ax1.tick_params(labelsize=8)

# Right: T sensitivity
T_vals = [0.3, 0.5, 0.7, 1.0]
delta_t = [-0.49, -0.49, 4.90, 0.49]

ax2.axhline(y=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
ax2.plot(T_vals, delta_t, 's-', color='#ff7f0e', linewidth=1.5, markersize=6, zorder=2)
ax2.plot(0.7, 4.90, '*', color='#d62728', markersize=14, zorder=3, markeredgecolor='black', markeredgewidth=0.5)
ax2.set_xlabel('Temperature ($T$)')
ax2.set_ylabel('SICA $\\Delta$ (pp)')
ax2.set_title('(b) T Sensitivity', fontsize=10)
ax2.set_ylim(-3, 6)
ax2.set_xticks(T_vals)
ax2.tick_params(labelsize=8)

plt.tight_layout()
plt.savefig('./figures/f3_kt_sensitivity.pdf', bbox_inches='tight')
plt.savefig('./figures/f3_kt_sensitivity.png', bbox_inches='tight', dpi=300)
plt.close()
print("Figure 3 saved.")
