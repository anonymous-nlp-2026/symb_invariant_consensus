import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({'font.size': 9, 'axes.labelsize': 10, 'figure.dpi': 300})

# Data: (SC%, delta_pp, label, color, marker)
data = [
    (54.41, 4.90, 'Mistral s42', '#1f77b4', 'o'),
    (56.37, 3.43, 'Mistral s123', '#1f77b4', 'o'),
    (56.37, -0.49, 'Mistral s456', '#1f77b4', 'o'),
    (39.33, -2.83, 'Mistral PW', '#1f77b4', '^'),
    (57.35, -0.49, 'Mistral T=0.3', '#1f77b4', 's'),
    (57.84, -0.49, 'Mistral T=0.5', '#1f77b4', 's'),
    (55.39, 0.49, 'Mistral T=1.0', '#1f77b4', 's'),
    (75.49, -0.98, 'Qwen2.5 FOLIO', '#d62728', 'D'),
    (78.50, -0.50, 'Qwen2.5 LogiQA', '#d62728', 'v'),
    (73.50, -2.00, 'Qwen2.5 PW', '#d62728', '^'),
    (80.39, 0.49, 'Qwen3 FOLIO', '#2ca02c', 'D'),
    (63.24, -2.45, 'LLaMA-8B', '#ff7f0e', 'o'),
    (75.49, 14.22, 'Oracle FOL', '#FFD700', '*'),
]

fig, ax = plt.subplots(figsize=(3.5, 3.0))

# Goldilocks zone shading
ax.axhspan(-20, 20, xmin=0, xmax=1, alpha=0.0)
ax.axvspan(50, 60, alpha=0.08, color='green', label='Goldilocks Zone')

# Zero line
ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)

for sc, delta, label, color, marker in data:
    size = 80 if marker == '*' else 40
    ax.scatter(sc, delta, c=color, marker=marker, s=size, edgecolors='black', linewidths=0.5, zorder=3)

# Legend by model family
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#1f77b4', markersize=7, label='Mistral-7B', markeredgecolor='black', markeredgewidth=0.5),
    Line2D([0], [0], marker='D', color='w', markerfacecolor='#d62728', markersize=7, label='Qwen2.5-14B', markeredgecolor='black', markeredgewidth=0.5),
    Line2D([0], [0], marker='D', color='w', markerfacecolor='#2ca02c', markersize=7, label='Qwen3-14B', markeredgecolor='black', markeredgewidth=0.5),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#ff7f0e', markersize=7, label='LLaMA-8B', markeredgecolor='black', markeredgewidth=0.5),
    Line2D([0], [0], marker='*', color='w', markerfacecolor='#FFD700', markersize=10, label='Oracle FOL', markeredgecolor='black', markeredgewidth=0.5),
]
ax.legend(handles=legend_elements, fontsize=7, loc='upper left', framealpha=0.9)

ax.set_xlabel('SC Accuracy (%)')
ax.set_ylabel('SICA $\\Delta$ (pp)')
ax.set_xlim(35, 85)
ax.set_ylim(-5, 16)
ax.tick_params(labelsize=8)

plt.tight_layout()
plt.savefig('./figures/f2_sc_vs_delta.pdf', bbox_inches='tight')
plt.savefig('./figures/f2_sc_vs_delta.png', bbox_inches='tight', dpi=300)
plt.close()
print("Figure 2 saved.")
