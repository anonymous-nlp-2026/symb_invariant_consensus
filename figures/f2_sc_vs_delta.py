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
    'legend.fontsize': 7,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'lines.linewidth': 1.5,
})

# Okabe-Ito colorblind-friendly palette
C_BLUE   = '#0072B2'
C_RED    = '#D55E00'
C_GREEN  = '#009E73'
C_ORANGE = '#E69F00'
C_GOLD   = '#F0E442'

data = [
    # (label, SC, delta, marker, color, size)
    ('Mistral FOLIO s42',    54.41,  +4.90, 'o', C_BLUE,   40),
    ('Mistral FOLIO s123',   56.37,  +3.43, 'o', C_BLUE,   40),
    ('Mistral FOLIO s456',   56.37,  -0.49, 'o', C_BLUE,   40),
    ('Mistral PW-600',       39.33,  -2.83, '^', C_BLUE,   40),
    ('Mistral FOLIO T=0.3',  57.35,  -0.49, 'o', C_BLUE,   40),
    ('Mistral FOLIO T=0.5',  57.84,  -0.49, 'o', C_BLUE,   40),
    ('Mistral FOLIO T=1.0',  55.39,  +0.49, 'o', C_BLUE,   40),
    ('Qwen2.5-14B FOLIO',    75.49,  -0.98, 's', C_RED,    40),
    ('Qwen2.5-14B LogiQA',   78.50,  -0.50, 'D', C_RED,    40),
    ('Qwen2.5-14B PW-600',   73.50,  -2.00, '^', C_RED,    40),
    ('Qwen3-14B FOLIO',      80.39,  +0.49, 's', C_GREEN,  40),
    ('LLaMA-8B FOLIO',       63.24,  -2.45, 'o', C_ORANGE, 40),
    ('Oracle FOL FOLIO',     75.49, +14.22, '*', C_GOLD,   120),
]

# Separate open-circle (T sensitivity) from filled
T_labels = {'Mistral FOLIO T=0.3', 'Mistral FOLIO T=0.5', 'Mistral FOLIO T=1.0'}

fig, ax = plt.subplots(figsize=(3.5, 3.0))

# Goldilocks zone
ax.axvspan(50, 60, alpha=0.08, color='#0072B2', zorder=0)
ax.text(55, 14.5, 'Goldilocks\nZone', ha='center', va='top',
        fontsize=7, color='#0072B2', alpha=0.6, fontstyle='italic')

# No-effect line
ax.axhline(0, color='gray', linestyle='--', linewidth=0.7, alpha=0.5, zorder=1)

# Plot each point
legend_entries = {}
for label, sc, delta, marker, color, size in data:
    is_open = label in T_labels
    facecolor = 'white' if is_open else color
    edgewidth = 1.2 if is_open else 0.5
    ax.scatter(sc, delta, marker=marker, s=size, c=facecolor,
               edgecolors=color, linewidths=edgewidth, zorder=3)

    # Build legend keys
    if 'Mistral' in label and 'T=' not in label and 'PW' not in label:
        key = 'Mistral-7B FOLIO'
    elif 'Mistral' in label and 'T=' in label:
        key = 'Mistral-7B (T sweep)'
    elif 'Mistral' in label and 'PW' in label:
        key = 'Mistral-7B PW-600'
    elif 'Qwen2.5' in label and 'FOLIO' in label:
        key = 'Qwen2.5-14B FOLIO'
    elif 'Qwen2.5' in label and 'LogiQA' in label:
        key = 'Qwen2.5-14B LogiQA'
    elif 'Qwen2.5' in label and 'PW' in label:
        key = 'Qwen2.5-14B PW-600'
    elif 'Qwen3' in label:
        key = 'Qwen3-14B FOLIO'
    elif 'LLaMA' in label:
        key = 'LLaMA-8B FOLIO'
    elif 'Oracle' in label:
        key = 'Oracle FOL'
    else:
        key = label

    if key not in legend_entries:
        legend_entries[key] = ax.scatter(
            [], [], marker=marker, s=size, c=facecolor,
            edgecolors=color, linewidths=edgewidth, label=key)

ax.set_xlim(35, 85)
ax.set_ylim(-5, 16)
ax.set_xlabel('SC Accuracy (%)')
ax.set_ylabel('SICA $\\Delta$ (pp)')

handles = list(legend_entries.values())
labels = list(legend_entries.keys())
ax.legend(handles, labels, loc='upper left', fontsize=6,
          framealpha=0.9, edgecolor='none', handletextpad=0.3,
          borderpad=0.4, labelspacing=0.3)

ax.grid(True, alpha=0.15, zorder=0)
plt.tight_layout()

outdir = './figures'
fig.savefig(f'{outdir}/f2_sc_vs_delta.pdf')
fig.savefig(f'{outdir}/f2_sc_vs_delta.png')
plt.close()
print('Saved f2_sc_vs_delta.pdf and .png')
