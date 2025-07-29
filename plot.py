import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

# uniform, 0.1, 0.2, 0.5, 0.9, 1.0

X = [0.0, 0.1, 0.2, 0.5, 0.9, 1.0]

cc_ds = {
    "relu": [0.35, 0.35, 0.36, 0.35, 0.32, 0.29],
    "jumprelu": [0.24, 0.35, 0.22, 0.32, 0.34, 0.37],
    "topk": [0.49, 0.51, 0.69, 0.54, 0.60, 0.53],
    "spade": [],
    "neuron": 0.43,
    "random": 0.13
}

cc_sp = {
    "relu": [0.52, 0.52, 0.52, 0.55, 0.58, 0.55],
    "jumprelu": [0.45, 0.61, 0.56, 0.56, 0.59, 0.51],
    "topk": [0.61, 0.63, 0.52, 0.46, 0.38, 0.39],
    "spade": [],
    "neuron": 0.58,
    "random": 0.08
}

cc_overall = {
    "relu": [0.37, 0.37, 0.37, 0.38, 0.38, 0.35],
    "jumprelu": [0.33, 0.47, 0.34, 0.35, 0.38, 0.37],
    "topk": [0.56, 0.56, 0.56, 0.54, 0.50, 0.47],
    "spade": [],
    "neuron": 0.46,
    "random": 0.12
}

colors = {"neuron": 'r', "random": 'k'}

# Create subplots
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Plot cc_ds
for method, values in cc_ds.items():
    if values:  # Only plot if values exist
        if type(values) is list and len(values) > 1:
            axes[0].plot(X, values, marker='o', label=method, linewidth=2)
        elif type(values) is float:
            axes[0].axhline(values, linestyle="--", label=method, linewidth=2, color=colors[method])
# axes[0].set_title('MCC (domain=science)')
axes[0].set_xlabel(r'$\rho$(domain=science, sentiment=positive)')
axes[0].set_ylabel('MCC (domain=science)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot cc_sp
for method, values in cc_sp.items():
    if values:  # Only plot if values exist
        if type(values) is list and len(values) > 1:
            axes[1].plot(X, values, marker='o', label=method, linewidth=2)
        elif type(values) is float:
            axes[1].axhline(values, linestyle="--", label=method, linewidth=2, color=colors[method])
# axes[1].set_title('MCC (sentiment=positive)')
axes[1].set_xlabel(r'$\rho$(domain=science, sentiment=positive)')
axes[1].set_ylabel('MCC (sentiment=positive)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Plot cc_overall
for method, values in cc_overall.items():
    if values:  # Only plot if values exist
        if type(values) is list and len(values) > 1:
            axes[2].plot(X, values, marker='o', label=method, linewidth=2)
        elif type(values) is float:
            axes[2].axhline(values, linestyle="--", label=method, linewidth=2, color=colors[method])
# axes[2].set_title('MCC (overall)')
axes[2].set_xlabel(r'$\rho$(domain=science, sentiment=positive)')
axes[2].set_ylabel('MCC (overall)')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("mcc_by_corr.pdf", format="pdf", bbox_inches="tight")