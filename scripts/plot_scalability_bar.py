"""Combined bar plot: FedQHD scalability for CartPole and LunarLander."""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams.update({
    'font.family':        'serif',
    'font.serif':         ['DejaVu Serif', 'Times New Roman', 'Times', 'serif'],
    'mathtext.fontset':   'dejavuserif',
    'font.size':          11,
    'axes.titlesize':     11,
    'axes.labelsize':     11,
    'xtick.labelsize':    10,
    'ytick.labelsize':    10,
    'legend.fontsize':    9.5,
    'figure.titlesize':   12,
    'axes.linewidth':     0.8,
    'grid.linewidth':     0.5,
})

N_VALUES = [1, 2, 5, 20]
LAST_N   = 30
BASE_DIR = Path("results/scalability")

# ── Shared palette: same N → same color across both panels ────────────────────
# coolwarm: cool blue → white-center → warm red; trim center whites by using
# only the outer portions so bars stay saturated and distinct
_t     = np.linspace(0.05, 0.95, len(N_VALUES))
COLORS = [plt.get_cmap("coolwarm")(_t[i]) for i in range(len(N_VALUES))]


def load_last_avg(env: str, n: int, method: str = "FedQHD") -> float:
    path = BASE_DIR / env / f"N_{n}" / f"{method}.json"
    with open(path) as f:
        data = json.load(f)
    return float(np.mean(data["reward_history"][-LAST_N:]))


def draw_panel(ax, rewards, n_values, colors, threshold, title, ylabel, yspan=None):
    x     = np.arange(len(n_values))
    width = 0.52

    for xi, h, c in zip(x, rewards, colors):
        rgba = np.array(c)
        edge = np.clip(rgba[:3] * 0.68, 0, 1).tolist() + [1.0]
        ax.bar(xi, h, width, color=c, edgecolor=edge,
               linewidth=0.9, zorder=2)

    # Value labels — offset proportional to reward range
    rng = (yspan[1] - yspan[0]) if yspan else (max(rewards) - min(rewards) + 1)
    offset = rng * 0.022
    for xi, h in zip(x, rewards):
        if h >= 0:
            ax.text(xi, h + offset, f"{h:.0f}",
                    ha="center", va="bottom", fontsize=11,
                    color="#1a1a1a", zorder=4)
        else:
            ax.text(xi, h - offset, f"{h:.0f}",
                    ha="center", va="top", fontsize=11,
                    color="#1a1a1a", zorder=4)

    # Reference lines
    ax.axhline(threshold, color="#c62828", linewidth=1.1, linestyle="--",
               alpha=0.85, zorder=1, label=f"Solved threshold = {threshold}")
    ax.axhline(0, color="black", linewidth=0.5, alpha=0.25, zorder=1)

    ax.set_xticks(x)
    ax.set_xticklabels([f"$N={n}$" for n in n_values])
    ax.set_xlim(-0.55, len(n_values) - 0.45)
    ax.set_xlabel("Number of Agents ($N$)", fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.set_title(title, fontsize=14, pad=8)

    if yspan:
        ax.set_ylim(*yspan)

    ax.legend(fontsize=11, framealpha=0.92, edgecolor="#cccccc", loc="upper left")
    ax.grid(axis="y", alpha=0.18, linestyle="--", linewidth=0.6, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.7)
    ax.spines["bottom"].set_linewidth(0.7)


# ── Load data ─────────────────────────────────────────────────────────────────
cp_rewards = [load_last_avg("CartPole",    n) for n in N_VALUES]
ll_rewards = [load_last_avg("LunarLander", n) for n in N_VALUES]

# ── Figure ────────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))

draw_panel(ax1, cp_rewards, N_VALUES, COLORS,
           threshold=475,
           title="CartPole-v1",
           ylabel=f"Avg. Reward (last {LAST_N} ep.)",
           yspan=(200, 530))

draw_panel(ax2, ll_rewards, N_VALUES, COLORS,
           threshold=200,
           title="LunarLander-v3",
           ylabel=f"Avg. Reward (last {LAST_N} ep.)",
           yspan=(100, 280))

# fig.suptitle("FedQHD Scalability with Number of Agents",
            #  fontsize=14, y=1.01)

plt.tight_layout(w_pad=3.5)

out_dir = Path("results/figures")
out_dir.mkdir(parents=True, exist_ok=True)
for ext in ("pdf", "png"):
    p = out_dir / f"scalability_bar_combined.{ext}"
    plt.savefig(p, dpi=600, bbox_inches="tight")
    print(f"Saved {p}")

plt.show()
