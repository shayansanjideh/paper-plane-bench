import json, glob, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
BG, INK, MUTED, GRID = "#111316", "#ecebe6", "#9a9a92", "#26292e"
LLM = [("sonnet", "Sonnet 5.5", "#3987e5"), ("opus", "Opus 5.5", "#d95926"), ("gpt-6.1-sol", "GPT-6.1 Sol", "#199e70"),
       ("gpt-6.1-terra", "GPT-6.1 Terra", "#c98500"), ("gpt-6-luna", "GPT-6 Luna", "#d55181"), ("haiku", "Haiku 5.5", "#9085e9")]
CTL = [("opus_blind", "Opus, blind (no feedback)", ":"), ("cma_warm", "CMA-ES from a classic glider", "--"), ("cma", "CMA-ES (random start)", "--"), ("random", "Random search", "-.")]
CEIL = 9.98   # CMA-ES with 600 tries
def load(tag):
    R = [json.load(open(f)) for f in sorted(glob.glob(f"results/{tag}/run*.json"))]
    return np.array([[a["best"] for a in r["attempts"]] for r in R])
def style(ax):
    ax.set_facecolor(BG)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"): ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=10); ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)

# 1) learning curves
fig, ax = plt.subplots(figsize=(10, 6.2), dpi=150, facecolor=BG); style(ax)
x = np.arange(1, 21)
ax.axhline(CEIL, color=MUTED, lw=1, ls=(0, (2, 3)))
ax.text(1.3, CEIL + 0.07, "optimizer's ceiling (CMA-ES with 600 tries)", color=MUTED, fontsize=9, va="bottom")
ends = []
for tag, lab, col in LLM:
    C = load(tag); m = C.mean(0); ax.plot(x, m, color=col, lw=2.2); ends.append((m[-1], lab, col))
for tag, lab, ls in CTL:
    C = load(tag); m = C.mean(0); ax.plot(x, m, color="#8b8f96" if tag != "opus_blind" else "#d95926", lw=1.6, ls=ls, alpha=0.95)
    ends.append((m[-1], lab, MUTED))
ends.sort(reverse=True); ys = []; prev = 10.75
for v, lab, col in ends:                      # de-collide end labels, top-down
    y = min(v, prev - 0.2); prev = y
    ys.append(y); ax.text(20.3, y, f"{lab}  {v:.2f}", color=INK if col != MUTED else MUTED, fontsize=9, va="center")
    ax.plot([20, 20.25], [v, y], color=GRID, lw=0.8)
ax.set_xlim(1, 20); ax.set_ylim(4, 10.6); ax.set_xticks([1, 5, 10, 15, 20])
ax.set_xlabel("attempt", color=MUTED); ax.set_ylabel("best distance so far (m), mean of runs", color=MUTED)
fig.text(0.07, 0.95, "20 tries to design the farthest-flying paper plane", color=INK, fontsize=16, weight="bold")
fig.text(0.07, 0.915, "Same simulator, same throw. LLMs: 3 runs each. Optimizers: 10 runs each.", color=MUTED, fontsize=11)
fig.subplots_adjust(left=0.07, right=0.72, top=0.88, bottom=0.1)
fig.savefig("charts/learning_curves.png", facecolor=BG); plt.close(fig)

# 2) best after 20 tries, with min-max whiskers
rows = [(lab, load(tag)[:, -1], col) for tag, lab, col in LLM] + [(lab, load(tag)[:, -1], "#8b8f96") for tag, lab, _ in CTL]
rows.sort(key=lambda r: r[1].mean())
fig, ax = plt.subplots(figsize=(9, 6), dpi=150, facecolor=BG); style(ax); ax.grid(axis="x", color=GRID, lw=0.8); ax.grid(axis="y", visible=False)
for i, (lab, v, col) in enumerate(rows):
    ax.plot([v.min(), v.max()], [i, i], color=col, lw=1.5, alpha=0.6)
    ax.scatter(v, [i] * len(v), color=col, s=18, alpha=0.5, zorder=3)
    ax.scatter([v.mean()], [i], color=col, s=80, zorder=4, edgecolor=BG, linewidth=2)
    ax.text(max(v.max(), CEIL) + 0.08, i, f"{v.mean():.2f} m", color=INK, fontsize=10, va="center")
ax.axvline(CEIL, color=MUTED, lw=1, ls=(0, (2, 3)))
ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows], color=INK, fontsize=10.5)
ax.set_xlim(4.5, 10.8); ax.set_xlabel("best distance after 20 tries (m); dots = individual runs", color=MUTED)
fig.text(0.02, 0.95, "Where everyone ended up", color=INK, fontsize=16, weight="bold")
fig.subplots_adjust(left=0.3, right=0.95, top=0.9, bottom=0.11)
fig.savefig("charts/final_best.png", facecolor=BG); plt.close(fig)
print("ok")
