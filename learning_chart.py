import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from charts import load
plt.rcParams["font.family"] = "DejaVu Sans"
INK, MUT = "#1b1f24", "#5d6573"
HL = [("sonnet","Claude Sonnet 5.5","#2f6fd6"),("opus","Claude Opus 5.5","#7a4fd0"),("gpt-6.1-sol","GPT-6.1 Sol","#e8692c")]
BG_ = [("gpt-6-luna","GPT-6 Luna"),("gpt-6.1-terra","GPT-6.1 Terra"),("haiku","Claude Haiku 5.5")]
CTL = [("cma_warm","Optimizer, classic start","#4a5260",(0,(5,3))),("random","Random guessing","#9aa0aa",(0,(1.5,2.5)))]
x = np.arange(1, 21)
fig = plt.figure(figsize=(12, 7.2), dpi=150, facecolor="white")
ax = fig.add_axes([0.07, 0.13, 0.68, 0.66])
ax.axhline(9.98, color=INK, lw=1, ls=(0,(3,3)))
ax.text(1.2, 10.04, "best possible ≈ 9.98 m", fontsize=9.5, color=MUT, va="bottom")
ends = []
for t, lab in BG_:
    m = load(t).mean(0); ax.plot(x, m, color="#c9ced6", lw=1.6, zorder=2); ends.append((m[-1], lab, "#8a909a", False))
for t, lab, col, ls in CTL:
    m = load(t).mean(0); ax.plot(x, m, color=col, lw=1.8, ls=ls, zorder=3); ends.append((m[-1], lab, col, False))
for t, lab, col in HL:
    m = load(t).mean(0); ax.plot(x, m, color=col, lw=3, zorder=4, solid_capstyle="round"); ends.append((m[-1], lab, col, True))
    k = int(np.argmax(m >= 9)); ax.plot(x[k], m[k], "o", color=col, ms=8, mec="white", mew=1.5, zorder=5)
ends.sort(key=lambda e: -e[0]); prev = 10.6
for v, lab, col, bold in ends:
    y = min(v, prev - 0.24); prev = y
    ax.text(20.4, y, f"{lab}  {v:.2f} m", color=col, fontsize=10.5, va="center", weight="bold" if bold else "normal", clip_on=False)
    ax.plot([20.05, 20.3], [v, y], color="#d5d9df", lw=0.8, clip_on=False)
ax.set_xlim(1, 20); ax.set_ylim(4, 10.4); ax.set_xticks([1, 5, 10, 15, 20])
ax.set_xlabel("Attempt", fontsize=10.5, color=MUT); ax.set_ylabel("Best distance so far, metres (mean of runs)", fontsize=10.5, color=MUT)
ax.grid(axis="y", color="#e6e8ec"); ax.tick_params(length=0, colors=MUT)
for s in ax.spines.values(): s.set_visible(False)
fig.text(0.03, 0.935, "How fast each model learned", fontsize=17, weight="bold", color=INK)
fig.text(0.03, 0.895, "Best distance so far after each try  ·  dot marks where the line first passes 9 m  ·  3 runs per model, 10 per baseline", fontsize=9.5, color=MUT)
fig.text(0.03, 0.04, "Simulation only, with air drag and no wind. Grey lines: the other three models.", fontsize=8.5, color=MUT)
fig.savefig("charts/learning_curves_light.png", facecolor="white")
