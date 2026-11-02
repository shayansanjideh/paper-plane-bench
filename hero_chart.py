import json, glob, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "DejaVu Sans"
P = [("gpt-6.1-sol","GPT-6.1 Sol","oai"),("opus","Claude Opus 5.5","ant"),("sonnet","Claude Sonnet 5.5","ant"),
     ("cma_warm","Optimizer, started from a classic glider","ctl"),("gpt-6-luna","GPT-6 Luna","oai"),("gpt-6.1-terra","GPT-6.1 Terra","oai"),
     ("opus_blind","Claude Opus 5.5, blind (never sees results)","ctl"),("haiku","Claude Haiku 5.5","ant"),
     ("cma","Optimizer, random start","ctl"),("random","Random guessing","ctl")]
C = {"ant": "#2f6fd6", "oai": "#e8692c", "ctl": "#8a8f98"}
rows = []
for t, lab, g in P:
    v = []
    for f in sorted(glob.glob(f"results/{t}/run*.json")):
        best = max([a["result"]["distance"] for a in json.load(open(f))["attempts"] if a["result"] and a["result"].get("valid")] or [0])
        v.append(best)
    v = np.array(v); rows.append((lab, g, v.mean(), v.min(), v.max(), len(v)))
rows.sort(key=lambda r: -r[2])
fig = plt.figure(figsize=(12, 7.2), dpi=150, facecolor="white")
ax = fig.add_axes([0.30, 0.17, 0.62, 0.64])
y = np.arange(len(rows))[::-1]
for yi, (lab, g, m, lo, hi, n) in zip(y, rows):
    ax.barh(yi, m, height=0.62, color=C[g], zorder=2)
    ax.plot([lo, hi], [yi, yi], color="#1b1f24", lw=1.1, zorder=3)
    for x in (lo, hi): ax.plot([x, x], [yi - .15, yi + .15], color="#1b1f24", lw=1.1, zorder=3)
    ax.text(0.12, yi, f"{m:.2f} m", va="center", ha="left", fontsize=10, color="white", weight="bold", zorder=4)
ax.axvline(9.98, color="#1b1f24", ls=(0, (3, 3)), lw=1, zorder=1)
ax.text(10.05, len(rows) - 0.45, "best possible\n≈ 9.98 m", ha="left", va="bottom", fontsize=9, color="#4a5260")
ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows], fontsize=10.5)
ax.set_xlim(0, 10.6); ax.set_xticks([0,2,4,6,8,10]); ax.set_ylim(-0.6, len(rows) + 0.1)
ax.set_xlabel("Best distance after 20 tries, metres (line: range across runs)", fontsize=10, color="#4a5260")
ax.grid(axis="x", color="#e6e8ec", zorder=0); ax.tick_params(length=0, colors="#4a5260")
for s in ax.spines.values(): s.set_visible(False)
fig.text(0.03, 0.935, "Paper plane design: AI models vs. an optimizer", fontsize=17, weight="bold", color="#1b1f24")
fig.text(0.03, 0.895, "Simulated flight with air drag, no wind  ·  A4 sheet, 9 fold settings  ·  20 tries each, seeing every result  ·  medium effort  ·  3 runs per model, 10 per control",
         fontsize=9.5, color="#5d6573")
hx = 0.03
for g, name in [("ant","Claude (Anthropic)"),("oai","GPT (OpenAI)"),("ctl","No-AI and blind baselines")]:
    fig.patches.append(matplotlib.patches.Rectangle((hx, 0.845), 0.011, 0.018, color=C[g], transform=fig.transFigure, figure=fig))
    fig.text(hx + 0.016, 0.847, name, fontsize=10, color="#1b1f24"); hx += 0.19
fig.text(0.03, 0.055, "Simulation only, not checked against real paper. Roll and yaw are simplified. Three runs per model: gaps under ~0.3 m are noise.\n"
         "Round 1 was rerun after every model found a loophole (a 5 mm keel). The simulator was written by an AI (Claude), which is also a contestant.",
         fontsize=8.5, color="#5d6573", linespacing=1.5)
fig.savefig("charts/hero.png", facecolor="white")
for r in rows: print(r)
