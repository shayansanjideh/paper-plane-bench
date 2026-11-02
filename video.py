"""Videos from run logs.

  python3 video.py evolution results/opus/run0.json out.mp4     # one model: fold -> throw, 20 attempts
  python3 video.py race results/ out.mp4                        # every model's best plane, thrown together
"""
import json, math, sys, textwrap, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plane as P, render as R

BG, INK, MUTED, GOLD, RED, BLUE = R.BG, R.INK, R.MUTED, "#f2b84b", "#e5534b", "#6cb6ff"
FPS = 30

def median_flight(design):
    g = P.analyse(design); am = P.aero_model(g)
    fl = [P.fly(g, am, *wb, record=True) for wb in P.WOBBLE]
    ds = [f["distance"] for f in fl]; i = int(np.argsort(ds)[len(ds) // 2])
    return fl[i]

def side_axes(ax, xmax):
    ax.set_facecolor(BG); ax.set_xlim(-0.3, xmax); ax.set_ylim(0, 3.2)
    for s in ax.spines.values(): s.set_visible(False)
    ax.tick_params(colors=MUTED, labelsize=8); ax.set_yticks([0, 1, 2, 3])
    ax.axhline(0, color="#3a3f47", lw=1.5)
    ax.set_xlabel("distance (m)", color=MUTED, fontsize=8); ax.set_ylabel("height (m)", color=MUTED, fontsize=8)
    for x in range(0, int(xmax) + 1, 2): ax.axvline(x, color="#20242b", lw=0.8, zorder=0)
    ax.plot([0, 0], [0, P.THROW["height"]], color=MUTED, lw=2)      # the thrower

def plane_icon(ax, x, z, th, color=INK, s=0.35):
    pts = np.array([[0.5, 0], [-0.5, 0.06], [-0.5, -0.06], [0.5, 0]]) * s
    c, si = math.cos(th), math.sin(th)
    P2 = pts @ np.array([[c, si], [-si, c]]) + [x, z]
    ax.fill(P2[:, 0], P2[:, 1], color=color, zorder=5)

def first_sentence(t, n=95):
    t = (t or "").strip().replace("\n", " ")
    return textwrap.shorten(t, n, placeholder="…")

def evolution_frames(log, label):
    atts = log["attempts"]
    flights, prev = [], []
    xmax = max(8.0, max(a["result"]["distance"] for a in atts) + 1.5)
    fig = plt.figure(figsize=(12.8, 7.2), dpi=100, facecolor=BG)
    ax3 = fig.add_axes([0.0, 0.08, 0.42, 0.78], projection="3d")
    ax2 = fig.add_axes([0.47, 0.22, 0.5, 0.56])
    axb = fig.add_axes([0.47, 0.06, 0.5, 0.07])
    best_d, best_i = 0.0, None
    def grab():
        fig.canvas.draw(); return np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    def header(i, a, extra=""):
        fig.texts.clear()
        fig.text(0.03, 0.93, label, color=INK, fontsize=20, weight="bold")
        fig.text(0.03, 0.885, f"attempt {i}/{len(atts)}   best so far {best_d:.2f} m{extra}", color=MUTED, fontsize=12)
        why = (a.get("design") or {}).get("rationale") if isinstance(a.get("design"), dict) else None
        if why: fig.text(0.47, 0.83, "“" + first_sentence(why) + "”", color=INK, fontsize=11, style="italic")
    def bars(upto):
        axb.cla(); axb.set_facecolor(BG); axb.set_xlim(0.4, len(atts) + 0.6); axb.set_ylim(0, xmax)
        axb.axis("off")
        for j, a in enumerate(atts[:upto]):
            d = a["result"]["distance"]
            axb.bar(j + 1, d, color=(GOLD if j + 1 == best_i else RED if not a["result"]["valid"] else "#4a515c"), width=0.7)
        axb.text(0.4, xmax * 1.05, "distance per attempt", color=MUTED, fontsize=8)
    def side(cur=None, k=None):
        ax2.cla(); side_axes(ax2, xmax)
        for j, f in prev:
            tr = np.array(f["traj"])
            ax2.plot(tr[:, 1], tr[:, 3], color=(GOLD if j == best_i else "#3b4250"), lw=(2 if j == best_i else 1), zorder=2)
        if cur is not None:
            tr = np.array(cur["traj"])[: k + 1]
            ax2.plot(tr[:, 1], tr[:, 3], color=BLUE, lw=2.2, zorder=4)
            plane_icon(ax2, tr[-1, 1], max(tr[-1, 3], 0.02), tr[-1, 4], color=INK)
    for i, a in enumerate(atts, 1):
        d = a["design"]; res = a["result"]
        # 1) folding
        if res["valid"]:
            frs = R.frames_for_design.__wrapped__ if hasattr(R.frames_for_design, "__wrapped__") else None
            stages, _ = R.fold_stages(d)
            nf = 4
            for st in stages:
                for t in range(nf + 1):
                    ax3.cla(); R.setup_3d(ax3, lim=100, elev=70, azim=-90, zlim=(-10, 60))
                    R.draw_polys(ax3, R.flat_frame_polys(st, t / nf)); header(i, a); side(); bars(i - 1); yield grab()
            na = 12
            for t in range(na + 1):
                u = t / na; polys, _ = R.assembled(d, min(u * 2, 1), max(0, u * 2 - 1))
                ax3.cla(); R.setup_3d(ax3, lim=100, elev=70 - 48 * u, azim=-90 + 30 * u, zlim=(-10, 110))
                R.draw_polys(ax3, [p for _, p in polys], [R.TAGCOL[k] for k, _ in polys]); header(i, a); side(); bars(i - 1); yield grab()
            # 2) flight, real-time x1.5
            f = median_flight(d); tr = f["traj"]; n = len(tr)
            steps = max(int(f["time"] / 1.5 * FPS), 8)
            for s in range(steps + 1):
                k = min(int(s / steps * (n - 1)), n - 1)
                header(i, a, f"   ·   this one: {tr[k][1]:.2f} m"); side(f, k); bars(i - 1); yield grab()
            if res["distance"] > best_d: best_d, best_i = res["distance"], i
            prev.append((i, f))
            for _ in range(int(0.35 * FPS)):
                header(i, a, f"   ·   this one: {res['distance']:.2f} m" + ("  NEW BEST" if best_i == i else "")); side(); bars(i); yield grab()
        else:
            ax3.cla(); R.setup_3d(ax3, lim=100, elev=70, azim=-90, zlim=(-10, 60))
            for _ in range(int(0.9 * FPS)):
                header(i, a, "   ·   can't be folded"); side(); bars(i)
                fig.text(0.03, 0.5, "✗ " + first_sentence(res.get("error", ""), 60), color=RED, fontsize=13); yield grab()
    # end card: best design spinning
    bd = atts[best_i - 1]["design"]; polys, _ = R.assembled(bd)
    for s in range(int(2.5 * FPS)):
        ax3.cla(); R.setup_3d(ax3, lim=100, elev=22, azim=-60 + 120 * s / (2.5 * FPS), zlim=(-10, 110))
        R.draw_polys(ax3, [p for _, p in polys], [R.TAGCOL[k] for k, _ in polys])
        header(best_i, atts[best_i - 1], f"   ·   best design: attempt {best_i}"); side(); bars(len(atts)); yield grab()
    plt.close(fig)

PAL = ["#6cb6ff", "#f2b84b", "#e5534b", "#57ab5a", "#b083f0", "#f69d50", "#76e3ea", "#ec775c", "#adbac7", "#8ddb8c"]

def race_frames(entries, title="Each model's best plane, same throw"):
    """entries: list of (label, design). All thrown at once (median throw), side view + plane gallery."""
    fl = [(lab, d, median_flight(d)) for lab, d in entries]
    fl.sort(key=lambda e: -e[2]["distance"])
    xmax = max(f["distance"] for _, _, f in fl) + 2.0
    T = max(f["time"] for _, _, f in fl)
    fig = plt.figure(figsize=(12.8, 7.2), dpi=100, facecolor=BG)
    ax2 = fig.add_axes([0.06, 0.12, 0.62, 0.66])
    n = len(fl); gal = []
    for k in range(n):
        r, c = divmod(k, 2)
        gal.append(fig.add_axes([0.70 + c * 0.15, 0.80 - (r + 1) * (0.78 / ((n + 1) // 2)), 0.15, 0.78 / ((n + 1) // 2)], projection="3d"))
    for k, (lab, d, f) in enumerate(fl):
        polys, _ = R.assembled(d); ax = gal[k]
        R.setup_3d(ax, lim=110, elev=22, azim=-60, zlim=(-10, 100)); ax.set_facecolor(BG)
        R.draw_polys(ax, [p for _, p in polys], [R.TAGCOL[t] for t, _ in polys])
        ax.set_title(f"{k + 1}. {lab}", color=PAL[k % len(PAL)], fontsize=9, pad=-2)
    def grab():
        fig.canvas.draw(); return np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    SLOW = 0.35  # sim seconds per video second
    steps = int(T / SLOW * FPS)
    for s in range(steps + int(3.5 * FPS)):
        tt = min(s / FPS * SLOW, T)
        ax2.cla(); side_axes(ax2, xmax); fig.texts.clear()
        fig.text(0.06, 0.92, title, color=INK, fontsize=18, weight="bold")
        fig.text(0.06, 0.87, f"A4 paper, thrown at 6 m/s from 1.8 m, no wind   ·   {SLOW:.2g}x speed", color=MUTED, fontsize=11)
        landed = []
        for k, (lab, d, f) in enumerate(fl):
            tr = np.array(f["traj"]); m = tr[:, 0] <= tt + 1e-9
            if m.sum() < 1: continue
            tr = tr[m]; col = PAL[k % len(PAL)]
            ax2.plot(tr[:, 1], tr[:, 3], color=col, lw=2)
            if tt < f["time"]: plane_icon(ax2, tr[-1, 1], max(tr[-1, 3], 0.02), tr[-1, 4], color=col)
            else: landed.append((f["distance"], lab, col))
        for j, (dist, lab, col) in enumerate(sorted(landed, reverse=True)):
            ax2.text(0.98, 0.96 - j * 0.07, f"{lab}  {dist:.2f} m", transform=ax2.transAxes, color=col, fontsize=12, ha="right", va="top", family="monospace")
        yield grab()
    plt.close(fig)

if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "evolution":
        log = json.load(open(sys.argv[2])); label = sys.argv[4] if len(sys.argv) > 4 else (log.get("model") or log["backend"])
        R.write_video(evolution_frames(log, label), sys.argv[3], FPS)
