"""Turns a design into pictures: fold-by-fold animation, the finished 3D plane, and its flight.

Frames are drawn with matplotlib (Agg) and piped to ffmpeg.
"""
import math, subprocess, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import plane as P

PAPER, PAPER_BACK, EDGE = "#f4f1ea", "#d9d3c4", "#55524b"
BG, INK, MUTED = "#0f1115", "#e9e7e1", "#8a8f98"

def rot(points, a, d, ang):
    """Rotate 3D points about the axis through a with unit direction d by ang (Rodrigues)."""
    p = points - a; c, s = math.cos(ang), math.sin(ang)
    return a + p * c + np.cross(d, p) * s + np.outer(p @ d, d) * (1 - c)

# ---------------------------------------------------------------- flat folding stages
def fold_stages(design):
    """List of stages: each = (facets_before, moving_flags, a, d) on the right half, in sheet coords."""
    d, _, log, _ = P.fold(design)
    sheet = [np.array(p, float) for p in [(0, 0), (P.W_HALF, 0), (P.W_HALF, P.L_SHEET), (0, P.L_SHEET)]]
    facets = [sheet]; stages = []
    for st in log:
        a, dv, n = np.array(st["a"]), np.array(st["d"]), np.array(st["n"])
        before, moving = [], []
        for f in facets:
            s, g = P.clip(f, a, n), P.clip(f, a, -n)
            if P.area(s) > 1e-6: before.append(s); moving.append(False)
            if P.area(g) > 1e-6: before.append(g); moving.append(True)
        stages.append((before, moving, a, dv, st["kind"]))
        facets = [(P.reflect(f, a, dv) if m else f) for f, m in zip(before, moving)]
    return stages, facets

def flat_frame_polys(stage, t):
    """3D polygons (both halves) for one fold stage at progress t in [0,1]."""
    before, moving, a, dv, _ = stage
    out = []
    for k, (f, m) in enumerate(zip(before, moving)):
        pts = np.c_[np.array(f), np.full(len(f), 0.05 * k)]
        if m:
            ang = math.pi * t
            pts = rot(pts, np.r_[a, 0], np.r_[dv, 0], -ang)
            pts[:, 2] = np.abs(pts[:, 2]) + 0.05 * k + (0.3 if t > 0.5 else 0)
        out.append(pts)
        out.append(pts * np.array([-1, 1, 1]))        # mirror half
    return out

# ---------------------------------------------------------------- finished plane in 3D
def assembled(design, tA=1.0, tB=1.0):
    """3D polygons of the plane. tA: fold-in-half progress, tB: wing-fold progress (both 0..1).
    Body frame at the end: x = right span, y = forward (nose +), z = up."""
    g = P.analyse(design); d = g["design"]
    a2, t2, n2, _ = g["hinge"]; a2, t2, n2 = map(np.array, (a2, t2, n2))
    s_fold = g["s_fold"]; e_mm = d["elevator_mm"]
    pieces = []
    for f, _ in g["keel"]: pieces.append(("keel", f))
    for f, _ in g["wing"]:
        # split wing by winglet line (s = s_fold) and elevator line (v = e_mm)
        wl_a = a2 + n2 * s_fold
        inner, outer = P.clip(f, wl_a, n2), P.clip(f, wl_a, -n2)
        for tag, part in (("wing", inner), ("winglet", outer)):
            if P.area(part) < 1e-6: continue
            if tag == "wing" and e_mm > 0.5:
                fwd = P.clip(part, np.array([0, e_mm]), np.array([0.0, -1.0])); aft = P.clip(part, np.array([0, e_mm]), np.array([0.0, 1.0]))
                if P.area(fwd) > 1e-6: pieces.append(("wing", fwd))
                if P.area(aft) > 1e-6: pieces.append(("elev", aft))
            else: pieces.append((tag, part))
    thA = math.pi / 2 * tA
    Gam = math.radians(d["dihedral_deg"])
    thB = (math.pi / 2 - Gam) * tB
    polys = []
    for side in (1, -1):
        for tag, f in pieces:
            F = np.array(f)                                  # (u, v)
            pts = np.c_[F[:, 0], F[:, 1], np.zeros(len(F))]  # flat: x=u, y=v, z=0
            hA = np.array([0.0, 0, 0]); dA = np.array([0.0, 1, 0])
            # fold in half: right half rotates about the crease so it stands up (u -> z)
            pts = rot(pts, hA, dA, -thA)
            if tag != "keel":
                # hinge axis after fold-in-half
                hp = rot(np.array([[a2[0], a2[1], 0]]), hA, dA, -thA)[0]
                hd = rot(np.array([[t2[0], t2[1], 0]]), np.zeros(3), dA, -thA)[0]
                pts = rot(pts, hp, hd, thB if hd[1] > 0 else -thB)
                if tag == "winglet" and tB > 0:
                    wp = rot(np.array([[a2[0] + n2[0] * s_fold, a2[1] + n2[1] * s_fold, 0]]), hA, dA, -thA)[0]
                    wp = rot(np.array([wp]), hp, hd, thB if hd[1] > 0 else -thB)[0]
                    pts = rot(pts, wp, hd, -(math.pi / 2 + Gam) * tB if hd[1] > 0 else (math.pi / 2 + Gam) * tB)
                if tag == "elev" and tB > 0:
                    de = math.radians(d["elevator_deg"]) * tB
                    ep = rot(np.array([[a2[0], e_mm, 0]]), hA, dA, -thA)[0]; ep = rot(np.array([ep]), hp, hd, thB if hd[1] > 0 else -thB)[0]
                    pts = rot(pts, ep, np.array([1.0, 0, 0]), de)
            pts[:, 0] *= side
            polys.append((tag, pts))
    return polys, g

# ---------------------------------------------------------------- drawing helpers
def setup_3d(ax, lim=170, elev=22, azim=-60, zlim=(-60, 120)):
    ax.set_facecolor(BG); ax.set_axis_off()
    ax.set_xlim(-lim, lim); ax.set_ylim(-20, 300); ax.set_zlim(*zlim)
    ax.set_box_aspect((2 * lim, 320, zlim[1] - zlim[0]))
    ax.view_init(elev=elev, azim=azim)

def draw_polys(ax, polys, colors=None):
    cols = []
    for i, p in enumerate(polys):
        cols.append(colors[i] if colors else PAPER)
    pc = Poly3DCollection(polys, facecolors=cols, edgecolors=EDGE, linewidths=0.6, alpha=0.97)
    ax.add_collection3d(pc)

TAGCOL = {"keel": PAPER_BACK, "wing": PAPER, "winglet": "#e8d9b8", "elev": "#f2c48d"}

def frames_for_design(design, fps=30, fold_s=0.35, assemble_s=0.8, hold_s=0.6, size=(6, 4.5), title=None, sub=None):
    """Yields RGB frames: sheet -> folds -> fold in half -> wing folds -> spin."""
    stages, _ = fold_stages(design)
    fig = plt.figure(figsize=size, dpi=120, facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 0.9], projection="3d")
    def text():
        if title: fig.text(0.04, 0.94, title, color=INK, fontsize=12, weight="bold")
        if sub: fig.text(0.04, 0.905, sub, color=MUTED, fontsize=9)
    def grab():
        fig.canvas.draw(); return np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    nf = max(int(fold_s * fps), 2)
    for st in stages:
        for i in range(nf + 1):
            ax.cla(); setup_3d(ax, lim=120, elev=70, azim=-90, zlim=(-10, 60)); fig.texts.clear(); text()
            draw_polys(ax, flat_frame_polys(st, i / nf)); yield grab()
    na = max(int(assemble_s * fps), 2)
    for i in range(na + 1):
        t = i / na; tA, tB = min(t * 2, 1), max(0, t * 2 - 1)
        polys, _ = assembled(design, tA, tB)
        ax.cla(); setup_3d(ax, lim=120, elev=70 - 48 * t, azim=-90 + 30 * t, zlim=(-10, 110)); fig.texts.clear(); text()
        draw_polys(ax, [p for _, p in polys], [TAGCOL[k] for k, _ in polys]); yield grab()
    polys, _ = assembled(design)
    nh = int(hold_s * fps)
    for i in range(nh):
        ax.cla(); setup_3d(ax, lim=120, elev=22, azim=-60 + 40 * i / max(nh, 1), zlim=(-10, 110)); fig.texts.clear(); text()
        draw_polys(ax, [p for _, p in polys], [TAGCOL[k] for k, _ in polys]); yield grab()
    plt.close(fig)

def still(design, path, elev=22, azim=-60, title=None):
    polys, g = assembled(design)
    fig = plt.figure(figsize=(5, 4), dpi=140, facecolor=BG)
    ax = fig.add_axes([0, 0, 1, 0.92], projection="3d"); setup_3d(ax, lim=120, elev=elev, azim=azim, zlim=(-10, 110))
    draw_polys(ax, [p for _, p in polys], [TAGCOL[k] for k, _ in polys])
    if title: fig.text(0.04, 0.94, title, color=INK, fontsize=11, weight="bold")
    fig.savefig(path, facecolor=BG); plt.close(fig)

def write_video(frames, path, fps=30):
    proc = None
    for fr in frames:
        if proc is None:
            h, w = fr.shape[:2]; h -= h % 2; w -= w % 2
            proc = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
                                     "-r", str(fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", path], stdin=subprocess.PIPE)
        proc.stdin.write(np.ascontiguousarray(fr[:h, :w]).tobytes())
    proc.stdin.close(); proc.wait()


# ---------------------------------------------------------------- printable fold instructions
def instructions(design, path, name="Plane"):
    """A one-page fold guide: numbered panels with the fold line dashed and the moving part shaded."""
    import matplotlib.patches as mp
    d = P.check(design)
    stages, final = fold_stages(d)
    words = {"corner": "Fold the top corners in so the top edges meet the centre crease.",
             "tip": "Fold the nose point down onto the plane along the dashed line, {d:.0f} mm below the tip."}
    n = len(stages) + 3
    cols = 3; rows = math.ceil(n / cols)
    fig, axs = plt.subplots(rows, cols, figsize=(11, 3.9 * rows), facecolor="white")
    axs = np.array(axs).ravel()
    def sheet_ax(ax):
        ax.set_aspect("equal"); ax.set_xlim(-115, 115); ax.set_ylim(-10, 310); ax.axis("off")
    def draw_flat(ax, facets, moving=None):
        for k, f in enumerate(facets):
            for sgn in (1, -1):
                F = np.array(f) * [sgn, 1]
                ax.add_patch(mp.Polygon(F, closed=True, fc=("#f2c48d" if moving and moving[k] else "#f4f1ea"), ec="#444", lw=0.8))
        ax.plot([0, 0], [0, 297], color="#999", lw=0.6, ls=":")
    def dashed(ax, a, dv, sgns=(1, -1)):
        for sgn in sgns:
            p0 = np.array(a) - 400 * np.array(dv); p1 = np.array(a) + 400 * np.array(dv)
            ax.plot([p0[0] * sgn, p1[0] * sgn], [p0[1], p1[1]], color="#d33", lw=1.4, ls="--")
    i = 0
    tipd = d["tip_fold_mm"]
    for st in stages:
        ax = axs[i]; sheet_ax(ax); before, moving, a, dv, kind = st
        draw_flat(ax, before, moving); dashed(ax, a, dv)
        ax.set_title(f"{i + 1}. " + textwrap_fill(words[kind].format(d=tipd)), fontsize=9, loc="left")
        i += 1
    # fold in half + wing lines, drawn on the folded half (side view of one side)
    ax = axs[i]; sheet_ax(ax); draw_flat(ax, final)
    g = P.analyse(d); a2, t2, n2, L = g["hinge"]
    p0 = np.array(a2); p1 = p0 + np.array(t2) * L
    for sgn in (1, -1):
        ax.plot([p0[0] * sgn, p1[0] * sgn], [p0[1], p1[1]], color="#d33", lw=1.4, ls="--")
    ax.set_title(f"{i + 1}. " + textwrap_fill(w=44, s=f"Fold in half along the centre (folds inside). Then fold each wing down on the dashed line: "
                                              f"{d['keel_front_mm']:.0f} mm from the bottom crease at the nose, {d['keel_rear_mm']:.0f} mm at the tail."), fontsize=9, loc="left")
    i += 1
    ax = axs[i]; ax.axis("off")
    extra = []
    if d["winglet_mm"] > 0.5: extra.append(f"Fold the outer {d['winglet_mm']:.0f} mm of each wing straight up (winglets).")
    if d["elevator_mm"] > 0.5 and abs(d["elevator_deg"]) > 0.5:
        extra.append(f"Bend the rear {d['elevator_mm']:.0f} mm of each wing {'up' if d['elevator_deg'] > 0 else 'down'} by about {abs(d['elevator_deg']):.0f}°.")
    extra.append(f"Open the wings so each one tilts {'up' if d['dihedral_deg'] >= 0 else 'down'} about {abs(d['dihedral_deg']):.0f}° from flat (dihedral).")
    ax.text(0, 0.95, f"{i + 1}. Finishing", fontsize=10, weight="bold", va="top", transform=ax.transAxes)
    ax.text(0, 0.82, "\n\n".join(textwrap_fill(e, 48) for e in extra), fontsize=9, va="top", transform=ax.transAxes)
    i += 1
    # 3D view
    fig.delaxes(axs[i]); ax = fig.add_subplot(rows, cols, i + 1, projection="3d")
    polys, _ = assembled(d)
    ax.set_axis_off(); ax.set_xlim(-110, 110); ax.set_ylim(-10, 300); ax.set_zlim(-10, 100); ax.set_box_aspect((220, 310, 110)); ax.view_init(22, -60)
    ax.add_collection3d(Poly3DCollection([p for _, p in polys], facecolors=[TAGCOL[k] for k, _ in polys], edgecolors="#444", linewidths=0.6))
    ax.set_title(f"{i + 1}. Done", fontsize=10, loc="left")
    for j in range(i + 1, len(axs)): axs[j].axis("off")
    fig.suptitle(name, fontsize=15, weight="bold", x=0.02, ha="left")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)

def textwrap_fill(s, w=52):
    import textwrap; return textwrap.fill(s, w)
