"""Paper airplane referee: folds a design from one A4 sheet, then throws it.

Only depends on numpy. Everything is in SI units inside; the design JSON is in mm/deg.

Geometry: we work on the right half of the sheet (u = distance from the centre crease,
0..105 mm; v = distance from the tail edge, 0..297 mm; the nose is at the top, v = 297).
Every fold is a reflection of convex paper facets across a line, so facets stay convex
and we can clip them with plain half-plane clipping. Layers/mass come from the facets.

Flight: a 2.5-D glider model. Pitch plane is a full rigid-body (x, z, pitch, pitch rate) with
lift, drag (skin friction + form + induced + flap) and pitching moment about the CG. Roll is a
1st-order bank mode whose stability depends on dihedral, sweep, keel and winglets; bank turns the
plane and tilts lift. Air resistance is always on. Wind is optional (headwind/gusts).
"""
import json, math
import numpy as np

W_HALF, L_SHEET = 105.0, 297.0          # A4 half-width and length, mm
GSM = 80.0                               # paper, g/m^2
RHO, NU, G = 1.20, 1.5e-5, 9.81          # air density, kinematic viscosity, gravity

PARAMS = {  # name: (low, high, kind, meaning)
    "nose_folds_1":   (1, 2, int,   "corner-to-centre folds at the start (1 = basic, 2 = classic dart)"),
    "tip_fold_mm":    (0, 120, float, "fold the nose tip back by this much after those folds (0 = sharp nose)"),
    "nose_folds_2":   (0, 1, int,   "one more corner-to-centre fold after the tip fold (Nakamura-style)"),
    "keel_front_mm":  (15, 80, float, "keel depth at the nose (distance from centre crease to wing fold; 15 mm minimum so you can hold it)"),
    "keel_rear_mm":   (15, 80, float, "keel depth at the tail (15 mm minimum)"),
    "winglet_mm":     (0, 40, float, "wingtips folded up vertically, height (0 = none)"),
    "elevator_mm":    (0, 40, float, "chord of the trailing-edge strip bent up or down"),
    "elevator_deg":   (-20, 25, float, "bend of that strip, + is up"),
    "dihedral_deg":   (-10, 25, float, "wings tilted up (+) or down (-) from flat"),
}
THROW = dict(height=1.8, speed=6.0, angle_deg=5.0)
# Same 5 throw wobbles for everyone (common random numbers): dspeed m/s, dangle deg, dalpha deg, roll deg
WOBBLE = [(0.0, 0.0, 0.0, 2.0), (-0.5, 3.0, 2.0, -4.0), (0.5, -3.0, -2.0, 5.0), (-0.3, -2.0, 3.0, -3.0), (0.4, 2.0, -3.0, 3.5)]


# ---------------------------------------------------------------- polygon helpers (convex)
def clip(poly, a, n):
    """Keep the part of convex polygon `poly` where n.(p-a) <= 0."""
    out = []
    if len(poly) == 0: return out
    d = [float(np.dot(n, p - a)) for p in poly]
    for i in range(len(poly)):
        p, q, dp, dq = poly[i], poly[(i + 1) % len(poly)], d[i], d[(i + 1) % len(poly)]
        if dp <= 0: out.append(p)
        if (dp < 0 < dq) or (dq < 0 < dp):
            out.append(p + (q - p) * (dp / (dp - dq)))
    return out

def area(poly):
    if len(poly) < 3: return 0.0
    x, y = np.array(poly).T
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

def centroid(poly):
    x, y = np.array(poly).T; x1, y1 = np.roll(x, -1), np.roll(y, -1)
    cr = x * y1 - x1 * y; A = cr.sum() / 2
    return np.array([((x + x1) * cr).sum() / (6 * A), ((y + y1) * cr).sum() / (6 * A)])

def second_moment(poly, c):
    """Polar second moment of area about point c in the v-direction only (for pitch inertia), mm^4."""
    P = np.array(poly) - c; x, y = P.T; x1, y1 = np.roll(x, -1), np.roll(y, -1)
    cr = x * y1 - x1 * y
    return abs(((y * y + y * y1 + y1 * y1) * cr).sum() / 12)

def span_at(poly, u, ax=0):
    """(min, max) of the other coordinate where convex poly crosses coordinate[ax] == u, or None."""
    P = np.asarray(poly); o = 1 - ax; vals = []
    for i in range(len(P)):
        p, q = P[i], P[(i + 1) % len(P)]
        lo, hi = min(p[ax], q[ax]), max(p[ax], q[ax])
        if lo - 1e-9 <= u <= hi + 1e-9:
            if hi - lo < 1e-9: vals += [p[o], q[o]]
            else: vals.append(p[o] + (q[o] - p[o]) * (u - p[ax]) / (q[ax] - p[ax]))
    return (min(vals), max(vals)) if vals else None

def reflect(poly, a, d):
    """Reflect points across the line through a with unit direction d."""
    out = []
    for p in poly:
        r = p - a; out.append(a + 2 * np.dot(r, d) * d - r)
    return out


# ---------------------------------------------------------------- folding
class Invalid(Exception): pass

def _outline_apex(facets):
    """Top of the centre crease and the direction of the leading edge leaving it."""
    pts = np.vstack([np.array(f) for f, _ in facets])
    on_c = pts[pts[:, 0] < 0.5]
    apex = np.array([0.0, on_c[:, 1].max()])
    # leading edge: for points near the apex, the upper boundary as u grows
    us = np.linspace(2, 25, 12); tops = []
    for u in us:
        best = -1e9
        for f, _ in facets:
            r = span_at(f, u)
            if r: best = max(best, r[1])
        tops.append(best)
    tops = np.array(tops); ok = tops > -1e8
    if ok.sum() < 2: raise Invalid("nose is too narrow to fold again")
    slope = np.polyfit(us[ok], tops[ok], 1)[0]
    e = np.array([1.0, slope]); e /= np.linalg.norm(e)
    return apex, e

def fold_corner(facets, log):
    """Fold the leading edge onto the centre crease (bisector through the apex)."""
    apex, e = _outline_apex(facets)
    b = e + np.array([0.0, -1.0]); b /= np.linalg.norm(b)
    n = np.array([-b[1], b[0]])                     # normal of fold line
    if np.dot(n, e) < 0: n = -n                       # n points to the side being folded (leading edge side)
    moved, out = [], []
    for f, L in facets:
        stay, go = clip(f, apex, n), clip(f, apex, -n)
        if area(stay) > 1e-6: out.append((stay, L))
        if area(go) > 1e-6:
            g = reflect(go, apex, b); out.append((g, L + 1)); moved.append(g)
    log.append(dict(kind="corner", a=apex.tolist(), d=b.tolist(), n=n.tolist()))
    spill = sum(area(clip(f, np.array([0.0, 0]), np.array([1.0, 0]))) for f, _ in out)
    if spill > 30: raise Invalid("a nose fold pushes paper past the centre crease")
    return out

def fold_tip(facets, d_mm, log):
    apex, _ = _outline_apex(facets)
    a = np.array([0.0, apex[1] - d_mm]); dvec = np.array([1.0, 0.0]); n = np.array([0.0, 1.0])
    out = []
    for f, L in facets:
        stay, go = clip(f, a, n), clip(f, a, -n)
        if area(stay) > 1e-6: out.append((stay, L))
        if area(go) > 1e-6: out.append((reflect(go, a, dvec), L + 1))
    log.append(dict(kind="tip", a=a.tolist(), d=dvec.tolist(), n=n.tolist()))
    return out

def check(design):
    d = {}
    for k, (lo, hi, kind, _) in PARAMS.items():
        if k not in design: raise Invalid(f"missing field {k}")
        try: x = float(design[k])
        except Exception: raise Invalid(f"{k} must be a number")
        if not (lo - 1e-9 <= x <= hi + 1e-9): raise Invalid(f"{k}={x} is outside {lo}..{hi}")
        d[k] = int(round(x)) if kind is int else x
    return d

def fold(design):
    """Returns the folded half-plane: facets with layers, fold log, wing-fold line, apex."""
    d = check(design)
    sheet = [np.array(p, float) for p in [(0, 0), (W_HALF, 0), (W_HALF, L_SHEET), (0, L_SHEET)]]
    facets, log = [(sheet, 1)], []
    for _ in range(d["nose_folds_1"]): facets = fold_corner(facets, log)
    if d["tip_fold_mm"] > 0.5:
        apex, _ = _outline_apex(facets)
        if d["tip_fold_mm"] > apex[1] * 0.6: raise Invalid("tip fold is longer than the plane can take")
        facets = fold_tip(facets, d["tip_fold_mm"], log)
    for _ in range(d["nose_folds_2"]): facets = fold_corner(facets, log)
    apex, _ = _outline_apex(facets)
    return d, facets, log, apex[1]


# ---------------------------------------------------------------- mass + planform
def analyse(design):
    d, facets, log, v_top = fold(design)
    kf, kr = d["keel_front_mm"], d["keel_rear_mm"]
    a = np.array([kr, 0.0]); t = np.array([kf - kr, v_top]); L_h = np.linalg.norm(t); t /= L_h   # hinge dir (to nose)
    n = np.array([t[1], -t[0]])                       # points outward (+u)
    if n[0] < 0: n = -n
    keel, wing = [], []
    for f, Ly in facets:
        k_, w_ = clip(f, a, n), clip(f, a, -n)
        if area(k_) > 1e-6: keel.append((k_, Ly))
        if area(w_) > 1e-6: wing.append((w_, Ly))
    if not wing: raise Invalid("keel is so deep there is no wing left")
    # wing coordinates: s = span distance from hinge, x = along hinge (toward nose)
    def to_w(P): R = np.array(P) - a; return np.c_[R @ n, R @ t]
    wingW = [(to_w(f), Ly) for f, Ly in wing]
    s_max = max(P[:, 0].max() for P, _ in wingW)
    h = min(d["winglet_mm"], s_max * 0.5)
    s_fold = s_max - h
    # raster planform (union of wing facets, ignoring layers), 1 mm grid in span
    S_grid = np.arange(0.5, s_fold, 1.0)
    if len(S_grid) < 8: raise Invalid("wings are too small (less than 8 mm of span)")
    LE, TE = np.full(len(S_grid), -1e9), np.full(len(S_grid), 1e9)
    for P, _ in wingW:
        for i, s in enumerate(S_grid):
            r = span_at(P, s)
            if r: LE[i] = max(LE[i], r[1]); TE[i] = min(TE[i], r[0])
    ok = LE > -1e8
    S_grid, LE, TE = S_grid[ok], LE[ok], TE[ok]; c = LE - TE
    if len(S_grid) < 8 or c.max() < 15: raise Invalid("wings are too small")
    S_half = c.sum() * 1.0                                          # mm^2 (gaps inside are ignored: fine)
    b_half = S_grid.max() + 0.5
    mac = (c ** 2).sum() / c.sum()
    x_le_mac = (LE * c).sum() / c.sum()                             # chord-weighted LE position
    S = 2 * S_half * 1e-6; b = 2 * b_half * 1e-3; AR = b * b / S
    sweep = math.degrees(math.atan2(LE[0] - LE[-1], S_grid[-1] - S_grid[0])) if len(S_grid) > 1 else 0
    # mass properties along hinge axis x (mm), using every layer
    mass_x = mass = I = 0.0; cents = []
    allW = wingW + [(to_w(f), Ly) for f, Ly in keel]
    for P, Ly in allW:
        A = area(list(P)); m = A; cx = centroid(list(P))[1]; mass += m; mass_x += m * cx; cents.append((P, Ly, A, cx))
    x_cg = mass_x / mass
    for P, Ly, A, cx in cents:
        I += second_moment(list(P), np.array([0, x_cg]))
    keel_area = sum(area(f) for f, _ in keel); wing_layers = sum(area(list(P)) for P, Ly in wingW) / (S_half + 1e-9)
    m_kg = 2 * mass * 1e-6 * GSM * 1e-3                             # both halves
    Iyy = 2 * I * 1e-12 * GSM * 1e-3 * 1.0 + m_kg * (0.01 ** 2)    # kg m^2 (+ small vertical spread)
    return dict(design=d, facets=facets, log=log, keel=keel, wing=wing, hinge=(a.tolist(), t.tolist(), n.tolist(), L_h),
                S=S, b=b, AR=AR, mac=mac * 1e-3, x_le_mac=x_le_mac * 1e-3, x_cg=x_cg * 1e-3, mass=m_kg, Iyy=Iyy,
                sweep_deg=sweep, keel_area=2 * keel_area * 1e-6, winglet=h * 1e-3, s_fold=s_fold, wing_layers=wing_layers,
                span_stations=(S_grid.tolist(), LE.tolist(), TE.tolist()))


# ---------------------------------------------------------------- aerodynamics
def aero_model(g):
    d = g["design"]; A = g["AR"]; b = g["b"]
    A_eff = A * (1 + 1.9 * g["winglet"] / max(b, 1e-3))             # winglets (Whitcomb-style rough)
    Gam = math.radians(d["dihedral_deg"])
    CLa = 2 * math.pi * A_eff / (2 + math.sqrt(A_eff ** 2 + 4)) * math.cos(Gam) ** 2   # Helmbold
    slender = min(max((3.0 - A) / 2.0, 0.0), 1.0)
    Kv = math.pi * slender                                           # Polhamus vortex lift (slender wings)
    a_stall = math.radians(10 + 18 * slender)
    ac_frac = 0.25 + 0.25 * slender                                  # a.c. moves aft on slender deltas
    x_ac = g["x_le_mac"] - ac_frac * g["mac"]
    x_cp_stall = g["x_le_mac"] - 0.45 * g["mac"]
    # elevator: thin-airfoil flap theory on the trailing-edge strip
    E = min(d["elevator_mm"] * 1e-3 / g["mac"], 0.45); de = math.radians(d["elevator_deg"])
    if E > 1e-4:
        th = math.acos(1 - 2 * E); tau = 1 - (th - math.sin(th)) / math.pi
        dCL_e = -CLa * tau * de                                      # up-bend loses lift
        dCm_e = 0.5 * math.sin(th) * (1 - math.cos(th)) * de * 2     # up-bend pitches nose up (paper flaps: x2 effectiveness)
    else: dCL_e = dCm_e = 0.0
    # drag pieces (per wing area)
    Swet = 2 * g["S"] * min(g["wing_layers"], 1.5) + 2 * g["keel_area"] + 4 * g["winglet"] * g["mac"]
    return dict(CLa=CLa, Kv=Kv, a_stall=a_stall, x_ac=x_ac, x_cp_stall=x_cp_stall, dCL_e=dCL_e, dCm_e=dCm_e,
                Swet=Swet, A_eff=A_eff, E=E, de=de, Gam=Gam)

def coeffs(alpha, V, g, am):
    """CL, CD, x of centre of pressure, stall blend."""
    Re = max(V * g["mac"] / NU, 2e3)
    Cf = 1.328 / math.sqrt(Re)                                       # laminar flat plate
    CD0 = Cf * am["Swet"] / g["S"] + 0.012                           # + paper edges, creases, roughness
    sa, ca = math.sin(alpha), math.cos(alpha)
    CL_att = am["CLa"] * sa * ca + am["Kv"] * ca * sa * abs(sa) + am["dCL_e"]
    CN_post = 1.8 * sa                                               # flat plate after stall
    CL_post, CD_post = CN_post * ca, CN_post * sa
    w = 1 / (1 + math.exp(-(abs(alpha) - am["a_stall"]) / math.radians(2.5)))   # 0 attached .. 1 stalled
    e = 0.8
    CD_att = CD0 + CL_att ** 2 / (math.pi * am["A_eff"] * e) + 1.2 * math.sin(am["de"]) ** 2 * am["E"]
    CL = (1 - w) * CL_att + w * CL_post
    CD = (1 - w) * CD_att + w * (CD_post + CD0)
    x_cp = (1 - w) * am["x_ac"] + w * am["x_cp_stall"]
    return CL, CD, x_cp, w

def roll_rate_const(g, am, CL=0.3):
    """Bank-mode rate (1/s). Negative = bank dies out, positive = plane rolls over into a spiral."""
    Clb = -am["CLa"] * am["Gam"] / 4                                 # dihedral
    Clb += -CL * math.tan(math.radians(max(g["sweep_deg"], 0))) / 4  # sweep (needs lift)
    Clb += -0.6 * g["keel_area"] / g["S"] * 0.3                       # keel hanging below the wing
    Clb += -am["CLa"] * g["winglet"] / max(g["b"], 1e-3) * 0.5       # winglets
    return 30.0 * (Clb + 0.03), Clb                                  # needs Clb < -0.03 to be stable


# ---------------------------------------------------------------- flight
def fly(g, am, dv=0.0, dang=0.0, dalpha=0.0, roll0=0.0, wind=None, dt=0.002, tmax=20.0, record=False):
    m, I, S, c = g["mass"], g["Iyy"], g["S"], g["mac"]
    V0 = THROW["speed"] + dv; gam0 = math.radians(THROW["angle_deg"] + dang)
    X, Y, Z = 0.0, 0.0, THROW["height"]; psi = 0.0
    vx, vz = V0 * math.cos(gam0), V0 * math.sin(gam0)
    th = gam0 + math.radians(dalpha); q = 0.0; phi = math.radians(roll0)
    lam, _ = roll_rate_const(g, am)
    Cmq = -(math.pi / 4 + 2 * am["CLa"] * ((g["x_cg"] - am["x_ac"]) / c) ** 2) - 0.6
    t, events, traj = 0.0, set(), []
    max_alpha = 0.0; looped = False
    def deriv(s, t):
        vx, vz, th, q, phi = s
        wx, wz = (wind(t) if wind else (0.0, 0.0))
        ux, uz = vx - wx, vz - wz
        V = math.hypot(ux, uz) + 1e-6; gam = math.atan2(uz, ux); al = th - gam
        al = (al + math.pi) % (2 * math.pi) - math.pi
        CL, CD, xcp, w = coeffs(al, V, g, am)
        qd = 0.5 * RHO * V * V
        L, D = qd * S * CL, qd * S * CD
        Lv = L * math.cos(phi)
        ax = (-D * math.cos(gam) - Lv * math.sin(gam)) / m
        az = (-D * math.sin(gam) + Lv * math.cos(gam)) / m - G
        CN = CL * math.cos(al) + CD * math.sin(al)
        M = qd * S * (CN * (xcp - g["x_cg"]) + c * am["dCm_e"] + c * Cmq * q * c / (2 * V))
        lam_eff = lam * min(max(V / 4.0, 0.3), 2.0)
        phid = lam_eff * math.sin(phi)
        return np.array([ax, az, q, M / I, phid]), dict(V=V, al=al, gam=gam, w=w, L=L)
    s = np.array([vx, vz, th, q, phi])
    while t < tmax:
        k1, info = deriv(s, t); k2, _ = deriv(s + dt / 2 * k1, t + dt / 2)
        k3, _ = deriv(s + dt / 2 * k2, t + dt / 2); k4, _ = deriv(s + dt * k3, t + dt)
        s = s + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4); t += dt
        vx, vz, th, q, phi = s
        Vh = vx                                                       # forward speed along heading
        psi += (G * math.tan(max(min(phi, 1.4), -1.4)) / max(info["V"], 0.5)) * dt
        X += Vh * math.cos(psi) * dt; Y += Vh * math.sin(psi) * dt; Z += vz * dt
        al = info["al"]; max_alpha = max(max_alpha, abs(al))
        if info["w"] > 0.5: events.add(("stall", round(t, 1)))
        if abs(th) > math.radians(100): looped = True
        if abs(phi) > math.radians(60): events.add(("roll", round(t, 1)))
        if info["gam"] < math.radians(-45) and t > 0.2: events.add(("dive", round(t, 1)))
        if record and int(round(t / dt)) % 10 == 0:
            traj.append((round(t, 3), X, Y, Z, th, phi, psi, info["al"]))
        if Z <= 0 or abs(phi) > math.radians(85) and Z < 0.3: break
        if Z > 15 or abs(X) > 200: break
    dist = math.hypot(X, Y)
    return dict(distance=dist, time=t, X=X, Y=Y, events=sorted(events, key=lambda e: e[1]), looped=looped, traj=traj,
                max_alpha_deg=math.degrees(max_alpha))

def summarize_events(evs, looped):
    first = {}
    for k, tt in evs: first.setdefault(k, tt)
    words = []
    if "stall" in first: words.append(f"stalled at {first['stall']:.1f} s")
    if "dive" in first: words.append(f"nose-dived at {first['dive']:.1f} s")
    if "roll" in first: words.append(f"rolled over at {first['roll']:.1f} s")
    if looped: words.append("looped")
    return ", ".join(words) or "smooth glide"

def evaluate(design, wind=None, record=False):
    """The referee. Returns a dict the models see (minus 'traj' and geometry)."""
    try:
        g = analyse(design)
    except Invalid as e:
        return dict(valid=False, error=str(e), distance=0.0)
    am = aero_model(g)
    flights = [fly(g, am, *wb, wind=wind, record=record) for wb in WOBBLE]
    ds = [f["distance"] for f in flights]
    med_i = int(np.argsort(ds)[len(ds) // 2])
    lam, Clb = roll_rate_const(g, am)
    return dict(valid=True, distance=float(np.median(ds)), throws=[round(float(x), 2) for x in ds],
                log=summarize_events(flights[med_i]["events"], flights[med_i]["looped"]),
                flight_time=round(float(flights[med_i]["time"]), 2),
                stats=dict(mass_g=round(float(g["mass"]) * 1e3, 2), wing_area_cm2=round(float(g["S"]) * 1e4, 1), span_mm=int(round(g["b"] * 1e3)),
                           aspect_ratio=round(float(g["AR"]), 2), static_margin=round(float((g["x_cg"] - am["x_ac"]) / g["mac"]), 3),
                           roll_stable=bool(lam < 0)),
                _flights=flights if record else None, _geom=g if record else None)

def feedback(res, best, left):
    if not res["valid"]:
        return f"INVALID: {res['error']}. Flew 0 m. Best so far: {best:.2f} m. Attempts left: {left}."
    s = res["stats"]
    return (f"Median distance: {res['distance']:.2f} m (5 throws: {', '.join(f'{x:.2f}' for x in res['throws'])}). "
            f"Flight log: {res['log']}, airborne {res['flight_time']} s. "
            f"Plane: {s['mass_g']} g, wing area {s['wing_area_cm2']} cm2, span {s['span_mm']} mm. "
            f"Best so far: {best:.2f} m. Attempts left: {left}.")

if __name__ == "__main__":
    import sys
    print(json.dumps({k: v for k, v in evaluate(json.loads(sys.argv[1])).items() if not k.startswith("_")}, indent=1))
