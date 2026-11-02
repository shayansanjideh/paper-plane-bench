"""Paper airplane showdown: a model gets 20 attempts against the shared referee (plane.py).

  python3 agent.py claude --model opus   --runs 3
  python3 agent.py codex  --model gpt-6.1-sol --runs 3
  python3 agent.py random --runs 3          # scripted controls (no LLM)
  python3 agent.py cma    --runs 3          # needs `pip install cma`
  python3 agent.py claude --model sonnet --blind --runs 3   # never sees results
Options: --jobs N (parallel runs, default 1), --tag NAME. Resumable: finished runs are skipped.
"""
import argparse, concurrent.futures as cf, json, pathlib, re, subprocess, sys, tempfile, time
import numpy as np
import plane as P

ROOT = pathlib.Path(__file__).parent
N_ATTEMPTS = 20

def knobs():
    out = []
    for k, (lo, hi, kind, why) in P.PARAMS.items():
        rng = f"{lo} to {hi}" + (" (whole number)" if kind is int else "")
        out.append(f'  "{k}": {rng}    // {why}')
    return "\n".join(out)

PROMPT = """You are designing a paper airplane. You have {n} attempts. After each attempt, a physics simulator folds your design, throws it, and tells you how far it flew. Your score is the distance of your BEST attempt, so use early attempts to learn.

THE SHEET
- One sheet of A4 paper: 210 x 297 mm, 80 g/m2 (about 5 g). No cutting, no tape, no extra weight. Folding only.

HOW YOUR DESIGN IS FOLDED (always in this order)
1. Fold the sheet's top corners to the centre crease, nose_folds_1 times (1 = basic nose, 2 = classic sharp dart).
2. Optionally fold the nose tip back by tip_fold_mm (makes a blunt, heavier nose).
3. Optionally fold the new top corners to the centre once more (nose_folds_2 = 1).
4. Fold the plane in half along the centre crease.
5. Fold each wing down along a line from keel_front_mm (at the nose) to keel_rear_mm (at the tail), measured from the centre crease. The part below that line is the keel/body.
6. Optionally fold the wingtips up into winglets, bend a trailing-edge strip up or down (elevator), and set the wings' dihedral.

THE THROW (fixed, you cannot change it)
- Launched by hand from 1.8 m high, at about 6 m/s, aimed 5 degrees up. Calm air, no wind.
- Each attempt is thrown 5 times, with the same 5 small hand variations (speed, angle, pitch, roll) every attempt. Your distance for an attempt is the median of the 5 throws.
- Distance is measured along the ground, in a straight line from the launch point to where the plane first touches down. Air resistance, stalls, pitching and rolling are all simulated.

THE DESIGN
Each turn, think as much as you like, then end your reply with exactly one JSON object in a ```json block with these fields:
{{
  "rationale": "one short sentence: what you changed and why",
{knobs}
}}

If a design can't be folded (for example the keel is so deep there is no wing), the attempt still counts, flies 0 m, and you get the reason.

AFTER EACH ATTEMPT you will receive the median distance, the 5 individual distances, a flight log (stall, nose-dive, roll-over, loop, or smooth glide, with times), the plane's mass, wing area and span, your best distance so far, and attempts left.

Begin with attempt 1."""

BLIND_NOTE = "\n\nNOTE: in this version you will NOT see any results until the end. Use your understanding of flight to make each attempt as good as you can."

def parse_design(text):
    blocks = re.findall(r"```json\s*(\{.*?\})\s*```", text, re.S) or re.findall(r"(\{[^{}]*\"keel_front_mm\"[^{}]*\})", text, re.S)
    for b in reversed(blocks):
        b = re.sub(r"//[^\n]*", "", b)
        try: return json.loads(b)
        except Exception: continue
    return None

class ClaudeChat:
    def __init__(self, model): self.dir = tempfile.mkdtemp(); self.sid = None; self.model = model
    def send(self, text):
        cmd = ["claude", "-p", "--output-format", "json", "--disallowedTools", "Bash,Edit,Write,Read,Glob,Grep,WebFetch,WebSearch,NotebookEdit,Task,Agent"] \
              + ["--effort", "medium"] + (["--model", self.model] if self.model else []) + (["--resume", self.sid] if self.sid else [])
        for attempt in range(3):
            r = subprocess.run(cmd, input=text, capture_output=True, text=True, cwd=self.dir, timeout=1200)
            if r.returncode == 0: break
            time.sleep(20 * (attempt + 1))
        if r.returncode != 0: raise RuntimeError(r.stderr[-1500:] or r.stdout[-1500:])
        j = json.loads(r.stdout); self.sid = j["session_id"]
        return j["result"], dict(cost=j.get("total_cost_usd"), turns=j.get("num_turns"))

class CodexChat:
    def __init__(self, model): self.dir = tempfile.mkdtemp(); self.history = []; self.model = model
    def send(self, text):
        self.history.append(("USER", text))
        convo = "\n\n".join(f"[{w}]\n{t}" for w, t in self.history) + "\n\n[ASSISTANT]\n"
        out = pathlib.Path(self.dir) / "last.txt"
        cmd = ["codex", "exec", "--skip-git-repo-check", "--sandbox", "read-only", "--json", "-c", 'model_reasoning_effort="medium"'] + (["-m", self.model] if self.model else []) + ["--output-last-message", str(out), "-"]
        for attempt in range(3):
            r = subprocess.run(cmd, input=convo, capture_output=True, text=True, cwd=self.dir, timeout=1200)
            if r.returncode == 0: break
            time.sleep(20 * (attempt + 1))
        if r.returncode != 0: raise RuntimeError(r.stderr[-1500:] or r.stdout[-1500:])
        reply = out.read_text() if out.exists() else r.stdout
        used_cmd = sum(1 for line in r.stdout.splitlines() if '"command_execution"' in line or '"exec_command' in line)
        self.history.append(("ASSISTANT", reply)); return reply, dict(tool_events=used_cmd)

# ---------------------------------------------------------------- scripted controls
K = list(P.PARAMS)
def decode(x):
    x = np.clip(x, 0, 1); return {k: round(P.PARAMS[k][0] + float(v) * (P.PARAMS[k][1] - P.PARAMS[k][0]), 2) for k, v in zip(K, x)}

def run_scripted(kind, run):
    rng = np.random.default_rng(1000 + run); log = []
    if kind == "random":
        xs = rng.random((N_ATTEMPTS, len(K)))
        for x in xs: log.append(decode(x))
        return [(d, None) for d in log]
    import cma
    x0 = rng.random(len(K)) * 0.5 + 0.25
    if kind == "cma_warm":   # start from a textbook wide-nose glider instead of a random point
        warm = dict(nose_folds_1=1, tip_fold_mm=0, nose_folds_2=0, keel_front_mm=20, keel_rear_mm=20, winglet_mm=0, elevator_mm=15, elevator_deg=5, dihedral_deg=5)
        x0 = np.array([(warm[k] - P.PARAMS[k][0]) / (P.PARAMS[k][1] - P.PARAMS[k][0]) for k in K])
    es = cma.CMAEvolutionStrategy(x0, 0.15 if kind == "cma_warm" else 0.25, {"bounds": [0, 1], "seed": 1000 + run, "popsize": 5, "verbose": -9})
    out = []
    while len(out) < N_ATTEMPTS:
        X = es.ask(); f = []
        for x in X:
            if len(out) >= N_ATTEMPTS: f.append(0.0); continue
            d = decode(x); r = P.evaluate(d); out.append((d, r)); f.append(-r["distance"])
        es.tell(X, f)
    return out

def strip(r): return {k: v for k, v in r.items() if not k.startswith("_")}

def episode(backend, model, run, blind=False):
    log = dict(backend=backend, model=model, run=run, blind=blind, attempts=[], best=0.0)
    if backend in ("random", "cma", "cma_warm"):
        for i, (d, r) in enumerate(run_scripted(backend, run)):
            r = r or P.evaluate(d); log["best"] = max(log["best"], r["distance"])
            log["attempts"].append(dict(i=i + 1, design=d, result=strip(r), best=log["best"]))
        return log
    chat = ClaudeChat(model) if backend == "claude" else CodexChat(model)
    msg = PROMPT.format(n=N_ATTEMPTS, knobs=knobs()) + (BLIND_NOTE if blind else "")
    for i in range(N_ATTEMPTS):
        t0 = time.time(); reply, meta = chat.send(msg); dt = time.time() - t0
        d = parse_design(reply)
        if d is None:
            r = dict(valid=False, error="no JSON design found in your reply", distance=0.0)
        else:
            r = strip(P.evaluate(d))
        log["best"] = max(log["best"], r["distance"])
        log["attempts"].append(dict(i=i + 1, design=d, result=r, best=log["best"], reply=reply, secs=round(dt, 1), meta=meta))
        left = N_ATTEMPTS - i - 1
        if blind:
            msg = f"Attempt {i + 1} recorded. Attempts left: {left}. Submit attempt {i + 2}." if left else "Done."
        else:
            msg = P.feedback(r, log["best"], left) + (f" Submit attempt {i + 2}." if left else "")
        if not left: break
    return log

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("backend"); ap.add_argument("--model", default=None)
    ap.add_argument("--runs", type=int, default=3); ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--tag", default=None); ap.add_argument("--blind", action="store_true"); ap.add_argument("--attempts", type=int, default=20)
    a = ap.parse_args()
    global N_ATTEMPTS; N_ATTEMPTS = a.attempts
    tag = a.tag or ((a.model or a.backend) + ("_blind" if a.blind else ""))
    out = ROOT / "results" / tag; out.mkdir(parents=True, exist_ok=True)
    todo = [r for r in range(a.runs) if not (out / f"run{r}.json").exists()]
    print(f"{len(todo)} runs to go -> results/{tag}/", flush=True)
    def work(r):
        L = episode(a.backend, a.model, r, a.blind); (out / f"run{r}.json").write_text(json.dumps(L, indent=1)); return L
    with cf.ThreadPoolExecutor(a.jobs) as ex:
        futs = {ex.submit(work, r): r for r in todo}
        for f in cf.as_completed(futs):
            r = futs[f]
            try: L = f.result(); print(f"  {tag} run{r}: best {L['best']:.2f} m  curve {[round(x['best'], 1) for x in L['attempts']]}", flush=True)
            except Exception as e: print(f"  FAIL {tag} run{r}: {str(e)[:300]}", file=sys.stderr, flush=True)

if __name__ == "__main__":
    main()
