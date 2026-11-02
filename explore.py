"""Landscape check: random designs + a long CMA-ES run, to find the ceiling and any exploits."""
import sys, json, numpy as np, cma
sys.path.insert(0, '.')
from plane import PARAMS, evaluate
K = list(PARAMS)
def dec(x):
    x = np.clip(x, 0, 1); return {k: PARAMS[k][0] + float(xi) * (PARAMS[k][1] - PARAMS[k][0]) for k, xi in zip(K, x)}
def f(x): return -evaluate(dec(x))["distance"]
rng = np.random.default_rng(0)
R = [(-f(x), x) for x in rng.random((200, len(K)))]
ds = np.array([r[0] for r in R]); print("random 200: valid", (ds > 0).mean(), "median", np.median(ds[ds > 0]), "max", ds.max())
es = cma.CMAEvolutionStrategy([0.5] * len(K), 0.25, {'bounds': [0, 1], 'seed': 1, 'maxfevals': 600, 'verbose': -9})
es.optimize(f)
x = es.result.xbest; r = evaluate(dec(x))
print("cma best", -es.result.fbest, json.dumps({k: round(v, 1) for k, v in dec(x).items()}))
print({k: v for k, v in r.items() if not k.startswith('_')})
