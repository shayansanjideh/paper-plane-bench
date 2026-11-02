import json, glob, os
TAGS = [("opus","Opus 5.5"),("sonnet","Sonnet 5.5"),("haiku","Haiku 5.5"),("gpt-6.1-sol","GPT-6.1 Sol"),
        ("gpt-6.1-terra","GPT-6.1 Terra"),("gpt-6-luna","GPT-6 Luna"),("cma_warm","CMA-ES warm"),("cma","CMA-ES cold"),("random","Random")]
KEYS = ["nose_folds_1","tip_fold_mm","nose_folds_2","keel_front_mm","keel_rear_mm","winglet_mm","elevator_mm","elevator_deg","dihedral_deg"]
def best(tag):
    top = None
    for f in sorted(glob.glob(f"results/{tag}/run*.json")):
        for a in json.load(open(f))["attempts"]:
            r = a["result"] or {}
            if r.get("valid") and (top is None or r["distance"] > top[0]):
                top = (r["distance"], {k: a["design"][k] for k in KEYS}, os.path.basename(f), a["i"])
    return top
if __name__ == "__main__":
    out = {t: dict(label=l, dist=b[0], design=b[1], run=b[2], i=b[3]) for t, l in TAGS for b in [best(t)]}
    json.dump(out, open("bests.json", "w"), indent=1)
    for t, v in out.items(): print(f"{v['label']:14s} {v['dist']:.2f} {v['run']} #{v['i']} {v['design']}")
