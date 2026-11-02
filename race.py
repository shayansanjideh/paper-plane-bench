import json, video as V, render as R
b = json.load(open("bests.json"))
tags = ["opus","sonnet","haiku","gpt-6.1-sol","gpt-6.1-terra","gpt-6-luna","cma_warm"]
ent = []
for t in tags:
    d = dict(b[t]["design"])
    for k in ("nose_folds_1","nose_folds_2"): d[k] = int(round(d[k]))
    ent.append((b[t]["label"], d))
R.write_video(V.race_frames(ent, "Every model's best plane, same throw"), "videos/race.mp4", V.FPS)
print("race done")
