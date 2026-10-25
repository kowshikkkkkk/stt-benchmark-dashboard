import time, numpy as np, pandas as pd
from faster_whisper import WhisperModel

d = pd.read_csv("analysis/clip_results_tagged.csv")
d["stall"] = (d.audio_s < 3) & (d.proc_s > 4)
rows = []
for variant, sub in [("all clips", d), ("excluding stalls", d[~d.stall])]:
    for m, g in sub.groupby("model"):
        slope, icpt = np.polyfit(g.audio_s, g.proc_s, 1)
        r2 = np.corrcoef(g.audio_s, g.proc_s)[0, 1] ** 2
        rows.append(dict(variant=variant, model=m, clips=len(g), fixed_cost_s=round(icpt, 3),
                         cost_per_audio_s=round(slope, 3), r2=round(r2, 3)))
fit = pd.DataFrame(rows)
fit.to_csv("dashboard/data/latency_fit.csv", index=False)
print(fit.to_string(index=False))

loads = []
for name in ["tiny.en", "base.en", "small.en"]:
    t = time.perf_counter()
    WhisperModel(name, device="cpu", compute_type="int8", cpu_threads=4)
    loads.append(dict(model=name, device="cpu", compute_type="int8",
                      load_s=round(time.perf_counter() - t, 2)))
pd.DataFrame(loads).to_csv("dashboard/data/model_load_cpu.csv", index=False)
print(pd.DataFrame(loads).to_string(index=False))
