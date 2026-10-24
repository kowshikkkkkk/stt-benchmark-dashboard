import shutil
import numpy as np
import pandas as pd

df = pd.read_csv("analysis/clip_results_tagged.csv")
df["dur_bin"] = pd.cut(df.audio_s, [0, 3, 6, 10, 20, 100],
                       labels=["0-3s", "3-6s", "6-10s", "10-20s", "20s+"])

b = (df.groupby(["model", "dur_bin"], observed=True)
       .agg(clips=("clip_id", "size"), audio_s=("audio_s", "sum"), proc_s=("proc_s", "sum"))
       .reset_index())
b["rtf"] = b.proc_s / b.audio_s
print("RTF by clip duration:")
print(b.pivot(index="dur_bin", columns="model", values="rtf").round(3))
print("\nclips per bin:")
print(b.pivot(index="dur_bin", columns="model", values="clips"))

fits = []
for m, g in df.groupby("model"):
    slope, icpt = np.polyfit(g.audio_s, g.proc_s, 1)
    r2 = np.corrcoef(g.audio_s, g.proc_s)[0, 1] ** 2
    fits.append(dict(model=m, fixed_cost_s=icpt, cost_per_audio_s=slope, r2=r2))
fits = pd.DataFrame(fits)
print("\nproc_s = fixed + per_second * audio_s:")
print(fits.round(3).to_string(index=False))

b.to_csv("dashboard/data/rtf_by_duration.csv", index=False)
fits.to_csv("dashboard/data/latency_fit.csv", index=False)
df.drop(columns=["hypothesis", "reference"], errors="ignore").to_csv("dashboard/data/fact_clips.csv", index=False)
for f in ["analysis/results.csv", "analysis/model_diffs.csv", "benchmark/out/model_load_cpu.csv"]:
    shutil.copy(f, "dashboard/data/")
print("\nexported to dashboard/data/")
