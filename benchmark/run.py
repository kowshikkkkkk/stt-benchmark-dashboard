import os, sys, time
import pandas as pd
from faster_whisper import WhisperModel
from tqdm import tqdm

MODELS = ["tiny.en", "base.en", "small.en"]
DEVICE = sys.argv[1] if len(sys.argv) > 1 else "cpu"
COMPUTE = "int8" if DEVICE == "cpu" else "float16"
THREADS = 4

man = pd.read_csv("data/manifest.csv")
os.makedirs("benchmark/out", exist_ok=True)

def transcribe(model, path):
    t0 = time.perf_counter()
    segments, _ = model.transcribe(path, language="en", beam_size=5,
                                   vad_filter=False)
    text = " ".join(s.text.strip() for s in segments)  # generator is lazy, so time it here
    return text, time.perf_counter() - t0

loads = []
for name in MODELS:
    out_path = f"benchmark/out/{name}_{DEVICE}.csv"
    if os.path.exists(out_path):
        print("skip (already done):", name)
        continue

    t0 = time.perf_counter()
    model = WhisperModel(name, device=DEVICE, compute_type=COMPUTE, cpu_threads=THREADS)
    load_s = time.perf_counter() - t0
    loads.append(dict(model=name, device=DEVICE, compute_type=COMPUTE, load_s=round(load_s, 2)))

    transcribe(model, man.path.iloc[0])  # warm-up, not recorded

    rows = []
    for r in tqdm(list(man.itertuples()), desc=name):
        hyp, proc = transcribe(model, r.path)
        rows.append(dict(clip_id=r.clip_id, model=name, device=DEVICE,
                         compute_type=COMPUTE, hypothesis=hyp,
                         audio_s=r.duration_s, proc_s=round(proc, 3),
                         rtf=round(proc / r.duration_s, 4)))
    pd.DataFrame(rows).to_csv(out_path, index=False)

if loads:
    pd.DataFrame(loads).to_csv(f"benchmark/out/model_load_{DEVICE}.csv", index=False)
print("done")
