import io, math, os
import pandas as pd
import soundfile as sf
from datasets import load_dataset, Audio
from tqdm import tqdm

ACCENTS = ["Mainstream US English", "Southern British English",
           "Indian English", "Nigerian English", "Jamaican English"]
PER_ACCENT = 60
MIN_WORDS = 3
SEED = 42

os.makedirs("data/audio", exist_ok=True)
ds = load_dataset("edinburghcstr/edacc")

parts = []
for split in ds:
    m = ds[split].select_columns(["speaker", "accent", "text"]).to_pandas()
    m["split"] = split
    m["idx"] = range(len(m))
    parts.append(m)
meta = pd.concat(parts, ignore_index=True)

val_spk = set(meta[meta.split == "validation"].speaker)
test_spk = set(meta[meta.split == "test"].speaker)
print("speakers present in both splits:", val_spk & test_spk)

meta = meta[meta.accent.isin(ACCENTS)]
tagged = meta.text.str.contains("<", regex=False)
print("dropped for tag markers:", int(tagged.sum()))
meta = meta[~tagged]
meta = meta[meta.text.str.split().str.len() >= MIN_WORDS]

chosen = []
for acc, g in meta.groupby("accent"):
    per_spk = math.ceil(PER_ACCENT / g.speaker.nunique())
    pick = g.sample(frac=1, random_state=SEED).groupby("speaker").head(per_spk)
    chosen.append(pick.sample(min(len(pick), PER_ACCENT), random_state=SEED))
chosen = pd.concat(chosen)

dec = {s: ds[s].cast_column("audio", Audio(decode=False)) for s in ds}
rows = []
for r in tqdm(list(chosen.itertuples())):
    a = dec[r.split][int(r.idx)]["audio"]
    data, sr = sf.read(io.BytesIO(a["bytes"]) if a.get("bytes") else a["path"])
    if data.ndim > 1:
        data = data.mean(axis=1)
    cid = f"{r.split}_{r.idx}"
    path = f"data/audio/{cid}.wav"
    sf.write(path, data, sr)
    rows.append(dict(clip_id=cid, path=path, speaker=r.speaker, accent=r.accent,
                     reference=r.text, duration_s=round(len(data) / sr, 2)))

out = pd.DataFrame(rows)
out.to_csv("data/manifest.csv", index=False)
print(out.groupby("accent").agg(clips=("clip_id", "size"),
      speakers=("speaker", "nunique"),
      minutes=("duration_s", lambda x: round(x.sum() / 60, 1))))
