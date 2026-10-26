import numpy as np, pandas as pd, jiwer
from whisper_normalizer.english import EnglishTextNormalizer

MODELS = ["tiny.en", "base.en", "small.en"]
DEVICE = "cpu"
B = 2000
rng = np.random.default_rng(42)
norm = EnglishTextNormalizer()

man = pd.read_csv("data/manifest.csv")
res = pd.concat([pd.read_csv(f"benchmark/out/{m}_{DEVICE}.csv") for m in MODELS])
df = res.merge(man[["clip_id", "speaker", "accent", "reference"]], on="clip_id")
df["hypothesis"] = df.hypothesis.fillna("").astype(str)
df["ref_n"] = df.reference.astype(str).map(norm)
df["hyp_n"] = df.hypothesis.map(norm)
df = df[df.ref_n.str.strip() != ""].copy()

def counts(ref, hyp):
    n = len(ref.split())
    if not hyp.strip():
        return 0, n, 0, n  # empty output: every ref word is a deletion
    o = jiwer.process_words(ref, hyp)
    return o.substitutions, o.deletions, o.insertions, n

c = [counts(r, h) for r, h in zip(df.ref_n, df.hyp_n)]
df[["sub", "dele", "ins", "ref_words"]] = pd.DataFrame(c, index=df.index)
df["errors"] = df["sub"] + df["dele"] + df["ins"]
df["wer_clip"] = df.errors / df.ref_words
df.to_csv("analysis/clip_results.csv", index=False)
df = df[df.audio_s <= 60].copy()  # accuracy scored on clips up to 60 s

rows, diffs = [], []
totE = np.zeros((B, len(MODELS)))
totR = np.zeros((B, len(MODELS)))
for acc, g in df.groupby("accent"):
    spk = sorted(g.speaker.unique())
    E = np.zeros((len(spk), len(MODELS)))
    R = np.zeros_like(E)
    for j, m in enumerate(MODELS):
        gm = g[g.model == m].groupby("speaker")[["errors", "ref_words"]].sum().reindex(spk).fillna(0)
        E[:, j] = gm.errors.values
        R[:, j] = gm.ref_words.values
    idx = rng.integers(0, len(spk), size=(B, len(spk)))
    bE, bR = E[idx].sum(axis=1), R[idx].sum(axis=1)
    totE += bE
    totR += bR
    boot = bE / bR
    for j, m in enumerate(MODELS):
        gm = g[g.model == m]
        lo, hi = np.percentile(boot[:, j], [2.5, 97.5])
        rows.append(dict(accent=acc, model=m, clips=len(gm), speakers=len(spk),
                         ref_words=int(R[:, j].sum()), wer=E[:, j].sum() / R[:, j].sum(),
                         ci_low=lo, ci_high=hi,
                         rtf_pooled=gm.proc_s.sum() / gm.audio_s.sum(),
                         rtf_median=gm.rtf.median()))
    d = boot[:, 2] - boot[:, 0]
    lo, hi = np.percentile(d, [2.5, 97.5])
    pt = E[:, 2].sum() / R[:, 2].sum() - E[:, 0].sum() / R[:, 0].sum()
    diffs.append(dict(accent=acc, small_minus_tiny=pt, ci_low=lo, ci_high=hi))

boot = totE / totR
for j, m in enumerate(MODELS):
    gm = df[df.model == m]
    lo, hi = np.percentile(boot[:, j], [2.5, 97.5])
    rows.append(dict(accent="ALL (stratified)", model=m, clips=len(gm),
                     speakers=gm.speaker.nunique(), ref_words=int(gm.ref_words.sum()),
                     wer=gm.errors.sum() / gm.ref_words.sum(), ci_low=lo, ci_high=hi,
                     rtf_pooled=gm.proc_s.sum() / gm.audio_s.sum(), rtf_median=gm.rtf.median()))

summary = pd.DataFrame(rows)
summary.to_csv("analysis/results.csv", index=False)
pd.DataFrame(diffs).to_csv("analysis/model_diffs.csv", index=False)

pd.set_option("display.width", 200)
print(summary.round(3).to_string(index=False))
print()
print(pd.DataFrame(diffs).round(3).to_string(index=False))
print("\nworst 8 clips for small.en:")
w = df[df.model == "small.en"].sort_values("wer_clip", ascending=False).head(8)
for r in w.itertuples():
    print(f"\n[{r.accent}] wer={r.wer_clip:.2f}\n REF: {r.ref_n}\n HYP: {r.hyp_n}")
