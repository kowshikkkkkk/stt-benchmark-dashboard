import time, pandas as pd
from faster_whisper import WhisperModel

d = pd.read_csv('analysis/clip_results_tagged.csv')

print("== short clips (<3 s): RTF with and without stalls (>4 s) ==")
for m, g in d[d.audio_s < 3].groupby('model'):
    o = g[g.proc_s > 4]
    rest_p, rest_a = g.proc_s.sum() - o.proc_s.sum(), g.audio_s.sum() - o.audio_s.sum()
    print(f"{m}: clips={len(g)} stalled={len(o)} RTF_all={g.proc_s.sum()/g.audio_s.sum():.2f} "
          f"RTF_without_stalls={rest_p/rest_a:.2f} median_proc_s={g.proc_s.median():.2f}")

print("\n== retry test: slow small.en clips, default vs temperature=0 ==")
slow = d[(d.model == 'small.en') & (d.audio_s < 3) & (d.proc_s > 4)].head(8)
man = pd.read_csv('data/manifest.csv').set_index('clip_id')
wm = WhisperModel('small.en', device='cpu', compute_type='int8', cpu_threads=4)
list(wm.transcribe(man.path.iloc[0], language='en')[0])
def timed(path, **kw):
    t = time.perf_counter()
    list(wm.transcribe(path, language='en', beam_size=5, **kw)[0])
    return time.perf_counter() - t
for r in slow.itertuples():
    p = man.loc[r.clip_id, 'path']
    print(f"{r.clip_id}: audio={r.audio_s:.2f}s original={r.proc_s:.1f}s "
          f"default_now={timed(p):.1f}s temp0={timed(p, temperature=0.0):.1f}s")

print("\n== implausible speech rate (>6 words/s) ==")
d['wps'] = d.ref_words / d.audio_s
bad = d.wps > 6
print("clips flagged per model:", d[bad].groupby('model').size().to_dict())
for m, g in d.groupby('model'):
    k = g[g.wps <= 6]
    print(f"{m}: WER all={g.errors.sum()/g.ref_words.sum():.3f} "
          f"WER without flagged={k.errors.sum()/k.ref_words.sum():.3f}")
