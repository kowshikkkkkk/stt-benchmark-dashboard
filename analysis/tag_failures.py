import re
import pandas as pd

df = pd.read_csv("analysis/clip_results.csv")
df["ref_n"] = df.ref_n.fillna("").astype(str)
df["hyp_n"] = df.hyp_n.fillna("").astype(str)

def has_new_repeat(ref, hyp):
    w = hyp.split()
    for n in (2, 3, 4):
        grams = [" ".join(w[i:i + n]) for i in range(len(w) - n + 1)]
        for g in set(grams):
            if grams.count(g) >= 3 and g not in ref:
                return True
    return False

def tag(r):
    if r.errors == 0:
        return "correct"
    if has_new_repeat(r.ref_n, r.hyp_n):
        return "hallucinated_repeat"
    if (set(re.findall(r"\d+", r.ref_n)) ^ set(re.findall(r"\d+", r.hyp_n))) - {"0"}:
        return "number"
    if r.ref_words <= 4:
        return "short_utterance"
    if r.ins > 0.5 * r.ref_words:
        return "extra_speech_inserted"
    if r.dele > 0.5 * r.ref_words:
        return "dropped_speech"
    return "other_substitution"

df["failure_type"] = df.apply(tag, axis=1)
df.to_csv("analysis/clip_results_tagged.csv", index=False)

bad = df[df.failure_type != "correct"]
print(pd.crosstab(bad.model, bad.failure_type))
print()
print(pd.crosstab(bad.accent, bad.failure_type))

lab = (df[(df.model == "small.en") & (df.errors > 0)]
       .sort_values("errors", ascending=False).head(80)
       [["clip_id", "accent", "failure_type", "ref_n", "hyp_n", "errors"]].copy())
lab["manual_tag"] = ""
lab.to_csv("analysis/failures_to_label.csv", index=False)
print("\nwrote analysis/failures_to_label.csv")
