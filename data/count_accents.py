from datasets import load_dataset

ds = load_dataset("edinburghcstr/edacc")
print(ds)
for split in ds:
    df = ds[split].select_columns(["speaker", "accent", "text"]).to_pandas()
    summary = df.groupby("accent").agg(clips=("text", "size"), speakers=("speaker", "nunique"))
    print("\n==", split, "==")
    print(summary.sort_values("clips", ascending=False).to_string())
