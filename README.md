# STT Benchmark Dashboard

Which open speech-to-text model works best for which accents, and what slows a voice pipeline down.

Status: benchmark and failure analysis done. Power BI dashboard and PostHog instrumentation in progress.

## Setup
- Data: EdAcc (Edinburgh International Accents of English Corpus), validation + test splits, accent labels assigned by linguists. License: CC-BY-SA per the Edinburgh DataShare record.
- Sample: 296 clips, 5 accents, 44 speakers, about 49 minutes of audio. Up to 60 clips per accent, spread across speakers. Clips under 3 words or with tag markers removed.
- Models: Whisper tiny.en, base.en, small.en via faster-whisper. CPU, int8, 4 threads, beam size 5, VAD off, forced English.
- Machine: Intel Core i5-10300H, 16 GB RAM, Windows. Warm-up run excluded.
- Scoring: Whisper English text normalizer on both reference and output. WER pooled per accent. 95% CIs from a speaker-level bootstrap (2000 resamples). Latency as real-time factor (processing time / audio duration).

## Results
Overall WER (stratified bootstrap, 95% CI):
- tiny.en: 23.6% (18.4-30.7), RTF 0.07
- base.en: 20.3% (15.5-27.1), RTF 0.15
- small.en: 15.4% (12.3-19.8), RTF 0.32

WER by accent (tiny / base / small):
- Indian English (5 speakers): 13.8% / 11.4% / 10.3%
- Jamaican English (6): 36.3% / 29.8% / 22.2%
- Mainstream US English (17): 17.9% / 19.0% / 15.4%
- Nigerian English (5): 37.6% / 31.0% / 20.5%
- Southern British English (11): 21.0% / 17.6% / 12.7%

Findings:
1. small.en beats tiny.en on every accent, and the bootstrap interval on the difference excludes zero for all five.
2. The gain from a bigger model is largest where the small model struggles most: Nigerian English (-17.1 points) and Jamaican English (-14.1 points), versus -2.5 for US English.
3. Latency roughly doubles with each size step. small.en costs about 4.5x the compute of tiny.en.
4. Short clips cost more per second of audio (median RTF is well above pooled RTF), consistent with a fixed per-call cost from Whisper's 30-second input window.
5. Most failing clips are ordinary mishearings. Digits stay a weak spot at every size, and hallucinated repeats are rare (2-6 clips of 296 per model).

## Limitations
- Small sample: 5-17 speakers per accent, so the intervals are wide. Accent rankings are NOT supported: the intervals overlap heavily.
- Indian English clips are longer on average, and WER on very short clips is noisy.
- EdAcc is conversational speech, so references contain partial words and disfluencies, and some clips have speech the reference does not cover.
- Latency is from one run on one laptop CPU, not repeated trials.
- Each failing clip gets one failure tag (first matching rule).

## Reproduce
python data/make_manifest.py
python benchmark/run.py cpu
python analysis/wer.py
python analysis/tag_failures.py

## Layout
- data/ manifest and dataset prep
- benchmark/ transcription runner and raw outputs
- analysis/ WER, RTF, bootstrap, failure tagging
- dashboard/ Power BI (coming)
- agent/ PostHog instrumentation (coming)
