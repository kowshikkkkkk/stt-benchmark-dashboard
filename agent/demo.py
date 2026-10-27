import os, time, uuid, random, asyncio
import pandas as pd
import edge_tts
from groq import Groq
from faster_whisper import WhisperModel
from posthog import Posthog

N_SESSIONS = 30
RUN = "run4"
STT_MODELS = ["tiny.en", "small.en"]
LLM_MODEL = os.environ.get("LLM_MODEL", "openai/gpt-oss-20b")
TTS_VOICE = "en-US-AriaNeural"
SYSTEM = ("You are a concise customer-support voice assistant. "
          "Reply in one or two short sentences.")

def ask(text):
    return llm.chat.completions.create(
        model=LLM_MODEL, max_tokens=400,
        messages=[{"role": "system", "content": SYSTEM},
                  {"role": "user", "content": text}],
        extra_body={"reasoning_effort": "low"})

ph = Posthog(os.environ["POSTHOG_KEY"], host=os.environ["POSTHOG_HOST"])
llm = Groq()  # reads GROQ_API_KEY
pre = ask("My order has not arrived yet.").choices[0].message.content or ""
assert pre.strip(), "LLM preflight returned empty text - raise max_tokens or drop extra_body"
print("LLM preflight ok:", LLM_MODEL, "|", pre[:60])

os.makedirs("agent/tts_out", exist_ok=True)
stt = {m: WhisperModel(m, device="cpu", compute_type="int8", cpu_threads=4)
       for m in STT_MODELS}

man = pd.read_csv("data/manifest.csv")
man = man[man.duration_s <= 12].sample(N_SESSIONS, random_state=7)
for m in STT_MODELS:  # warm-up, not recorded
    list(stt[m].transcribe(man.path.iloc[0], language="en")[0])

log = []
def emit(sid, event, **props):
    ph.capture(distinct_id=sid, event=event, properties=props)
    log.append(dict(event=event, ts=round(time.time(), 3), **props))

def ms(t0):
    return round((time.perf_counter() - t0) * 1000)

for r in man.itertuples():
    sid = f"test-{uuid.uuid4().hex[:8]}"
    model = random.choice(STT_MODELS)
    base = dict(run=RUN, session_id=sid, stt_model=model, accent=r.accent, audio_s=r.duration_s)
    emit(sid, "conversation_started", **base)
    t_all = time.perf_counter()
    stage = "stt"
    try:
        t0 = time.perf_counter()
        segs, _ = stt[model].transcribe(r.path, language="en", beam_size=5)
        text = " ".join(s.text.strip() for s in segs)
        emit(sid, "stt_done", latency_ms=ms(t0), chars=len(text), **base)
        if not text.strip():
            emit(sid, "conversation_dropped", reason="empty_transcript", stage="stt", **base)
            print(sid, model, "dropped: empty transcript")
            continue

        stage = "llm"
        t0 = time.perf_counter()
        reply = ask(text).choices[0].message.content or ""
        if not reply.strip():
            raise ValueError("empty LLM reply")
        emit(sid, "llm_done", latency_ms=ms(t0), chars=len(reply), **base)

        stage = "tts"
        t0 = time.perf_counter()
        asyncio.run(edge_tts.Communicate(reply, TTS_VOICE).save(f"agent/tts_out/{sid}.mp3"))
        emit(sid, "tts_done", latency_ms=ms(t0), **base)

        emit(sid, "conversation_completed", total_latency_ms=ms(t_all), **base)
        print(sid, model, "ok")
    except Exception as e:
        emit(sid, "conversation_dropped", reason="error", stage=stage,
             error=str(e)[:120], **base)
        print(sid, model, "dropped at", stage, "-", str(e)[:100])
    time.sleep(2)

ph.flush()
ph.shutdown()
os.makedirs("dashboard/data", exist_ok=True)
df = pd.DataFrame(log)
df.to_csv("dashboard/data/agent_events.csv", index=False)
print("events logged:", len(df))
print(df.event.value_counts().to_string())
