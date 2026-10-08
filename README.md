# Tracy — Voice-Based Python Tutor

Tracy is a voice assistant that quizzes you on Python. Press Start, say which
Python topic you want to practise, answer her questions out loud, and get
spoken feedback plus a score (0–10) after every answer. End the session
whenever you like for a final score and a short summary of what to work on.

Built for the group project of *Programming & Introduction to Speech
Technology* (Speech Technology).

---

## The pipeline

```
 microphone ──► ASR ──────────────► LLM ─────────────────► TTS ──────────► speaker
 (browser)     faster-whisper       Groq (via LangChain)    edge-tts          + text on screen
               local, free          free API tier           free, online
```

| Step | Tool | Runs where | Cost |
|---|---|---|---|
| Speech → text (ASR) | `faster-whisper` (Whisper `base`, int8) | locally, on your CPU | free |
| Text → answer (LLM) | Groq API, e.g. `llama-3.3-70b-versatile` | Groq's servers | free tier |
| Answer → speech (TTS) | `edge-tts` (Microsoft neural voice `en-US-AriaNeural`) | Microsoft's servers | free |
| Interface | Streamlit | locally or Streamlit Cloud | free |

---

## Files

| File | What it does |
|---|---|
| `config.py` | Every tunable setting: Whisper model size, ASR vocabulary prompt, TTS voice, assistant name |
| `asr_utils.py` | Speech-to-text: decodes the browser recording and transcribes it with `faster-whisper` |
| `llm_chains.py` | All LLM work (LangChain + Groq): checking the topic, asking questions, scoring answers, the final summary |
| `tts_utils.py` | Text-to-speech: turns Tracy's answers into mp3 audio with `edge-tts` |
| `app.py` | The Streamlit interface and the Start → Topic → Quiz → Result flow |
| `asr_test/` | Small experiment that measures whether the ASR vocabulary prompt helps (see section 5) |

The logic is written as plain functions. The only classes are the Pydantic
models in `llm_chains.py`, which describe the exact shape of the LLM's answers
(e.g. `QuizTurn` with `score`, `feedback` and `next_question`).

---

## 1. Setup

You need a free Groq API key: https://console.groq.com/keys

Two ways to provide it (pick one):
- **Easiest:** do nothing now. When the app opens it asks you to paste the key.
  The key is kept only in your browser session and is never saved.
- **Optional:** copy `.env.example` to `.env` and paste the key there; the app
  then skips the key prompt on your own machine. `.env` is in `.gitignore`,
  so it is never pushed to GitHub.

Install the Python packages (no separate `ffmpeg` install is needed - `av`
bundles its own ffmpeg libraries):
```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

## 2. Run it

```bash
python -m streamlit run app.py
```
Your browser opens at http://localhost:8501. Stop the app with **Ctrl+C**.

Always use `python -m streamlit` (not bare `streamlit`), so the app runs with
the Python of your active venv (see Troubleshooting).

**Flow**
1. Click **▶️ Start Quiz**. Tracy greets you and asks which topic you want.
2. Click the microphone, say e.g. *"lists and tuples"* or *"mixed"*, click stop.
3. Tracy asks a question out loud. Answer through the microphone. You get
   spoken feedback, a score and the next question, on repeat.
4. **🏁 End & get results** gives a final score and summary.
   **🔁 Change topic** continues with another topic (your scores are kept).

**What you see on screen**
- The **orb** turns green and pulses only while Tracy is actually speaking;
  the line under it says when it is your turn. **🔊 Replay** plays the last
  answer again (and appears as **▶ Play answer** if the browser blocks autoplay).
- The current **question** is the largest element. Above it: Tracy's
  **feedback** with a coloured score badge, and (small) what the speech
  recogniser heard, so you can spot recognition errors.
- **Previous questions** (expandable) lists every question, your answer,
  the score and the feedback.

---

## 3. How it works, file by file

**`asr_utils.py`** — Streamlit's built-in microphone widget (`st.audio_input`)
returns the recording as bytes. We decode those ourselves with the `av`
library into 16 kHz mono audio (the format Whisper expects) and pass that to
`faster-whisper`. We deliberately do not use faster-whisper's own
`decode_audio()`: it calls `av.open(..., metadata_errors=...)`, an argument
that newer `av` versions removed, which crashed the app. A failed
transcription returns `""` instead of crashing, and the full error is printed
in the terminal.

Whisper also receives an **`initial_prompt`** (`config.ASR_PROMPT`): a short
sentence full of Python terms. Whisper treats it as text spoken just before
the recording, which nudges it towards spellings like *tuple*, *elif* and
*lambda* instead of *"two pull"* or *"else if"*. Switch it off with
`USE_ASR_PROMPT = False` in `config.py`.

**`llm_chains.py`** — the "brain". Each function builds a prompt with
LangChain's `ChatPromptTemplate`, sends it to a Groq model and forces the
answer into a Pydantic model, so the app gets real Python objects
(`result.score`, `result.feedback`) instead of free text. If the model's
structured answer fails (Plan A, tool calling), the code asks for plain JSON
and validates it itself (Plan B). `resolve_model_name()` asks Groq which
models your key can use and picks the best available one. The system prompt
tells the model to answer in short spoken sentences without markdown,
because everything is read aloud.

**`tts_utils.py`** — `edge-tts` generates speech as mp3 bytes in memory.
Because `edge-tts` is async and Streamlit is not, one background thread keeps
an event loop running for all speech jobs. Each request has a 20-second
timeout and one retry; if speech still fails, the answer is shown as text only.

**`app.py`** — ties everything together. Streamlit re-runs the whole script
after every click, so everything that must be remembered (stage, topic,
scores, conversation history) lives in `st.session_state`. The audio is
played inside the orb component, so JavaScript can make the orb pulse exactly
while the sound plays.

---

## 4. Timing per step

After every answer the terminal prints how long each step took, e.g.
```
[TIMING] ASR 1.2 s | LLM 0.9 s | TTS 0.6 s
```
Useful for judging the user experience (and for the report). To compare
speed and accuracy, try `WHISPER_MODEL_SIZE = "small"` in `config.py`.

---

## 5. ASR experiment: does the vocabulary prompt help?

`asr_test/compare_asr_prompt.py` transcribes the same recordings twice -
without and with `config.ASR_PROMPT` - and compares both with the expected
sentences using the **Word Error Rate** (WER: share of words that are wrong,
missing or extra; 0.0 is perfect, lower is better).

1. Record the 12 sentences in `asr_test/sentences.csv`, one file per sentence
   (e.g. with Voice Memos).
2. Save them in `asr_test/recordings/` as `01.m4a`, `02.m4a`, … `12.m4a`
   (`.wav` and `.mp3` also work).
3. Run, with the venv active:
   ```bash
   python asr_test/compare_asr_prompt.py
   ```
Per sentence you see both transcriptions, at the bottom the average WER of
both versions. Everything is also saved to `asr_test/results.csv`.

---

## 6. Online version (Streamlit Community Cloud)

The app can run online for free, so others can use it with just a link.
Every visitor enters their own Groq key; no key is stored in the code.

1. Push the code to a GitHub repository you have access to.
2. Go to https://share.streamlit.io, sign in with GitHub, click **Create app**.
3. Choose the repository, the branch and `app.py` as main file; under
   *Advanced settings* pick Python 3.11.
4. Click **Deploy** (the first build takes a few minutes).

Keep `WHISPER_MODEL_SIZE = "base"` there: the free servers have a slow CPU
and little memory. After a while without visitors the app goes to sleep and
needs ~30 seconds to wake up, so open it before a demo.

---

## 7. Troubleshooting

- **`TypeError: open() got an unexpected keyword argument 'metadata_errors'`
  or `file does not start with RIFF id`**: Python is probably running from
  conda's `base` environment instead of your venv. Check with `which python`;
  if the path does not contain `venv`, run:
  ```bash
  conda deactivate
  source venv/bin/activate
  pip install --force-reinstall --no-cache-dir faster-whisper
  python -m streamlit run app.py
  ```
- **App keeps asking for the API key / "key was rejected"**: re-copy the key
  from https://console.groq.com/keys (no spaces). A `.env` file must be named
  exactly `.env` and sit next to `app.py`.
- **"I didn't catch that" / empty transcription**: check that the browser
  allowed microphone access (icon in the address bar) and that you speak
  before pressing stop. Errors from the speech recogniser are printed in the
  terminal as `[ASR ERROR]`.
- **LLM errors**: shown as a yellow banner at the top of the page; the quiz
  continues with a fallback where possible.
- **No sound**: some browsers block autoplay on the first interaction. Click
  **▶ Play answer** under the orb; after that, autoplay normally works.
