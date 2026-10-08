"""
config.py — every tunable setting lives here, nowhere else.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# OPTIONAL: only used when running locally with a .env file. On the deployed
# app each visitor types their own key into the page instead (see app.py).
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

# --- Speech-to-text ---
# "base" is a good speed/accuracy balance on a normal laptop CPU.
# Try "small" if you want noticeably better accuracy and don't mind an
# extra second or so of latency per answer.
WHISPER_MODEL_SIZE = "base"
WHISPER_LANGUAGE = "en"

# Vocabulary hint for Whisper ("initial_prompt"). Whisper treats this text as
# if it had been said just before the recording, which nudges it towards
# these spellings. Without it, Python terms are often misheard (e.g. "tuple"
# -> "toople"/"two pull", "elif" -> "else if"/"Eliff", "dict" -> "dick").
# Written as a natural sentence on purpose: Whisper copes better with that
# than with a bare word list. Keep it short - Whisper only uses the last
# ~224 tokens, and a very long prompt can make it "echo" the prompt back on
# near-silent recordings.
# Set USE_ASR_PROMPT = False to compare results with and without the hint.
USE_ASR_PROMPT = True
ASR_PROMPT = (
    "A student answers a Python quiz question about variables, integers, floats, "
    "strings, booleans, lists, tuples, dictionaries, dicts, sets, None, if, elif, "
    "else, for loops, while loops, range, break, continue, functions, def, return, "
    "arguments, parameters, *args, **kwargs, lambda, list comprehensions, classes, "
    "objects, self, __init__, inheritance, methods, modules, import, try, except, "
    "exceptions, f-strings, slicing, indexing, len, append, print and input."
)

# --- Text-to-speech ---
TTS_VOICE = "en-US-AriaNeural"
TTS_RATE = "+5%"

# --- LLM ---
# Used only if we can't reach Groq's model-listing endpoint at all (see
# llm_chains.py's resolve_model_name) - normally the app figures out which
# model your specific account can use automatically.
LLM_MODEL_FALLBACK = "llama-3.3-70b-versatile"

# --- Assistant identity ---
ASSISTANT_NAME = "Tracy"
