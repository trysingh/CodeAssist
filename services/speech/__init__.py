"""Voice input (speech-to-text -> clean English prompt) and optional read-aloud. Routes: views/voice_routes.py. Setup: docs/VOICE_SETUP.md.

    triggers   split_trigger()                pure: spoken "go ahead now" at the end of an utterance
    providers  GeminiStt / CloudflareStt / OpenAICompatStt   one HTTP call each: wav bytes -> SttResult
    resolver   resolve_chain(runner)          which providers to try, in order (empty = the browser's own speech recognition)
    refine     refine_to_english(text, runner)  tidy a transcript with the user's chat runner; never raises
    service    voice_config / transcribe_audio / refine_text / speak   what the routes call
"""
from services.speech.triggers import split_trigger, parse_phrases   # noqa: F401
