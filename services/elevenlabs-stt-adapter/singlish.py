"""
Singlish and Manglish linguistic assets for the ElevenLabs STT adapter.

Provides three exports consumed by main.py:
  - KEYTERMS: curated term list for ElevenLabs Scribe v2 keyterm biasing.
  - REFINEMENT_SYSTEM_PROMPT: system prompt for LLM post-ASR correction.
  - deterministic_normalize: conservative, zero-latency transcript fixups.

Linguistic note: Singlish/Manglish discourse particles (lah, leh, lor, …)
and code-switched vocabulary are *features* of the speech variety, not errors.
Nothing in this module translates, removes, or standardises them.
"""

import re

# ---------------------------------------------------------------------------
# KEYTERMS
# ElevenLabs Scribe v2 keyterm biasing: terms the model should recognise.
# Rules: ≤100 entries, each ≤50 chars and ≤5 words, lowercase unless a
# proper noun.  Spelling variants are intentional — ASR may produce any of
# them for the same spoken token.
# ---------------------------------------------------------------------------

KEYTERMS: list[str] = [
    # ---- sentence-final / discourse particles ----------------------------
    "lah",
    "leh",
    "lor",
    "meh",
    "hor",
    "mah",
    "sia",
    "siah",
    "ah",
    "hah",
    "liao",
    "wah",
    "eh",
    # ---- singlish function words (carry pragmatic meaning in CSE) --------
    "one",
    "what",
    "already",
    "also",
    # ---- exclamations / emotional fillers --------------------------------
    "alamak",
    "walao",
    "walau",
    "wah lao",
    "aiyoh",
    "aiyah",
    "fuyoh",
    "sian",
    # ---- attitudinal / character descriptors ----------------------------
    "kiasu",
    "kiasi",
    "paiseh",
    "bojio",
    "atas",
    "blur",
    "steady",
    "garang",
    "terror",
    "sabo",
    "gila",
    "cincai",
    # ---- positive affect ------------------------------------------------
    "shiok",
    "syiok",
    "swee",
    # ---- social identity & address terms --------------------------------
    "ang moh",
    "ah beng",
    "ah lian",
    "auntie",
    "uncle",
    # ---- actions / states -----------------------------------------------
    "makan",
    "tapau",
    "dabao",
    "tabao",
    "chope",
    "kena",
    "lepak",
    "gostan",
    "tahan",
    "buay tahan",
    "potong",
    "kacau",
    "habis",
    "pakat",
    # ---- descriptors / evaluatives --------------------------------------
    "jialat",
    "teruk",
    "rojak",
    "agak agak",
    # ---- food & drink (high-frequency in SG/MY conversation) -----------
    "kopi",
    "kopi o",
    "teh",
    "teh tarik",
    "kopitiam",
    "hawker",
    "mamak",
    "ondeh",
    # ---- places / local referents ---------------------------------------
    "void deck",
    "lobang",
    # ---- multi-word expressions (≤5 words, ≤50 chars) ------------------
    "can or not",
    "cannot make it",
    "like that",
    "last time",
    "on lah",
    "catch no ball",
    "confirm plus chop",
]

# Deduplicate in insertion order (safety net; no semantic duplicates above).
KEYTERMS = list(dict.fromkeys(KEYTERMS))


# ---------------------------------------------------------------------------
# REFINEMENT_SYSTEM_PROMPT
# Instructs the LLM corrector to fix ASR errors while preserving all
# Singlish/Manglish vocabulary and register intact.
# ---------------------------------------------------------------------------

REFINEMENT_SYSTEM_PROMPT: str = (
    "You are a post-ASR transcript corrector for Singlish and Manglish speech. "
    "Fix only clear phonetic mis-transcriptions and obvious speech-recognition errors "
    "(e.g. a Malay, Hokkien, Cantonese, or Tamil word mangled into a wrong English word, "
    "or a discourse particle mis-spelled in a way that changes its identity). "
    "Singlish and Manglish discourse particles — lah, leh, lor, meh, hor, mah, sia, siah, "
    "ah, hah, liao, wah, eh — and all colloquial or borrowed vocabulary "
    "(shiok, kiasu, paiseh, makan, chope, kena, tapau, alamak, walao, aiyoh, jialat, "
    "buay tahan, agak agak, and every similar Malay/Hokkien/Cantonese/Tamil borrowing) "
    "are CORE FEATURES of the speech variety, NOT transcription errors: preserve them "
    "verbatim, exactly as spoken, with no translation, no standardisation to formal English, "
    "no removal, and no explanation. "
    "Keep the speaker's original word order, informal register, and sentence structure. "
    "If the transcript already looks correct, return it unchanged. "
    "NEVER add, remove, summarise, translate, or explain anything. "
    "Output ONLY the corrected transcript text — no quotes, no markdown, no preamble. "
    "Correct examples you must not alter: "
    "'can lah', 'makan already lah', 'so shiok sia', "
    "'eh you blur blur one leh', 'walao this jialat lah'."
)


# ---------------------------------------------------------------------------
# deterministic_normalize
# Conservative, zero-latency fixups applied before or after ASR.
# Only corrections with essentially zero false-positive risk are applied.
# ---------------------------------------------------------------------------

_MULTI_SPACE: re.Pattern[str] = re.compile(r" {2,}")


def deterministic_normalize(text: str) -> str:
    """Apply conservative, unambiguous normalizations to ASR transcript text.

    Currently applies:
    - Collapse runs of two or more consecutive spaces into a single space.
    - Strip leading and trailing whitespace.

    All other corrections are intentionally omitted: Singlish particles and
    colloquial vocabulary have too many surface forms to normalise safely
    without context, and the risk of false positives outweighs any benefit.

    Guarantees:
    - Idempotent: f(f(x)) == f(x) for all x.
    - Near-identity on plain English: deterministic_normalize('hello world')
      returns 'hello world' unchanged.
    """
    result = _MULTI_SPACE.sub(" ", text)
    result = result.strip()
    return result
