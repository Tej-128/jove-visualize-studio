"""
tts_preprocessor.py
────────────────────
Full pre-processing pipeline for K12 ElevenLabs TTS.

Strategy (conservative, no-SSML rollback):
- Convert math symbols / units / LaTeX / chemistry into plain spoken English.
- Preserve script wording exactly. Never add teacher phrases such as "put that into".
- Do NOT use SSML at all. No <break> tags are inserted anywhere.
- Keep pacing natural by using normal punctuation, paragraph breaks, and spoken
  math descriptions such as "two plus two equals four".
- Keep temperature/unit handling intact: 4.8°F becomes 4.8 degrees Fahrenheit.
- DO NOT post-process audio. DO NOT change voice speeds.

Usage:
    from tts_preprocessor import prepare_script_for_tts
    chunks, debug_log = prepare_script_for_tts(raw_script, debug=True)
"""

import re
import logging

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# PACING POLICY — NO SSML
# ─────────────────────────────────────────────────────────────────────────────

# Do not insert SSML tags. ElevenLabs can speak SSML markup literally or behave
# inconsistently depending on model/API handling, which can sound like gibberish.
# We rely only on plain text, punctuation, and paragraph breaks.

# ─────────────────────────────────────────────────────────────────────────────
# CHUNKING CONSTANT — avoid unnecessary restarts for normal K12 scripts
# ─────────────────────────────────────────────────────────────────────────────
#
# Most scripts are below the practical 5,000-character target. Keeping this at
# 5000 avoids unnecessary extra ElevenLabs calls and avoids artificial stitch
# points. Only genuinely long scripts are split, and only at safe boundaries.
MAX_CHARS_PER_CHUNK = 5000


# ══════════════════════════════════════════════════════════════════════════════
# COMMON TEXT REPAIR
# ══════════════════════════════════════════════════════════════════════════════

_MOJIBAKE_REPLACEMENTS = {
    # Currency — must precede all â‚... entries.
    'â‚¬': '€',
    'â‚¹': '₹',
    'â‚£': '£',
    # Degree / superscripts.
    'Â°': '°', 'Â±': '±', 'Â²': '²', 'Â³': '³',
    # Dashes / punctuation.
    'â€“': '–', 'â€”': '—', 'â€¢': '•',
    # Comparison operators.
    'â‰¤': '≤', 'â‰¥': '≥', 'â‰ ': '≠', 'â‰ˆ': '≈', 'â‰…': '≅', 'âˆ¼': '∼',
    # Math symbols.
    'âˆš': '√', 'âˆ†': '∆', 'â–³': '△',
    'â†’': '→', 'â†←': '←', 'â†”': '↔',
    # Greek.
    'Î¼': 'μ',
    # Subscripts — specific sequences only.
    'â‚€': '₀', 'â‚‚': '₂', 'â‚ƒ': '₃', 'â‚„': '₄', 'â‚…': '₅',
    'â‚†': '₆', 'â‚‡': '₇', 'â‚ˆ': '₈', 'â‚‰': '₉',
}


def repair_common_text_artifacts(text: str) -> str:
    for bad, good in _MOJIBAKE_REPLACEMENTS.items():
        text = text.replace(bad, good)
    text = text.replace('\u2212', '-')  # Unicode minus
    return text


# ══════════════════════════════════════════════════════════════════════════════
# GEMINI FALLBACK — SURGICAL FRAGMENT-LEVEL, opt-in only
#
# Sends ONLY the unresolved fragments to Gemini 2.5 Flash, never the whole script.
# Toggle: ENABLE_GEMINI_SYMBOL_FALLBACK = true in Streamlit secrets.
# Safe by default: if disabled or any call fails, the original fragment is kept
# and the rest of the script is never affected.
# ══════════════════════════════════════════════════════════════════════════════


def _find_unresolved_fragments(text: str) -> list:
    """
    Identify fragments of text that the deterministic rules couldn't fully convert.
    Returns a list of (start_index, end_index, fragment) tuples.

    A fragment is "unresolved" if it contains:
    - Non-ASCII characters (other than smart quotes which we accept)
    - Math-like patterns that survived our rules (orphaned LaTeX, unusual sub/super combinations)
    - Sequences of unusual punctuation

    Whitelist of accepted non-ASCII chars: smart quotes, em/en dashes, ellipsis,
    paragraph mark — these don't need Gemini's help.
    """
    accepted_non_ascii = set('\u2018\u2019\u201c\u201d\u2013\u2014\u2026\u00b6')
    fragments = []

    # Pattern 1: contiguous runs of non-ASCII characters (excluding whitelist).
    # We expand each run to include adjacent word/number context for Gemini.
    i = 0
    while i < len(text):
        ch = text[i]
        if not ch.isascii() and ch not in accepted_non_ascii:
            # Found start of unresolved span. Extend backwards to include preceding word.
            start = i
            while start > 0 and not text[start - 1].isspace() and text[start - 1] not in '.!?,;:':
                start -= 1
            # Extend forwards until we hit whitespace boundary after the bad chars end.
            j = i
            while j < len(text) and (not text[j].isascii() or text[j] in accepted_non_ascii
                                      or (text[j].isalnum() and j > i)):
                if text[j].isspace():
                    break
                j += 1
            # Expand forward to include following word for context.
            while j < len(text) and not text[j].isspace() and text[j] not in '.!?,;:':
                j += 1
            fragments.append((start, j, text[start:j]))
            i = j
        else:
            i += 1

    # Pattern 2: orphaned LaTeX commands (\foo without proper handler).
    for m in re.finditer(r'\\[A-Za-z]+(?:\{[^}]*\})?', text):
        # Skip if already inside a fragment we found.
        if any(s <= m.start() < e for s, e, _ in fragments):
            continue
        fragments.append((m.start(), m.end(), m.group(0)))

    # Dedupe and sort by position.
    fragments = sorted(set(fragments), key=lambda x: x[0])
    return fragments


def _gemini_convert_fragment(fragment: str, context: str = "") -> str:
    """
    Send a single fragment to Gemini 2.5 Flash and return its spoken-English
    conversion. On any failure, returns the fragment unchanged so the pipeline
    is never blocked.
    """
    try:
        import streamlit as st
        import google.generativeai as genai

        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
        model = genai.GenerativeModel("gemini-2.5-flash")

        prompt = (
            "You are converting one short fragment of a TTS script.\n\n"
            "Convert mathematical symbols, equations, special characters, or unusual\n"
            "notation in the fragment into plain spoken English.\n\n"
            "Strict rules:\n"
            "- Output ONLY the converted fragment, nothing else.\n"
            "- No markdown, no quotes, no preamble, no explanation.\n"
            "- Preserve all regular words exactly as written.\n"
            "- Do not add teacher phrases, connectors, or commentary.\n"
            "- Do not change punctuation, capitalization, or spacing of regular words.\n"
            "- If the fragment is already plain English with no symbols, return it unchanged.\n\n"
            f"Context (do not modify, just for reference): {context!r}\n"
            f"Fragment to convert: {fragment!r}\n\n"
            "Converted fragment:"
        )

        response = model.generate_content(prompt)
        result = response.text.strip()

        # Strip any wrapping quotes Gemini sometimes adds despite instructions.
        if len(result) >= 2 and result[0] in '"\'' and result[-1] == result[0]:
            result = result[1:-1]

        # Sanity check: if Gemini's output is wildly longer than the fragment,
        # it probably went off-script. Fall back to the original fragment.
        if len(result) > max(len(fragment) * 4, len(fragment) + 60):
            logger.warning(
                f"Gemini output suspiciously long for fragment {fragment!r}; "
                f"using original fragment instead."
            )
            return fragment

        # Empty result is also suspicious.
        if not result.strip():
            return fragment

        logger.info(f"Gemini fragment convert: {fragment!r} -> {result!r}")
        return result

    except Exception as e:
        logger.warning(f"Gemini fragment conversion failed for {fragment!r}: {e}")
        return fragment


def gemini_symbol_fallback(text: str) -> str:
    """
    Surgical fallback: only fragments the deterministic rules couldn't handle
    are sent to Gemini. The rest of the text is untouched.

    Behavior:
    - If no unresolved fragments are found, returns text unchanged.
    - If fragments exist but ENABLE_GEMINI_SYMBOL_FALLBACK is false, logs and
      returns text unchanged (deterministic-only mode).
    - If enabled, each fragment is sent individually with surrounding context.
      Gemini's reply replaces ONLY that fragment in the original text.
    - Any per-fragment failure is silent: the fragment stays as-is, the pipeline
      continues, and other fragments are still processed.

    Why this is safe for the universal case:
    - Gemini never sees deterministic-converted equations like "x plus y equals 50",
      so it cannot rephrase them.
    - Gemini cannot add or remove paragraphs / numbered points, since fragments
      are single tokens or short phrases.
    - No SSML is ever inserted, so Gemini never sees break tags either.
    """
    fragments = _find_unresolved_fragments(text)
    if not fragments:
        return text

    # Check the toggle. If disabled, log what we would have sent but don't call.
    try:
        import streamlit as st
        enabled = st.secrets.get("ENABLE_GEMINI_SYMBOL_FALLBACK", False)
    except Exception:
        enabled = False

    if not enabled:
        sample = [f for _, _, f in fragments[:3]]
        logger.warning(
            f"Found {len(fragments)} unresolved fragment(s) but Gemini fallback is "
            f"disabled. Examples: {sample}. Set ENABLE_GEMINI_SYMBOL_FALLBACK=true in "
            f"secrets to enable surgical fallback."
        )
        return text

    # Process fragments from end to start so earlier indices remain valid
    # as we splice replacements in.
    result_text = text
    successes = 0
    for start, end, fragment in reversed(fragments):
        # Get ~30 chars of context on each side for Gemini.
        ctx_start = max(0, start - 30)
        ctx_end = min(len(result_text), end + 30)
        context = result_text[ctx_start:ctx_end]

        converted = _gemini_convert_fragment(fragment, context)
        if converted != fragment:
            result_text = result_text[:start] + converted + result_text[end:]
            successes += 1

    logger.info(
        f"Gemini surgical fallback: {successes}/{len(fragments)} fragments converted."
    )
    return result_text


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — STRIP METADATA AND NON-SPOKEN STRUCTURAL ELEMENTS
# ══════════════════════════════════════════════════════════════════════════════

def _safe_paren_strip(m: re.Match) -> str:
    content = m.group(1).strip()
    if re.search(r'[\d\+\=<>±√∑∫°%\-]', content): return m.group(0)
    if len(content) <= 3 and content.islower():    return m.group(0)
    stage = {'beat','pause','note','music','cut','vo','sfx',
             'emphasis','slow','fast','softly','loudly'}
    if any(w in content.lower().split() for w in stage): return ''
    if content.isupper(): return ''
    return m.group(0)


def strip_metadata_sections(text: str) -> str:
    for p in [
        r'(?ims)^\s*(?:#+\s*)?\[?pronunciation(?:\s+guide|\s+notes)?\]?\s*:?\s*$[\s\S]*?(?=^\s*$|^\s*(?:script|lesson|references|metadata|problem)\b|\Z)',
        r'(?ims)^\s*\[pronunciation[^\]]*\]\s*$[\s\S]*?(?=^\s*$|^\s*(?:script|lesson|references|metadata|problem)\b|\Z)',
        r'(?ims)^\s*#+\s*pronunciation[\s\S]*?(?=^\s*#+\s*(?:script|lesson|references|metadata|problem)\b|\Z)',
    ]:
        text = re.sub(p, '', text)
    text = re.sub(r'(?im)^\s*[A-Za-z][A-Za-z\s-]{1,60}:\s*[A-Z]{1,8}(?:[-\s][A-Za-z]{1,8}){1,6}\s*$', '', text)
    text = re.sub(r'\[[A-Z][A-Z\s]{0,30}\]', '', text)
    text = re.sub(r'\(([A-Za-z][A-Za-z\s,\']{0,40})\)', _safe_paren_strip, text)
    for key in ['word count','voice','script id','id','unit',
                'module','chapter','topic','version','date','subject']:
        text = re.sub(rf'(?im)^{key}\s*:.*$', '', text)
    text = re.sub(r'(?m)^\s*\d+\.\s+', '', text)
    text = re.sub(r'(?m)^\s*[•\-\*]\s+', '', text)
    return text.strip()


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — CHEMICAL FORMULAS
# ══════════════════════════════════════════════════════════════════════════════

_CHEM = [
    ('H₂SO₄','sulfuric acid'),('H₂SO4','sulfuric acid'),('H2SO4','sulfuric acid'),
    ('HNO₃','nitric acid'),('HNO3','nitric acid'),
    ('C₆H₁₂O₆','glucose'),('C6H12O6','glucose'),
    ('CaCO₃','calcium carbonate'),('CaCO3','calcium carbonate'),
    ('NaHCO₃','sodium bicarbonate'),('NaHCO3','sodium bicarbonate'),
    ('Ca(OH)₂','calcium hydroxide'),('Ca(OH)2','calcium hydroxide'),
    ('NaOH','sodium hydroxide'),('KOH','potassium hydroxide'),
    ('NaCl','sodium chloride'),('KCl','potassium chloride'),('MgO','magnesium oxide'),
    ('CO₂','C O 2'),('CO2','C O 2'),
    ('H₂O₂','hydrogen peroxide'),('H2O2','hydrogen peroxide'),
    ('H₂O','H 2 O'),('H2O','H 2 O'),
    ('NH₃','N H 3'),('NH3','N H 3'),
    ('CH₄','C H 4'),('CH4','C H 4'),
    ('O₂','O 2'),('O2','O 2'),
    ('N₂','N 2'),('N2','N 2'),
    ('H₂','H 2'),('H2','H 2'),
    ('Cl₂','Cl 2'),('Cl2','Cl 2'),
    ('SO₂','S O 2'),('SO2','S O 2'),
    ('NO₂','N O 2'),('NO2','N O 2'),
]

def convert_chemical_formulas(text: str) -> str:
    for formula, spoken in _CHEM:
        text = text.replace(formula, f' {spoken} ')
    return text


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — TYPOGRAPHY / PUNCTUATION
# ══════════════════════════════════════════════════════════════════════════════

def fix_typography(text: str) -> str:
    text = re.sub(r'(\d+)\s*[\u2013\u2014]\s*(\d+)', r'\1 to \2', text)
    text = text.replace('\u2018', "'").replace('\u2019', "'")
    text = text.replace('\u201c', '"').replace('\u201d', '"')
    text = text.replace('\u2026', ' and so on')
    return text


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — COORDINATE NOTATION
# ══════════════════════════════════════════════════════════════════════════════

def convert_coordinate_notation(text: str) -> str:
    """A(0, 3) → A at 0 comma 3"""
    def replace_coord(m):
        x = _number_token_to_words(m.group(2).strip())
        y = _number_token_to_words(m.group(3).strip())
        return f"{m.group(1)} at {x} comma {y}"
    return re.sub(r'\b([A-Z])\((-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\)', replace_coord, text)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — INLINE MATH EXPRESSIONS
# ══════════════════════════════════════════════════════════════════════════════

def convert_inline_math(text: str) -> str:
    """
    Convert operators to spoken English. NO SSML inserted at this stage.
    """
    text = text.replace('≠', ' is not equal to ')
    text = text.replace('≈', ' is approximately equal to ')
    text = text.replace('≤', ' is less than or equal to ')
    text = text.replace('≥', ' is greater than or equal to ')

    # Hyphens in compound names and unit modifiers are spoken as spaces.
    # Keep single-variable expressions such as x-4 and 12-x mathematical.
    text = re.sub(r'(?<=[A-Za-z]{2})-(?=\d)|(?<=\d)-(?=[A-Za-z]{2})', ' ', text)

    # Digit-adjacent unspaced minus: 12-4, 12-x, x-4
    text = re.sub(r'(?<=\d)-(?=[A-Za-z0-9])', ' minus ', text)
    text = re.sub(r'(?<=[A-Za-z])-(?=\d)', ' minus ', text)

    # Letter-minus-letter in equation context (x-y=2, x-y+z=10, a-b=c+d)
    def _fix_math_minus(m: re.Match) -> str:
        return re.sub(r'(?<=[A-Za-z0-9])-(?=[A-Za-z0-9])', ' minus ', m.group(0))
    text = re.sub(
        r'\b[A-Za-z0-9]+(?:-[A-Za-z0-9]+)+(?:\s*[=+*/]\s*[A-Za-z0-9][A-Za-z0-9\s+\-*/=.]*)+',
        _fix_math_minus, text,
    )

    text = re.sub(r'\s*\+\s*', ' plus ', text)
    text = re.sub(r'(?<=[\w\d])\s+-\s+(?=[\w\d])', ' minus ', text)
    text = re.sub(r'\s*=\s*', ' equals ', text)
    text = re.sub(r'\bequals\s+equals\b', 'equals', text)
    return text


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6 — UNITS
# ══════════════════════════════════════════════════════════════════════════════

_UNITS = [
    (r'(-?\d+\.?\d*)\s*°\s*F\b', r'\1 degrees Fahrenheit'),
    (r'(-?\d+\.?\d*)\s*°\s*C\b', r'\1 degrees Celsius'),
    (r'(-?\d+\.?\d*)\s*°\s*K\b', r'\1 degrees Kelvin'),
    (r'(-?\d+\.?\d*)\s*°',        r'\1 degrees'),
    (r'(\d+\.?\d*)\s*km\b',       r'\1 kilometers'),
    (r'(\d+\.?\d*)\s*cm\b',       r'\1 centimeters'),
    (r'(\d+\.?\d*)\s*mm\b',       r'\1 millimeters'),
    (r'(\d+\.?\d*)\s*nm\b',       r'\1 nanometers'),
    (r'(\d+\.?\d*)\s*μm\b',       r'\1 micrometers'),
    (r'(\d+\.?\d*)\s*mi\b',       r'\1 miles'),
    (r'(\d+\.?\d*)\s*ft\b',       r'\1 feet'),
    (r'(\d+\.?\d*)\s*yd\b',       r'\1 yards'),
    (r'(\d+\.?\d*)\s*in\b',       r'\1 inches'),
    (r'(\d+\.?\d*(?:[eE][+\-]?\d+)?)\s*km/h\b', r'\1 kilometers per hour'),
    (r'(\d+\.?\d*(?:[eE][+\-]?\d+)?)\s*m/s\b',  r'\1 meters per second'),
    (r'(\d+\.?\d*)\s*mph\b',      r'\1 miles per hour'),
    (r'(\d+\.?\d*)\s*ft/s\b',     r'\1 feet per second'),
    (r'(\d+\.?\d*)\s*m\b',        r'\1 meters'),
    (r'(\d+\.?\d*)\s*km²',        r'\1 square kilometers'),
    (r'(\d+\.?\d*)\s*m²',         r'\1 square meters'),
    (r'(\d+\.?\d*)\s*cm²',        r'\1 square centimeters'),
    (r'(\d+\.?\d*)\s*mm²',        r'\1 square millimeters'),
    (r'(\d+\.?\d*)\s*ft²',        r'\1 square feet'),
    (r'(\d+\.?\d*)\s*in²',        r'\1 square inches'),
    (r'(\d+\.?\d*)\s*km³',        r'\1 cubic kilometers'),
    (r'(\d+\.?\d*)\s*m³',         r'\1 cubic meters'),
    (r'(\d+\.?\d*)\s*cm³',        r'\1 cubic centimeters'),
    (r'(\d+\.?\d*)\s*mm³',        r'\1 cubic millimeters'),
    (r'(\d+\.?\d*)\s*mL\b',       r'\1 milliliters'),
    (r'(\d+\.?\d*)\s*ml\b',       r'\1 milliliters'),
    (r'(\d+\.?\d*)\s*L\b',        r'\1 liters'),
    (r'(\d+\.?\d*)\s*gal\b',      r'\1 gallons'),
    (r'(\d+\.?\d*)\s*qt\b',       r'\1 quarts'),
    (r'(\d+\.?\d*)\s*pt\b',       r'\1 pints'),
    (r'(\d+\.?\d*)\s*fl\s*oz\b',  r'\1 fluid ounces'),
    (r'(\d+\.?\d*)\s*oz\b',       r'\1 ounces'),
    (r'(\d+\.?\d*)\s*kg\b',       r'\1 kilograms'),
    (r'(\d+\.?\d*)\s*mg\b',       r'\1 milligrams'),
    (r'(\d+\.?\d*)\s*μg\b',       r'\1 micrograms'),
    (r'(\d+\.?\d*)\s*lbs?\b',     r'\1 pounds'),
    (r'(\d+\.?\d*)\s*g\b',        r'\1 grams'),
    (r'(\d+\.?\d*)\s*t\b',        r'\1 tonnes'),
    (r'(\d+\.?\d*)\s*ms\b',       r'\1 milliseconds'),
    (r'(\d+\.?\d*)\s*μs\b',       r'\1 microseconds'),
    (r'(\d+\.?\d*)\s*ns\b',       r'\1 nanoseconds'),
    (r'(\d+\.?\d*)\s*sec\b',      r'\1 seconds'),
    (r'(\d+\.?\d*)\s*min\b',      r'\1 minutes'),
    (r'(\d+\.?\d*)\s*hr\b',       r'\1 hours'),
    (r'(\d+\.?\d*)\s*s\b',        r'\1 seconds'),
    (r'(\d+\.?\d*)\s*h\b',        r'\1 hours'),
    (r'(\d+\.?\d*)\s*kWh\b',      r'\1 kilowatt hours'),
    (r'(\d+\.?\d*)\s*kW\b',       r'\1 kilowatts'),
    (r'(\d+\.?\d*)\s*kJ\b',       r'\1 kilojoules'),
    (r'(\d+\.?\d*)\s*kcal\b',     r'\1 kilocalories'),
    (r'(\d+\.?\d*)\s*cal\b',      r'\1 calories'),
    (r'(\d+\.?\d*)\s*W\b',        r'\1 watts'),
    (r'(\d+\.?\d*)\s*J\b',        r'\1 joules'),
    (r'(\d+\.?\d*)\s*kV\b',       r'\1 kilovolts'),
    (r'(\d+\.?\d*)\s*mV\b',       r'\1 millivolts'),
    (r'(\d+\.?\d*)\s*V\b',        r'\1 volts'),
    (r'(\d+\.?\d*)\s*mA\b',       r'\1 milliamperes'),
    (r'(\d+\.?\d*)\s*A\b',        r'\1 amperes'),
    (r'(\d+\.?\d*)\s*MΩ\b',       r'\1 mega-ohms'),
    (r'(\d+\.?\d*)\s*kΩ\b',       r'\1 kilo-ohms'),
    (r'(\d+\.?\d*)\s*Ω\b',        r'\1 ohms'),
    (r'(\d+\.?\d*)\s*μF\b',       r'\1 microfarads'),
    (r'(\d+\.?\d*)\s*nF\b',       r'\1 nanofarads'),
    (r'(\d+\.?\d*)\s*F\b',        r'\1 farads'),
    (r'(\d+\.?\d*)\s*mH\b',       r'\1 millihenries'),
    (r'(\d+\.?\d*)\s*H\b',        r'\1 henries'),
    (r'(\d+\.?\d*)\s*GHz\b',      r'\1 gigahertz'),
    (r'(\d+\.?\d*)\s*MHz\b',      r'\1 megahertz'),
    (r'(\d+\.?\d*)\s*kHz\b',      r'\1 kilohertz'),
    (r'(\d+\.?\d*)\s*Hz\b',       r'\1 hertz'),
    (r'(\d+\.?\d*)\s*kPa\b',      r'\1 kilopascals'),
    (r'(\d+\.?\d*)\s*Pa\b',       r'\1 pascals'),
    (r'(\d+\.?\d*)\s*atm\b',      r'\1 atmospheres'),
    (r'(\d+\.?\d*)\s*mmHg\b',     r'\1 millimeters of mercury'),
    (r'(\d+\.?\d*)\s*kN\b',       r'\1 kilonewtons'),
    (r'(\d+\.?\d*)\s*N\b',        r'\1 newtons'),
    (r'(\d+\.?\d*)\s*%',          r'\1 percent'),
    (r'(\d+\.?\d*)\s*‰',          r'\1 per mille'),
    (r'\$(\d[\d,]*(?:\.\d+)?)',   r'\1 dollars'),
    (r'£(\d[\d,]*(?:\.\d+)?)',    r'\1 pounds sterling'),
    (r'€(\d[\d,]*(?:\.\d+)?)',    r'\1 euros'),
    (r'¥(\d[\d,]*(?:\.\d+)?)',    r'\1 yen'),
    (r'₹(\d[\d,]*(?:\.\d+)?)',    r'\1 rupees'),
]

def convert_units(text: str) -> str:
    for pattern, replacement in _UNITS:
        text = re.sub(pattern, replacement, text)
    text = text.replace('°F', 'degrees Fahrenheit')
    text = text.replace('°C', 'degrees Celsius')
    text = text.replace('°K', 'degrees Kelvin')
    return text


# ══════════════════════════════════════════════════════════════════════════════
# STEP 7 — LATEX
# ══════════════════════════════════════════════════════════════════════════════

def strip_latex_wrappers(text: str) -> str:
    text = re.sub(r'\\\[|\\\]|\$\$', ' ', text)
    text = re.sub(r'\\\(|\\\)', ' ', text)
    text = re.sub(r'(?<!\$)\$(?!\$)', ' ', text)
    text = re.sub(r'\\(text|mathrm|mathbf|mathit|boldsymbol|left|right|big|Big|bigg|Bigg)\s*', '', text)
    text = re.sub(r'\\(displaystyle|textstyle|scriptstyle)\s*', '', text)
    return text

def convert_latex_fractions(text: str) -> str:
    def rep(m): return f"{m.group(1).strip()} over {m.group(2).strip()}"
    for cmd in (r'\\frac', r'\\dfrac', r'\\tfrac'):
        text = re.sub(cmd + r'\{([^}]+)\}\{([^}]+)\}', rep, text)
    return text

def convert_latex_roots(text: str) -> str:
    nth = {'2':'square root of','3':'cube root of','4':'fourth root of'}
    def rep(m): return f"{nth.get(m.group(1), m.group(1)+'th root of')} {m.group(2).strip()}"
    text = re.sub(r'\\sqrt\[(\d+)\]\{([^}]+)\}', rep, text)
    text = re.sub(r'\\sqrt\{([^}]+)\}', r'square root of \1', text)
    return text

def convert_latex_exponents(text: str) -> str:
    pw = {'2':'squared','3':'cubed'}
    def rep(m): return f"{m.group(1)} {pw.get(m.group(2), 'to the power of '+m.group(2))}"
    text = re.sub(r'(\w[\w.]*)\^\{([^}]+)\}', rep, text)
    text = re.sub(r'(\w[\w.]*)\^(\d+)', rep, text)
    return text

def convert_latex_subscripts(text: str) -> str:
    text = re.sub(r'(\w[\w.]*)_\{([^}]+)\}', r'\1 sub \2', text)
    text = re.sub(r'(\w[\w.]*)_(\w+)', r'\1 sub \2', text)
    text = re.sub(r'(\w[\w.]*)\{([^}]+)\}', r'\1 sub \2', text)
    return text

_GREEK = {
    r'\\alpha':'alpha', r'\\beta':'beta', r'\\gamma':'gamma', r'\\delta':'delta',
    r'\\epsilon':'epsilon', r'\\varepsilon':'epsilon', r'\\zeta':'zeta', r'\\eta':'eta',
    r'\\theta':'theta', r'\\vartheta':'theta', r'\\iota':'iota', r'\\kappa':'kappa',
    r'\\lambda':'lambda', r'\\mu':'mu', r'\\nu':'nu', r'\\xi':'xi', r'\\pi':'pi',
    r'\\varpi':'pi', r'\\rho':'rho', r'\\varrho':'rho', r'\\sigma':'sigma',
    r'\\varsigma':'sigma', r'\\tau':'tau', r'\\upsilon':'upsilon', r'\\phi':'phi',
    r'\\varphi':'phi', r'\\chi':'chi', r'\\psi':'psi', r'\\omega':'omega',
    r'\\Gamma':'Gamma', r'\\Delta':'Delta', r'\\Theta':'Theta', r'\\Lambda':'Lambda',
    r'\\Xi':'Xi', r'\\Pi':'Pi', r'\\Sigma':'Sigma', r'\\Upsilon':'Upsilon',
    r'\\Phi':'Phi', r'\\Psi':'Psi', r'\\Omega':'Omega',
}

def convert_latex_greek(text: str) -> str:
    for cmd, spoken in _GREEK.items():
        text = re.sub(cmd + r'\b', f' {spoken}', text)
    return text

_LATEX_OPS = {
    r'\\times':' times ', r'\\cdot':' times ', r'\\div':' divided by ',
    r'\\pm':' plus or minus ', r'\\mp':' minus or plus ',
    r'\\leq':' is less than or equal to ', r'\\geq':' is greater than or equal to ',
    r'\\neq':' is not equal to ', r'\\approx':' is approximately equal to ',
    r'\\equiv':' is equivalent to ', r'\\sim':' is similar to ',
    r'\\propto':' is proportional to ', r'\\infty':' infinity ',
    r'\\partial':' partial derivative of ', r'\\nabla':' del ',
    r'\\forall':' for all ', r'\\exists':' there exists ',
    r'\\in':' is in ', r'\\notin':' is not in ',
    r'\\subset':' is a subset of ', r'\\supset':' is a superset of ',
    r'\\cup':' union ', r'\\cap':' intersection ',
    r'\\int':' integral of ', r'\\sum':' sum of ', r'\\prod':' product of ',
    r'\\lim':' limit of ', r'\\log':' log of ', r'\\ln':' natural log of ',
    r'\\sin':' sine of ', r'\\cos':' cosine of ', r'\\tan':' tangent of ',
    r'\\sec':' secant of ', r'\\csc':' cosecant of ', r'\\cot':' cotangent of ',
    r'\\arcsin':' arc sine of ', r'\\arccos':' arc cosine of ', r'\\arctan':' arc tangent of ',
    r'\\sinh':' hyperbolic sine of ', r'\\cosh':' hyperbolic cosine of ',
    r'\\tanh':' hyperbolic tangent of ',
    r'\\max':' maximum ', r'\\min':' minimum ',
    r'\\gcd':' greatest common divisor of ',
    r'\\ldots':' and so on', r'\\cdots':' and so on',
    r'\\to':' approaches ', r'\\rightarrow':' approaches ',
    r'\\Rightarrow':' implies ', r'\\Leftrightarrow':' if and only if ',
}

def convert_latex_operators(text: str) -> str:
    for cmd, spoken in _LATEX_OPS.items():
        text = re.sub(cmd + r'\b', spoken, text)
    return text


# ══════════════════════════════════════════════════════════════════════════════
# STEP 8 — UNICODE
# ══════════════════════════════════════════════════════════════════════════════

_UNICODE = [
    ('×',' times '), ('÷',' divided by '), ('±',' plus or minus '),
    ('∓',' minus or plus '), ('·',' times '),
    ('≠',' is not equal to '), ('≈',' is approximately equal to '),
    ('≡',' is equivalent to '), ('≤',' is less than or equal to '),
    ('≥',' is greater than or equal to '),
    ('≪',' is much less than '), ('≫',' is much greater than '),
    ('√',' square root of '), ('∛',' cube root of '), ('∜',' fourth root of '),
    ('∫',' integral of '), ('∬',' double integral of '), ('∮',' contour integral of '),
    ('∑',' the sum of '), ('∏',' the product of '),
    ('∂',' partial derivative of '), ('∇',' del '),
    ('∞',' infinity '), ('∝',' is proportional to '),
    ('∆',' delta '), ('Δ',' delta '),
    ('∀',' for all '), ('∃',' there exists '),
    ('∈',' is an element of '), ('∉',' is not an element of '),
    ('∅',' the empty set '), ('⊂',' is a subset of '), ('⊃',' is a superset of '),
    ('⊆',' is a subset of or equal to '), ('⊇',' is a superset of or equal to '),
    ('∪',' union '), ('∩',' intersection '),
    ('α','alpha'),('β','beta'),('γ','gamma'),('δ','delta'),('ε','epsilon'),
    ('ζ','zeta'),('η','eta'),('θ','theta'),('ι','iota'),('κ','kappa'),
    ('λ','lambda'),('μ','mu'),('ν','nu'),('ξ','xi'),('π','pi'),
    ('ρ','rho'),('σ','sigma'),('τ','tau'),('υ','upsilon'),('φ','phi'),
    ('χ','chi'),('ψ','psi'),('ω','omega'),
    ('Γ','Gamma'),('Θ','Theta'),('Λ','Lambda'),('Ξ','Xi'),('Π','Pi'),
    ('Σ','Sigma'),('Υ','Upsilon'),('Φ','Phi'),('Ψ','Psi'),('Ω','Omega'),
    ('⁰',' to the power of 0'),('¹',' to the power of 1'),
    ('²',' squared'),('³',' cubed'),
    ('⁴',' to the power of 4'),('⁵',' to the power of 5'),
    ('⁶',' to the power of 6'),('⁷',' to the power of 7'),
    ('⁸',' to the power of 8'),('⁹',' to the power of 9'),
    ('₀',' sub 0'),('₁',' sub 1'),('₂',' sub 2'),('₃',' sub 3'),
    ('₄',' sub 4'),('₅',' sub 5'),('₆',' sub 6'),('₇',' sub 7'),
    ('₈',' sub 8'),('₉',' sub 9'),
    ('½','one half'),('⅓','one third'),('⅔','two thirds'),
    ('¼','one quarter'),('¾','three quarters'),
    ('⅕','one fifth'),('⅖','two fifths'),('⅗','three fifths'),('⅘','four fifths'),
    ('⅙','one sixth'),('⅚','five sixths'),
    ('⅛','one eighth'),('⅜','three eighths'),('⅝','five eighths'),('⅞','seven eighths'),
    ('→',' approaches '),('←',' from '),('↔',' corresponds to '),
    ('⇒',' implies '),('⇔',' if and only if '),
    ('↑',' increases '),('↓',' decreases '),
    ('△',' triangle '),
    ('°',' degrees'),
    ('′',' prime'),('″',' double prime'),
    ('∠',' angle '),('⊥',' is perpendicular to '),('∥',' is parallel to '),
    ('□',' square '),('○',' circle '),
    ('≅',' is congruent to '),('∼',' is similar to '),
    ('∴',' therefore '),('∵',' because '),
    ('‰',' per mille '),('‱',' per ten thousand '),
]

def convert_unicode_symbols(text: str) -> str:
    for symbol, spoken in _UNICODE:
        text = text.replace(symbol, spoken)
    return text


# ══════════════════════════════════════════════════════════════════════════════
# STEP 9 — ARITHMETIC
# ══════════════════════════════════════════════════════════════════════════════

def convert_arithmetic(text: str) -> str:
    text = re.sub(r'(\d+\.?\d*)[eE]\+?(-?\d+)', r'\1 times 10 to the power of \2', text)
    text = re.sub(r'(\d+\.?\d*)\s*times\s*10\s*\^(-?\d+)', r'\1 times 10 to the power of \2', text)
    text = re.sub(r'(?<!\d)(\d+)\s*:\s*(\d+)(?!\d)', r'\1 to \2', text)
    pw = {'2':'squared','3':'cubed'}
    def caret(m): return f"{m.group(1)} {pw.get(m.group(2), 'to the power of '+m.group(2))}"
    text = re.sub(r'(\w[\w.]*)\^(\d+)', caret, text)
    text = re.sub(r'(?<!\|)\|([^|\n]{1,40})\|(?!\|)', r'the absolute value of \1', text)
    for frac, spoken in [
        ('7/8','seven eighths'),('5/8','five eighths'),('3/8','three eighths'),
        ('1/8','one eighth'),('5/6','five sixths'),('1/6','one sixth'),
        ('4/5','four fifths'),('3/5','three fifths'),('2/5','two fifths'),
        ('1/5','one fifth'),('3/4','three quarters'),('1/4','one quarter'),
        ('2/3','two thirds'),('1/3','one third'),('1/2','one half'),
        ('1/9','one ninth'),('1/7','one seventh'),('1/10','one tenth'),
    ]:
        text = re.sub(r'(?<!\w)' + re.escape(frac) + r'(?!\w)', spoken, text)
    text = re.sub(r'(?<!\w)(\d+)/(\d+)(?!\w)', r'\1 over \2', text)
    text = re.sub(r'(?<=[\w\d])\s*\*\s*(?=[\w\d])', ' times ', text)
    text = re.sub(r'(?<=\s)/(?=\s)', ' divided by ', text)
    named = {
        '1 over 2':'one half','1 over 3':'one third','2 over 3':'two thirds',
        '1 over 4':'one quarter','3 over 4':'three quarters',
        '1 over 5':'one fifth','1 over 6':'one sixth',
        '1 over 8':'one eighth','1 over 10':'one tenth',
    }
    for expr, spoken in named.items():
        text = re.sub(r'(?<!\w)' + re.escape(expr) + r'(?!\w)', spoken, text)
    return text


# ══════════════════════════════════════════════════════════════════════════════
# STEP 10 — NEGATIVE NUMBERS
# ══════════════════════════════════════════════════════════════════════════════

def convert_negative_numbers(text: str) -> str:
    text = re.sub(r'(?m)^(-\d+\.?\d*)', lambda m: f'negative {m.group(1)[1:]}', text)
    text = re.sub(r'(?<=[=+×÷(,\s])(-\d+\.?\d*)', lambda m: f'negative {m.group(1)[1:]}', text)
    return text


# ══════════════════════════════════════════════════════════════════════════════
# STEP 11 — NUMBER FORMATTING
# ══════════════════════════════════════════════════════════════════════════════

def convert_number_formatting(text: str) -> str:
    return re.sub(r'(\d),(\d{3})(?!\d)', r'\1\2', text)



# ══════════════════════════════════════════════════════════════════════════════
# STEP 12 — SPOKEN NUMBER DESCRIPTIONS INSIDE MATH PHRASES
# ══════════════════════════════════════════════════════════════════════════════

_SMALL_NUMBERS = {
    0: 'zero', 1: 'one', 2: 'two', 3: 'three', 4: 'four', 5: 'five',
    6: 'six', 7: 'seven', 8: 'eight', 9: 'nine', 10: 'ten',
    11: 'eleven', 12: 'twelve', 13: 'thirteen', 14: 'fourteen',
    15: 'fifteen', 16: 'sixteen', 17: 'seventeen', 18: 'eighteen',
    19: 'nineteen', 20: 'twenty', 30: 'thirty', 40: 'forty',
    50: 'fifty', 60: 'sixty', 70: 'seventy', 80: 'eighty', 90: 'ninety',
}

def _int_to_words(n: int) -> str:
    if n in _SMALL_NUMBERS:
        return _SMALL_NUMBERS[n]
    if 20 < n < 100:
        tens = (n // 10) * 10
        ones = n % 10
        return _SMALL_NUMBERS[tens] + ('' if ones == 0 else ' ' + _SMALL_NUMBERS[ones])
    if 100 <= n < 1000:
        hundreds = n // 100
        rest = n % 100
        return _SMALL_NUMBERS[hundreds] + ' hundred' + ('' if rest == 0 else ' ' + _int_to_words(rest))
    return str(n)

def _number_token_to_words(token: str) -> str:
    negative = token.startswith('-')
    core = token[1:] if negative else token
    if '.' in core:
        left, right = core.split('.', 1)
        left_words = _int_to_words(int(left)) if left else 'zero'
        right_words = ' '.join(_SMALL_NUMBERS.get(int(d), d) for d in right if d.isdigit())
        spoken = f"{left_words} point {right_words}".strip()
    else:
        spoken = _int_to_words(int(core))
    return ('negative ' + spoken) if negative else spoken

def convert_numbers_in_math_phrases_to_words(text: str) -> str:
    """
    Convert numbers only in already-detected math phrases to spoken words.

    This mirrors the team's manual cleanup style:
      2 plus 2 equals 4  ->  two plus two equals four
      2 plus 2 minus 5   ->  two plus two minus five

    It deliberately avoids ordinary measurements/units unless the same sentence
    contains math operators, so temperature handling like "4.8 degrees Fahrenheit"
    remains intact.
    """
    math_word_pattern = re.compile(
        r'\b(plus|minus|equals|times|divided by|over|squared|cubed|to the power of)\b',
        re.IGNORECASE,
    )

    def process_sentence(m: re.Match) -> str:
        sentence = m.group(0)
        if not math_word_pattern.search(sentence):
            return sentence
        return re.sub(
            r'(?<![A-Za-z0-9.])(-?\d+(?:\.\d+)?)(?![A-Za-z0-9])(?!\.\d)(?!\s*(?:degrees|degree|percent|seconds?|minutes?|hours?|meters?|inches?|feet|kilometers?|centimeters?|millimeters?|grams?|kilograms?|pounds?|liters?|milliliters?))',
            lambda n: _number_token_to_words(n.group(1)),
            sentence,
        )

    return re.sub(r'[^\n]+', process_sentence, text)

# ══════════════════════════════════════════════════════════════════════════════
# STEP 12 — FINAL PUNCTUATION & SPACING CLEANUP
# ══════════════════════════════════════════════════════════════════════════════

def clean_for_tts(text: str) -> str:
    text = re.sub(r'[{}\\]', ' ', text)
    text = re.sub(r'[ \t]{2,}', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Promote single-newline sentence breaks to paragraph breaks so each
    # numbered point remains a natural pacing unit without SSML.
    text = re.sub(r'([.!?])\n(?=[A-Z])', r'\1\n\n', text)
    # Fallback: alphanumeric → capital across a newline (no terminal punctuation)
    # becomes ". " for proper sentence flow.
    text = re.sub(r'([a-zA-Z0-9])\n(?=[A-Z])', r'\1. ', text)
    return text.strip()


# ══════════════════════════════════════════════════════════════════════════════
# STEP 13 — NATURAL EQUATION PACING WITHOUT SSML
# ══════════════════════════════════════════════════════════════════════════════

def apply_natural_equation_pacing(text: str) -> str:
    """
    Plain-text pacing only. No SSML, no added teacher phrases.

    What this does:
    1. Separates coefficients: 2y -> 2 y, so the voice reads "two why".
    2. Replaces colon-before-equation with a comma: "we write: x equals 50"
       -> "we write, x equals 50".
    3. Normalizes spacing around punctuation.

    What this never does:
    - It never inserts <break> tags.
    - It never inserts phrases like "put that into" or "this gives".
    - It never changes voice speed or temperature/unit conversions.
    """
    text = re.sub(r'(?<![A-Za-z0-9])([2-9]|\d{2,})([A-Za-z])\b', r'\1 \2', text)
    text = re.sub(
        r'\b(write|writes|wrote|written|get|gets|got|have|has|obtain|obtains|find|finds|calculate|calculates|solve|solves|simplify|simplifying)\s*:\s+(?=.{0,140}?\b(equals|plus|minus|times|divided|over)\b)',
        lambda m: m.group(1) + ', ',
        text,
        flags=re.IGNORECASE,
    )
    # Defensive cleanup: remove any SSML break tags left by older pasted text.
    text = re.sub(r'\s*<break\s+time=["\'][^"\']+["\']\s*/>\s*', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'\s+([.,;:!?])', r'\1', text)
    text = re.sub(r'[ \t]{2,}', ' ', text)
    return text.strip()


def preserve_point_breaks_without_ssml(text: str) -> str:
    """
    Keep original paragraph boundaries as plain blank lines. This gives the TTS
    model normal text structure without any SSML markup.
    """
    paragraphs = [p.strip() for p in re.split(r'\n\n+', text) if p.strip()]
    return '\n\n'.join(paragraphs)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 14 — MERGE SHORT SENTENCES
# ══════════════════════════════════════════════════════════════════════════════

def merge_short_sentences(text: str, min_words: int = 8) -> str:
    """
    Merge short sentences within a paragraph for smoother pacing.
    Never merges across paragraphs (\n\n) and never merges into math sentences.
    """
    paragraphs = re.split(r'\n\n+', text)
    processed = []
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        # Skip merging explicit math-heavy paragraphs; they are already structured.
        if re.search(r'\b(equals|plus|minus|times|divided by|over)\b', para, re.IGNORECASE):
            processed.append(para)
            continue
        sentences = re.split(r'(?<=[.!?])\s+', para)
        result = []
        i = 0
        while i < len(sentences):
            s = sentences[i]
            if len(s.split()) < min_words and s.endswith('.') and i + 1 < len(sentences):
                next_s = sentences[i + 1]
                next_has_math = bool(re.search(
                    r'\bequals\b|[=+×÷±√]|\b(?:substitute|simplifying|from the)\b',
                    next_s, re.IGNORECASE))
                if next_has_math:
                    result.append(s)
                else:
                    if next_s and next_s[0].isupper():
                        next_s = next_s[0].lower() + next_s[1:]
                    sentences[i + 1] = s[:-1] + ', ' + next_s
            else:
                result.append(s)
            i += 1
        processed.append(' '.join(result))
    return '\n\n'.join(processed)


# ══════════════════════════════════════════════════════════════════════════════
# STEP 15 — CHUNKING AT SENTENCE BOUNDARIES
# ══════════════════════════════════════════════════════════════════════════════

def _split_long_piece(piece: str, max_chars: int) -> list:
    piece = piece.strip()
    parts = []
    while len(piece) > max_chars:
        cut = piece.rfind(' ', 0, max_chars)
        if cut < int(max_chars * 0.6):
            cut = max_chars
        parts.append(piece[:cut].strip())
        piece = piece[cut:].strip()
    if piece:
        parts.append(piece)
    return parts


def chunk_at_sentence_boundaries(text: str, max_chars: int = MAX_CHARS_PER_CHUNK) -> list:
    """
    Smart chunking that splits at paragraph (numbered-point) boundaries only.

    Rules:
    1. Target chunk size: max_chars (default 5000). Soft target, not strict.
    2. NEVER splits mid-sentence or mid-point. Boundaries land between complete
       numbered points/paragraphs where possible.
    3. No SSML markers are added or stripped because the pipeline is plain text.
    4. If a single paragraph somehow exceeds max_chars, falls back to splitting
       at sentence boundaries — last resort only.
    """
    paragraphs = [p.strip() for p in re.split(r'\n\n+', text) if p.strip()]
    if not paragraphs:
        return [text.strip()] if text.strip() else []

    chunks = []
    current_parts = []

    def current_len() -> int:
        """Length of current_parts joined with \n\n separators."""
        if not current_parts:
            return 0
        return sum(len(p) for p in current_parts) + 2 * (len(current_parts) - 1)

    def close_current_chunk():
        """Finalize current_parts into a chunk, stripping boundary breaks."""
        parts = list(current_parts)
        if parts:
            chunks.append('\n\n'.join(parts).strip())
        current_parts.clear()

    for para in paragraphs:
        para_len = len(para)

        # Fallback: a single paragraph larger than max_chars (rare). Close any
        # in-progress chunk, then split this oversized paragraph at sentences.
        if para_len > max_chars:
            close_current_chunk()
            sentences = re.split(r'(?<=[.!?])\s+', para)
            sub_chunk = ''
            for s in sentences:
                s = s.strip()
                if not s:
                    continue
                pieces = _split_long_piece(s, max_chars) if len(s) > max_chars else [s]
                for piece in pieces:
                    candidate = (sub_chunk + ' ' + piece).strip() if sub_chunk else piece
                    if len(candidate) <= max_chars:
                        sub_chunk = candidate
                    else:
                        if sub_chunk:
                            chunks.append(sub_chunk.strip())
                        sub_chunk = piece
            if sub_chunk:
                chunks.append(sub_chunk.strip())
            continue

        # Would adding this paragraph push current chunk over the limit?
        sep_len = 2 if current_parts else 0
        if current_parts and current_len() + sep_len + para_len > max_chars:
            close_current_chunk()
        current_parts.append(para)

    close_current_chunk()
    return chunks if chunks else [text.strip()]


# ══════════════════════════════════════════════════════════════════════════════
# MASTER FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def prepare_script_for_tts(raw_text: str, debug: bool = False) -> tuple:
    """
    Full conservative pre-processing pipeline. Returns (chunks, debug_log).
    """
    log = {}
    def rec(name, t):
        if debug: log[name] = t
        return t

    t = repair_common_text_artifacts(raw_text); t = rec('01_mojibake', t)
    t = strip_metadata_sections(t);             t = rec('02_metadata', t)
    t = convert_chemical_formulas(t);           t = rec('03_chemicals', t)
    t = fix_typography(t);                      t = rec('04_typography', t)
    t = convert_coordinate_notation(t);         t = rec('05_coords', t)
    t = convert_units(t);                       t = rec('06_units', t)

    t = strip_latex_wrappers(t)
    t = convert_latex_fractions(t)
    t = convert_latex_roots(t)
    t = convert_latex_exponents(t)
    t = convert_latex_subscripts(t)
    t = convert_latex_greek(t)
    t = convert_latex_operators(t);             t = rec('07_latex', t)

    t = convert_unicode_symbols(t);             t = rec('08_unicode', t)
    t = convert_inline_math(t);                 t = rec('09_inline_math', t)
    t = convert_arithmetic(t);                  t = rec('10_arithmetic', t)
    t = convert_negative_numbers(t);            t = rec('11_negatives', t)
    t = convert_number_formatting(t);           t = rec('12_numformat', t)
    t = convert_numbers_in_math_phrases_to_words(t); t = rec('13_math_number_words', t)
    t = clean_for_tts(t);                       t = rec('14_cleanup_before_fallback', t)

    # Gemini fallback is opt-in and runs before final plain-text pacing.
    t = rec('15_gemini_disabled', t)

    # Natural equation pacing: punctuation only, no SSML and no added phrases.
    t = apply_natural_equation_pacing(t);       t = rec('16_natural_pacing', t)

    # Natural pacing can separate coefficients like 2y -> 2 y, so run
    # math-number wording once more after that surgical change.
    t = convert_numbers_in_math_phrases_to_words(t); t = rec('17_math_number_words_final', t)

    # Preserve paragraph boundaries without inserting any markup.
    t = preserve_point_breaks_without_ssml(t);  t = rec('18_plain_paragraphs', t)

    # Merge short prose sentences only when they are not equation-heavy.
    t = merge_short_sentences(t);               t = rec('19_merge_short', t)

    chunks = chunk_at_sentence_boundaries(t)
    if debug: log['20_chunks'] = chunks

    logger.info(f"Preprocessed {len(raw_text)} chars -> {len(chunks)} chunk(s)")
    return chunks, log


# ══════════════════════════════════════════════════════════════════════════════
# SELF-TEST
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import logging
    logging.basicConfig(level=logging.WARNING)
    G='\033[92m'; R='\033[91m'; B='\033[1m'; X='\033[0m'

    tests = {
        'No SSML anywhere': {
            'input': 'From the first condition, we write: x + y = 50.',
            'must_contain': ['we write, x plus y equals fifty'],
            'must_not_contain': ['<break', '/>', 'put that', 'this gives'],
        },
        'Manual-style arithmetic speech': {
            'input': 'We write 2 + 2 = 4. Then 2 + 2 - 5.',
            'must_contain': ['two plus two equals four', 'two plus two minus five'],
            'must_not_contain': ['2 + 2', '<break'],
        },
        'Equation variables preserved': {
            'input': 'Substitute x = y + 10 into the equation x + y = 50.',
            'must_contain': ['x equals y plus ten', 'x plus y equals fifty'],
            'must_not_contain': ['put that', 'this gives', '<break'],
        },
        'Temperature units preserved': {
            'input': 'The temperature is 4.8°F, and it drops another 1.4°F. We write 4.8 - 1.4.',
            'must_contain': ['4.8 degrees Fahrenheit', '1.4 degrees Fahrenheit', 'four point eight minus one point four'],
            'must_not_contain': ['Â°F', '°F', '<break'],
        },
        'Mojibake repaired': {
            'input': 'The temperature is 4.8Â°F and the range is 70â€“74. â‰¥ means at least.',
            'must_contain': ['4.8 degrees Fahrenheit', '70 to 74', 'is greater than or equal to'],
            'must_not_contain': ['Â', 'â', '<break'],
        },
        'Triangle and coordinates': {
            'input': '△ABC has vertices at A(0, 3), B(3, 3), and C(3, 0).',
            'must_contain': ['triangle ABC', 'A at zero comma three'],
            'must_not_contain': ['△', '<break'],
        },
        'No unnecessary chunking under 5000 chars': {
            'input': 'Sentence one. Sentence two. ' * 40,
            'custom_check': 'single_chunk',
        },
    }

    def run_test(name, tc):
        chunks, _ = prepare_script_for_tts(tc['input'])
        out = '\n\n'.join(chunks)
        errors = []
        for item in tc.get('must_contain', []):
            if item not in out:
                errors.append(f"missing: {item!r}")
        for item in tc.get('must_not_contain', []):
            if item in out:
                errors.append(f"should not contain: {item!r}")
        if tc.get('custom_check') == 'single_chunk' and len(chunks) != 1:
            errors.append(f"expected 1 chunk, got {len(chunks)}")
        if errors:
            print(f"{R}FAIL{X} {name}")
            for e in errors:
                print('  -', e)
            print('  OUT:', out[:500])
            return False
        print(f"{G}PASS{X} {name}")
        return True

    print(f"{B}Running tts_preprocessor self-tests...{X}")
    passed = sum(run_test(name, tc) for name, tc in tests.items())
    total = len(tests)
    print(f"\n{B}{passed}/{total} tests passed.{X}")
    raise SystemExit(0 if passed == total else 1)



