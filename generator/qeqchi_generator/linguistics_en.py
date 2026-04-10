"""English-specific constants and morphology."""

from typing import Dict, Tuple
import re
from .utils import _s

# -----------------------
# Constants / placeholders
# -----------------------

_PERSONS = ("1sg", "2sg", "3sg", "1pl", "2pl", "3pl")

DEFAULT_PLACEHOLDERS_EN: Dict[str, str] = {
    "WH_EN": "Why",
    "WH_LOC_EN": "Where",
    "WH_TIME_EN": "When",
    "GOAL_REL_EN": "to",
    "EXIST_EN": "There is",
    "EXIST_PL_EN": "There are",
    "EXIST_NEG_EN": "There is no",
    "EXIST_NEG_PL_EN": "There are no",
}

# --- Determiners and possessives (used for adjective placement and article agreement) ---
EN_DETERMINERS = {
    "the", "a", "an",
    "this", "that", "these", "those",
    "my", "your", "our", "their", "his", "her", "its"
}

# --- Pronouns
EN_SUBJ_PRONOUNS = {"1sg":"I","2sg":"you","3sg":"he","1pl":"we","2pl":"you","3pl":"they"}
EN_POSSESSIVE    = {"1sg":"my","2sg":"your","3sg":"his","1pl":"our","2pl":"your","3pl":"their"}

AGENT_BY_EN = {"1sg":"by me", "2sg":"by you", "3sg":"by him", "1pl":"by us", "2pl":"by you", "3pl":"by them"}

# Gendered realizations
EN_3SG_PRONOUN_BY_GENDER = {"m": "he", "f": "she"}
EN_3SG_AGENT_BY_GENDER   = {"m": "by him", "f": "by her"}

# --- Nouns
_EN_IRREG_PLURALS = {
    "deer":"deer","goose":"geese","mouse":"mice","ox":"oxen","sheep":"sheep",
    "tooth":"teeth","foot":"feet",
    "ear of corn":"ears of corn",
    "man":"men","woman":"women","child":"children","person":"people",
    "young man":"young men","lonely person":"lonely people","short person":"short people",
    "brother-in-law":"brothers-in-law","sister-in-law":"sisters-in-law",
    "son-in-law":"sons-in-law","daughter-in-law":"daughters-in-law",
    "wife":"wives","countryman":"countrymen","cactus":"cacti",
}

# --- Verbs
_EN_3SG_OVERRIDE = {"be":"is","have":"has","do":"does","go":"goes"}

# --- BE ---
EN_BE        = {"1sg":"am",  "2sg":"are", "3sg":"is",  "1pl":"are", "2pl":"are", "3pl":"are"}
EN_BE_PST    = {"1sg":"was", "2sg":"were","3sg":"was", "1pl":"were","2pl":"were","3pl":"were"}
EN_BE_NEG    = {"1sg":"am not","2sg":"are not","3sg":"is not","1pl":"are not","2pl":"are not","3pl":"are not"}
EN_BE_NEG_CT = {"1sg":"am not","2sg":"aren’t", "3sg":"isn’t", "1pl":"aren’t","2pl":"aren’t","3pl":"aren’t"}
EN_BE_FUT    = {p: "will be" for p in _PERSONS}

# --- HAVE ---
EN_HAVE      = {"1sg":"have","2sg":"have","3sg":"has","1pl":"have","2pl":"have","3pl":"have"}
EN_HAVE_PST  = {p: "had" for p in _PERSONS}
EN_HAVE_FUT  = {p: "will have" for p in _PERSONS}
# NOTE: possession-progressive is generally unidiomatic; keep map for completeness but DON'T use for V_POSS.
EN_HAVE_PROG = {p: f"{EN_BE[p]} having" for p in _PERSONS}
EN_HAVE_NEG  = {"1sg":"haven’t","2sg":"haven’t","3sg":"hasn’t","1pl":"haven’t","2pl":"haven’t","3pl":"haven’t"}

# --- DO (auxiliary) ---
EN_DO_AFF    = {"1sg":"do","2sg":"do","3sg":"does","1pl":"do","2pl":"do","3pl":"do"}
EN_DO_NEG    = {"1sg":"don’t","2sg":"don’t","3sg":"doesn’t","1pl":"don’t","2pl":"don’t","3pl":"don’t"}
EN_DO_PST    = {p: "did" for p in _PERSONS}
EN_DO_NEG_PST= {p: "didn’t" for p in _PERSONS}

# --- WILL
EN_WILL    = {p: "will" for p in _PERSONS}
EN_WILL_NEG = {p: "will not" for p in _PERSONS}

# -----------------------
# Tiny helpers 
# -----------------------

_VOWELS = "aeiou"

def _is_cvc(low: str) -> bool:
    """Consonant–vowel–consonant word-final pattern (no w/x/y as final)."""
    return bool(re.search(r"^[^aeiou]*[aeiou][^aeiouwxy]$", low))

def _past_regular(w: str, low: str) -> Tuple[str, str]:
    """Regular ED formation shared by past & participle."""
    if low.endswith("e"):        past = w + "d"
    elif re.search(r"[^aeiou]y$", low):
                                 past = w[:-1] + "ied"
    elif re.search(r"[aeiou]c$", low):
                                 past = w + "ked"
    elif _is_cvc(low):           past = w + w[-1] + "ed"
    else:                        past = w + "ed"
    return past, past

def _ing_regular(w: str, low: str) -> str:
    """Regular ING formation."""
    if low.endswith("ie"):                return w[:-2] + "ying"
    if re.search(r"(?<![aeoy])e$", low):  return w[:-1] + "ing"
    if re.search(r"[aeiou]c$", low):      return w + "king"
    if _is_cvc(low):                       return w + w[-1] + "indg".replace("indg","ing")  # safe single pass
    return w + "ing"

def _indef_article(base: str) -> str:
    """Lightweight 'a/an' without handling all edge cases like 'hour'/'university'."""
    return "an" if (base[:1].lower() in _VOWELS) else "a"

def _en_do(person: str, *, negative: bool = False) -> str:
    """Shared helper for do/does and don't/doesn't."""
    is_3sg = (person or "3sg").strip().lower() == "3sg"
    if negative:
        return "doesn't" if is_3sg else "don't"
    return "does" if is_3sg else "do"

# -----------------------
# Verb morphology
# -----------------------

# Auxiliaries

def en_do_aff(person: str = "3sg") -> str:
    return _en_do(person, negative=False)

def en_do_neg(person: str = "3sg") -> str:
    return _en_do(person, negative=True)

# Verbs

def en_present_3sg(base: str) -> str:
    b = (base or "").strip()
    if not b: return b
    low = b.lower()
    if low in _EN_3SG_OVERRIDE: return _EN_3SG_OVERRIDE[low]
    if re.search(r"(s|sh|ch|x|z|o)$", b): return b + "es"
    if re.search(r"[^aeiou]y$", b):      return b[:-1] + "ies"
    return b + "s"

def en_ing(base: str) -> str:
    w = (base or "").strip()
    if not w: return w
    return _ing_regular(w, w.lower())

def en_past_pp(base: str) -> tuple[str, str]:
    w = (base or "").strip()
    if not w: return "", ""
    return _past_regular(w, w.lower())

def en_imperative(base_or_inf: str, person: str = "2sg") -> str:
    w = (base_or_inf or "").strip()
    if not w: return w
    if (person or "2sg").strip().lower() == "1pl":
        return f"let's {w}"
    return w

def en_directive_forms_en(base_or_inf: str, imp_person: str | None) -> tuple[str, str]:
    """
    Build English directive forms (most pseudo-imperatives) as a pair:
        (affirmative, negative),
    driven by imp_person (1sg/1pl/2sg/2pl/3sg/3pl, etc.).

    Mapping:
      - 2sg / 2pl / default → 'eat', 'don't eat'
      - 1pl                → 'let's eat', 'let's not eat'
      - 1sg, 3sg, 3pl      → 'should eat', 'shouldn't eat'
    """
    w = (base_or_inf or "").strip()
    if not w:
        return "", ""

    p = (imp_person or "2sg").strip().lower()

    if p in {"1sg", "3sg", "3pl"}:
        # 'I should eat', 'He should eat', 'They should eat'
        return f"should {w}", f"shouldn't {w}"

    if p == "1pl":
        # 'Let's eat', 'Let's not eat'
        aff = en_imperative(w, "1pl")  # "let's w"
        neg = f"let's not {w}"
        return aff, neg

    if p == "2pl":
        # English doesn't morphologically distinguish 2sg vs 2pl imperatives.
        # Plurality can be expressed by a pronoun in the template if desired.
        return en_imperative(w, "2sg"), f"don't {w}"

    # Default: 2nd-person singular imperative
    return en_imperative(w, "2sg"), f"don't {w}"

# -----------------------
# Pronouns
# -----------------------

def en_subj_pronoun(person: str = "3sg", gender: str | None = None, rng=None) -> str:
    """
    English subject pronoun with optional gender control.
    - For 3sg, gender can be "m" or "f". If missing, and rng is provided, gender is randomized.
    """
    p = (person or "3sg").strip().lower()

    if p == "3sg":
        g = (gender or "").strip().lower() or None
        if g not in {"m", "f"} and rng is not None:
            g = rng.choice(["m", "f"])
        if g in {"m", "f"}:
            return EN_3SG_PRONOUN_BY_GENDER[g]
        return EN_SUBJ_PRONOUNS["3sg"]

    return EN_SUBJ_PRONOUNS.get(p, EN_SUBJ_PRONOUNS["3sg"])


def agent_by_en(person: str = "3sg", gender: str | None = None, rng=None) -> str:
    """
    Return the English agent phrase for passive.
    - For 3sg, gender can be "m"/"f" (by him/by her). If missing, and rng is provided, gender is randomized.
    """
    p = (person or "3sg").strip().lower()

    if p == "3sg":
        g = (gender or "").strip().lower() or None
        if g not in {"m", "f"} and rng is not None:
            g = rng.choice(["m", "f"])
        if g in {"m", "f"}:
            return EN_3SG_AGENT_BY_GENDER[g]

    return AGENT_BY_EN.get(p, AGENT_BY_EN["3sg"])

# -----------------------
# Nouns
# -----------------------

def pluralize_en(w: str) -> str:
    w = (w or "").strip()
    if not w: return w
    lw = w.lower()
    if lw in _EN_IRREG_PLURALS: return _EN_IRREG_PLURALS[lw]
    if re.search(r"(s|sh|ch|x|z)$", lw): return w + "es"
    if re.search(r"[^aeiou]y$", lw):     return w[:-1] + "ies"
    return w + "s"

def build_np_en(
    noun: Dict,
    *,
    plural: bool = False,
    definite: bool = True,
    possessed_person: str | None = None,
    as_subject: bool = False,   # accepted for signature parity; EN has no subject-article rule
    ref_gender: str | None = None,  # optional gender hint ('f' → use gloss_en_f when available)
) -> str:
    """
    Build an English NP. Handles:
      - possession via EN_POSSESSIVE
      - pluralization for countable nouns
      - definite ('the') vs. indefinite (a/an) vs. bare (uncountable/plural-only)
    Note: `as_subject` is ignored for English; included only to keep a uniform call shape
    across linguistics_* modules.

    If `ref_gender == "f"` and the noun row provides a `gloss_en_f`, that feminine form
    is used instead of `gloss_en`. This is mainly for gendered occupations like
    actor/actress, waiter/waitress, etc.
    """
    # Use _s(...) so NaN / floats from pandas don't break .strip()
    base_m = _s(noun.get("gloss_en")).strip()
    base_f = _s(noun.get("gloss_en_f")).strip()
    if not base_m and not base_f:
        return ""

    use_f = (ref_gender or "").lower() == "f"
    base = base_f if (use_f and base_f) else base_m
    if not base:
        # Fallback if we had only gloss_en_f but no gloss_en and ref_gender wasn't 'f'
        base = base_f

    status = _s(noun.get("countability_en")).strip().lower()

    if possessed_person:
        poss = EN_POSSESSIVE.get(possessed_person, "my")
        form = pluralize_en(base) if (plural and status == "countable") else base
        return f"{poss} {form}"

    if plural and status == "countable":
        base = pluralize_en(base)

    if definite:
        return f"the {base}".strip()

    # For indefinite plurals, use bare plural (no a/an)
    if plural:
        return base

    # Bare singular for uncountable and plural-only nouns
    if status in {"uncountable", "plural-only"}:
        return base

    # Indefinite singular count noun: a/an X
    return f"{_indef_article(base)} {base}"

def build_num_np_en(noun: Dict, numeral_row: Dict) -> str:
    num  = (numeral_row.get("gloss_en") or "").strip()
    base = (noun.get("gloss_en") or "").strip()
    if not num or not base:
        return f"{num} {base}".strip()

    is_one = num.lower() in {"1", "one"}
    if is_one:
        return f"{num} {base}"

    status = (noun.get("countability_en", "") or "").strip().lower()
    if status == "uncountable":
        return f"some {base}"
    if status == "plural-only":
        # e.g. "two scissors", not "two scissorses"
        return f"{num} {base}"

    return f"{num} {pluralize_en(base)}"

# -----------------------
# Adjectives
# -----------------------

def _adj_er_form_en(base: str) -> str:
    """
    Build a regular '-er' comparative form for a simple adjective.

    Very lightweight heuristics:
      - happy -> happier (y -> ier, except for 'ay/ey/oy')
      - big   -> bigger (CVC doubling for short forms)
      - else  -> base + 'er'
    """
    low = base.lower()

    # y -> ier (but avoid 'day' -> 'daiier'; only non-ay/ey/oy)
    if low.endswith("y") and not low.endswith(("ay", "ey", "oy")):
        return base[:-1] + "ier"

    # rough CVC pattern for short adjectives: big, sad, hot, etc.
    # (non-vowel, vowel, consonant)
    if re.match(r"^[^aeiou]*[aeiou][bcdfghjklmnpqrstvwxyz]$", low):
        return base + base[-1] + "er"

    # default: tall -> taller
    return base + "er"


def build_adj_comparative_en(adj_row: dict) -> str:
    """
    Return the English comparative form for an adjective row.

    Expects (optionally) a 'comp_en' column in the adjectives CSV:
      - 'er'   -> force '-er' comparative (including spelling rules)
      - 'more' -> force periphrastic 'more ADJ'
      - any other non-empty value -> treated as the full comparative surface
        (e.g. 'better', 'worse', 'farther', 'further').

    If 'comp_en' is empty or missing, a simple length-based heuristic is used:
      - short, single-word adjectives -> '-er'
      - others -> 'more ADJ'
    """
    base = _s(adj_row.get("gloss_en")).strip().lower()
    if not base:
        return ""

    comp_flag = _s(adj_row.get("comp_en")).strip().lower()

    # Explicit full form (irregulars etc.), e.g. 'better', 'worse'
    if comp_flag and comp_flag not in {"er", "more"}:
        return comp_flag

    # Force periphrastic 'more'
    if comp_flag == "more":
        return f"more {base}"

    # Force synthetic '-er'
    if comp_flag == "er":
        return _adj_er_form_en(base)

    # Default heuristic if column is empty:
    #   - short simple adjectives: '-er'
    #   - otherwise: 'more ADJ'
    if len(base) <= 5 and " " not in base:
        return _adj_er_form_en(base)

    return f"more {base}"

# -----------------------
# Verb renderer
# -----------------------

def render_verb_bundle_en(vrow: dict, key_prefix: str, *, person: str="3sg", imp_person: str | None = None) -> Dict[str,str]:
    if not vrow:
        return {}
    base_en = (vrow.get("gloss_en") or "").strip()

    # Prefer CSV-provided irregulars; fall back to rule-based forms
    irr_past = _s(vrow.get("gloss_en_past")).strip()
    irr_pp   = _s(vrow.get("gloss_en_pp")).strip()
    rule_past, rule_pp = en_past_pp(base_en)
    past_en = irr_past or rule_past
    pp_en   = irr_pp   or rule_pp

    out = {}
    out[f"{key_prefix}_EN"]       = (en_present_3sg(base_en) if person == "3sg" else base_en)
    out[f"{key_prefix}_BASE_EN"]  = base_en
    out[f"{key_prefix}_PST_EN"]   = past_en
    out[f"{key_prefix}_PP_EN"]    = pp_en
    out[f"{key_prefix}_PROG_EN"]  = en_ing(base_en)
    out[f"{key_prefix}_IMP_EN"]  = en_imperative(base_en, imp_person or "2sg")

    # Auxiliaries (for use in templates: {BE_EN} {V_*_PROG_EN}, {HAVE_EN} {V_*_PP_EN}, etc.)
    out["BE_EN"]        = EN_BE.get(person, "is")
    out["BE_PST_EN"]    = EN_BE_PST.get(person, "was")
    out["BE_FUT_EN"]    = EN_BE_FUT.get(person, "will be")
    out["BE_NEG_EN"]    = EN_BE_NEG.get(person, "is not")
    out["BE_NEG_CT_EN"] = EN_BE_NEG_CT.get(person, EN_BE_NEG.get(person, "is not"))

    out["HAVE_EN"]      = EN_HAVE.get(person, "have")
    out["HAVE_PST_EN"]  = EN_HAVE_PST.get(person, "had")
    out["HAVE_FUT_EN"]  = EN_HAVE_FUT.get(person, "will have")
    out["HAVE_NEG_EN"]  = EN_HAVE_NEG.get(person, "haven’t")

    out["DO_AFF_EN"]     = EN_DO_AFF.get(person, "do")
    out["DO_NEG_EN"]     = EN_DO_NEG.get(person, "don’t")
    out["DO_PST_EN"]     = EN_DO_PST.get(person, "did")
    out["DO_NEG_PST_EN"] = EN_DO_NEG_PST.get(person, "didn’t")
    out["WILL_EN"]     = EN_WILL.get(person, "will")
    out["WILL_NEG_EN"] = EN_WILL_NEG.get(person, "won’t")
    
    return out

# Adjective placement
def embed_adj_into_np_in_template_en(tmpl: str, repl: Dict[str, str]) -> tuple[str, Dict[str, str]]:
    """
    Rewrites '{ADJ_EN} {NP_*_EN}' → '{NP_*_EN__WITH_ADJ_*}', where the value places
    the adjective *after* the determiner: e.g. 'the hard axe'.
    """
    if not tmpl or "ADJ_EN" not in tmpl:
        return tmpl, repl

    np_pat = r"\{(?P<np>(?:POSS_)?(?:AGENT_)?(?:THEME_)?(?:GOAL_)?NP(?:_[A-Z_]+)?_EN)\}"
    rx = re.compile(rf"\{{ADJ_EN\}}\s+{np_pat}")

    idx = 0
    out = tmpl
    for m in list(rx.finditer(tmpl)):
        np_key = m.group("np")
        adj = (repl.get("ADJ_EN") or "").strip()
        np_val = (repl.get(np_key) or "").strip()
        if not adj or not np_val:
            continue

        def _place_en(np_phrase: str, adj_word: str) -> str:
            parts = np_phrase.split()
            if not parts:
                return np_phrase
            if parts[0].lower() in EN_DETERMINERS:
                # determiner first → determiner + adjective + rest
                return " ".join([parts[0], adj_word] + parts[1:])
            return f"{adj_word} {np_phrase}"

        new_val = _place_en(np_val, adj)
        synth_key = f"{np_key}__WITH_ADJ_{idx}"
        idx += 1
        repl[synth_key] = new_val
        out = out.replace(m.group(0), "{" + synth_key + "}", 1)

    return out, repl

def clean_english_surface(text: str) -> str:
    """
    Post-process English surface forms after template assembly.

    Conservative cleanup only:
    - normalize whitespace
    - remove obvious doubled determiners/auxiliaries
    - remove stray unresolved placeholders
    """
    if not text:
        return text

    t = text.strip()

    # Normalize whitespace
    t = re.sub(r"\s+", " ", t)

    # Remove stray placeholders defensively
    t = re.sub(r"\{[^}]+\}", "", t)

    # Collapse obvious doubled determiners
    t = re.sub(r"\b(the|a|an|this|that|these|those) \1\b", r"\1", t, flags=re.IGNORECASE)

    # Collapse obvious doubled auxiliaries
    t = re.sub(r"\b(is|are|was|were|do|does|did|will|have|has|had) \1\b", r"\1", t, flags=re.IGNORECASE)

    # Normalize whitespace again after deletions
    t = re.sub(r"\s+", " ", t).strip()

    return t
