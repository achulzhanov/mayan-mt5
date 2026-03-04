"""Q’eqchi’-specific constants and morphology."""

from typing import Dict, Tuple, Optional
import re
from .utils import _is_vowel_initial

_APOS_CHARS = ("'", "’")

DEFAULT_PLACEHOLDERS: Dict[str, str] = {
    "NEG_PRE": "Moko",
    "NEG_SUF": "ta",
    "YN_Q_MARKER": "Ma",
    "WH": "K'a'ut",
    "WH_LOC": "B'ar",
    "WH_TIME": "Jo’q'e",
    "EXIST": "Wan",
    "EXIST_PL": "Wankeb'",
    "COMP": "chi ru",
    "BENEF_REL": "",
    "GOAL_REL": "",
}

# Pronouns
SUBJ_PRONOUNS = {"1sg":"laa'in","2sg":"laa'at","3sg":"a'an","1pl":"laa'o","2pl":"laa'ex","3pl":"heb'a'an"}
AFFIX_PRONOUNS = {"1sg":"in","2sg":"at","3sg":"","1pl":"o","2pl":"ex","3pl":"eeb'"}
REFL_PRONOUNS = {"1sg":"wib’","2sg":"aawib’","3sg":"rib’", "1pl":"qib’","2pl":"eerib’","3pl":"rib’eb"}

# Nouns
_NONPOSSESSED_EXCEPTIONS = {
    "na'": "na'bej",
    "tat": "tatej",          # or "tatejb'ej" depending on dialect
    "chi'": "chi'bej",
    "q'ab'": "q'ab'",        # zero-suffix case
    "wi'": "wi'b'ej",
    "kok": "kokb'ej",
    "xkool": "xkoolb'ej",
    "b'och": "b'och'b'ej",
    "b'aq": "b'aqb'ej",
    "tzo'm": "tzo'mb'ej",
    "xpe'": "xpe'b'ej",
}

# Verb affixes
# Conjugation by person
SET_A_PRECON = {"1sg":"in","2sg":"aa","3sg":"x","1pl":"qa","2pl":"ee","3pl_prefix":"e'x","3pl_suffix":"eb’"}
SET_A_PREVOW = {"1sg":"w","2sg":"aa","3sg":"r","1pl":"qa","2pl":"eer","3pl_prefix":"e'r","3pl_suffix":"eb’"}
SET_B = {"1sg":"in","2sg":"at","3sg":"","1pl":"oo","2pl":"ex","3pl":"e'"}
# Tense marker: present tense
INTR_PRES_PREFIX = {"1sg": "nakin","2sg": "nakat","3sg": "na","1pl": "nako","2pl": "nakex","3pl": "nake'"}
# Tense markers: future tense
SET_A_PRECON_FUT = {"1sg":"tin","2sg":"taa","3sg":"tix","1pl":"taqa","2pl":"tee","3pl":"te'x"}
SET_A_PREVOW_FUT = {"1sg":"tinw","2sg":"taaw","3sg":"tar","1pl":"taaqa","2pl":"teer","3pl":"te'r"}
SET_B_FUT = {"1sg":"tin","2sg":"tat","3sg":"taa","1pl":"too","2pl":"tex","3pl":"te'"}
# Tense marker: perfective (comparable to past)
KEK_TAM_PFV_DEFAULT  = "x"  # Can be "x" or "x-" (hyphen is stylistic)
# Aspect markers: imperative
POS_INTR_IMP_PREFIX_PREVOW = {"1sg": "chin", "2sg": "chat", "3sg": "ch", "1pl": "cho’", "2pl": "chee’", "3pl": "che’"}
POS_INTR_IMP_PREFIX_PRECON = {"1sg": "chin", "2sg": "chat", "3sg": "chi", "1pl": "cho", "2pl": "chee","3pl": "che’"}
POS_TR_IMP_PREFIX_PREVOW = {"1sg": "chiw", "2sg": "chaaw", "3sg": "chir", "1pl": "chiq", "2pl": "cheer", "3pl": "che’r"}
POS_TR_IMP_PREFIX_PRECON = {"1sg": "chin", "2sg": "chaa", "3sg": "chix", "1pl": "chiqa", "2pl": "cheeb’", "3pl": "che’x"}
# Aspect markers: negative imperative
NEG_INTR_IMP_PREFIX = {"1sg": "min", "2sg": "mat", "3sg": "ma", "1pl": "mako’", "2pl": "mex", "3pl": "make’"}
NEG_TR_IMP_PREFIX_PRECON = {"1sg": "min", "2sg": "maa", "3sg": "mix", "1pl": "miqa", "2pl": "mee", "3pl": "make’x"}
NEG_TR_IMP_PREFIX_PREVOW = {"1sg": "miw", "2sg": "maaw", "3sg": "mir", "1pl": "miq", "2pl": "meer", "3pl": "me’r"}

# Statives in future tense (Stewart 112)
# For statives which are adjectives / transitive participles / intransitive participles:
# only one future suffix, always realized as -aq.
STATIVE_FUT_AQ = "aq"
# For stative participles / positionals:
# -q  (indicative)
# -kaq (future optative)
# -qaq (future doubt; must occur with a doubt particle such as maare or ta na)
STATIVE_FUT_IND = "aq"
STATIVE_FUT_OPT = "kaq"
STATIVE_FUT_DBT = "qaq"

STATIVE_DOUBT_PARTICLES = ["maare", "ta na"]

# Auxiliary: for progressive
AUX_PROG = {"1sg": "yookin","2sg": "yookat","3sg": "yoo","1pl": "yooko","2pl": "yookex","3pl": "yookeb’"}
AUX_PROG_FUT = {"1sg": "yooqin","2sg": "yooqat","3sg": "yoo","1pl": "yooqo","2pl": "yooqex","3pl": "yooqeb’"}

# Pronoun constructions
# Pronoun with "with" (used in have" constructions, as in "own, possess")
WITH_PRON = {"1sg":"wik'in","2sg":"aawik'in","3sg":"rik'in","1pl":"qik'in","2pl":"eerik'in","3pl":"rik'ineb'"}
# Agentive relational noun b'aan “by X” (used in passive agent phrases)
AGENT_BY = {"1sg":"inb'aan", "2sg": "aa'baan","3sg":"x'baan", "1pl":"qa'baan", "2pl":"eeb'aan", "3pl":"xb'aaneb"}

def _join_kek(root: str, suffix: str) -> str:
    r = (root or ""); s = (suffix or "")
    if r and s and r[-1] in _APOS_CHARS and s.startswith("'"):
        s = s[1:]
    return r + s

def choose_set_a_allomorph(lemma_kek: str, is_vowel_initial_override: Optional[bool] = None) -> Dict[str, str]:
    is_vowel = is_vowel_initial_override if is_vowel_initial_override is not None else _is_vowel_initial(lemma_kek)
    return SET_A_PREVOW if is_vowel else SET_A_PRECON

# Construction of names: Adding laj and li x for male and female names
def kek_name_surface(name_dict: dict) -> str:
    """
    KEK proper-name surface rule used inside sentences:
      - male:   'laj ' + Name      → 'laj Mateo'
      - female: 'li x' + Name      → 'li xRosa'
    """
    txt = (name_dict or {}).get("text", "") or ""
    gen = (name_dict or {}).get("gender", "")
    if gen == "m":
        return f"laj {txt}".strip()
    if gen == "f":
        return f"li x{txt}".strip()
    return txt

def _np_nonposs_kek(lemma_kek: str, noun_class: str | None = None) -> str:
    lemma_kek = (lemma_kek or "").strip()
    if not lemma_kek:
        return lemma_kek

    # Irregular non-possessed forms
    if lemma_kek in _NONPOSSESSED_EXCEPTIONS:
        return _NONPOSSESSED_EXCEPTIONS[lemma_kek]

    allow_suffix = (noun_class or "").strip() in {"kinship", "body_part"}
    if not allow_suffix:
        return lemma_kek

    suffix = "b'ej" if _is_vowel_initial(lemma_kek) else "'ej"
    return _join_kek(lemma_kek, suffix)

def _np_possessed_kek(
    lemma_kek: str,
    person: str,
    use_prevowel: bool,
    *,
    definite: bool = False,
) -> str:
    """
    Build the possessed Q'eqchi' NP core (Set A prefix + noun),
    optionally adding the definite article 'li' in front.

    For now we keep 'li' separate orthographically:
      in + punit   → inpunit
      li + inpunit → li inpunit

    More stylistic fusions like 'linpunit', 'lixna'', 'laawu' are handled
    later in style_kek_possessives.
    """
    lemma_kek = (lemma_kek or "").strip()
    if not lemma_kek:
        return lemma_kek

    set_a = SET_A_PREVOW if use_prevowel else SET_A_PRECON
    pref = set_a.get(person, "")
    base = f"{pref}{lemma_kek}"

    if not definite:
        return base

    # Definite possessed NP: article 'li' + possessed form
    return f"li {base}"

def build_possessed_np(noun: Dict, person: str="1sg") -> Tuple[str, str]:
    from .linguistics_en import EN_POSSESSIVE
    kek_poss = _np_possessed_kek(
        noun["lemma_kek"],
        person,
        use_prevowel=_is_vowel_initial(noun["lemma_kek"]),
    )
    en_poss  = f"{EN_POSSESSIVE.get(person,'my')} {noun['gloss_en']}"
    return kek_poss, en_poss

def style_kek_possessives(sentence: str) -> str:
    """
    Post-process a Q'eqchi' sentence to implement ALMG-style possessive
    morphology with the definite article 'li'.

    Targets patterns where a definite article is followed by a possessed NP:
        li inC…   → linC…      (1sg, consonant-initial noun)
        li aaC…   → laaC…      (2sg, consonant-initial noun)
        li eeC…   → leeC…      (2pl, consonant-initial noun)
        li eerV…  → lee V…     (2pl, vowel-initial noun)
        li aaV…   → laa V…     (2sg, vowel-initial noun)

    Vowel-initial nouns keep the noun separate (no *liwu'uj, *laawaq'):
        li aautz   → laa utz
        li eerutz  → lee utz

    3sg (x-), 1pl (qa-), and 3pl (x-…eb') are left untouched.
    """
    sentence = (sentence or "").strip()
    if not sentence:
        return sentence

    tokens = sentence.split()
    out: list[str] = []
    i = 0

    while i < len(tokens):
        tok = tokens[i]

        if tok == "li" and i + 1 < len(tokens):
            cand = tokens[i + 1]
            handled = False

            # 1sg: in- + C-root  → linC…
            if cand.startswith("in") and len(cand) > 2:
                lemma = cand[2:]
                if not _is_vowel_initial(lemma):
                    out.append("l" + cand)   # li + inC… → linC…
                    i += 2
                    handled = True

            if handled:
                continue

            # 2sg: aa- + root
            if cand.startswith("aa") and len(cand) > 2:
                lemma = cand[2:]
                if _is_vowel_initial(lemma):
                    # Vowel-initial noun: 'li aautz' → 'laa utz'
                    out.append("laa")
                    out.append(lemma)
                else:
                    # Consonant-initial noun: 'li aaxaab’' → 'laaxaab’'
                    out.append("l" + cand)
                i += 2
                handled = True

            if handled:
                continue

            # 2pl: ee-/eer- + root
            if cand.startswith("eer") and len(cand) > 3:
                lemma = cand[3:]
                if _is_vowel_initial(lemma):
                    # Vowel-initial: 'li eerutz' → 'lee utz'
                    out.append("lee")
                    out.append(lemma)
                else:
                    # (Theoretically odd, but handle generically.)
                    out.append("l" + cand)
                i += 2
                handled = True

            elif cand.startswith("ee") and len(cand) > 2:
                lemma = cand[2:]
                if _is_vowel_initial(lemma):
                    # Vowel-initial: 'li eetz' → 'lee etz'
                    out.append("lee")
                    out.append(lemma)
                else:
                    # Consonant-initial: 'li eexaab’' → 'leexaab’'
                    out.append("l" + cand)
                i += 2
                handled = True

            if handled:
                continue

        # Default: keep token as is
        out.append(tok)
        i += 1

    return " ".join(out)

def pluralize_kek_poss_np(kek_poss_np: str, style: str="pre") -> str:
    return f"{kek_poss_np}eb'" if style == "post" else f"eb' {kek_poss_np}"

def pluralize_kek_np_bare_or_def(noun: Dict, definite: bool=True) -> str:
    k = _np_nonposs_kek(noun["lemma_kek"], noun.get("class"))
    return f"eb' {k}" if not k.startswith("eb' ") else k

def _dedupe_eb(text: str) -> str:
    return re.sub(r"(?:^|\s)eb'\s+eb'(\s|$)", r" eb'\1", text).strip()

def build_np_kek(
    noun: Dict,
    *,
    plural: bool = False,
    definite: bool = True,
    possessed_person: str | None = None,
    poss_plural_style: str = "auto",
    as_subject: bool = False,
) -> str:
    """
    Build a Q'eqchi' NP surface form.
    - Handles possessive morphology, plural markers, definiteness.
    - Injects subject article 'li' for singular, non-possessed subject NPs.
    - For possessed NPs, definiteness now also controls a preposed 'li':
        definite   → 'li inpunit'  ('my hat')
        indefinite → 'inpunit'
    """
    lemma = (noun.get("lemma_kek") or "").strip()
    uncount_kek = (noun.get("countability_kek") or "").strip().lower() == "uncountable"

    # Possessed path
    if possessed_person:
        kek_poss = _np_possessed_kek(
            lemma,
            possessed_person,
            use_prevowel=_is_vowel_initial(lemma),
            definite=definite,
        )
        if plural and not uncount_kek:
            if poss_plural_style == "auto":
                import random
                poss_plural_style = random.choice(["pre", "post"])
            form = _dedupe_eb(pluralize_kek_poss_np(kek_poss, poss_plural_style))
        else:
            form = _dedupe_eb(kek_poss)
        # Note: 'li' is now added *inside* _np_possessed_kek when definite=True.
        return form

    # Non-possessed path
    if plural and not uncount_kek:
        form = _dedupe_eb(pluralize_kek_np_bare_or_def(noun, definite=definite))
    else:
        form = _np_nonposs_kek(lemma, noun.get("class"))

    # Subject article rule (only for singular, non-possessed subjects)
    if as_subject and (not plural) and (not form.lower().startswith("li ")):
        form = f"li {form}".strip()

    return form


def build_pred_np_kek(noun: Dict, *, person: str = "3sg", plural: bool = False) -> str:
    """
    Predicate-NP for stative identity clauses: bare noun + affixed pronoun.

    Example:
      winq + 1sg(in) -> "winq-in"

    Constraints per your requirement:
      - NO 'li' (never a subject NP here)
      - NO agentive name wrappers
      - Just the noun stem (non-possessed form rules still apply where relevant)
    """
    p = (person or "3sg").strip().lower()
    aff = AFFIX_PRONOUNS.get(p, "")

    # Build a bare noun core (no subject article, no definiteness machinery).
    lemma = (noun.get("lemma_kek") or "").strip()
    if not lemma:
        return ""

    core = _np_nonposs_kek(lemma, noun.get("class"))

    # Optional plural marker for predicate NPs
    uncount_kek = (noun.get("countability_kek") or "").strip().lower() == "uncountable"
    if plural and (not uncount_kek):
        core = _dedupe_eb(pluralize_kek_np_bare_or_def(noun, definite=False))

    if not aff:
        return core

    # Use hyphenated attachment for predicate identity forms (winq-in).
    return f"{core}-{aff}"

def build_num_np_kek(
    noun: Dict,
    numeral_row: Dict,
    *,
    as_subject: bool = False,
    possessed_person: Optional[str] = None,
) -> str:
    num_kek = (numeral_row.get("lemma_kek") or "").strip()

    def _is_one_en(s: str) -> bool:
        low = (s or "").strip().lower()
        return low in {"1", "one"}

    # Decide plurality from the numeral (EN gloss is already present in numerals CSV)
    is_plural = not _is_one_en(numeral_row.get("gloss_en"))

    if possessed_person:
        # For possessum in HAVE/TENER templates:
        # - KEK: possessed morphology must reflect the possessor (= subject person)
        # - Force indefinite (no 'li') because EN/ES are indefinite objects
        np = build_np_kek(
            noun,
            plural=is_plural,
            definite=False,
            possessed_person=possessed_person,
            as_subject=False,
        )
    else:
        # Legacy: numeral + bare noun; optionally add 'li' in subject position for 'one'
        bare = (noun.get("lemma_kek") or "").strip()
        if as_subject and _is_one_en(numeral_row.get("gloss_en")) and not bare.lower().startswith("li "):
            bare = f"li {bare}".strip()
        np = bare

    return f"{num_kek} {np}".strip()

# -----------------------
# verb renderer
# -----------------------

def kek_infinitive(infinitive_kek: str, transitivity: str) -> str:
    """
    Return the Q’eqchi’ infinitive used for progressive.
    This expects the infinitive form from the verbs CSV
    (column `kek_gloss`).

    Reflexives are stored with a lexical 'ib’' marker in the CSV/XLSX, but
    surface Q’eqchi’ uses the possessed reflexive noun (wib’/awib’/rib’/...)
    after the verb, so we strip the marker here.
    """
    inf = (infinitive_kek or "").strip()
    if kek_is_reflexive_infinitive(inf):
        inf = kek_strip_reflexive_marker(inf)
    return inf

def _starts_with_a(s: str) -> bool:
    """Detect if the (cleaned) stem starts with 'a' (skip leading apostrophes)."""
    ss = (s or "").lstrip()
    i = 0
    while i < len(ss) and ss[i] in _APOS_CHARS:
        i += 1
    return (i < len(ss)) and (ss[i].lower() == "a")

def kek_is_reflexive_infinitive(infinitive_kek: str) -> bool:
    """
    Reflexive verbs are identified by the stored infinitive ending in `ib’` (or `ib'`).
    """
    s = (infinitive_kek or "").strip()
    if not s:
        return False
    s = s.replace("ib'", "ib’")
    return s.endswith("ib’")

def kek_strip_reflexive_marker(infinitive_kek: str) -> str:
    """
    Remove the lexical reflexive marker from the stored KEK infinitive.

    In our verbs CSV/XLSX, reflexives may be stored Spanish-style as:
        "kanasink ib’"  or "kanasink ib'"  or occasionally "... 'ib"

    But in surface Q’eqchi’, the reflexive is realized as a possessed relational noun
    following the verb (wib’/awib’/rib’/...), so we must not keep the lexical "ib’".
    """
    s = (infinitive_kek or "").strip()
    if not s:
        return s

    # Normalize straight apostrophe to curly for matching, but keep output as-is aside from removal.
    norm = s.replace("ib'", "ib’").replace("'ib", "’ib")

    # Remove final token: "ib’" or "’ib" (with or without a preceding space)
    # Examples:
    #   "kanasink ib’" -> "kanasink"
    #   "kanasinkib’"  -> "kanasink" (defense-in-depth)
    #   "kanasink ’ib" -> "kanasink"
    for tok in (" ib’", "ib’", " ’ib", "’ib"):
        if norm.endswith(tok):
            cut_len = len(tok)
            return s[:-cut_len].rstrip()

    return s

def _kek_attach_reflexive_suffix(form: str, person: str, *, is_reflexive: bool) -> str:
    """
    Spanish-style lexical reflexive handling, but Q'eqchi' reflexive is a possessed
    relational noun that follows the verb: xwil wib’ ...
    """
    f = (form or "").strip()
    if not f or not is_reflexive:
        return f
    p = (person or "3sg").strip().lower()
    pron = REFL_PRONOUNS.get(p, REFL_PRONOUNS["3sg"])
    return f"{f} {pron}"

def kek_build_intransitive(*, gloss_kek: str, person: str, tam: str = "") -> str:
    """
    Canonical Q'eqchi' intransitive form:
      TAM + Set B + intransitive stem
    IMPORTANT: Stem basis is ALWAYS the infinitive/citation form in gloss_kek.
    """
    
    stem = (gloss_kek or "").strip()
    if not stem:
        return ""

    p = (person or "3sg").strip().lower()
    tam = (tam or "").strip()

    if tam == "":
        pres = INTR_PRES_PREFIX.get(p, INTR_PRES_PREFIX.get("3sg", "na"))
        return f"{pres}{stem}"

    set_b = SET_B.get(p, "")
    return f"{tam}{set_b}{stem}"

def kek_conjugate(base_kek: str, noun_class: str, person: str, transitivity: str,
                  obj_person: str | None = None, tam_prefix: str = "") -> str:
    """
    Conjugate a Q’eqchi’ verb base by person, transitivity, and TAM.

    - For transitive/ditransitive: base is the stem from verbs CSV (`lemma_kek`).
    - For intransitive: build TAM + Set B + stem using the citation form from verbs CSV (`gloss_kek`).
    """
    kv = (base_kek or "").strip()
    if not kv:
        return kv

    trans = (transitivity or "").strip().lower()
    p = (person or "3sg").strip().lower()

    set_a = choose_set_a_allomorph(kv)

    if trans in {"tr", "ditr"}:
        # Transitive/ditransitive: TAM + Set A(subj) + Set B(obj) + stem (+ 3pl suffix where applicable)
        if p == "3pl":
            subj_prefix = set_a.get("3pl_prefix", "")
            suffix = set_a.get("3pl_suffix", "")
        else:
            subj_prefix = set_a.get(p, "")
            suffix = ""
        obj_prefix = SET_B.get((obj_person or "").strip().lower(), "") if obj_person else ""
        return f"{tam_prefix}{subj_prefix}{obj_prefix}{kv}{suffix}"

    # Intransitive: TAM + Set B + intransitive stem
    # NOTE: `kv` is expected to be the citation/infinitive form from `gloss_kek`.
    tam = (tam_prefix or "").strip()
    return kek_build_intransitive(gloss_kek=kv, person=p, tam=tam)

def _k_to_q_future(stem: str) -> str:
    """
    Q'eqchi' future alternation: final -k/-nk realised with 'q'.
    This is applied *after* stem derivation.
    """
    s = (stem or "").strip()
    if not s:
        return s

    if s.endswith("nk"):
        # ...nk → ...nq
        return s[:-1] + "q"
    if s.endswith("k"):
        # ...k → ...q
        return s[:-1] + "q"
    return s

def kek_future_form(stem: str, vclass: str, person: str = "3sg",
                    transitivity: str = "intr") -> str:
    """
    Build simple future forms using SET_A/SET_B_FUT and vowel/consonant alternation.

    Caller provides the appropriate base:
    - intr: may be the stored infinitive (`gloss_kek`) if you want finite intr forms to use it
    - tr/ditr: typically the stem (`lemma_kek`)

    Applies final k → q / nk → nq for future.
    """
    base_stem = (stem or "").strip()
    if not base_stem:
        return ""

    # Apply the future k→q alternation
    fut_stem = _k_to_q_future(base_stem)

    # choose prefix family
    set_a  = SET_A_PRECON_FUT
    set_a_v = SET_A_PREVOW_FUT
    set_b  = SET_B_FUT

    # detect vowel-initial stem
    is_vowel = fut_stem[0].lower() in "aeiouáéíóú'"

    # select prefix based on person and transitivity
    if (transitivity or "").strip().lower() == "intr":
        pref = set_b.get(person, "")
    else:
        pref = (set_a_v if is_vowel else set_a).get(person, "")

    # concatenate (no TAM particle for future)
    return f"{pref}{fut_stem}"

def kek_imperative(
    stem_kek: str,
    person: str = "2sg",
    transitivity: str = "intr",
    *,
    infinitive_kek: Optional[str] = None,
) -> str:
    """
    Positive imperative formation in Q’eqchi’.

    IMPORTANT:
    - Imperatives must be built from the full infinitive/citation form from the verbs CSV
      (passed in as `infinitive_kek` == `gloss_kek`).
    - Do NOT truncate to a bare stem.
    """
    stem = (stem_kek or "").strip()
    inf_csv = (infinitive_kek or "").strip()

    # We require the CSV infinitive for imperatives.
    if not inf_csv:
        raise ValueError("kek_imperative(): missing required infinitive_kek (gloss_kek in kek_verbs.csv).")

    p = (person or "2sg").strip().lower()

    # Prefix allomorphy should depend on what the prefix actually attaches to (the infinitive surface).
    is_vowel = _is_vowel_initial(inf_csv) if inf_csv else _is_vowel_initial(stem)

    if (transitivity or "intr").strip().lower() == "intr":
        table = POS_INTR_IMP_PREFIX_PREVOW if is_vowel else POS_INTR_IMP_PREFIX_PRECON
    else:
        table = POS_TR_IMP_PREFIX_PREVOW if is_vowel else POS_TR_IMP_PREFIX_PRECON

    pre = table.get(p, table.get("2sg"))
    return f"{pre}{inf_csv}"


def kek_imperative_negative(
    stem_kek: str,
    person: str = "2sg",
    *,
    infinitive_kek: Optional[str] = None,
) -> str:
    """
    Negative imperative formation in Q’eqchi’ for **intransitive** verbs.

    IMPORTANT:
    - Must be built from the full infinitive/citation form from the verbs CSV
      (passed in as `infinitive_kek` == `gloss_kek`), not from the bare stem.
    """
    inf_csv = (infinitive_kek or "").strip()
    if not inf_csv:
        raise ValueError("kek_imperative_negative(): missing required infinitive_kek (gloss_kek in kek_verbs.csv).")

    p = (person or "2sg").strip().lower()
    pre = NEG_INTR_IMP_PREFIX.get(p, "ma")

    # 3sg "ma" -> "ma'" before a-/’a- (apply to the infinitive surface).
    if pre == "ma" and _starts_with_a(inf_csv):
        pre = "ma'"

    return f"{pre}{inf_csv}"


def kek_imperative_negative_tr(
    stem_kek: str,
    person: str = "2sg",
    *,
    infinitive_kek: Optional[str] = None,
) -> str:
    """
    Negative imperative formation in Q’eqchi’ for **transitive and ditransitive** verbs.

    IMPORTANT:
    - Must be built from the full infinitive/citation form from the verbs CSV
      (passed in as `infinitive_kek` == `gloss_kek`), not from the bare stem.
    """
    inf_csv = (infinitive_kek or "").strip()
    if not inf_csv:
        raise ValueError("kek_imperative_negative_tr(): missing required infinitive_kek (gloss_kek in kek_verbs.csv).")

    p = (person or "2sg").strip().lower()
    is_vowel = _is_vowel_initial(inf_csv) if inf_csv else _is_vowel_initial(stem_kek)

    table = NEG_TR_IMP_PREFIX_PREVOW if is_vowel else NEG_TR_IMP_PREFIX_PRECON
    pre = table.get(p, table.get("2sg"))

    return f"{pre}{inf_csv}"

def kek_progressive(
    infinitive_kek: str,
    person: str,
    transitivity: str,
    *,
    aux_map: Dict[str, str] | None = None
) -> str:
    """
    Build the progressive periphrasis from a stored infinitive:

        {aux(person)} + ' chi ' + infinitive

    The infinitive should come from the verbs CSV (`kek_gloss`).

    `aux_map` lets callers swap the auxiliary inventory (e.g., future progressive).
    """
    table = aux_map or AUX_PROG
    aux = table.get((person or "3sg").strip().lower(), table.get("3sg", "yoo"))
    inf = kek_infinitive(infinitive_kek, transitivity)
    return f"{aux} chi {inf}"

def render_verb_bundle_kek(
    vrow: dict,
    key_prefix: str,
    *,
    person: str = "3sg",
    transitivity: str = "intr",
    obj_person: str | None = None,
    env: dict | None = None
) -> Dict[str, str]:
    if not vrow:
        return {}

    # Explicit split between stem and infinitive from the verbs CSV
    stem = (vrow.get("lemma_kek") or "").strip()         # verb stem (finite base for tr/ditr)
    vclass = (vrow.get("class") or "").strip()
    infinitive = (vrow.get("gloss_kek") or "").strip()   # citation/infinitive form (finite base for intr)

    trans = (transitivity or "").strip().lower()

    # Choose the base that feeds finite morphology:
    # - intr: conjugate from citation/infinitive form in `gloss_kek` (Stewart-style nak-/na- + Set B)
    # - tr/ditr: conjugate from lemma_kek as before.
    if trans == "intr":
        finite_base = infinitive if infinitive else stem
    else:
        finite_base = stem if stem else infinitive

    # Reflexive if infinitive ends in `ib’` (or `ib'`)
    is_refl = kek_is_reflexive_infinitive(infinitive)

    # Clean the stored infinitive for surface forms (progressive/imperative/etc.)
    infinitive_core = kek_strip_reflexive_marker(infinitive) if is_refl else infinitive
    
    # Grammar: for reflexives, Set B on the verb must be 3sg/ø (patient = "self")
    eff_obj_person = obj_person
    if is_refl and (transitivity or "").strip().lower() in {"tr", "ditr"}:
        eff_obj_person = "3sg"

    # Use environment overrides if present; otherwise fall back to module defaults
    tam_pfv = (env.get("KEK_TAM_PFV") if (env and "KEK_TAM_PFV" in env) else KEK_TAM_PFV_DEFAULT)

    # Negation markers fall back to DEFAULT_PLACEHOLDERS
    neg_pre = (env.get("NEG_PRE") if env and "NEG_PRE" in env else DEFAULT_PLACEHOLDERS["NEG_PRE"])
    neg_suf = (env.get("NEG_SUF") if env and "NEG_SUF" in env else DEFAULT_PLACEHOLDERS["NEG_SUF"])

    out: Dict[str, str] = {}

    # Finite forms (wrapped for reflexives)
    v_pres = kek_conjugate(finite_base, vclass, person, transitivity, eff_obj_person, tam_prefix="")
    v_pst  = kek_conjugate(finite_base, vclass, person, transitivity, eff_obj_person, tam_prefix=tam_pfv)
    out[f"{key_prefix}_KEK"]     = _kek_attach_reflexive_suffix(v_pres, person, is_reflexive=is_refl)
    out[f"{key_prefix}_PST_KEK"] = _kek_attach_reflexive_suffix(v_pst,  person, is_reflexive=is_refl)

    # Imperatives from the stem + stored infinitive (also wrapped)
    imp = kek_imperative(
        stem,
        person,
        transitivity,
        infinitive_kek=infinitive_core,
    )
    out[f"{key_prefix}_IMP"]    = _kek_attach_reflexive_suffix(imp, person, is_reflexive=is_refl)
    out[f"{key_prefix}_IMP_PL"] = out[f"{key_prefix}_IMP"]

    imp_p = (env.get("IMP_PERSON") if (env and "IMP_PERSON" in env) else person) or "2sg"
    if (transitivity or "").strip().lower() in {"tr", "ditr"}:
        neg_imp = kek_imperative_negative_tr(
            stem,
            imp_p,
            infinitive_kek=infinitive_core,
        )
    else:
        neg_imp = kek_imperative_negative(
            stem,
            imp_p,
            infinitive_kek=infinitive_core,
        )
    
    out[f"{key_prefix}_IMP_NEG"]    = _kek_attach_reflexive_suffix(neg_imp, imp_p, is_reflexive=is_refl)
    out[f"{key_prefix}_IMP_NEG_PL"] = out[f"{key_prefix}_IMP_NEG"]

    # Future from the stem (wrapped)
    fut = kek_future_form(finite_base, vclass, person, transitivity)
    out[f"{key_prefix}_FUT_KEK"] = _kek_attach_reflexive_suffix(fut, person, is_reflexive=is_refl)
    out[f"{key_prefix}_FUT"]     = out[f"{key_prefix}_FUT_KEK"]

    # Progressive from the stored infinitive (wrapped)
    prog = kek_progressive(infinitive_core, person, transitivity)
    out[f"{key_prefix}_PROG_KEK"] = _kek_attach_reflexive_suffix(prog, person, is_reflexive=is_refl)

    # Future progressive (same progressive stem, different auxiliary)
    prog_fut = kek_progressive(infinitive_core, person, transitivity, aux_map=AUX_PROG_FUT)
    out[f"{key_prefix}_FUT_PROG_KEK"] = _kek_attach_reflexive_suffix(prog_fut, person, is_reflexive=is_refl)
    
    # Non-specific passive (-man) for transitive verbs ---
    # We treat the -man form as an intransitive neutral verb (vin-class).
    # (Left unwrapped intentionally; reflexive semantics generally should not propagate to this derived passive.)
    if (transitivity or "").strip().lower() in {"tr", "ditr"} and stem:
        passive_stem = f"{stem}man"
        out["V_TR_PASS"]      = kek_conjugate(passive_stem, vclass, person, "intr", None, tam_prefix="")
        out["V_TR_PST_PASS"]  = kek_conjugate(passive_stem, vclass, person, "intr", None, tam_prefix=tam_pfv)
        out["V_TR_FUT_PASS"]  = kek_future_form(passive_stem, vclass, person, "intr")

    # Convenience aliases
    out[f"{key_prefix}"]     = out[f"{key_prefix}_KEK"]
    out[f"{key_prefix}_PST"] = out[f"{key_prefix}_PST_KEK"]

    return out

# Sub-clause constructions for affixing pronouns, "Have [something]", "by [someone]" and negation ##

def style_kek_affix_pronouns(sentence: str) -> str:
    """
    Attach KEK clitic/affix pronouns (in/at/o/ex/eb') to the preceding token.
    Example: "tz'iib' in." -> "tz'iib'in."
    Only affects occurrences that are separated by whitespace.
    """
    import re

    s = (sentence or "").strip()
    if not s:
        return s

    # Match: <non-space> + spaces + (affix) + boundary
    # Keep it conservative: only join known affixes.
    affixes = ("in", "at", "o", "ex", "eb'")
    pat = r"(\S)\s+(" + "|".join(re.escape(a) for a in affixes) + r")\b"
    return re.sub(pat, r"\1\2", s)

def kek_with_pron(person: str = "3sg") -> str:
    p = (person or "3sg").strip().lower()
    return WITH_PRON.get(p, WITH_PRON["3sg"])

def kek_agent_by(person: str = "3sg") -> str:
    p = (person or "3sg").strip().lower()
    return AGENT_BY.get(p, AGENT_BY["3sg"])

def style_kek_moko_ta(sentence: str) -> str:
    """
    Post-process Q'eqchi' negation with 'moko ... ta'.

    Linguistic constraint:
      - 'moko' and 'ta' can only enclose a single word.
    Implementation:
      - For each 'moko' token, find the next 'ta'.
      - If 'ta' is not already immediately after the first word following 'moko',
        move that 'ta' so the sequence becomes:
            moko + <first_word_after_moko> + ta + (rest...)
    Robustness fix:
      - Treat 'ta' as found even if followed by trailing punctuation (e.g., 'ta.').
      - Preserve that punctuation when moving the token.
    """
    sentence = (sentence or "").strip()
    if not sentence:
        return sentence

    tokens = sentence.split()

    # Accept ta with optional trailing punctuation attached (common in templates: "ta.")
    _ta_re = re.compile(r"^ta([.?!,;:]+)?$", flags=re.IGNORECASE)

    def _ta_match(tok: str):
        return _ta_re.match(tok or "")

    pending_final_punct: Optional[str] = None

    i = 0
    while i < len(tokens):
        if (tokens[i] or "").lower() == "moko":
            ta_idx = None
            ta_m = None
            for j in range(i + 1, len(tokens)):
                m = _ta_match(tokens[j])
                if m:
                    ta_idx = j
                    ta_m = m
                    break

            # Need at least one word after moko and a later ta
            if ta_idx is not None and i + 1 < len(tokens):
                desired_ta_pos = i + 2  # moko [i], first word [i+1], then ta [i+2]

                if ta_idx != desired_ta_pos:
                    ta_token = tokens.pop(ta_idx)

                    # If ta had trailing punctuation, strip it off before moving.
                    # If it is sentence-final punctuation, remember it for the end.
                    m = ta_m or _ta_match(ta_token)
                    punct = (m.group(1) if m else None) or ""
                    if punct:
                        ta_token = "ta"  # normalize

                        # Only treat . ? ! as final punctuation to move to sentence end.
                        # (Keep other punctuation attached to ta if you ever use it, but most templates won't.)
                        if any(ch in punct for ch in ".?!"):
                            # Keep the last final punctuation char if multiple are present (e.g., "?!")
                            finals = [ch for ch in punct if ch in ".?!"]
                            pending_final_punct = finals[-1] if finals else pending_final_punct

                    insert_at = min(desired_ta_pos, len(tokens))
                    tokens.insert(insert_at, ta_token)
                    i = insert_at + 1
                else:
                    i += 1
            else:
                i += 1
        else:
            i += 1

    # If we captured a sentence-final punctuation mark from "ta.", attach it to the end.
    if pending_final_punct and tokens:
        last = tokens[-1]
        if not last or last[-1] not in ".?!":
            tokens[-1] = f"{last}{pending_final_punct}"

    return " ".join(tokens)