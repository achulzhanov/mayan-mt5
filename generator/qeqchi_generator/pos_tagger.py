"""pos_tagger.py — POS and semantic tagging for Q'eqchi' sentences.

Produces two parallel annotation strings in the format:
    "Word (TAG) Word (TAG) ..."

where TAG is either a UPOS tag (for POS) or a BIO semantic class (for semantics).

Rules:
  - 17 Universal Dependencies (UPOS) tags only.
  - Semantic tags use BIO format: B-CLASS for content words, O for everything else.
  - Punctuation (. , ! ?) → PUNCT.  Glottal-stop apostrophe (') is a consonant
    and stays attached to its word — never tagged separately as PUNCT.
  - Implied/pro-dropped pronouns are NOT tagged (only surface tokens).
  - AFFIX_PRONOUN tokens fuse with the preceding content word; the fused form
    keeps the content word's tag.
  - Possessed NPs undergo the same possessive contractions as
    style_kek_possessives(); the annotation mirrors those fusions.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Semantic label map  (CSV `class` column → BIO class name)
# ---------------------------------------------------------------------------
# Keys are normalised to lowercase with underscores (matching enrich_noun_semantics).
_SEMANTIC_LABEL: Dict[str, str] = {
    "animal_domesticated": "ANIMAL-DOMESTICATED",
    "animal_wild":         "ANIMAL-WILD",
    "artifact":            "ARTIFACT",
    "beverage":            "BEVERAGE",
    "body_part":           "BODY-PART",
    "body part":           "BODY-PART",   # legacy CSV spelling
    "building":            "BUILDING",
    "clothing":            "CLOTHING",
    "container":           "CONTAINER",
    "corn":                "CORN",
    "emotion":             "EMOTION",
    "food":                "FOOD",
    "fruit":               "FRUIT",
    "furniture":           "FURNITURE",
    "human":               "HUMAN",
    "insect":              "INSECT",
    "kinship":             "KINSHIP",
    "language":            "LANGUAGE",
    "occupation":          "OCCUPATION",
    "place":               "LOCATION",
    "plant":               "PLANT",
    "substance":           "SUBSTANCE",
    "tool":                "TOOL",
    "vehicle":             "VEHICLE",
    "wood":                "WOOD",
}


def _sem_label(cls: str) -> str:
    """Return 'B-LABEL' for a known class, else 'O'."""
    if not cls:
        return "O"
    label = _SEMANTIC_LABEL.get(cls.strip().lower())
    return f"B-{label}" if label else "O"


# ---------------------------------------------------------------------------
# Known KEK function-word surfaces → UPOS
# These are tokens that appear either as standalone words or as the first
# token inside a multi-word noun/name surface produced by build_np_kek /
# kek_name_surface.
# ---------------------------------------------------------------------------
_KEK_FUNCTION_WORDS: Dict[str, str] = {
    # --- Determiners (class markers, articles, plural marker) ---
    "li":       "DET",
    "laj":      "DET",   # male proper-name prefix
    "aj":       "DET",   # agentive nominaliser (occupations, some animals)
    "eb'":      "DET",   # plural marker
    "laa":      "DET",   # contracted 2sg possessed + li
    "lee":      "DET",   # contracted 2pl possessed + li
    # --- Negation particles ---
    "moko":     "PART",
    "ta":       "PART",
    # --- Yes/no question marker ---
    "ma":       "PART",
    # --- Doubt particles ---
    "maare":    "PART",
    # --- Question / wh-words ---
    "k'a'ut":   "ADV",
    "b'ar":     "ADV",
    "jo'q'e":   "ADV",
    # --- Existential copula ---
    "wan":      "VERB",
    "wankeb'":  "VERB",
    # --- Standalone subject pronouns ---
    "laa'in":   "PRON",
    "laa'at":   "PRON",
    "a'an":     "PRON",
    "laa'o":    "PRON",
    "laa'ex":   "PRON",
    "heb'a'an": "PRON",
    # --- Reflexive pronouns ---
    "wib'":     "PRON",
    "aawib'":   "PRON",
    "rib'":     "PRON",
    "qib'":     "PRON",
    "eerib'":   "PRON",
    "rib'eb":   "PRON",
    # --- Comitative "with" pronouns (have/possession constructions) ---
    "wik'in":      "PRON",
    "aawik'in":    "PRON",
    "rik'in":      "PRON",
    "qik'in":      "PRON",
    "eerik'in":    "PRON",
    "rik'ineb'":   "PRON",
    # --- Comparative adposition ---
    "chi":      "ADP",
    "ru":       "ADP",   # appears in "chi ru" (than)
}

# Tokens within a noun/name surface that mark a DET prefix before the noun head.
# The FIRST token of a multi-word surface is checked against this set.
_DET_PREFIXES = frozenset({"li", "laj", "aj", "eb'", "laa", "lee", "lin"})

# Affix-pronoun tokens that fuse with the preceding word via style_kek_affix_pronouns.
_AFFIX_TOKENS = frozenset({"in", "at", "o", "ex", "eb'"})

# Punctuation characters that are split off as separate PUNCT tokens.
_PUNCT_CHARS = frozenset(".,!?;:")


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

def tokenize_kek(sent: str) -> List[str]:
    """
    Split a final Q'eqchi' sentence into annotation tokens.

    - Splits on whitespace.
    - Strips leading and trailing punctuation (.!?,;:) from each whitespace-token,
      emitting them as separate tokens.
    - Does NOT split on apostrophe/glottal-stop (') — it stays attached to the word.
    """
    tokens: List[str] = []
    for raw in sent.split():
        leading: List[str] = []
        while raw and raw[0] in _PUNCT_CHARS:
            leading.append(raw[0])
            raw = raw[1:]
        trailing: List[str] = []
        while raw and raw[-1] in _PUNCT_CHARS:
            trailing.append(raw[-1])
            raw = raw[:-1]
        tokens.extend(leading)
        if raw:
            tokens.append(raw)
        tokens.extend(reversed(trailing))
    return tokens


# ---------------------------------------------------------------------------
# Slot → (pos_tag, sem_tag) builder
# ---------------------------------------------------------------------------

def _build_slot_tags(ai: dict) -> Dict[str, Optional[Tuple[str, str]]]:
    """
    Build a mapping  slot_name → (upos_tag, sem_tag)  from the annotation_info
    dict returned by render().  None means the slot is not a KEK surface token
    (EN/ES-only or internal metadata).
    """
    tags: Dict[str, Optional[Tuple[str, str]]] = {}

    # ---- Noun slot families ------------------------------------------------
    # (base_slot_names, row_dict)
    _np_families: List[Tuple[List[str], Optional[dict]]] = [
        (["NP", "PRED_NP", "NUM_NP"],                    ai.get("np_noun")),
        (["POSS_NP", "THEME_POSS_NP"],                   ai.get("poss_noun")),
        (["POSSESSUM_NP", "NUM_POSSESSUM_NP"],            ai.get("possessum_row")),
        (["THEME_NP", "REF_NP_DEF", "NUM_THEME_NP"],     ai.get("theme_row")),
        (["GOAL_NP", "NUM_GOAL_NP"],                     ai.get("goal_row")),
        (["AGENT_NP", "NUM_AGENT_NP"],                   ai.get("agent_row")),
        (["OCCUPATION"],                                  ai.get("occ_row")),
    ]

    _NP_VAR_SUFFIXES = [
        "", "_PL", "_PL_INDEF", "_PL_PRE", "_PL_POST", "_INDEF", "_DEF",
    ]

    for base_slots, row in _np_families:
        cls = str((row or {}).get("class") or "").strip().lower()
        sem = _sem_label(cls)
        pt = ("NOUN", sem)
        for base in base_slots:
            for suf in _NP_VAR_SUFFIXES:
                tags[f"{base}{suf}"] = pt

    # NAME → PROPN; PLACE → PROPN + B-LOCATION
    tags["NAME"] = ("PROPN", "O")
    tags["PLACE"] = ("PROPN", "B-LOCATION")

    # ---- Verb slot families ------------------------------------------------
    _VERB_PREFIXES = ["V_INTR", "V_TR", "V_DITR", "V_POSS_TR"]
    _VERB_SUFFIXES = [
        "", "_KEK", "_PST", "_PST_KEK", "_FUT", "_FUT_KEK",
        "_PROG", "_PROG_KEK", "_PP", "_PP_KEK",
        "_IMP", "_IMP_KEK", "_IMP_NEG", "_IMP_NEG_KEK",
        "_NEG", "_NEG_KEK", "_BASE", "_BASE_KEK",
    ]
    for vp in _VERB_PREFIXES:
        for vs in _VERB_SUFFIXES:
            tags[f"{vp}{vs}"] = ("VERB", "O")

    # Auxiliary progressives
    tags["AUX_PROG_KEK"]     = ("VERB", "O")
    tags["AUX_PROG_FUT_KEK"] = ("VERB", "O")

    # ---- Other content slots -----------------------------------------------
    tags["ADJ"]          = ("ADJ",  "O")
    tags["ADV"]          = ("ADV",  "O")
    tags["NUM"]          = ("NUM",  "O")
    tags["SUBJ_PRONOUN"] = ("PRON", "O")
    tags["AFFIX_PRONOUN"] = ("PRON", "O")   # will fuse with preceding token

    # ---- Function / particle slots -----------------------------------------
    tags["NEG_PRE"]      = ("PART", "O")   # Moko
    tags["NEG_SUF"]      = ("PART", "O")   # ta
    tags["YN_Q_MARKER"]  = ("PART", "O")   # Ma
    tags["DOUBT_PART"]   = ("PART", "O")
    tags["WH"]           = ("ADV",  "O")
    tags["WH_LOC"]       = ("ADV",  "O")
    tags["WH_TIME"]      = ("ADV",  "O")
    tags["EXIST"]        = ("VERB", "O")
    tags["EXIST_PL"]     = ("VERB", "O")
    tags["COMP"]         = ("ADP",  "O")
    tags["BENEF_REL"]    = ("ADP",  "O")
    tags["GOAL_REL"]     = ("ADP",  "O")
    tags["WITH_PRON"]    = ("PRON", "O")
    tags["AGENT_BY"]     = ("ADP",  "O")

    # Stative future suffixes — they are bare suffixes (no leading space in
    # templates), so they concatenate directly with the preceding token.
    # Treat as PART; their fusion is handled like AFFIX_PRONOUN.
    tags["ST_FUT_AQ"]  = ("PART", "O")
    tags["ST_FUT_IND"] = ("PART", "O")
    tags["ST_FUT_OPT"] = ("PART", "O")
    tags["ST_FUT_DBT"] = ("PART", "O")

    # ---- Metadata slots — no KEK surface token -----------------------------
    for k in (
        "IMP_PERSON",
        "REF_IS_SUBJ_EN", "REF_IS_SUBJ_ES", "REF_IS_PL",
        "REF_GENDER", "REF_GENDER_ES",
        "BE_EN", "BE_NEG_EN", "BE_FUT_EN",
        "COP_ES", "COP_NEG_ES", "COP_FUT_ES", "COP_NEG_FUT_ES",
        "V_SER_ES", "DO_NEG_EN", "DO_AFF_EN", "IR_ES",
        "SUBJ_PRONOUN_EN", "SUBJ_PRONOUN_ES",
        "SUBJ_PRON_EN", "SUBJ_PRON_ES",
        "OCCUPATION_EN", "OCCUPATION_ES",
        "ADJ_EN", "ADJ_ES", "ADJ_COMP_EN",
        "ADV_EN", "ADV_ES",
        "NUM_EN", "NUM_ES",
        "AGENT_BY_EN", "AGENT_BY_ES",
        "EXIST_EN", "EXIST_NEG_EN", "EXIST_ES", "EXIST_NEG_ES",
        "EXIST_PL_EN", "EXIST_NEG_PL_EN", "EXIST_PL_ES", "EXIST_NEG_PL_ES",
        "V_POSS_TR_EN", "V_POSS_TR_ES",
        "V_POSS_TR_PST_EN", "V_POSS_TR_PST_ES",
        "V_POSS_TR_FUT_EN", "V_POSS_TR_FUT_ES",
        "V_POSS_TR_PRG_EN", "V_POSS_TR_PRG_ES",
        "V_POSS_TR_PP_EN",  "V_POSS_TR_BASE_EN", "V_POSS_TR_BASE_ES",
    ):
        tags[k] = None

    return tags


# ---------------------------------------------------------------------------
# Template expansion
# ---------------------------------------------------------------------------

_SLOT_RE = re.compile(r'\{([A-Z0-9_]+)\}')


def _expand_template(
    kek_tmpl: str,
    repl: dict,
    slot_tags: Dict[str, Optional[Tuple[str, str]]],
) -> List[Tuple[str, str, str]]:
    """
    Walk kek_tmpl segment by segment and produce a list of
    (surface_token, pos_tag, sem_tag) tuples, representing the
    pre-post-processing annotation sequence.

    Slots ending in _EN or _ES are skipped (they don't appear in KEK output).
    Empty surface expansions (3sg AFFIX_PRONOUN, empty BENEF_REL, etc.) are skipped.
    """
    # Split into alternating [literal, slot, literal, slot, …] segments.
    segments = _SLOT_RE.split(kek_tmpl)
    # After re.split with one capture group, odd indices are slot names, even are literals.

    result: List[Tuple[str, str, str]] = []

    for idx, seg in enumerate(segments):
        if not seg:
            continue

        if idx % 2 == 1:
            # --- Slot segment ---
            slot_name = seg
            if slot_name.endswith("_EN") or slot_name.endswith("_ES"):
                continue

            tag_pair = slot_tags.get(slot_name)
            if tag_pair is None:
                # Metadata-only slot, no KEK surface.
                continue

            pos_tag, sem_tag = tag_pair
            surface = str(repl.get(slot_name, "")).strip()
            if not surface:
                continue

            # Tokenise the surface on whitespace (glottal-stop stays attached).
            sub_tokens = surface.split()
            for i, tok in enumerate(sub_tokens):
                if not tok:
                    continue
                tok_lower = tok.lower()
                # First sub-token of a multi-token surface may be a determiner prefix.
                if i == 0 and tok_lower in _DET_PREFIXES and len(sub_tokens) > 1:
                    result.append((tok, "DET", "O"))
                elif tok_lower in _KEK_FUNCTION_WORDS and i == 0:
                    # Single-token surface that happens to be a known function word
                    # (e.g. EXIST → "wan").  Trust the slot tag rather than the
                    # function-word lookup, since slot_tags already encodes intent.
                    result.append((tok, pos_tag, sem_tag))
                else:
                    result.append((tok, pos_tag, sem_tag))

        else:
            # --- Literal text segment (static template text) ---
            for tok in seg.split():
                if not tok:
                    continue
                # Strip leading/trailing punctuation from the literal token.
                lead: List[str] = []
                while tok and tok[0] in _PUNCT_CHARS:
                    lead.append(tok[0])
                    tok = tok[1:]
                trail: List[str] = []
                while tok and tok[-1] in _PUNCT_CHARS:
                    trail.append(tok[-1])
                    tok = tok[:-1]
                for p in lead:
                    result.append((p, "PUNCT", "O"))
                if tok:
                    tok_lower = tok.lower()
                    upos = _KEK_FUNCTION_WORDS.get(tok_lower, "X")
                    result.append((tok, upos, "O"))
                for p in reversed(trail):
                    result.append((p, "PUNCT", "O"))

    return result


# ---------------------------------------------------------------------------
# Fusion replays  (mirror linguistics_kek post-processing on annotation list)
# ---------------------------------------------------------------------------

def _apply_possessives_fusion(
    tokens: List[Tuple[str, str, str]],
) -> List[Tuple[str, str, str]]:
    """
    Mirror style_kek_possessives():
    Fuse 'li' + possessed-prefix forms into a single token, keeping the
    possessed noun's tag.
    """
    # Import locally to avoid circular imports at module level.
    from .linguistics_kek import _is_vowel_initial  # type: ignore[attr-defined]

    out: List[Tuple[str, str, str]] = []
    i = 0
    while i < len(tokens):
        surf, pos, sem = tokens[i]

        if surf.lower() == "li" and i + 1 < len(tokens):
            ns, np_, nsem = tokens[i + 1]

            # 1sg  li + inC… → linC…
            if ns.startswith("in") and len(ns) > 2 and not _is_vowel_initial(ns[2:]):
                out.append(("l" + ns, np_, nsem))
                i += 2
                continue

            # 2sg  li + aaV…  → laa + V…   (vowel-initial stays separate)
            #      li + aaC…  → laaC…       (consonant-initial fuses)
            if ns.startswith("aa") and len(ns) > 2:
                lemma = ns[2:]
                if _is_vowel_initial(lemma):
                    out.append(("laa", "DET", "O"))
                    out.append((lemma, np_, nsem))
                else:
                    out.append(("l" + ns, np_, nsem))
                i += 2
                continue

            # 2pl  li + eerV… → lee + V…   (vowel-initial)
            #      li + eerC… → leerC…      (consonant-initial, uncommon)
            if ns.startswith("eer") and len(ns) > 3:
                lemma = ns[3:]
                if _is_vowel_initial(lemma):
                    out.append(("lee", "DET", "O"))
                    out.append((lemma, np_, nsem))
                else:
                    out.append(("l" + ns, np_, nsem))
                i += 2
                continue

            # 2pl  li + eeC… → leeC…
            #      li + eeV… → lee + V…
            if ns.startswith("ee") and len(ns) > 2:
                lemma = ns[2:]
                if _is_vowel_initial(lemma):
                    out.append(("lee", "DET", "O"))
                    out.append((lemma, np_, nsem))
                else:
                    out.append(("l" + ns, np_, nsem))
                i += 2
                continue

        out.append((surf, pos, sem))
        i += 1

    return out


def _apply_affix_fusion(
    tokens: List[Tuple[str, str, str]],
) -> List[Tuple[str, str, str]]:
    """
    Mirror style_kek_affix_pronouns():
    Fuse affix-pronoun tokens (in/at/o/ex/eb') with the immediately preceding
    token, keeping the preceding token's tag.
    """
    out: List[Tuple[str, str, str]] = []
    for surf, pos, sem in tokens:
        if surf in _AFFIX_TOKENS and out:
            ps, pp, psem = out[-1]
            out[-1] = (ps + surf, pp, psem)
        else:
            out.append((surf, pos, sem))
    return out


def _apply_moko_ta_reorder(
    tokens: List[Tuple[str, str, str]],
) -> List[Tuple[str, str, str]]:
    """
    Mirror style_kek_moko_ta():
    Ensure 'ta' (possibly with attached PUNCT) sits immediately after the
    first content word following 'moko'.
    """
    # Work on surfaces only; re-attach tags afterward.
    result = list(tokens)
    i = 0
    while i < len(result):
        surf, pos, sem = result[i]
        if surf.lower() == "moko":
            # Find the next 'ta' (may have trailing PUNCT)
            ta_idx = None
            for j in range(i + 1, len(result)):
                s = result[j][0].lower()
                if s == "ta" or (s.startswith("ta") and s[2:] and all(c in _PUNCT_CHARS for c in s[2:])):
                    ta_idx = j
                    break
            if ta_idx is not None and i + 1 < len(result):
                desired = i + 2
                if ta_idx != desired:
                    ta_entry = result.pop(ta_idx)
                    result.insert(desired, ta_entry)
        i += 1
    return result


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def build_kek_annotation(
    kek_sent: str,
    annotation_info: dict,
) -> Tuple[str, str]:
    """
    Build POS and semantic annotation strings for a final Q'eqchi' sentence.

    Args:
        kek_sent:        The final assembled Q'eqchi' sentence string.
        annotation_info: Dict returned by render() when annotate=True.
                         Must contain keys: kek_tmpl, repl, np_noun,
                         poss_noun, possessum_row, theme_row, goal_row,
                         agent_row, occ_row, chosen_v_intr, chosen_v_tr,
                         chosen_v_ditr, chosen_adj, chosen_adv, chosen_num,
                         chosen_name_dict, place_dict.

    Returns:
        (pos_kek, semantic_kek) — two strings in "Word (TAG)" format.
    """
    kek_tmpl: str = annotation_info.get("kek_tmpl", "")
    repl: dict    = annotation_info.get("repl", {})

    # 1. Build slot → tag mapping.
    slot_tags = _build_slot_tags(annotation_info)

    # 2. Expand template into pre-fusion annotation list.
    raw_tokens = _expand_template(kek_tmpl, repl, slot_tags)

    # 3. Apply possessive contractions (mirrors style_kek_possessives).
    fused = _apply_possessives_fusion(raw_tokens)

    # 4. Apply affix-pronoun fusions (mirrors style_kek_affix_pronouns).
    fused = _apply_affix_fusion(fused)

    # 5. Reorder moko/ta (mirrors style_kek_moko_ta).
    fused = _apply_moko_ta_reorder(fused)

    # 6. Align annotation surfaces with the final kek_sent tokens so that
    #    capitalisation in the output matches the sentence exactly.
    final_tokens = _align_with_sentence(fused, kek_sent)

    # 7. Serialise.
    pos_parts:  List[str] = []
    sem_parts:  List[str] = []
    for surf, pos, sem in final_tokens:
        if surf:
            pos_parts.append(f"{surf} ({pos})")
            sem_parts.append(f"{surf} ({sem})")

    return " ".join(pos_parts), " ".join(sem_parts)


def _align_with_sentence(
    annotated: List[Tuple[str, str, str]],
    kek_sent: str,
) -> List[Tuple[str, str, str]]:
    """
    Replace annotation surface strings with the corresponding tokens from
    the final kek_sent so that capitalisation matches exactly.

    If the token counts match (case-insensitive comparison), we use kek_sent
    tokens for surfaces.  If they diverge (a rare edge-case not covered by
    the fusion rules), we fall back to the annotation surfaces unchanged.
    """
    sent_tokens = tokenize_kek(kek_sent)

    if len(sent_tokens) != len(annotated):
        # Mismatch — return annotation as-is (surface may differ slightly in
        # capitalisation, but tags are correct).
        return annotated

    result: List[Tuple[str, str, str]] = []
    for sent_tok, (_, pos, sem) in zip(sent_tokens, annotated):
        result.append((sent_tok, pos, sem))
    return result
