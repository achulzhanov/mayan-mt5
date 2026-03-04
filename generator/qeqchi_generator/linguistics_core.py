"""linguistics_core.py – language-independent utilities and a minimal registry."""

from typing import Dict, Tuple, Optional, List, Callable
import random
import re
from . import utils

# -----------------------
# General data / helpers
# -----------------------

PERSONS = ["1sg", "2sg", "3sg", "1pl", "2pl", "3pl"]

# Classes of nouns that are typically *not* possessable cross-linguistically
NON_POSSESSABLE_CLASSES = {
    "animal_wild", "human", "insect", "occupation", 
    "weather", "phenomenon", "time", "season", "celestial",
    "language", "mass", "measurement", "quantity",
    "abstract", "event", "disease"
}
# Classes that are *obligatorily* possessed (kinship terms, body parts, etc.).
# This is needed for Mayan languages.
OBLIGATORY_POSSESSED_CLASSES = {"kinship", "body_part"}

NAMES_HUMAN = [
    {"text": "Ajpop", "gender": "m"},
    {"text": "Alejandro", "gender": "m"},
    {"text": "Ana", "gender": "f"},
    {"text": "Carlos", "gender": "m"},
    {"text": "Carmen", "gender": "f"},
    {"text": "Claudia", "gender": "f"},
    {"text": "Diego", "gender": "m"},
    {"text": "Elena", "gender": "f"},
    {"text": "Ixkik’", "gender": "f"},
    {"text": "José", "gender": "m"},
    {"text": "Juan", "gender": "m"},
    {"text": "Jwaan", "gender": "f"},
    {"text": "Jwan", "gender": "m"},
    {"text": "Kalich", "gender": "m"},
    {"text": "Kax", "gender": "m"},
    {"text": "Konsep", "gender": "f"},
    {"text": "Liiy", "gender": "f"},
    {"text": "Lis", "gender": "f"},
    {"text": "Lucas", "gender": "m"},
    {"text": "María", "gender": "f"},
    {"text": "Mateo", "gender": "m"},
    {"text": "Merse", "gender": "f"},
    {"text": "Miguel", "gender": "m"},
    {"text": "Patricia", "gender": "f"},
    {"text": "Pedro", "gender": "m"},
    {"text": "Rosa", "gender": "f"},
    {"text": "Sofia", "gender": "f"},
    {"text": "Winston", "gender": "m"},
    {"text": "Yadira", "gender": "f"},
]

def pick_name(prefer_gender: str | None = None) -> dict:
    pool = [n for n in NAMES_HUMAN if prefer_gender and n["gender"] == prefer_gender] or NAMES_HUMAN
    return random.choice(pool)

def name_to_noun_row(name_dict: dict) -> dict:
    txt = name_dict["text"]
    g   = (name_dict.get("gender") or "m").lower()

    return {
        # Surface forms
        "lemma_kek": txt,
        "gloss_en":  txt,
        "gloss_es":  txt,

        # Core semantic class
        "class": "human",
        "is_human": True,
        "possessability": "optional",

        # Gender for ES agreement
        "gender": g,
        "gender_es": g,

        # Optional fields used in adjective selection
        "has_color": False,

        # Stable unique identity
        "id": f"name::{txt}",
    }

NAMES_PLACE = [
    {"text":"Cobán"},
    {"text":"San Pedro Carchá"},
    {"text":"San Juan Chamelco"},
    {"text":"San Cristóbal Verapaz"},
    {"text":"Tactic"},
    {"text":"Tucurú"},
    {"text":"Panzós"},
    {"text":"Chisec"},
    {"text":"Fray Bartolomé de las Casas"},
    {"text":"Raxruhá"},
    {"text":"Santa Catalina La Tinta"},
    {"text":"Senahú"},
    {"text":"Lanquín"},
    {"text":"Cahabón"},
    {"text":"Salamá"},
    {"text":"Rabinal"},
    {"text":"Cubulco"},
    {"text":"Purulhá"},
    {"text":"El Estor"},
    {"text":"Livingston"},
    {"text":"Morales"},
    {"text":"Poptún"},
    {"text":"Dolores"},
    {"text":"San Luis"},
    {"text":"Melchor de Mencos"},
    {"text":"San Benito"},
    {"text":"Flores"},
    {"text":"Chajul"},
    {"text":"Uspantán"},
    {"text":"Santa María Nebaj"},
    {"text":"Mixco"},
    {"text":"Villa Nueva"},
    {"text":"Petapa"},
    {"text":"Quetzaltenango"},
    {"text":"San Juan Sacatepéquez"},
    {"text":"Escuintla"},
    {"text":"Huehuetenango"},
    {"text":"Chimaltenango"},
]

def pick_place() -> dict:
    """Pick a random place dict from NAMES_PLACE (untranslated place names)."""
    return random.choice(NAMES_PLACE)

# -----------------------
# Semantics helpers
# -----------------------

def norm_transitivity_set(s: str) -> set[str]:
    toks = re.split(r"[;,/| ]+", utils._s(s))
    valid = {"intr", "tr", "ditr"}
    return {t for t in (utils._s(x) for x in toks) if t in valid}

def enrich_noun_semantics(df):
    """
    Enrich noun semantics + normalize key categorical fields.
    """
    if df is None or df.empty:
        return df

    # Normalize 'class' and a few booleans up front to make downstream matching robust
    if "class" in df.columns:
        df["class"] = df["class"].astype(str).str.strip().str.lower()

    def _is(cls):
        return utils._s(cls)

def derive_possessability_from_class(noun_class: str) -> str:
    """Return 'obligatory', 'non-possessable', or 'optional' based on noun class."""
    cls = utils._s(noun_class)
    if cls in OBLIGATORY_POSSESSED_CLASSES:
        return "obligatory"
    if cls in NON_POSSESSABLE_CLASSES:
        return "non-possessable"
    return "optional"

def normalize_countability(val: str) -> str:
    """
    Normalize CSV values ('count', 'mass', 'plural') into
    'countable', 'uncountable', 'plural-only'.
    """
    v = utils._s(val).strip().lower()
    if not v:
        return v

    if v in {"count", "countable", "c"}:
        return "countable"
    if v in {"mass", "uncountable", "unc", "noncount", "non-count"}:
        return "uncountable"
    if v in {"plural", "plural-only", "plurale tantum", "plurale-tantum", "pl-only"}:
        return "plural-only"

    return v

def enrich_noun_semantics(df):
    """
    Normalize and enrich noun semantics for all languages.
    """
    if df is None or df.empty:
        return df

    # Normalize class
    if "class" in df.columns:
        df["class"] = (
            df["class"].astype(str).str.strip().str.lower()
        )

    # Possessability
    df["possessability"] = df["class"].apply(derive_possessability_from_class)

    # Human & color properties
    df["is_human"] = df.apply(
        lambda r: "1" if utils._s(r.get("class")) in {"human","kinship","occupation"} 
                  else (r.get("is_human") or "0"),
        axis=1
    )
    df["has_color"] = df.apply(
        lambda r: "1" if utils._s(r.get("class")) in {"artifact","building","clothing","food","tool","vehicle"} 
                  else (r.get("has_color") or "0"),
        axis=1
    )

    # Normalize countability for each language
    for suffix in ("kek", "en", "es"):
        col = f"countability_{suffix}"
        if col in df.columns:
            df[col] = df[col].apply(normalize_countability)

    # Normalize loan flag
    df["loan"] = df["loan"].apply(
        lambda x: "1" if utils._s(x) in {"1","true","yes"} else "0"
    )

    return df

def noun_class(noun_row: Dict) -> str:
    return utils._s((noun_row or {}).get("class")).strip()

def class_in(noun_cls: str, req_list: List[str]) -> bool:
    req = {utils._s(x) for x in (req_list or []) if utils._s(x)}
    return noun_cls in req

def verb_arg_constraints(vrow: Optional[Dict]) -> Dict[str, set]:
    """
    Normalize constraint sets to lowercase/trim so they match normalized noun classes.

    Canonical source columns (from semantics_verbs.csv):

        agent_class_any_of   → agent_allow
        agent_class_none_of  → agent_ban
        theme_class_any_of   → theme_allow
        theme_class_none_of  → theme_ban
        goal_class_any_of    → goal_allow
        goal_none_of         → goal_ban
    """
    empty = {
        "agent_allow": set(), "agent_ban": set(),
        "theme_allow": set(), "theme_ban": set(),
        "goal_allow":  set(), "goal_ban":  set(),
    }
    if not vrow:
        return empty

    def _to_set(val):
        raw = [utils._s(x) for x in re.split(r"[;,/| ]+", utils._s(val)) if x]
        return {x.strip().lower() for x in raw if x.strip()}

    return {
        # Agent
        "agent_allow": _to_set(vrow.get("agent_class_any_of")),
        "agent_ban":   _to_set(vrow.get("agent_class_none_of")),
        # Theme
        "theme_allow": _to_set(vrow.get("theme_class_any_of")),
        "theme_ban":   _to_set(vrow.get("theme_class_none_of")),
        # Goal
        "goal_allow":  _to_set(vrow.get("goal_class_any_of")),
        "goal_ban":    _to_set(vrow.get("goal_class_none_of") or vrow.get("goal_none_any_of")),
    }

def adj_compatible_with_noun(adj_row: Dict, noun_row: Dict, noun_class: Optional[str] = None) -> bool:
    if adj_row is None or noun_row is None:
        return False
    aget = adj_row.get if hasattr(adj_row, "get") else (lambda k, d=None: adj_row[k] if k in adj_row else d)
    nget = noun_row.get if hasattr(noun_row, "get") else (lambda k, d=None: noun_row[k] if k in noun_row else d)

    allowed = {x.strip().lower() for x in utils._split_list(utils._s(aget("class_any_of")))}
    banned  = {x.strip().lower() for x in utils._split_list(utils._s(aget("class_none_of")))}
    ncls = (noun_class or utils._s(nget("class"))).strip().lower()

    if allowed and ncls not in allowed:
        return False
    if banned and ncls in banned:
        return False

    at = utils._s(aget("adj_type")).strip().lower()
    has_color_norm = utils._s(nget("has_color")).strip().lower() in {"1","true","yes"}
    if at == "color" and not has_color_norm:
        return False
    return True

def adverb_ok(row: dict, tense_flags: dict, verb_category: str) -> bool:
    """
    Decide whether an adverb is compatible with a given verb context.
    Accepts multiple verb categories separated by | or ; and compares loosely.
    """
    # Normalize categories from verbs (may contain several, e.g. "v_action|v_conflict")
    vcat_parts = {
        p.strip().lower()
        for p in re.split(r"[|;/, ]+", (verb_category or ""))
        if p.strip()
    }

    # --- Tense compatibility ---
    def _norm_tense_key(x: str) -> str:
        t = (x or "").strip().lower()
        return {
            "present": "prs", "past": "pst", "future": "fut",
            "progressive": "prg", "continuous": "prg",
            "perfect": "prf",
        }.get(t, t)

    tense_key = "allow_tense" if row.get("allow_tense") else ("tense_compat" if row.get("tense_compat") else None)
    if tense_key and tense_flags:
        allowed_raw = [x for x in re.split(r"[;,/| ]+", utils._s(row[tense_key])) if x]
        allowed = {_norm_tense_key(x) for x in allowed_raw}
        if allowed and not any(tense_flags.get(t) for t in allowed):
            return False

    # --- Category matching ---
    ok = set()
    if row.get("class_any_of"):
        ok = {x.strip().lower() for x in re.split(r"[;,/| ]+", utils._s(row["class_any_of"])) if x}
        # must match at least one
        if vcat_parts and not (vcat_parts & ok):
            return False

    if row.get("class_none_of"):
        ban = {x.strip().lower() for x in re.split(r"[;,/| ]+", utils._s(row["class_none_of"])) if x}
        # must not overlap any banned
        if vcat_parts & ban:
            return False

    return True