"""Spanish-specific constants and morphology (incl. reflexives and periphrastic future)."""

from typing import Dict, Tuple
import re
import math # Needed for _s() NaN check
from .linguistics_core import PERSONS

# ------------------
# Generic utilities
# ------------------

def _s(val) -> str:
    """NaN/None-safe string: returns '' for None/NaN, else str(val).strip()."""
    if val is None:
        return ""
    if isinstance(val, float) and math.isnan(val):
        return ""
    return str(val).strip()

# -----------------
# Placeholders
# -----------------

DEFAULT_PLACEHOLDERS_ES: Dict[str, str] = {
    "WH_ES": "Por qué",
    "WH_LOC_ES": "Dónde",
    "WH_TIME_ES": "Cuándo",
    "BENEF_REL_ES": "",
    "GOAL_REL_ES": "a",
    "EXIST_ES": "hay",
    "EXIST_PL_ES": "hay",
    "EXIST_NEG_ES": "No hay",
    "EXIST_NEG_PL_ES": "No hay",
}

# --- Determiners and possessives (used for adjective placement, article agreement, etc.) ---
ES_DETERMINERS = {
    "el", "la", "los", "las",
    "un", "una", "unos", "unas",
    "mi", "mis", "tu", "tus", "su", "sus",
    "nuestro", "nuestra", "nuestros", "nuestras",
    "vuestro", "vuestra", "vuestros", "vuestras"
}

# --- Pronouns
ES_SUBJ_PRONOUNS = {"1sg":"yo","2sg":"tú","3sg":"él","1pl":"nosotros","2pl":"ustedes","3pl":"ellos"}
ES_POSSESSIVE   = {"1sg":"mi","2sg":"tu","3sg":"su","1pl":"nuestro","2pl":"su","3pl":"su"}
_ES_REFLEXIVE_PRON = {"1sg": "me","2sg": "te","3sg": "se","1pl": "nos","2pl": "os","3pl": "se"}

# Gendered realizations
ES_3SG_PRONOUN_BY_GENDER = {"m": "él", "f": "ella"}
ES_3PL_PRONOUN_BY_GENDER = {"m": "ellos", "f": "ellas"}
ES_3SG_AGENT_BY_GENDER   = {"m": "por él", "f": "por ella"}
ES_3PL_AGENT_BY_GENDER   = {"m": "por ellos", "f": "por ellas"}

# Agent phrase “por X” for passive constructions
AGENT_BY_ES = {"1sg":"por mí", "2sg":"por ti", "3sg":"por él", "1pl":"por nosotros", "2pl":"por ustedes", "3pl":"por ellos"}

# --- Verbs
ES_TU_IMP_IRREG = {
    "venir": "ven", "hacer": "haz", "decir": "di", "poner": "pon",
    "salir": "sal", "tener": "ten", "ir": "ve", "ser": "sé",
}

ES_PRESENT_ENDINGS = {
    "ar": {"1sg":"o","2sg":"as","3sg":"a","1pl":"amos","2pl":"áis","3pl":"an"},
    "er": {"1sg":"o","2sg":"es","3sg":"e","1pl":"emos","2pl":"éis","3pl":"en"},
    "ir": {"1sg":"o","2sg":"es","3sg":"e","1pl":"imos","2pl":"ís","3pl":"en"},
}
ES_PRESENT_IRREG = {
    "caber":{"1sg":"quepo","2sg":"cabes","3sg":"cabe","1pl":"cabemos","2pl":"cabéis","3pl":"caben"},
    "caer":{"1sg":"caigo","2sg":"caes","3sg":"cae","1pl":"caemos","2pl":"caéis","3pl":"caen"},
    "conducir":{"1sg":"conduzco","2sg":"conduces","3sg":"conduce","1pl":"conducimos","2pl":"conducís","3pl":"conducen"},
    "conocer":{"1sg":"conozco","2sg":"conoces","3sg":"conoce","1pl":"conocemos","2pl":"conocéis","3pl":"conocen"},
    "dar":{"1sg":"doy","2sg":"das","3sg":"da","1pl":"damos","2pl":"dais","3pl":"dan"},
    "decir":{"1sg":"digo","2sg":"dices","3sg":"dice","1pl":"decimos","2pl":"decís","3pl":"dicen"},
    "dormir":{"1sg":"duermo","2sg":"duermes","3sg":"duerme","1pl":"dormimos","2pl":"dormís","3pl":"duermen"},
    "estar":{"1sg":"estoy","2sg":"estás","3sg":"está","1pl":"estamos","2pl":"estáis","3pl":"están"},
    "hacer":{"1sg":"hago","2sg":"haces","3sg":"hace","1pl":"hacemos","2pl":"hacéis","3pl":"hacen"},
    "haber":{"1sg":"he","2sg":"has","3sg":"ha","1pl":"hemos","2pl":"habéis","3pl":"han"},
    "ir":{"1sg":"voy","2sg":"vas","3sg":"va","1pl":"vamos","2pl":"vais","3pl":"van"},
    "jugar":{"1sg":"juego","2sg":"juegas","3sg":"juega","1pl":"jugamos","2pl":"jugáis","3pl":"juegan"},
    "oír":{"1sg":"oigo","2sg":"oyes","3sg":"oye","1pl":"oímos","2pl":"oís","3pl":"oyen"},
    "pedir":{"1sg":"pido","2sg":"pides","3sg":"pide","1pl":"pedimos","2pl":"pedís","3pl":"piden"},
    "pensar":{"1sg":"pienso","2sg":"piensas","3sg":"piensa","1pl":"pensamos","2pl":"pensáis","3pl":"piensan"},
    "poder":{"1sg":"puedo","2sg":"puedes","3sg":"puede","1pl":"podemos","2pl":"podéis","3pl":"pueden"},
    "poner":{"1sg":"pongo","2sg":"pones","3sg":"pone","1pl":"ponemos","2pl":"ponéis","3pl":"ponen"},
    "querer":{"1sg":"quiero","2sg":"quieres","3sg":"quiere","1pl":"queremos","2pl":"queréis","3pl":"quieren"},
    "salir":{"1sg":"salgo","2sg":"sales","3sg":"sale","1pl":"salimos","2pl":"salís","3pl":"salen"},
    "saber":{"1sg":"sé","2sg":"sabes","3sg":"sabe","1pl":"sabemos","2pl":"sabéis","3pl":"saben"},
    "ser":{"1sg":"soy","2sg":"eres","3sg":"es","1pl":"somos","2pl":"sois","3pl":"son"},
    "tener":{"1sg":"tengo","2sg":"tienes","3sg":"tiene","1pl":"tenemos","2pl":"tenéis","3pl":"tienen"},
    "traer":{"1sg":"traigo","2sg":"traes","3sg":"trae","1pl":"traemos","2pl":"traéis","3pl":"traen"},
    "valer":{"1sg":"valgo","2sg":"vales","3sg":"vale","1pl":"valemos","2pl":"valéis","3pl":"valen"},
    "venir":{"1sg":"vengo","2sg":"vienes","3sg":"viene","1pl":"venimos","2pl":"venís","3pl":"vienen"},
    "ver":{"1sg":"veo","2sg":"ves","3sg":"ve","1pl":"vemos","2pl":"veis","3pl":"ven"},
}
ES_GERUND_IRREG = {"ir":"yendo","poder":"pudiendo","leer":"leyendo"}

ES_PP_IRREG     = {"abrir":"abierto","cubrir":"cubierto","decir":"dicho","escribir":"escrito","hacer":"hecho","morir":"muerto","poner":"puesto","romper":"roto","ver":"visto","volver":"vuelto"}

_ES_PRET_ENDINGS = {
    "ar": {"1sg":"é","2sg":"aste","3sg":"ó","1pl":"amos","2pl":"asteis","3pl":"aron"},
    "er": {"1sg":"í","2sg":"iste","3sg":"ió","1pl":"imos","2pl":"isteis","3pl":"ieron"},
    "ir": {"1sg":"í","2sg":"iste","3sg":"ió","1pl":"imos","2pl":"isteis","3pl":"ieron"},
}

_ES_PRET_IRREG = {
    "andar":{"1sg":"anduve","2sg":"anduviste","3sg":"anduvo","1pl":"anduvimos","2pl":"anduvisteis","3pl":"anduvieron"},
    "caber":{"1sg":"cupe","2sg":"cupiste","3sg":"cupo","1pl":"cupimos","2pl":"cupisteis","3pl":"cupieron"},
    "dar":{"1sg":"di","2sg":"diste","3sg":"dio","1pl":"dimos","2pl":"disteis","3pl":"dieron"},
    "decir":{"1sg":"dije","2sg":"dijiste","3sg":"dijo","1pl":"dijimos","2pl":"dijisteis","3pl":"dijeron"},
    "hacer":{"1sg":"hice","2sg":"hiciste","3sg":"hizo","1pl":"hicimos","2pl":"hicisteis","3pl":"hicieron"},
    "haber":{"1sg":"hube","2sg":"hubiste","3sg":"hubo","1pl":"hubimos","2pl":"hubisteis","3pl":"hubieron"},
    "ir":{"1sg":"fui","2sg":"fuiste","3sg":"fue","1pl":"fuimos","2pl":"fuisteis","3pl":"fueron"},
    "poner":{"1sg":"puse","2sg":"pusiste","3sg":"puso","1pl":"pusimos","2pl":"pusisteis","3pl":"pusieron"},
    "poder":{"1sg":"pude","2sg":"pudiste","3sg":"pudo","1pl":"pudimos","2pl":"pudisteis","3pl":"pudieron"},
    "querer":{"1sg":"quise","2sg":"quisiste","3sg":"quiso","1pl":"quisimos","2pl":"quisisteis","3pl":"quisieron"},
    "saber":{"1sg":"supe","2sg":"supiste","3sg":"supo","1pl":"supimos","2pl":"supisteis","3pl":"supieron"},
    "ser":{"1sg":"fui","2sg":"fuiste","3sg":"fue","1pl":"fuimos","2pl":"fuisteis","3pl":"fueron"},
    "tener":{"1sg":"tuve","2sg":"tuviste","3sg":"tuvo","1pl":"tuvimos","2pl":"tuvisteis","3pl":"tuvieron"},
    "traer":{"1sg":"traje","2sg":"trajiste","3sg":"trajo","1pl":"trajimos","2pl":"trajisteis","3pl":"trajeron"},
    "venir":{"1sg":"vine","2sg":"viniste","3sg":"vino","1pl":"vinimos","2pl":"vinisteis","3pl":"vinieron"},
    "ver":{"1sg":"vi","2sg":"viste","3sg":"vio","1pl":"vimos","2pl":"visteis","3pl":"vieron"},

    # -ducir / -ducir family (j-preterite)
    "conducir":{"1sg":"conduje","2sg":"condujiste","3sg":"condujo","1pl":"condujimos","2pl":"condujisteis","3pl":"condujeron"},
    "introducir":{"1sg":"introduje","2sg":"introdujiste","3sg":"introdujo","1pl":"introdujimos","2pl":"introdujisteis","3pl":"introdujeron"},
    "producir":{"1sg":"produje","2sg":"produjiste","3sg":"produjo","1pl":"produjimos","2pl":"produjisteis","3pl":"produjeron"},
    "reducir":{"1sg":"reduje","2sg":"redujiste","3sg":"redujo","1pl":"redujimos","2pl":"redujisteis","3pl":"redujeron"},
    "traducir":{"1sg":"traduje","2sg":"tradujiste","3sg":"tradujo","1pl":"tradujimos","2pl":"tradujisteis","3pl":"tradujeron"},

    # Stem-changer that is common in simple corpora
    "vestir":{"1sg":"vestí","2sg":"vestiste","3sg":"vistió","1pl":"vestimos","2pl":"vestisteis","3pl":"vistieron"},
}

_ES_SUBJ_ENDINGS = {
    "ar": {"1sg":"e","2sg":"es","3sg":"e","1pl":"emos","2pl":"éis","3pl":"en"},
    "er": {"1sg":"a","2sg":"as","3sg":"a","1pl":"amos","2pl":"áis","3pl":"an"},
    "ir": {"1sg":"a","2sg":"as","3sg":"a","1pl":"amos","2pl":"áis","3pl":"an"},
}
_ES_SUBJ_PRESENT_IRREG = {
    "ser":{"1sg":"sea","2sg":"seas","3sg":"sea","1pl":"seamos","2pl":"seáis","3pl":"sean"},
    "ir":{"1sg":"vaya","2sg":"vayas","3sg":"vaya","1pl":"vayamos","2pl":"vayáis","3pl":"vayan"},
    "estar":{"1sg":"esté","2sg":"estés","3sg":"esté","1pl":"estemos","2pl":"estéis","3pl":"estén"},
    "haber":{"1sg":"haya","2sg":"hayas","3sg":"haya","1pl":"hayamos","2pl":"hayáis","3pl":"hayan"},
    "saber":{"1sg":"sepa","2sg":"sepas","3sg":"sepa","1pl":"sepamos","2pl":"sepáis","3pl":"sepan"},
    "dar":{"1sg":"dé","2sg":"des","3sg":"dé","1pl":"demos","2pl":"deis","3pl":"den"},
}

# ----------------------
# Normalization helpers
# ----------------------

def _es_norm_person(person: str) -> str:
    """"
    Normalizes a person code to the canonical inventory (1sg…3pl), defaulting to 3sg.
    """
    p = (person or "3sg").lower().strip()
    return p if p in PERSONS else "3sg"

def _es_norm_person_label_or_code(person: str) -> str:
    """
    Accepts Spanish labels (tu/usted/ustedes/vosotros/nosotros) or PERSONS codes.
    """
    lab = (person or "").strip().lower()
    label_map = {
        "yo": "1sg", "tú": "2sg", "tu": "2sg", "él": "3sg", "ella": "3sg",
        "nosotros": "1pl", "vosotros": "2pl", "ustedes": "3pl", "usted": "3sg",
    }
    if lab in label_map:
        return label_map[lab]
    return _es_norm_person(lab)

def _es_person_for_conjugation(person: str) -> str:
    """
    Returns the person used for verb endings. Project convention:
    for Guatemala, map 2pl to 3pl to model ‘ustedes’ rather than ‘vosotros’ endings.
    """
    p = _es_norm_person(person)
    return "3pl" if p == "2pl" else p

def _es_norm_inf(inf: str) -> tuple[str, str | None]:
    """
    Normalizes the infinitive and return its conjugation class (ar/er/ir) if detectable.
    """
    low = (inf or "").strip().lower()
    if not low: return "", None
    m = re.search(r"(ar|er|ir)$", low)
    return low, (m.group(1) if m else None)

def _split_reflexive(inf: str) -> tuple[str, bool]:
    """
    Detects reflexive infinitives ending in -se.
    """
    s = (inf or "").strip().lower()
    if s.endswith("se") and len(s) > 3: return s[:-2], True
    return s, False

def _reflexive_pronoun(person: str) -> str:
    return _ES_REFLEXIVE_PRON.get(person, "se")

def _es_pick_gendered_form(gender: str | None, rng, by_gender: dict[str, str]) -> str | None:
    """
    Pick a gendered Spanish form (m/f) from `by_gender`.

    If `gender` is not provided or invalid and `rng` is provided, a random gender is chosen.
    Returns None if no valid gender can be resolved.
    """
    g = (gender or "").strip().lower() or None
    if g not in {"m", "f"} and rng is not None:
        g = rng.choice(["m", "f"])
    if g in {"m", "f"}:
        return by_gender[g]
    return None

# ------------------
# Verb conjugation
# -----------------

def _es_apply_regular(low: str, conj_class: str, endings: dict, person: str | None = None) -> str:
    root = low[:-2]
    return root + (endings[conj_class][person] if person else endings[conj_class])

def _es_pick_irregular(irr_map: dict, low: str, person: str | None = None) -> str | None:
    """
    Looks up irregular verb forms.
    """
    irr = irr_map.get(low)
    if irr is None: return None
    if isinstance(irr, dict): return irr.get(person or "3sg") or irr.get("3sg")
    return irr

def _es_make_nonfinite(inf: str, irr_map: dict, endings_map: dict) -> str:
    """
    Shared helper for gerund/participle: strip reflexive, honor irregulars, else apply class suffix.
    """
    base, _ = _split_reflexive(inf)
    low, cls = _es_norm_inf(base)
    if not low:
        return low
    irr = _es_pick_irregular(irr_map, low)
    if irr:
        return irr
    if not cls:
        return (base or "").strip()
    return _es_apply_regular(low, cls, endings_map)

def _es_conjugate(
    inf: str,
    person: str,
    irr_map: dict,
    endings: dict,
    *,
    orth_fix_1sg=None,
    attach_reflexive_to: str = "finite"  # "finite" | "infinitive" | "none"
) -> str:
    base, is_refl = _split_reflexive(inf)
    low, cls = _es_norm_inf(base)
    if not low:
        return low

    # p = person used for finite endings (2pl→3pl mapping for standard Latin American usage)
    p_finite = _es_person_for_conjugation(person)
    # p_pron = person used for reflexive clitics (NEVER remap 2pl to 3pl)
    p_pron   = _es_norm_person(person)

    # Reflexive attached to infinitive? (e.g., "vas a oxidarte")
    if attach_reflexive_to == "infinitive":
        if is_refl:
            pron = _reflexive_pronoun(p_pron)
            return f"{base}{pron}"
        return base

    irr = _es_pick_irregular(irr_map, low, p_finite)
    if irr:
        form = irr
    elif not cls:
        form = (base or "").strip()
    else:
        if p_finite == "1sg" and orth_fix_1sg:
            orth = orth_fix_1sg(low)
            form = orth if orth else _es_apply_regular(low, cls, endings, p_finite)
        else:
            form = _es_apply_regular(low, cls, endings, p_finite)

    if not is_refl or attach_reflexive_to == "none":
        return form

    # Attach correct reflexive pronoun based on the original person (keeps "os" for 2pl)
    pron = _reflexive_pronoun(p_pron)
    return f"{pron} {form}"

# ---------------------------
# Noun/noun phrase building
# ---------------------------

def _es_det(definite: bool, plural: bool, gloss_es: str, gender: str|None=None) -> str:
    """
    Returns an article + noun phrase for gloss_es, selecting definite/indefinite and number,
    and using gender (m/f) to choose the correct article.
    """
    g = (gender or 'm').lower()
    if definite:
        art = ('los' if g=='m' else 'las') if plural else ('el' if g=='m' else 'la')
        return f"{art} {gloss_es}"
    else:
        art = ('unos' if g=='m' else 'unas') if plural else ('un' if g=='m' else 'una')
        return f"{art} {gloss_es}"

def pluralize_es(word: str) -> str:
    w = (word or "").strip()
    if not w: return w
    lw = w.lower()
    if re.search(r"[aeiouáéíóú]$", lw): return w + "s"
    if re.search(r"z$", lw): return w[:-1] + "ces"
    return w + "es"

def _noun_es_bare(noun: Dict, plural: bool) -> str:
    base = _s(noun.get("gloss_es"))
    if not base: 
        return base
    status = _s(noun.get("countability_es")).lower()
    if plural and status == "countable":
        return pluralize_es(base)
    return base

def build_np_es(
    noun: Dict,
    *,
    plural: bool = False,
    definite: bool = True,
    possessed_person: str | None = None,
    as_subject: bool = False,         # kept for call-shape parity
    ref_gender: str | None = None,    # NEW: allow callers to request feminine agreement
) -> str:
    return build_np_pluralized_es(
        noun,
        plural=plural,
        definite=definite,
        possessed_person=possessed_person,
        ref_gender=ref_gender,
    )

def build_np_pluralized_es(
    noun: Dict, *,
    plural: bool=False,
    definite: bool=True,
    possessed_person: str|None=None,
    ref_gender: str|None=None,   # lets callers force feminine agreement for occupations
) -> str:
    # Choose feminine surface if requested and available (e.g., 'traductora')
    base_m = _s(noun.get('gloss_es'))
    base_f = _s(noun.get('gloss_es_f'))
    use_f  = (ref_gender or '').lower() == 'f'
    base   = (base_f if (use_f and base_f) else base_m)

    # Article gender: prefer ref_gender if given; else fall back to noun’s gender_es
    _g_src = _s(ref_gender) or _s(noun.get('gender_es')) or 'm'
    gender = _g_src.lower()

    status = _s(noun.get("countability_es")).lower()
    plural_only = (status == "plural-only")
    uncount = (status == "uncountable")

    if possessed_person:
        # Surface form of the possessed noun
        form = (pluralize_es(base) if (plural and status == "countable") else base)

        # Use centralized helper for possessive determiner agreement
        poss = possessive_det_es(
            possessed_person,
            plural=plural,
            gender=gender,
        )
        return f"{poss} {form}".strip()

    if plural and status == "countable":
        base = pluralize_es(base)

    if not definite and (uncount or plural_only):
        return base

    return _es_det(definite, (plural and status == "countable"), base, gender=gender)


# --------------------
# Pronouns
# --------------------

def es_subj_pronoun(person: str = "3sg", gender: str | None = None, rng=None) -> str:
    """
    Spanish subject pronoun with optional gender control.
    - For 3sg: él/ella
    - For 3pl: ellos/ellas (if gender not provided and rng is provided, randomize)
    """
    p = (person or "3sg").strip().lower()

    if p == "3sg":
        picked = _es_pick_gendered_form(gender, rng, ES_3SG_PRONOUN_BY_GENDER)
        return picked if picked is not None else ES_SUBJ_PRONOUNS["3sg"]

    if p == "3pl":
        picked = _es_pick_gendered_form(gender, rng, ES_3PL_PRONOUN_BY_GENDER)
        return picked if picked is not None else ES_SUBJ_PRONOUNS["3pl"]

    return ES_SUBJ_PRONOUNS.get(p, ES_SUBJ_PRONOUNS["3sg"])

def possessive_det_es(
    possessed_person: str,
    *,
    plural: bool,
    gender: str | None = None
) -> str:
    """
    Returns the properly inflected Spanish possessive determiner
    (mi/mis, tu/tus, su/sus, nuestro/a/os/as, vuestro/a/os/as)
    based on:
      - possessor person code (1sg, 2sg, 3sg, 1pl, 2pl, 3pl),
      - number of the possessed noun (plural),
      - gender of the possessed noun (m/f).
    """
    person = (possessed_person or "").strip().lower()
    base = ES_POSSESSIVE.get(person, "mi")
    g = (gender or "m").lower()

    # Simple series: mi, tu, su → mi/mis, tu/tus, su/sus
    if base in {"mi", "tu", "su"}:
        return base + ("s" if plural else "")

    # nuestro / nuestra / nuestros / nuestras
    if base.startswith("nuestro") or base.startswith("nuestra"):
        if g == "f":
            return "nuestras" if plural else "nuestra"
        return "nuestros" if plural else "nuestro"

    # vuestro / vuestra / vuestros / vuestras
    if base.startswith("vuestro") or base.startswith("vuestra"):
        if g == "f":
            return "vuestras" if plural else "vuestra"
        return "vuestros" if plural else "vuestro"

    # Fallback: if plural and base doesn't already end in -s, add it.
    if plural and not base.endswith("s"):
        return base + "s"
    return base

def es_agent_by(person: str = "3sg", gender: str | None = None, rng=None) -> str:
    """
    Return the Spanish agent phrase for passive.
    - For 3sg: por él / por ella
    - For 3pl: por ellos / por ellas
    """
    p = (person or "3sg").strip().lower()

    if p == "3sg":
        picked = _es_pick_gendered_form(gender, rng, ES_3SG_AGENT_BY_GENDER)
        return picked if picked is not None else AGENT_BY_ES["3sg"]

    if p == "3pl":
        picked = _es_pick_gendered_form(gender, rng, ES_3PL_AGENT_BY_GENDER)
        return picked if picked is not None else AGENT_BY_ES["3pl"]

    return AGENT_BY_ES.get(p, AGENT_BY_ES["3sg"])

# ------------
# Verb forms
# ------------

def pick_copula_es(adj_row: Dict, is_plural: bool) -> str:
    """
    Returns the copula *type* ('ser' or 'estar') based on adjective, not a conjugated surface form.
    Conjugation is done by es_present(..., person) in the generator.
    """
    rule = (adj_row.get("copula_es") or "ser").strip().lower()
    if rule not in {"ser", "estar", "either"}:
        rule = "ser"

    # Deterministic default for 'either': choose 'ser' (stative/classification)
    # Location and other special contexts are handled elsewhere.
    return "estar" if rule == "estar" else "ser"

def pick_copula_neg_es(adj_row: Dict, person: str, is_plural: bool) -> str:
    """
    Negative copula surface: 'no' + conjugated COP_ES.
    """
    cop_type = pick_copula_es(adj_row, is_plural)
    return f"no {es_present(cop_type, person)}"

def spanish_copula_for_location(noun_class: str) -> str:
    """
    Determines whether Spanish should use 'ser' or 'estar'
    in 'Where' (¿Dónde...?) location questions based on noun class.

    Parameters
    ----------
    noun_class : str
        The semantic class from the noun CSV (row['class']).

    Returns
    -------
    str
        "ser" or "estar"
    """
    if not noun_class:
        # Default fallback: ESTAR is safer for physical location
        return "estar"

    cls = noun_class.strip().lower()

    # Classes that use ESTAR for location (physical entities, people, places)
    estar_classes = {
        "animal_domesticated", "animal_wild", "artifact", "beverage", "body_part",
        "building", "clothing", "container", "corn", "food", "fruit", "human",
        "insect", "kinship", "person", "plant", "profession", "place", "substance",
        "tool", "vegetable", "vehicle", "wood",
    }

    # Events take SER in 'where' questions (¿Dónde es...?)
    ser_classes = {"event"}

    if cls in ser_classes:
        return "ser"
    if cls in estar_classes:
        return "estar"

    # Fallback: ESTAR remains the safer choice for unspecified classes
    return "estar"

def _es_pres_orth_1sg(low: str) -> str | None:
    """
    Minimal orthographic present-1sg fixes (yo-form), used as a verb-class layer:
    - -cer / -cir -> -zco  (conocer -> conozco, desaparecer -> desaparezco)
    - -ger / -gir -> -jo   (proteger -> protejo, dirigir -> dirijo)
    """
    if low.endswith(("cer", "cir")) and len(low) > 3:
        # root = low[:-2] (e.g. desaparec), swap final c -> zc, then + o
        root = low[:-2]
        if root.endswith("c"):
            return root[:-1] + "zc" + "o"
    if low.endswith(("ger", "gir")) and len(low) > 3:
        root = low[:-2]
        if root.endswith("g"):
            return root[:-1] + "j" + "o"
    return None

def es_present(inf: str, person: str = "3sg") -> str:
    return _es_conjugate(
        inf,
        person,
        ES_PRESENT_IRREG,
        ES_PRESENT_ENDINGS,
        orth_fix_1sg=_es_pres_orth_1sg,
        attach_reflexive_to="finite",
    )

def _es_pret_orth_1sg(low: str) -> str | None:
    """
    Preterite 1sg orthographic override builder.

    Called only when person == 1sg inside _es_conjugate().
    Must return the FULL 1sg form, or None to fall back to regular endings.
    """
    if not low:
        return None
    low = low.strip().lower()

    # -car -> -qué (buscar -> busqué)
    if low.endswith("car"):
        return low[:-3] + "qué"
    # -gar -> -gué (llegar -> llegué)
    if low.endswith("gar"):
        return low[:-3] + "gué"
    # -zar -> -cé (empezar -> empecé)
    if low.endswith("zar"):
        return low[:-3] + "cé"

    return None

def es_past(inf: str, person: str = "3sg") -> str:
    return _es_conjugate(inf, person, _ES_PRET_IRREG, _ES_PRET_ENDINGS, orth_fix_1sg=_es_pret_orth_1sg, attach_reflexive_to="finite")

def es_gerund(inf: str) -> str:
    return _es_make_nonfinite(inf, ES_GERUND_IRREG, {"ar": "ando", "er": "iendo", "ir": "iendo"})

def es_pp(inf: str) -> str:
    return _es_make_nonfinite(inf, ES_PP_IRREG, {"ar": "ado", "er": "ido", "ir": "ido"})

# Aux wrapper, can be extended later
def es_aux_present(aux: str, person: str = "3sg") -> str:
    return es_present(aux, person)

# Back-compat wrappers (no change to callers)
def es_ir_present(person: str = "3sg") -> str:
    return es_aux_present("ir", person)

def es_estar_present(person: str = "3sg") -> str:
    return es_aux_present("estar", person)

def es_future_periphrastic(inf: str, person: str = "3sg") -> str:
    p = _es_person_for_conjugation(person)
    ir = es_ir_present(p)
    inf_or_infcl = _es_conjugate(inf, person, irr_map={}, endings={}, orth_fix_1sg=None, attach_reflexive_to="infinitive")
    return f"{ir} a {inf_or_infcl}"

def es_progressive(inf: str, person: str = "3sg") -> str:
    p = _es_norm_person(person)
    base, is_refl = _split_reflexive(inf)
    estar = es_estar_present(p)
    ger = es_gerund(base)
    if is_refl:
        pron = _reflexive_pronoun(p)
        return f"{pron} {estar} {ger}"
    return f"{estar} {ger}"

def es_future_progressive(inf: str, person: str = "3sg") -> str:
    p = _es_norm_person(person)

    base, is_refl = _split_reflexive(inf)

    ir = es_ir_present(p)
    estar = es_estar_present(p)
    ger = es_gerund(base)

    if is_refl:
        pron = _reflexive_pronoun(p)
        return f"{pron} {ir} a {estar} {ger}"

    return f"{ir} a {estar} {ger}"

def es_subj_present(inf: str, person: str = "3sg") -> str:
    """
    Very compact present-subjunctive generator:
    - Handles a tiny set of high-value irregulars.
    - Else derives the subjunctive stem from the 1sg-present (yo-form),
      which automatically captures orthographic classes (-zco, -jo, etc.).
    """
    p = _es_norm_person(person)
    raw = (inf or "").strip().lower()
    base = raw[:-2] if raw.endswith("se") else raw

    low, cls = _es_norm_inf(base)
    if not low:
        return low

    irr = _ES_SUBJ_PRESENT_IRREG.get(low)
    if irr:
        return irr.get(p, irr.get("3sg"))

    if not cls:
        return (inf or "").strip()

    # Build stem from yo-form present (without reflexive attachment)
    yo = _es_conjugate(
        base,
        "1sg",
        ES_PRESENT_IRREG,
        ES_PRESENT_ENDINGS,
        orth_fix_1sg=_es_pres_orth_1sg,
        attach_reflexive_to="none",
    )
    stem = yo[:-1] if yo.endswith("o") and len(yo) > 1 else low[:-2]

    end = _ES_SUBJ_ENDINGS.get(cls, {}).get(p, "")
    return stem + end

def es_imperative(inf: str, person: str = "2sg", negative: bool = False) -> str:
    """
    Unified Spanish imperative builder.
    - `person` may be a canonical code ('2sg', '3pl', ...) or a Spanish label
      ('tú', 'usted', 'ustedes', 'nosotros', ...); all are normalized via
      _es_norm_person_label_or_code.
    - Works for both plain and reflexive infinitives:
        oxidar   → habla/ no hables pattern
        oxidarse → oxídate / no te oxides pattern (without accent logic)
    - Negative imperatives use present subjunctive:
        no hables, no coma, no vivan
        no te oxides, no se oxide, no se oxiden
    """
    canon = _es_norm_person_label_or_code(person)

    raw = (inf or "").strip().lower()
    is_refl = raw.endswith("se")
    base_inf = raw[:-2] if is_refl else raw  # strip trailing 'se' if reflexive

    # Negative imperatives
    if negative:
        # Present subjunctive built from the non-reflexive infinitive
        subj = es_subj_present(base_inf, canon)
        if not is_refl:
            # Non-reflexive: no + subj
            return f"no {subj}"
        # Reflexive: no + clitic + subj
        pron = _ES_REFLEXIVE_PRON.get(canon, "se")
        return f"no {pron} {subj}"

    # ----- Affirmative imperatives -----
    low, cls = _es_norm_inf(base_inf)
    if not low:
        return low

    # Base (non-reflexive) imperative form for this person
    if canon == "2sg":
        # Affirmative tú imperatives (irregular list, else root+a/e)
        if low in ES_TU_IMP_IRREG:
            form = ES_TU_IMP_IRREG[low]
        elif cls in {"ar", "er", "ir"}:
            root = low[:-2]
            form = root + ("a" if cls == "ar" else "e")
        else:
            form = low
    elif canon in {"3sg", "3pl", "1pl"}:
        # Usted / Ustedes / Nosotros → present subjunctive
        form = es_subj_present(base_inf, canon)
    else:
        # Safety fallback (also used if you ever pass 1sg/2pl)
        form = es_subj_present(base_inf, canon)

    # Non-reflexive: done
    if not is_refl:
        return form

    # Reflexive: attach clitic at the end (no accent juggling)
    pron = _ES_REFLEXIVE_PRON.get(canon, "se")
    return f"{form}{pron}"

# ---------------------------
# Nouns
# ---------------------------

def _is_one_es(s: str) -> bool:
    low = (s or "").strip().lower()
    return low in {"1", "uno", "un", "una"}

def _uno_agree_es(numeral: str, gender_es: str) -> str:
    low = (numeral or "").strip().lower()
    if low in {"1", "uno"}:
        return "una" if (gender_es or "m").lower() == "f" else "un"
    return numeral

def build_num_np_es(noun: Dict, numeral_row: Dict) -> str:
    num_es   = _s(numeral_row.get("gloss_es"))
    gender   = (_s(noun.get("gender_es")) or "m").lower()
    nform    = _uno_agree_es(num_es, gender)
    is_one   = _is_one_es(num_es)

    if not is_one and (_s(noun.get("countability_es")).lower() == "uncountable"):
        return f"algo de {_s(noun.get('gloss_es'))}".strip()

    return f"{nform} {_noun_es_bare(noun, plural=not is_one)}".strip()

def pick_subject_noun_row_es(
    es_tmpl: str,
    np_noun: Dict | None,
    poss_noun: Dict | None,
    name_row_for_subject: Dict | None
) -> Dict | None:
    """
    Pick the subject NP row for Spanish agreement based on which subject-like
    placeholder appears in the ES template. Preference order:
      1) {POSS_NP...} if present and poss_noun is available
      2) {NP...} / {AGENT_NP...} if present and np_noun is available
      3) {NAME...} if present and name_row_for_subject is available
      4) Fallback to first non-None among (np_noun, poss_noun, name_row_for_subject)
    """
    t = (es_tmpl or "")
    if "{POSS_NP"   in t and poss_noun: return poss_noun
    if "{AGENT_NP"  in t and np_noun:   return np_noun
    if "{NP"        in t and np_noun:   return np_noun
    if "{NAME"      in t and name_row_for_subject: return name_row_for_subject
    return np_noun or poss_noun or name_row_for_subject

def subject_agreement_es(
    es_tmpl: str,
    np_noun: Dict | None,
    poss_noun: Dict | None,
    name_row_for_subject: Dict | None,
    inferred_person: str,
) -> Tuple[Dict | None, str, bool]:
    """
    Central helper for Spanish subject agreement.

    Returns:
        (subj_row_es, subj_gender_es, subj_is_plural_es)
    where:
        - subj_row_es: the noun-row used as subject in Spanish (or None)
        - subj_gender_es: 'm' or 'f' (defaults to 'm')
        - subj_is_plural_es: True if the subject is plural for agreement
    """
    subj_row_es = pick_subject_noun_row_es(
        es_tmpl,
        np_noun,
        poss_noun,
        name_row_for_subject,
    )
    subj_gender_es = (subj_row_es.get("gender_es", "m") if subj_row_es else "m")
    subj_is_plural_es = (inferred_person or "").strip().lower() in {"1pl", "2pl", "3pl"}
    return subj_row_es, subj_gender_es, subj_is_plural_es

def adj_agree_es(adj_row: Dict, base_es: str, gender: str, plural: bool) -> str:
    base = (base_es or "").strip()
    if not base: return base
    inv = str(adj_row.get('adj_invariable_es') or '').strip().lower() in {'1','true','yes'}
    if inv: return base
    if (gender or 'm').lower() == 'f':
        fem_col = _s(adj_row.get('gloss_es_f'))
        if fem_col:
            base = fem_col
        elif re.search(r'o$', base):
            base = re.sub(r'o$', 'a', base)
    if plural:
        base = pluralize_es(base)
    return base

# -----------------------
# verb renderer
# -----------------------

def render_verb_bundle_es(vrow: dict | None, key_prefix: str, *, person: str = "3sg", imp_person: str | None = None) -> Dict[str, str]:
    # Safety: allow caller to pass None (e.g., template doesn’t actually realize this verb)
    if not vrow:
        return {}

    base_es = _s(vrow.get("gloss_es")).strip()
    if not base_es:
        return {}
    out: Dict[str, str] = {}

    # Core finite/non-finite
    out[f"{key_prefix}_ES"]       = es_present(base_es, person)
    out[f"{key_prefix}_PST_ES"]   = es_past(base_es, person)
    out[f"{key_prefix}_PROG_ES"]  = es_progressive(base_es, person)
    out[f"{key_prefix}_PP_ES"]    = es_pp(base_es)
    out[f"{key_prefix}_FUT_ES"]   = es_future_periphrastic(base_es, person)
    out[f"{key_prefix}_FUT_PROG_ES"]  = es_future_progressive(base_es, person)
    out[f"{key_prefix}_BASE_ES"]  = base_es  # symmetry with EN

    # Person-specific imperatives
    out[f"{key_prefix}_IMP_TU_ES"]          = es_imperative(base_es, "tu", False)
    out[f"{key_prefix}_IMP_TU_NEG_ES"]      = es_imperative(base_es, "tu", True)
    out[f"{key_prefix}_IMP_USTED_ES"]       = es_imperative(base_es, "usted", False)
    out[f"{key_prefix}_IMP_USTED_NEG_ES"]   = es_imperative(base_es, "usted", True)
    out[f"{key_prefix}_IMP_USTEDES_ES"]     = es_imperative(base_es, "ustedes", False)
    out[f"{key_prefix}_IMP_USTEDES_NEG_ES"] = es_imperative(base_es, "ustedes", True)

    # Legacy aliases follow imp_person if provided; else default to tú
    ip = (imp_person or "").strip().lower()
    if ip in {"2sg", "tu"}:
        out[f"{key_prefix}_IMP_ES"]     = out[f"{key_prefix}_IMP_TU_ES"]
        out[f"{key_prefix}_IMP_NEG_ES"] = out[f"{key_prefix}_IMP_TU_NEG_ES"]
    elif ip in {"3sg", "usted"}:
        out[f"{key_prefix}_IMP_ES"]     = out[f"{key_prefix}_IMP_USTED_ES"]
        out[f"{key_prefix}_IMP_NEG_ES"] = out[f"{key_prefix}_IMP_USTED_NEG_ES"]
    elif ip in {"3pl", "ustedes", "2pl"}:  # LatAm plural address
        out[f"{key_prefix}_IMP_ES"]     = out[f"{key_prefix}_IMP_USTEDES_ES"]
        out[f"{key_prefix}_IMP_NEG_ES"] = out[f"{key_prefix}_IMP_USTEDES_NEG_ES"]
    elif ip in {"1pl", "nosotros"}:
        out[f"{key_prefix}_IMP_ES"]     = es_imperative(base_es, "nosotros", False)
        out[f"{key_prefix}_IMP_NEG_ES"] = es_imperative(base_es, "nosotros", True)
    else:
        out[f"{key_prefix}_IMP_ES"]     = out[f"{key_prefix}_IMP_TU_ES"]
        out[f"{key_prefix}_IMP_NEG_ES"] = out[f"{key_prefix}_IMP_TU_NEG_ES"]

    return out

# ----------------------
# Post-processing
# ----------------------

# Adjective placement
def embed_adj_into_np_in_template_es(tmpl: str, repl: Dict[str, str]) -> tuple[str, Dict[str, str]]:
    """
    Rewrites '{ADJ_ES} {NP_*_ES}' → '{NP_*_ES__WITH_ADJ_*}',
    making the adjective follow the noun: 'la milpa grande'.
    """
    if not tmpl or "ADJ_ES" not in tmpl:
        return tmpl, repl

    np_pat = r"\{(?P<np>(?:POSS_)?(?:AGENT_)?(?:THEME_)?(?:GOAL_)?NP(?:_[A-Z_]+)?_ES)\}"
    rx = re.compile(rf"\{{ADJ_ES\}}\s+{np_pat}")

    idx = 0
    out = tmpl
    for m in list(rx.finditer(tmpl)):
        np_key = m.group("np")
        adj = (repl.get("ADJ_ES") or "").strip()
        np_val = (repl.get(np_key) or "").strip()
        if not adj or not np_val:
            continue

        def _place_es(np_phrase: str, adj_word: str) -> str:
            parts = np_phrase.split()
            if not parts:
                return np_phrase
            return f"{np_phrase} {adj_word}"

        new_val = _place_es(np_val, adj)
        synth_key = f"{np_key}__WITH_ADJ_{idx}"
        idx += 1
        repl[synth_key] = new_val
        out = out.replace(m.group(0), "{" + synth_key + "}", 1)

    return out, repl
