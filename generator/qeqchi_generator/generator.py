from __future__ import annotations
import math
import pandas as pd
import random
import re
from pathlib import Path
from typing import Dict, List, Optional

# Streamlined module importing
from . import utils
from . import linguistics_core, linguistics_en, linguistics_es, linguistics_kek

"""Q'eqchi' Sentence Generator"""

def _compute_template_counts(
    probs: List[float],
    total_n: int,
) -> List[int]:
    """
    Given a list of probabilities (not necessarily normalized) and a target
    total_n, return integer counts for each index such that:

    - sum(counts) == total_n
    - Every template with p > 0 gets at least 1 sentence,
      IF total_n >= number_of_templates_with_p>0.

    If total_n < num_positive_probs, we do the best we can:
    we assign 1 to as many as possible, 0 to the rest.
    """
    if not probs:
        return []

    # Clean probabilities: NaN/negative -> 0
    clean = [p if (p is not None and p > 0) else 0.0 for p in probs]

    # Indices of templates that have positive probability
    positive_indices = [i for i, p in enumerate(clean) if p > 0]
    num_positive = len(positive_indices)

    # If everything is zero, fall back to uniform
    if num_positive == 0:
        clean = [1.0] * len(clean)
        positive_indices = list(range(len(clean)))
        num_positive = len(positive_indices)

    # Edge case: if total_n is smaller than the number of positive-prob templates,
    # we *cannot* give each one at least 1. Best-effort: give 1 to the first total_n.
    if total_n <= 0:
        return [0] * len(clean)
    if total_n <= num_positive:
        counts = [0] * len(clean)
        for i in positive_indices[:total_n]:
            counts[i] = 1
        return counts

    # Step 1: give 1 to every positive-prob template
    counts = [0] * len(clean)
    for i in positive_indices:
        counts[i] = 1

    remaining = total_n - num_positive
    if remaining == 0:
        return counts

    # Step 2: distribute the remaining sentences according to probabilities
    # among the positive-prob templates
    s = sum(clean[i] for i in positive_indices)
    # Shouldn't happen, but just in case:
    if s <= 0:
        # uniform over positive_indices
        extra_per = remaining / num_positive
        raw_extra = [extra_per] * num_positive
    else:
        raw_extra = [(clean[i] / s) * remaining for i in positive_indices]

    base_extra = [math.floor(x) for x in raw_extra]
    extra_counts = base_extra[:]
    diff = remaining - sum(base_extra)

    if diff > 0:
        # Assign leftover "diff" to largest fractional parts
        fracs = [r - math.floor(r) for r in raw_extra]
        order = sorted(range(len(fracs)), key=lambda k: fracs[k], reverse=True)
        for k in order[:diff]:
            extra_counts[k] += 1

    # Apply the extra counts back to the main counts array
    for idx_in_list, tmpl_idx in enumerate(positive_indices):
        counts[tmpl_idx] += extra_counts[idx_in_list]

    return counts

class QeqchiGenerator:

    """
    Generates synthetic Q'eqchi'/English/Spanish sentence pairs based on templates and semantic rules.
    This class loads linguistic data (nouns, verbs, adjectives, etc.) and sentence
    templates, then uses constraints and compatibility rules to generate grammatically
    plausible and semantically coherent sentence triplets.
    """
    
    def __init__(self, data_root: Path, lang: str = "kek", rng_seed: int = 1337):
        """
        Initializes the generator by loading all required data files.

        Args:
            data_root (Path): Root directory containing:
                - 'templates/{lang}_templates.csv' for sentence templates
                - '{lang}/{lang}_*.csv' for lexical semantics (nouns, verbs, etc.)
            lang (str): ISO-like language code used in filenames and folders (e.g. 'kek').
            rng_seed (int): Integer seed for the RNG to ensure reproducibility.
        """
        self.data_root = Path(data_root)
        self.lang = (lang or "kek").lower()

        # -------- Templates --------
        templates_path = self.data_root / "templates" / f"{self.lang}_templates.csv"
        # Use low_memory=False to avoid massive DtypeWarning spam for wide CSVs
        self.templates = pd.read_csv(templates_path, low_memory=False)
        
        # -------- Lexical CSVs from data/{lang}/ ------
        lex_root = data_root / self.lang

        self.nouns = pd.read_csv(lex_root / f"{self.lang}_nouns.csv")
        # enrich noun semantics right away
        self.nouns = linguistics_core.enrich_noun_semantics(self.nouns)

        self.verbs = pd.read_csv(lex_root / f"{self.lang}_verbs.csv")
        # normalize verb membership in transitivity classes right away
        self.verbs["_transitivity_set"] = self.verbs["transitivity"].apply(
            linguistics_core.norm_transitivity_set
        )

        self.adjs     = pd.read_csv(lex_root / f"{self.lang}_adjectives.csv")
        self.adverbs  = pd.read_csv(lex_root / f"{self.lang}_adverbs.csv")
        self.numerals = pd.read_csv(lex_root / f"{self.lang}_numerals.csv")

        self.rng = random.Random(rng_seed)
        
        # Round-robin index for possessed NPs
        self._possessor_rr_idx = 0

    ## PICKERS ##
        """
        _pick_noun_for_slot selects a random noun appropriate for a specific slot in a template.

        Filtering order:
            1. Base filters (e.g., {'possessability': 'obligatory'}).
            2. Verb-driven constraints (Option A: agent/theme/goal allow-lists).
            3. Verb-less fallback defaults (for copular / zero-verb templates).

        Args:
            slot: Template slot name (e.g., 'SUBJECT_NP', 'THEME_NP', 'GOAL_NP').
            template_id: Numeric template ID (kept for interface consistency).
            base_filters: Dict of key-value filters for noun selection.
            verb_arg: Dict returned by _verb_arg_constraints() if a verb is present.

        Returns:
            A random noun row (dict) or None if no valid noun is found.
        """
    
    def _pick_noun_for_slot(
        self,
        slot: str,
        base_filters: Optional[Dict[str, object]] = None,
        verb_arg: Optional[Dict[str, set]] = None,
        verb_meta: Optional[Dict] = None,
    ) -> Optional[Dict]:
        df = self.nouns.copy()

        # Ensure normalized class column exists (defense-in-depth)
        if "class" in df.columns:
            df["class"] = df["class"].astype(str).str.strip().str.lower()
    
        # --- 1) Apply base filters (string/enum, set-of-values, or boolean-like) ---
        def _bool_norm(x):
            s = utils._s(x).lower()
            if s in {"1","true","yes"}:  return True
            if s in {"0","false","no"}:  return False
            return None
        
        if base_filters:
            for k, v in base_filters.items():
                if k not in df.columns:
                    continue
                # Normalize string columns to lowercase/trim for comparison
                if df[k].dtype == object:
                    col = df[k].astype(str).str.strip().str.lower()
                else:
                    col = df[k]

                if isinstance(v, (set, list, tuple)):
                    vals = {utils._s(x).strip().lower() for x in v}
                    df = df[col.isin(vals)]
                else:
                    vb = v if isinstance(v, bool) else None
                    if vb is not None:
                        df = df[df[k].map(_bool_norm) == vb]
                    else:
                        df = df[col == utils._s(v).strip().lower()]

        # --- 2) Verb-driven constraints (agent/theme/goal) ---
        verb_arg = verb_arg or {}
        if (slot.startswith("AGENT") or slot.startswith("SUBJECT") or
            slot.startswith("NP") or slot.startswith("POSS_NP")):
            allow = {utils._s(x).strip().lower() for x in (verb_arg.get("agent_allow") or [])}
            ban   = {utils._s(x).strip().lower() for x in (verb_arg.get("agent_ban")   or [])}
            if allow: df = df[df["class"].isin(allow)]
            if ban:   df = df[~df["class"].isin(ban)]

        if slot.startswith("THEME"):
            allow = {utils._s(x).strip().lower() for x in (verb_arg.get("theme_allow") or [])}
            ban   = {utils._s(x).strip().lower() for x in (verb_arg.get("theme_ban")   or [])}
            if allow: df = df[df["class"].isin(allow)]
            if ban:   df = df[~df["class"].isin(ban)]

        if slot.startswith("GOAL") or slot.startswith("RECIPIENT"):
            allow = {utils._s(x).strip().lower() for x in (verb_arg.get("goal_allow") or [])}
            ban   = {utils._s(x).strip().lower() for x in (verb_arg.get("goal_ban")   or [])}
            if allow: df = df[df["class"].isin(allow)]
            if ban:   df = df[~df["class"].isin(ban)]
        
        if slot == "PRED_NP":
            # If we want to be conservative: pick human-ish nouns that work as identity predicates
            # (grandfather, man, woman, etc.), reactivate following line, otherwise keep "pass"
            # base_filters["class"] = {"human", "person", "kinship",}
            pass
        
        return self._pick_weighted(df)

    # ---- Adjective pick  ----
    def _pick_adjective_for_subject(self, subj_row: Dict) -> Optional[Dict]:
        noun_cls = linguistics_core.noun_class(subj_row) if hasattr(linguistics_core, "noun_class") else utils._s(subj_row.get("class"))
        cands = [
            a.to_dict() for _, a in self.adjs.iterrows()
            if linguistics_core.adj_compatible_with_noun(a, subj_row, noun_cls)
        ]
        
        return self._pick_weighted(cands)
    
    # --- Verb pick ----
    def _pick_verb_by_class(self, cls: str) -> Optional[Dict]:
        """Sample a verb whose normalized _transitivity_set contains cls."""
        df = self.verbs[self.verbs["_transitivity_set"].apply(lambda s: cls in s)]
        if df.empty:
            return None
        row = df.sample(n=1, random_state=self.rng.randint(0, 2**31-1)).iloc[0]
        return row.to_dict()
    
    def pick_v_intr(self) -> Optional[Dict]:
        """Picks a random intransitive verb."""
        return self._pick_verb_by_class("intr")

    def pick_v_tr(self) -> Optional[Dict]:
        """Picks a random transitive verb."""
        return self._pick_verb_by_class("tr")

    def pick_v_ditr(self) -> Optional[Dict]:
        """Picks a random ditransitive verb."""
        return self._pick_verb_by_class("ditr")

    def pick_random_noun(self, filters: Optional[Dict[str, str]] = None) -> Optional[Dict]:
        """
        Picks a random noun row, optionally applying simple equality filters.

        Args:
            filters (Optional[Dict[str, str]]): A dictionary of column names and
                                                required string values (e.g., {'possessability': 'obligatory'}).

        Returns:
            Optional[Dict]: A dictionary representing the chosen noun row, or None if
                            no noun matches the filters or the nouns list is empty.
        """
        df = self.nouns.copy()
        if filters:
            for k, v in filters.items():
                if k in df.columns:
                    df = df[df[k].astype(str) == str(v)]
        return self._pick_weighted(df)

    def pick_numeral(self) -> Optional[Dict]:
        """Picks a random numeral row."""
        if self.numerals.empty:
            return None
        row = self.numerals.iloc[self.rng.randrange(len(self.numerals))]
        return row.to_dict()

    def pick_adverb(self, tense_flags, verb_category: str = "", required_category: Optional[str] = None) -> Optional[Dict]:
        """
        Select an adverb compatible with the given tense_flags and verb_category.
        If required_category is provided, restrict to adverbs whose category matches (case-insensitive).
        """
        req = (required_category or "").strip().lower() or None

        ok = []
        for _, r in self.adverbs.iterrows():
            if req:
                cat = utils._s(r.get("category")).strip().lower()
                if cat != req:
                    continue
            if linguistics_core.adverb_ok(r, tense_flags, verb_category):
                ok.append(r)

        return self._pick_weighted(ok)
    
    def pick_time_adverb_future(self) -> Optional[Dict]:
        """
        Selects a time adverb compatible with future tense.
        Used specifically for future stative templates.
        """
        # 1. Filter adverbs for future compatibility
        future_candidates = []
        
        # We iterate over the dataframe rows
        for _, row in self.adverbs.iterrows():
            # Handle potential NaN/empty values safely
            tense_compat = str(row.get("tense_compat", "")).lower()
            
            # Check if 'fut' is in the compatibility list (e.g. "fut;prs")
            if "fut" in tense_compat:
                future_candidates.append(row.to_dict())
        
        # 2. Use the Zipfian weighted picker
        return self._pick_weighted(future_candidates)
    
    def _pick_weighted(self, candidates) -> Optional[Dict]:
        """
        Universal weighted picker.
        Accepts:
          - pd.DataFrame (Nouns, Verbs)
          - List of Dicts OR List of Series (Adjectives, Adverbs)

        Returns:
          - A single row (as a Dict).
          - None if input is empty.
        """
        import math

        chosen = None

        def _clean_weight(x) -> float:
            """
            Convert to a finite, non-negative float.
            NaN / inf / non-numeric -> 0.0
            Negative values -> 0.0
            """
            try:
                w = float(x)
            except Exception:
                return 0.0
            if not math.isfinite(w):
                return 0.0
            return w if w > 0.0 else 0.0

        # --- CASE A: Input is a DataFrame ---
        if isinstance(candidates, pd.DataFrame):
            if candidates.empty:
                return None

            if "p_weight" in candidates.columns:
                weights = candidates["p_weight"].map(_clean_weight).tolist()
                total_w = sum(weights)

                if total_w > 0.0:
                    chosen = self.rng.choices(
                        population=candidates.to_dict("records"),
                        weights=weights,
                        k=1
                    )[0]
                else:
                    chosen = candidates.iloc[self.rng.randrange(len(candidates))].to_dict()
            else:
                chosen = candidates.iloc[self.rng.randrange(len(candidates))].to_dict()

        # --- CASE B: Input is a List ---
        elif isinstance(candidates, list):
            if not candidates:
                return None

            first_item = candidates[0]

            def get_w(item):
                if isinstance(item, pd.Series):
                    return item.get("p_weight", 0.0)
                return item.get("p_weight", 0.0) if isinstance(item, dict) else 0.0

            use_weights = _clean_weight(get_w(first_item)) > 0.0

            if use_weights:
                weights = [_clean_weight(get_w(x)) for x in candidates]
                total_w = sum(weights)

                if total_w > 0.0:
                    chosen = self.rng.choices(candidates, weights=weights, k=1)[0]
                else:
                    chosen = self.rng.choice(candidates)
            else:
                chosen = self.rng.choice(candidates)

        if chosen is None:
            return None

        if isinstance(chosen, pd.Series):
            return chosen.to_dict()

        return chosen

## SEMANTICS CHECK ##
    def _validate_semantics(self, tid: int, chosen: Dict) -> bool:
        """
        Performs final lightweight semantic checks before returning the rendered sentence.
        This acts as a last-resort filter for issues not easily caught during selection,
        such as ensuring theme and goal aren't identical nouns or checking color adjective
        compatibility.
        """
        # Existing heuristics (if you have additional checks above, keep them here)
        # ...

        # Theme and goal should not be the same surface (if both exist)
        theme_row = chosen.get("theme_row")
        goal_row  = chosen.get("goal_row")
        if theme_row and goal_row:
            # Compare using a unique identifier if available, otherwise fallback to surface form
            theme_id = theme_row.get("id", utils._s(theme_row.get("lemma_kek") or theme_row.get("gloss_en")))
            goal_id  = goal_row.get("id",  utils._s(goal_row.get("lemma_kek")  or goal_row.get("gloss_en")))
            if theme_id and goal_id and theme_id == goal_id:
                return False  # Avoid "He gave the dog to the dog"

        # If ditransitive and template has goal but none filled
        if bool(chosen.get("verb_ditr")) and chosen.get("template_has_goal") and not goal_row:
            return False

        # If transitive verb and template has THEME but none filled → reject
        if bool(chosen.get("verb_tr")) and chosen.get("template_has_theme") and not theme_row:
            return False

        # ---- enforce verb argument class constraints against filled NPs ----
        verb_meta = chosen.get("verb_tr") or chosen.get("verb_ditr")
        if verb_meta:
            cons = linguistics_core.verb_arg_constraints(verb_meta)

            def _norm_class(row: Optional[Dict]) -> str:
                return utils._s((row or {}).get("class")).strip().lower() if row else ""

            def _violates(row: Optional[Dict], allowed: set, banned: set) -> bool:
                if row is None:
                    return False
                cls = _norm_class(row)
                if allowed and cls not in allowed:
                    return True
                if banned and cls in banned:
                    return True
                return False

            subj_row = chosen.get("subj")

            if _violates(subj_row, cons["agent_allow"], cons["agent_ban"]):
                return False
            if _violates(theme_row, cons["theme_allow"], cons["theme_ban"]):
                return False
            if _violates(goal_row,  cons["goal_allow"],  cons["goal_ban"]):
                return False

        return True
 
## SLOT UTILITIES ##
    def _extract_slots(self, tmpl_row: Dict) -> set:
        """Return a set of all {SLOTS} present across KEK/EN/ES for this template row."""
        slots = set()
        text = (utils._s(tmpl_row.get("kek","")) + " " +
                utils._s(tmpl_row.get("en",""))  + " " +
                utils._s(tmpl_row.get("es","")))
        for m in re.finditer(r"\{[A-Z0-9_]+(?:_[A-Z0-9_]+)?\}", text):
            slots.add(m.group(0))
        return slots

    def _slot_present_any(self, name: str, tmpl_tokens: set) -> bool:
        """True if any token starting with {name or exactly {name} is in the cached set."""
        prefix = "{"+name+"_"
        bare   = "{"+name+"}"
        return any(tok.startswith(prefix) or tok == bare for tok in tmpl_tokens)

    def _slot_family_present(self, family: str, tmpl_tokens: set) -> bool:
        """
        Return True if any slot from a verb family (including imperative, progressive,
        or participle variants) is present in the cached template token set.
        Example: family='V_TR' matches {V_TR}, {V_TR_IMP_EN}, {V_TR_PST_ES}, etc.
        """
        if not tmpl_tokens:
            return False
        family_prefix = "{" + family + "_"
        family_bare = "{" + family + "}"
        for tok in tmpl_tokens:
            if tok.startswith(family_prefix) or tok == family_bare:
                return True
        return False

    ## --- Allowed tenses for time-related adverbs
    def _compute_tense_flags(self, tokens: set) -> Dict[str, bool]:
        """
        Build a simple tense flag map for adverb filtering.
        Keys should match values used in adverbs CSV `allow_tense` (e.g., present/past/future/progressive/perfect).
        """
        has_past  = any("_PST_"  in t for t in tokens)
        has_fut   = any("_FUT_"  in t for t in tokens)
        has_prog  = any("_PROG_" in t for t in tokens)
        has_pp    = any("_PP_"   in t for t in tokens)

        # If no marked tense is present, assume simple present.
        present = not (has_past or has_fut or has_prog or has_pp)

        return {
            "prs":     present,
            "pst":     has_past,
            "fut":     has_fut,
            "prg":     has_prog,
            "prf":     has_pp,
        }

    # --- Person inference (subject) ---
    def _infer_person(self, tmpl_tokens: set, default_person: Optional[str] = None) -> str:
        """
        Decide the subject person for agreement.
        1) Use provided default_person if valid.
        2) Otherwise default to '3sg'.
        3) If template marks plural subject ({SUBJ_PRONOUN_PL} or {SUBJ_NP_PL}),
           force plural: keep same person if known (1→1pl, 2→2pl, 3→3pl); otherwise 3pl.
        """
        p = (default_person or "").strip().lower()
        valid = set(linguistics_core.PERSONS)
        if p not in valid:
            p = "3sg"

        is_plural = (self._slot_present_any("SUBJ_PRONOUN_PL", tmpl_tokens)
                     or self._slot_present_any("SUBJ_NP_PL", tmpl_tokens))
        if is_plural:
            if p.endswith("sg"):
                p = p.replace("sg", "pl")
            elif p not in valid or not p.endswith("pl"):
                p = "3pl"
        return p    
    
    def _subject_is_plural(self, tmpl_tokens: set, person: Optional[str] = None) -> bool:
        """
        Return True if the subject is plural, determined only by SUBJECT-side
        markers in the template (NP/AGENT/NAME/POSS_NP), or by an explicit plural
        'person'. Never looks at THEME/GOAL object plurality.
        """
        # Respect explicit person if given
        if person and str(person).strip().endswith("pl"):
            return True

        # Subject-side plural tokens across languages
        subject_pl_tokens = {
            "{AGENT_NP_PL}",     "{AGENT_NP_PL_EN}",     "{AGENT_NP_PL_ES}",
            "{NP_PL}",           "{NP_PL_EN}",           "{NP_PL_ES}",
            "{NP_PL_INDEF}",     "{NP_PL_INDEF_EN}",     "{NP_PL_INDEF_ES}",
            "{POSS_NP_PL}",      "{POSS_NP_PL_EN}",      "{POSS_NP_PL_ES}",
        }
        return any(tok in tmpl_tokens for tok in subject_pl_tokens)

    ## SENTENCE RENDERING ##
    def render(
        self,
        template_id: str | int,
        *,
        person: str = "3sg",
    ) -> Optional[Dict[str, str]]:
        """
        Renders a single trilingual sentence (KEK/EN/ES) from the id of a template.
        Returns a dict with keys: id, kek, en, es, and several diagnostics fields.
        May return None if the template cannot be instantiated (e.g., constraints conflict).
        """
        np_noun: Optional[Dict] = None
        poss_noun: Optional[Dict] = None
        poss_person: Optional[str] = None

        recs = self.templates[self.templates["id"] == template_id]
        if recs.empty:
            return None
        t = recs.iloc[0].to_dict()
        kek_tmpl, en_tmpl = t["kek"], t["en"]
        es_tmpl = t.get("es", "") or ""

        # Extract all {SLOTS} once across KEK/EN/ES templates
        tokens = self._extract_slots(t)

        both_tmpl = f"{kek_tmpl} {en_tmpl} {es_tmpl}"  # Keep for _fill_np_variants

        # --- Copular stative predicate context (pronoun/name subject) ---
        # Intended target: "I am a grandfather." / "Yo soy un abuelo." / (KEK predicate nominal)
        is_copular_any = (
            ("{BE_EN}" in en_tmpl) or ("{BE_NEG_EN}" in en_tmpl) or ("{BE_FUT_EN}" in en_tmpl) or
            ("{V_SER_ES}" in es_tmpl) or
            ("{COP_ES}" in es_tmpl) or ("{COP_NEG_ES}" in es_tmpl) or
            ("{COP_FUT_ES}" in es_tmpl) or ("{COP_NEG_FUT_ES}" in es_tmpl)
        )

        subj_is_pron_or_name_any = any(tag in (kek_tmpl + en_tmpl + es_tmpl) for tag in (
            "{SUBJ_PRONOUN}", "{SUBJ_PRONOUN_EN}", "{SUBJ_PRONOUN_ES}",
            "{SUBJ_PRON}", "{SUBJ_PRON_EN}", "{SUBJ_PRON_ES}",
            "{NAME}", "{NAME_EN}", "{NAME_ES}",
        ))

        # In this context, {NP_*} is interpreted as a predicate nominal, not a subject NP.
        copular_pred_np_context = bool(is_copular_any and subj_is_pron_or_name_any)

        # Noun-class filter for predicate nominals in copular statives.
        # Broad-but-safe set; supports older labels too.
        np_filters_for_copular_pred = None
        ## For stricter filtering of subjects in copula sentences, activate following code:
       # if copular_pred_np_context:
       #     np_filters_for_copular_pred = {
       #         "class": {"person", "human", "kinship"}
       #     }
        
        # Possession context? (any language contains a V_POSS_TR_* placeholder)
        possession_context = (
            self._slot_family_present("V_POSS_TR", tokens)
        )

        # --- replacement dict ---
        repl = {}

        # --- Names (human-only, un-translated) ---
        uses_name = self._slot_present_any("NAME", tokens)
        chosen_name_dict = None
        name_row_for_subject = None

        if uses_name:
            chosen_name_dict = linguistics_core.pick_name(prefer_gender=None)
            name_row_for_subject = linguistics_core.name_to_noun_row(chosen_name_dict)
            # KEK gets language-specific surface; EN/ES keep bare name
            repl["NAME"]    = linguistics_kek.kek_name_surface(chosen_name_dict)
            repl["NAME_EN"] = chosen_name_dict["text"]
            repl["NAME_ES"] = chosen_name_dict["text"]

        # --- Places (un-translated) ---
        # Used for "origin" predicates like: aj {PLACE} {AFFIX_PRONOUN}.
        # We provide KEK/EN/ES variants for template convenience.
        uses_place = self._slot_present_any("PLACE", tokens)
        if uses_place:
            place_dict = linguistics_core.pick_place()
            repl["PLACE"]    = place_dict["text"]
            repl["PLACE_EN"] = place_dict["text"]
            repl["PLACE_ES"] = place_dict["text"]
            
        # --- Verbs -----------------------------------------------------------------
        needs_intr = self._slot_family_present("V_INTR", tokens)
        needs_tr   = (
            self._slot_family_present("V_TR", tokens)
            or self._slot_family_present("V_POSS_TR", tokens)
        )
        needs_ditr = self._slot_family_present("V_DITR", tokens)
        chosen_v_intr = self.pick_v_intr() if needs_intr else None
        chosen_v_tr   = self.pick_v_tr()   if self._slot_family_present("V_TR", tokens) else None
        chosen_v_ditr = self.pick_v_ditr() if needs_ditr else None

        # --- subject / person for this template ---
        # Subject field in templates file may contain rich labels like:
        #   "1sg", "2sg", "1pl", "2pl",
        #   "3sg - Name", "3sg - Noun", "3pl - Noun def",
        #   "3pl - Noun indef", "3pl - Possessed noun", etc.
        # We always extract the leading person code (1sg/2sg/3sg/1pl/2pl/3pl).
        subj_raw = utils._s(t.get("Subject")).strip()
        valid_persons = set(linguistics_core.PERSONS)

        subj_person: Optional[str] = None
        if subj_raw:
            low = subj_raw.lower()
            # Direct match first
            if low in valid_persons:
                subj_person = low
            else:
                # Accept composite labels like "3sg - Name", "3pl - Noun def", etc.
                m = re.match(r"^(1sg|2sg|3sg|1pl|2pl|3pl)\b", low)
                if m:
                    subj_person = m.group(1)

        mood_raw = utils._s(t.get("Mood") or t.get("mood"))
        is_imperative = bool(mood_raw and mood_raw.strip().upper().startswith("IMP"))

        # Imperative person for all three languages; defaults to 2nd person
        # with number decided by subject plurality if Subject is empty.
        tmpl_imp_person: Optional[str] = None
        if is_imperative:
            if subj_person in {"1sg", "1pl", "2sg", "2pl", "3sg", "3pl"}:
                tmpl_imp_person = subj_person
            else:
                # No explicit Subject person: choose 2sg vs 2pl from the template
                is_subj_pl = self._subject_is_plural(tokens, person=subj_person or person)
                tmpl_imp_person = "2pl" if is_subj_pl else "2sg"

        # Default person used later for agreement & pronoun logic.
        # The actual `inferred_person` is finalized in the integrated pronoun block
        # further down.
        default_person_for_agreement = tmpl_imp_person or subj_person or person

        # Infer object person for agreement/clitics when there is a theme NP
        obj_person_tr   = "3sg" if self._slot_present_any("THEME_NP", tokens) and needs_tr   else None
        obj_person_ditr = "3sg" if self._slot_present_any("THEME_NP", tokens) and needs_ditr else None

        # How many object NPs can we realize for this template instance?
        obj_capacity = 0
        if chosen_v_ditr:
            obj_capacity = 2
        elif needs_tr:
            obj_capacity = 1
        # intr stays 0

        # Track how many object NPs we already filled (theme/goal)
        objects_filled = 0

        # Verb-argument constraints (for noun picking)
        verb_row = chosen_v_ditr or chosen_v_tr or chosen_v_intr
        verb_arg = linguistics_core.verb_arg_constraints(verb_row)

        # ========== NP VARIANTS ==========
        def _fill_np_variants(base_slot: str, noun_row: Dict, possessed_person: Optional[str] = None) -> None:
            """
            Fills all variants (plural/definiteness/language) of a base NP slot found in templates.

            Special case:
            - PRED_NP: noun predicate used with pronoun+copula stative templates.
              KEK must be the *bare noun lemma only* (no 'li', no class articles like 'aj').
              EN/ES default to indefinite in singular ("a/an", "un/una"), and no article in plural.
            """
            if noun_row is None:
                return

            pattern = re.compile(
                r"\{("
                + re.escape(base_slot)
                + r"(?:_(?:PL|INDEF|DEF|[A-Z]+))*"
                + r"(?:_(?:EN|ES))?"
                + r")\}"
            )

            for match in pattern.finditer(both_tmpl):
                full_slot_name = match.group(1)
                is_en = full_slot_name.endswith("_EN")
                is_es = full_slot_name.endswith("_ES")
                bare_slot = full_slot_name[:-3] if (is_en or is_es) else full_slot_name

                # ---- Parse flags ----
                tail = full_slot_name[:-3] if (is_en or is_es) else full_slot_name
                parts = tail.split("_")
                flags = set(parts[1:]) if len(parts) > 1 else set()

                plural = "PL" in flags

                # Default definite unless explicitly INDEF (existing contract).
                # NOTE: For PRED_NP we override this default below.
                explicit_indef = "INDEF" in flags
                definite = not explicit_indef

                # Q'eqchi' possessed-plural style flags
                poss_plural_style = "auto"
                if "PRE" in flags:
                    poss_plural_style = "pre"
                elif "POST" in flags:
                    poss_plural_style = "post"

                # Object NPs inside possession context default to indefinite (legacy behavior)
                if (
                    not explicit_indef
                    and possession_context
                    and base_slot in ("NP", "THEME_NP", "GOAL_NP")
                ):
                    definite = False

                # Subject role informs KEK 'li' handling
                as_subject = base_slot in ("NP", "AGENT_NP")

                # ========== SPECIAL CASE: PRED_NP ==========
                if base_slot == "PRED_NP":
                    # KEK: must be *pure lemma only*, because pronoun suffix attaches to it (winq-in),
                    # and we must not output 'li'/'aj' class markers here.
                    kek_form = (noun_row.get("lemma_kek") or "").strip()

                    # EN/ES: predicate NPs default to indefinite in singular and bare in plural.
                    # We treat them as non-subjects for determiner logic.
                    en_form = linguistics_en.build_np_en(
                        noun_row,
                        plural=plural,
                        definite=False,        # force indefinite/bare
                        as_subject=False,
                    ) or (noun_row.get("gloss_en") or "").strip()

                    es_form = linguistics_es.build_np_es(
                        noun_row,
                        plural=plural,
                        definite=False,        # force indefinite/bare
                        as_subject=False,
                    ) or (noun_row.get("gloss_es") or "").strip()

                # ========== DEFAULT PATH (all other NP-like slots) ==========
                else:
                    if base_slot in {"POSS_NP", "THEME_POSS_NP", "GOAL_POSS_NP"}:
                        possessor = possessed_person or "1sg"
                        kek_form = linguistics_kek.build_np_kek(
                            noun_row,
                            plural=plural,
                            definite=definite,
                            possessed_person=possessor,
                            poss_plural_style=poss_plural_style,
                            as_subject=as_subject,
                        ) or (noun_row.get("lemma_kek") or "").strip()
        
                        en_form = linguistics_en.build_np_en(
                            noun_row,
                            plural=plural,
                            definite=definite,
                            possessed_person=possessor,
                            as_subject=as_subject,
                        ) or (noun_row.get("gloss_en") or "").strip()

                        es_form = linguistics_es.build_np_es(
                            noun_row,
                            plural=plural,
                            definite=definite,
                            possessed_person=possessor,
                            as_subject=as_subject,
                        ) or (noun_row.get("gloss_es") or "").strip()
                    elif base_slot == "POSSESSUM_NP":
                        # KEK: morphologically possessed, person must match the subject.
                        # EN/ES: plain indefinite object NP (possession expressed by HAVE/TENER).
                        possessor = possessed_person or "3sg"

                        kek_form = linguistics_kek.build_np_kek(
                            noun_row,
                            plural=plural,
                            definite=False,                  # force indefinite for object-of-possession
                            possessed_person=possessor,
                            poss_plural_style=poss_plural_style,
                            as_subject=False,
                        ) or (noun_row.get("lemma_kek") or "").strip()

                        en_form = linguistics_en.build_np_en(
                            noun_row,
                            plural=plural,
                            definite=False,                  # force indefinite article in singular
                            as_subject=False,
                        ) or (noun_row.get("gloss_en") or "").strip()

                        es_form = linguistics_es.build_np_es(
                            noun_row,
                            plural=plural,
                            definite=False,                  # force un/una in singular
                            as_subject=False,
                        ) or (noun_row.get("gloss_es") or "").strip()
                    else:
                        kek_form = linguistics_kek.build_np_kek(
                            noun_row,
                            plural=plural,
                            definite=definite,
                            possessed_person=None,
                            poss_plural_style=poss_plural_style,
                            as_subject=as_subject,
                        ) or (noun_row.get("lemma_kek") or "").strip()

                        en_form = linguistics_en.build_np_en(
                            noun_row,
                            plural=plural,
                            definite=definite,
                            as_subject=as_subject,
                        ) or (noun_row.get("gloss_en") or "").strip()

                        es_form = linguistics_es.build_np_es(
                            noun_row,
                            plural=plural,
                            definite=definite,
                            as_subject=as_subject,
                        ) or (noun_row.get("gloss_es") or "").strip()

                # ---- Write surfaces ----
                surface = en_form if is_en else (es_form if is_es else kek_form)
                if not surface:
                    surface = (noun_row.get("gloss_en") or noun_row.get("lemma_kek") or "").strip()
        
                repl[full_slot_name] = surface

                # Ensure base aliases exist
                repl.setdefault(bare_slot, kek_form or (noun_row.get("lemma_kek") or "").strip())
                repl.setdefault(f"{bare_slot}_EN", en_form or (noun_row.get("gloss_en") or "").strip())
                repl.setdefault(f"{bare_slot}_ES", es_form or (noun_row.get("gloss_es") or "").strip())
               
        # ========== NUM + NP VARIANTS ==========
        def _fill_num_np_variants(
            num_base_slot: str,
            noun_row: Dict,
            numeral_row: Dict,
            possessed_person: Optional[str] = None
        ) -> None:
            """
            Fills combined numeral+NP slots like {NUM_NP}, {NUM_NP_EN}, {NUM_NP_ES}.
            Also supports {NUM_THEME_NP*}, {NUM_GOAL_NP*}, and {NUM_POSSESSUM_NP*}.
            """
            if noun_row is None or numeral_row is None:
                return

            pattern = re.compile(r"\{(" + re.escape(num_base_slot) + r"(?:_[A-Z_]+)?(?:_(?:EN|ES))?)\}")

            for match in pattern.finditer(both_tmpl):
                full_slot_name = match.group(1)
                is_en = full_slot_name.endswith("_EN")
                is_es = full_slot_name.endswith("_ES")

                is_possessum = (num_base_slot == "NUM_POSSESSUM_NP")

                # KEK: if NUM_POSSESSUM_NP, pass possessed_person through to build possessed numeral NP
                kek_form = linguistics_kek.build_num_np_kek(
                    noun_row,
                    numeral_row,
                    as_subject=(num_base_slot in {"NUM_NP", "NUM_AGENT_NP"}),
                    possessed_person=(possessed_person if is_possessum else None),
                )

                en_form = linguistics_en.build_num_np_en(noun_row, numeral_row)
                es_form = linguistics_es.build_num_np_es(noun_row, numeral_row)

                repl[full_slot_name] = (en_form if is_en else (es_form if is_es else kek_form))

        # Initialize and (if needed) pick the base NP and the possessed NP
        np_noun: Optional[Dict] = None
        poss_noun: Optional[Dict] = None
        poss_person: Optional[str] = None  # possessor; defaulted later if None

        # Possessum row (the object that is possessed in HAVE/TENER constructions)
        possessum_row: Optional[Dict] = None
        
        # Pick a plain NP when the template uses {NP...} OR {PRED_NP...}
        # (PRED_NP is the predicate nominal slot for copular statives with affixed pronouns)
        if self._slot_present_any("NP", tokens) or self._slot_present_any("PRED_NP", tokens):
            # Use the same conservative noun-class filter in copular predicate contexts
            slot_for_pick = "PRED_NP" if self._slot_present_any("PRED_NP", tokens) else "NP"
            np_noun = self._pick_noun_for_slot(
                slot_for_pick,
                base_filters=np_filters_for_copular_pred,
                verb_arg=verb_arg,
                verb_meta=verb_row
            )

        # Pick a possessable noun if the template uses {POSS_NP...}
        if self._slot_present_any("POSS_NP", tokens):
            # Only nouns that are possessable
            poss_noun = self._pick_noun_for_slot(
                "POSS_NP",
                base_filters={"possessability": {"optional", "obligatory"}},
                verb_arg=verb_arg
            )
            # Cycle through PERSONS unless the caller pins --person; ensures coverage of my/your/his/our/your/their
            if person in linguistics_core.PERSONS:
                poss_person = person
            else:
                poss_person = linguistics_core.PERSONS[self._possessor_rr_idx % len(linguistics_core.PERSONS)]
                self._possessor_rr_idx += 1

        # Pick a possessum noun if the template uses {POSSESSUM_NP...}
        if self._slot_present_any("POSSESSUM_NP", tokens):
            possessum_row = self._pick_noun_for_slot(
                "POSSESSUM_NP",
                base_filters={"possessability": {"optional", "obligatory"}},
                verb_arg=verb_arg,
                verb_meta=verb_row,
            )
    
        # Subject proxy for semantic checks (READ *after* initialization/selection)
        subj_for_adj_adv_checks = np_noun or poss_noun or name_row_for_subject
        
        # Templates like "{ADJ}{AFFIX_PRONOUN}." have no NP/NAME/POSS_NP, but still require an adjective.
        # Provide a generic "person" proxy with RANDOM gender to avoid bias.
        if subj_for_adj_adv_checks is None:
            gender_es = random.choice(["m", "f"])
            subj_for_adj_adv_checks = {
                "class": "person",
                "has_color": "0",
                "gender_es": gender_es,
            }

        # --- Subject NP selection ---
        if self._slot_present_any("NP", tokens) and np_noun is None:
            np_noun = self._pick_noun_for_slot("NP", base_filters=np_filters_for_copular_pred, verb_arg=verb_arg, verb_meta=verb_row)

        # Now fill the actual surface forms
        # Prefer PRED_NP if present; otherwise fill NP as usual.
        if self._slot_present_any("PRED_NP", tokens) and np_noun is not None:
            _fill_np_variants("PRED_NP", np_noun)
        elif self._slot_present_any("NP", tokens) and np_noun is not None:
            _fill_np_variants("NP", np_noun)

        if self._slot_present_any("POSS_NP", tokens) and poss_noun is not None:
            _fill_np_variants("POSS_NP", poss_noun, possessed_person=poss_person or "1sg")

        # Placeholders for theme/goal
        theme_row = None
        goal_row = None

        # Adverb (picked later)
        chosen_adv = None

        # Numerals
        chosen_num = self.pick_numeral() if self._slot_present_any("NUM", tokens) else None

        # --- Fill NP slots based on presence and constraints ---
        if self._slot_present_any("POSS_NP", tokens) and poss_noun is not None:
            _fill_np_variants("POSS_NP", poss_noun, possessed_person=(poss_person or "1sg"))

        # Generic NP
        if self._slot_present_any("NP", tokens) and np_noun is not None:
            _fill_np_variants("NP", np_noun)

        if self._slot_present_any("THEME_NP", tokens):
            if obj_capacity >= 1:
                # If this is a possession context, avoid human nouns as objects
                base_filters_for_poss = {"is_human": False} if possession_context else None
                theme_row = self._pick_noun_for_slot(
                    "THEME_NP",
                    base_filters=base_filters_for_poss,
                    verb_arg=verb_arg,
                    verb_meta=verb_row
                )
                if theme_row is not None:
                    # --- Auto-possess obligatorily possessed theme nouns ---
                    is_oblig_poss = utils._s(theme_row.get("possessability")).lower() == "obligatory"
                    if is_oblig_poss:
                        poss_person_theme = person if person in linguistics_core.PERSONS else "3sg"
                        _fill_np_variants("THEME_NP", theme_row, possessed_person=poss_person_theme)
                    else:
                        _fill_np_variants("THEME_NP", theme_row)
                    objects_filled += 1
            else:  # Clear theme slots if intransitive
                for tok in ["THEME_NP", "THEME_NP_EN", "THEME_NP_ES",
                            "THEME_NP_PL", "THEME_NP_PL_EN", "THEME_NP_PL_ES"]:
                    repl.setdefault(tok, "")

        # --- Coreference placeholder family ---
        # REF_NP_DEF should reuse the most relevant prior noun row:
        # 1) THEME_NP if present, else 2) POSSESSUM_NP (HAVE/TENER object) if present.
        ref_row = theme_row if theme_row is not None else possessum_row

        if self._slot_present_any("REF_NP_DEF", tokens):
            if ref_row is not None:
                _fill_np_variants("REF_NP_DEF", ref_row)
            else:
                for tok in [
                    "REF_NP_DEF", "REF_NP_DEF_EN", "REF_NP_DEF_ES",
                    "REF_NP_DEF_PL", "REF_NP_DEF_PL_EN", "REF_NP_DEF_PL_ES",
                ]:
                    repl.setdefault(tok, "")

        # REF_NP_DEF is typically the subject of the 2nd clause; store metadata for later blocks (copula + adjective).
        ref_is_subject_en = bool(re.search(r"\.\s*\{REF_NP_DEF(?:_PL)?_EN\}\s*\{BE_EN\}", en_tmpl or ""))
        ref_is_subject_es = bool(re.search(r"\.\s*\{REF_NP_DEF(?:_PL)?_ES\}\s*\{COP_ES\}", es_tmpl or ""))

        ref_is_plural = (
            "{REF_NP_DEF_PL}" in tokens or "{REF_NP_DEF_PL_EN}" in tokens or "{REF_NP_DEF_PL_ES}" in tokens
        )

        repl["REF_IS_SUBJ_EN"] = "1" if ref_is_subject_en else "0"
        repl["REF_IS_SUBJ_ES"] = "1" if ref_is_subject_es else "0"
        repl["REF_IS_PL"] = "1" if ref_is_plural else "0"

        # Spanish gender for referent-driven agreement
        if ref_row is not None:
            g = str(ref_row.get("gender_es") or "").strip().lower()
            if g in {"m", "f"}:
                repl["REF_GENDER_ES"] = g
                
        if self._slot_present_any("THEME_POSS_NP", tokens):
            if objects_filled < obj_capacity:
                # Reuse same constraints as THEME_NP
                base_filters_for_poss = {"is_human": False} if possession_context else None
                theme_poss_row = self._pick_noun_for_slot(
                    "THEME_POSS_NP",
                    base_filters=base_filters_for_poss,
                    verb_arg=verb_arg,
                    verb_meta=verb_row
                )
                if theme_poss_row is not None:
                    # default possessor: subject’s person; fall back to 3sg
                    poss_person_theme = person if person in linguistics_core.PERSONS else "3sg"
                    _fill_np_variants("THEME_POSS_NP", theme_poss_row, possessed_person=poss_person_theme)
                    objects_filled += 1
            else:  # Clear if intransitive / no capacity
                for tok in ["THEME_POSS_NP", "THEME_POSS_NP_EN", "THEME_POSS_NP_ES",
                            "THEME_POSS_NP_PL", "THEME_POSS_NP_PL_EN", "THEME_POSS_NP_PL_ES"]:
                    repl.setdefault(tok, "")

        if self._slot_present_any("GOAL_NP", tokens):
            if objects_filled < obj_capacity:
                selected_goal = None
                for _ in range(5):
                    cand = self._pick_noun_for_slot("GOAL_NP", verb_arg=verb_arg, verb_meta=verb_row)
                    if not cand:
                        break

                    theme_id = theme_row.get("id") if theme_row else None
                    goal_id = cand.get("id")
                    if theme_id is None or goal_id != theme_id:
                        selected_goal = cand
                        break

                if selected_goal is not None:
                    goal_row = selected_goal
                    _fill_np_variants("GOAL_NP", selected_goal)
                    objects_filled += 1
            else:  # Clear goal slots if not allowed/filled
                for tok in ["GOAL_NP", "GOAL_NP_EN", "GOAL_NP_ES", "GOAL_NP_PL",
                            "GOAL_NP_PL_EN", "GOAL_NP_PL_ES",
                            "GOAL_REL", "GOAL_REL_EN", "GOAL_REL_ES"]:
                    repl.setdefault(tok, "")

        if self._slot_present_any("AGENT_NP", tokens):
            agent_row = name_row_for_subject if name_row_for_subject else self._pick_noun_for_slot(
                "AGENT_NP",
                verb_arg=verb_arg,
                verb_meta=verb_row
            )
            if agent_row is not None:
                _fill_np_variants("AGENT_NP", agent_row)

        # ---- Existential ----
        if self._slot_present_any("EXIST", tokens):
            # Decide plurality
            def _num_is_plural(nrow: Optional[Dict]) -> bool:
                if not nrow:
                    return False
                en = (str(nrow.get("gloss_en") or "").strip().lower())
                es = (str(nrow.get("gloss_es") or "").strip().lower())
                return not (en in {"1", "one"} or es in {"1", "uno", "un", "una"})

            plural = False
            if chosen_num and ("{NUM_NP_EN}" in en_tmpl or "{NUM_EN}" in en_tmpl):
                plural = _num_is_plural(chosen_num)
            if not plural and re.search(r"\{(?:NP|POSS_NP|NAME)_PL(?:_[A-Z_]+)?_EN\}", en_tmpl):
                plural = True
            if not plural and any(k.endswith("_PL_EN") or ("_PL_" in k and k.endswith("_EN")) for k in repl.keys()):
                plural = True

            # pull templates from linguistics constants
            ph_kek = linguistics_kek.DEFAULT_PLACEHOLDERS
            ph_en  = linguistics_en.DEFAULT_PLACEHOLDERS_EN
            ph_es  = linguistics_es.DEFAULT_PLACEHOLDERS_ES

            # KEK existential (sing/pl)
            repl["EXIST"] = (ph_kek["EXIST_PL"] if plural else ph_kek["EXIST"])

            # EN / ES existential (sing/pl + neg)
            repl["EXIST_EN"]     = ph_en["EXIST_PL_EN"]     if plural else ph_en["EXIST_EN"]
            repl["EXIST_NEG_EN"] = ph_en["EXIST_NEG_PL_EN"] if plural else ph_en["EXIST_NEG_EN"]
            repl["EXIST_ES"]     = ph_es["EXIST_PL_ES"]     if plural else ph_es["EXIST_ES"]
            repl["EXIST_NEG_ES"] = ph_es["EXIST_NEG_PL_ES"] if plural else ph_es["EXIST_NEG_ES"]

        # --- Combined NUM+NP slots ---
        if chosen_num:
            # 1) Bare NUM_NP: ensure we have a base NP even if the template
            # never uses a plain {NP...} placeholder.
            if self._slot_present_any("NUM_NP", tokens):
                if np_noun is None:
                    np_noun = self._pick_noun_for_slot("NP", base_filters=np_filters_for_copular_pred, verb_arg=verb_arg, verb_meta=verb_row)
                    if np_noun is not None and self._slot_present_any("NP", tokens):
                        # Also populate plain NP variants if the template happens
                        # to contain them (keeps behavior consistent).
                        _fill_np_variants("NP", np_noun)
                if np_noun is not None:
                    _fill_num_np_variants("NUM_NP", np_noun, chosen_num)

            # 2) NUM_THEME_NP: allow numeric theme NP even if there is no plain THEME_NP.
            if self._slot_present_any("NUM_THEME_NP", tokens):
                theme_was_new = False
                if theme_row is None and obj_capacity >= (objects_filled + 1):
                    # If this is a possession context, avoid human nouns as objects
                    base_filters_for_poss = {"is_human": False} if possession_context else None
                    theme_row = self._pick_noun_for_slot(
                        "THEME_NP",
                        base_filters=base_filters_for_poss,
                        verb_arg=verb_arg,
                        verb_meta=verb_row,
                    )
                    if theme_row is not None:
                        theme_was_new = True
                        # Mirror the auto-possession behavior from the THEME_NP block
                        is_oblig_poss = utils._s(theme_row.get("possessability")).lower() == "obligatory"
                        if is_oblig_poss:
                            poss_person_theme = (
                                default_person_for_agreement
                                if default_person_for_agreement in linguistics_core.PERSONS
                                else "3sg"
                            )
                            if self._slot_present_any("THEME_NP", tokens):
                                _fill_np_variants("THEME_NP", theme_row, possessed_person=poss_person_theme)
                        else:
                            if self._slot_present_any("THEME_NP", tokens):
                                _fill_np_variants("THEME_NP", theme_row)
                        objects_filled += 1

                if theme_row is not None:
                    _fill_num_np_variants("NUM_THEME_NP", theme_row, chosen_num)
                    
            # 3) NUM_GOAL_NP: allow numeric goal NP even if there is no plain GOAL_NP.
            if self._slot_present_any("NUM_GOAL_NP", tokens):
                goal_was_new = False
                if goal_row is None and objects_filled < obj_capacity:
                    selected_goal = None
                    for _ in range(5):
                        cand = self._pick_noun_for_slot("GOAL_NP", verb_arg=verb_arg, verb_meta=verb_row)
                        if not cand:
                            break

                        theme_id = theme_row.get("id") if theme_row else None
                        goal_id = cand.get("id")
                        if theme_id is None or goal_id != theme_id:
                            selected_goal = cand
                            break

                    if selected_goal is not None:
                        goal_row = selected_goal
                        goal_was_new = True
                        if self._slot_present_any("GOAL_NP", tokens):
                            _fill_np_variants("GOAL_NP", selected_goal)
                        objects_filled += 1

                if goal_row is not None:
                    _fill_num_np_variants("NUM_GOAL_NP", goal_row, chosen_num)
        
        # --- START: INTEGRATED PRONOUN LOGIC BLOCK ---

        # Check if the template needs any subject pronoun (new-style or legacy)
        needs_any_pron = (
            self._slot_present_any("SUBJ_PRONOUN", tokens) or
            self._slot_present_any("SUBJ_PRONOUN_EN", tokens) or
            self._slot_present_any("SUBJ_PRONOUN_ES", tokens) or
            self._slot_present_any("SUBJ_PRON", tokens)      or
            self._slot_present_any("SUBJ_PRON_EN", tokens)   or
            self._slot_present_any("SUBJ_PRON_ES", tokens)
        )

        # Decide whether the subject is plural, using the central helper that
        # looks only at SUBJECT-side markers (NP/AGENT/NAME) and the person hint.
        # Prefer the person inferred from the Subject field, fall back to the
        # caller-provided `person` if needed.
        subj_is_plural_generic = self._subject_is_plural(
            tokens,
            person=subj_person or person
        )

        # Choose person:
        # Always allow the template's plural markers to upgrade a singular default person.
        if needs_any_pron:
            # Reuse the existing inference helper so SUBJECT plural markers
            # (NP_PL / AGENT_NP_PL / NAME_PL) properly flip 3sg → 3pl.
            inferred_person = self._infer_person(
                tokens,
                default_person=default_person_for_agreement
            )
        else:
            # No explicit subject pronoun in the template → simple 3sg/3pl choice
            # based solely on whether the subject is plural.
            inferred_person = "3pl" if subj_is_plural_generic else "3sg"

        # --- START: PRONOUN FILLING (tokens-based, self-contained) ---

        # Determine which pronoun slots are actually present in this template
        needs_kek_pron = self._slot_present_any("SUBJ_PRONOUN", tokens)
        needs_kek_affix  = self._slot_present_any("AFFIX_PRONOUN", tokens)
        
        needs_en_pron  = self._slot_present_any("SUBJ_PRONOUN_EN", tokens)
        needs_es_pron  = self._slot_present_any("SUBJ_PRONOUN_ES", tokens)

        # Choose a shared referential gender once per sentence (only relevant for 3sg),
        # then realize EN/ES from that shared value (avoids EN/ES mismatch).
        ref_gender = None  # "m" | "f" | None
        if inferred_person == "3sg" and (needs_en_pron or needs_es_pron):
            # If we already computed/stored it earlier in this run, reuse it.
            ref_gender = (repl.get("REF_GENDER") or "").strip().lower() or None

            # If the template contains {NAME...}, we treat NAME as the default antecedent
            # for gendered pronouns (EN/ES). This is important for two-sentence templates
            # where the pronoun in sentence 2 refers back to the NAME in sentence 1.
            template_has_name = (
                self._slot_present_any("NAME", tokens) or
                self._slot_present_any("NAME_EN", tokens) or
                self._slot_present_any("NAME_ES", tokens)
            )

            # Heuristic: if we have a Spanish-gendered noun/name, follow it.
            # Otherwise, randomize ONCE and share across EN + ES.
            if ref_gender not in {"m", "f"}:
                g = None

                # Prefer NAME gender if NAME is present (coreference default).
                rows_in_preference_order = (
                    (name_row_for_subject, np_noun, poss_noun) if template_has_name
                    else (np_noun, poss_noun, name_row_for_subject)
                )

                for row in rows_in_preference_order:
                    if row and str(row.get("gender_es") or "").strip().lower() in {"m", "f"}:
                        g = str(row.get("gender_es")).strip().lower()
                        break

                ref_gender = g or self.rng.choice(["m", "f"])

            repl["REF_GENDER"] = ref_gender
            
        # Fill pronouns based on inferred_person
        if needs_kek_pron:
            repl["SUBJ_PRONOUN"] = linguistics_kek.SUBJ_PRONOUNS.get(inferred_person, "")

        if needs_kek_affix:
            repl["AFFIX_PRONOUN"] = linguistics_kek.AFFIX_PRONOUNS.get(inferred_person, "")

        if needs_en_pron:
            repl["SUBJ_PRONOUN_EN"] = linguistics_en.en_subj_pronoun(
                person=inferred_person,
                gender=ref_gender,
                rng=self.rng
            )

        if needs_es_pron:
            repl["SUBJ_PRONOUN_ES"] = linguistics_es.es_subj_pronoun(
                person=inferred_person,
                gender=ref_gender,
                rng=self.rng
            )

        # Passive agent “by X” phrases (AGENT_BY* placeholders).
        if (
            self._slot_present_any("AGENT_BY", tokens)
            or self._slot_present_any("AGENT_BY_EN", tokens)
            or self._slot_present_any("AGENT_BY_ES", tokens)
        ):
            agent_person = inferred_person or default_person_for_agreement or "3sg"
            repl.setdefault("AGENT_BY", linguistics_kek.kek_agent_by(agent_person))
            repl.setdefault("AGENT_BY_EN", linguistics_en.agent_by_en(agent_person, gender=ref_gender, rng=self.rng))
            repl.setdefault("AGENT_BY_ES", linguistics_es.es_agent_by(agent_person, gender=ref_gender, rng=self.rng))
        
        # --- END: PRONOUN FILLING ---
        # --- END: INTEGRATED PRONOUN LOGIC BLOCK ---

        # NEW: Fill possessum NP surfaces now that we know inferred_person (the possessor)
        if possessum_row is not None and self._slot_present_any("POSSESSUM_NP", tokens):
            _fill_np_variants("POSSESSUM_NP", possessum_row, possessed_person=inferred_person)
    
        # Fill numeral+possessum surfaces
        if chosen_num and self._slot_present_any("NUM_POSSESSUM_NP", tokens):
            if possessum_row is None:
                possessum_row = self._pick_noun_for_slot(
                    "POSSESSUM_NP",
                    base_filters={"possessability": {"optional", "obligatory"}},
                    verb_arg=verb_arg,
                    verb_meta=verb_row,
                )
            if possessum_row is not None:
                _fill_num_np_variants(
                    "NUM_POSSESSUM_NP",
                    possessum_row,
                    chosen_num,
                    possessed_person=inferred_person
                )
        
        # --- Q'eqchi' stative future suffixes (for future stative templates) ---
        if (
            ("{ST_FUT_AQ}" in tokens) or
            ("{ST_FUT_IND}" in tokens) or
            ("{ST_FUT_OPT}" in tokens) or
            ("{ST_FUT_DBT}" in tokens) or
            ("{DOUBT_PART}" in tokens)
        ):
            repl.setdefault("ST_FUT_AQ", linguistics_kek.STATIVE_FUT_AQ)
            repl.setdefault("ST_FUT_IND", linguistics_kek.STATIVE_FUT_IND)
            repl.setdefault("ST_FUT_OPT", linguistics_kek.STATIVE_FUT_OPT)
            repl.setdefault("ST_FUT_DBT", linguistics_kek.STATIVE_FUT_DBT)

        if "{DOUBT_PART}" in tokens:
            repl.setdefault("DOUBT_PART", self.rng.choice(linguistics_kek.STATIVE_DOUBT_PARTICLES))
        
        # Imperative person has been resolved earlier into tmpl_imp_person.
        # Make it available to per-language renderers via env.
        # Q'eqchi' uses env['IMP_PERSON'] for positive/negative imperative forms.
        # English uses imp_person to build directive forms (IMP / IMP_NEG).
        # Spanish uses imp_person to select tú / usted / ustedes / nosotros forms.
        if tmpl_imp_person:
            repl["IMP_PERSON"] = tmpl_imp_person

        # --- Verb rendering (per-language; merge dicts) ---
        if chosen_v_intr:
            repl.update(linguistics_kek.render_verb_bundle_kek(
                chosen_v_intr, "V_INTR",
                person=inferred_person, transitivity="intr", obj_person=None, env=repl
            ))
            repl.update(linguistics_en.render_verb_bundle_en(
                chosen_v_intr, "V_INTR",
                person=inferred_person, imp_person=tmpl_imp_person
            ))
            repl.update(linguistics_es.render_verb_bundle_es(
                chosen_v_intr, "V_INTR",
                person=inferred_person, imp_person=tmpl_imp_person
            ))

        if chosen_v_tr:
            repl.update(linguistics_kek.render_verb_bundle_kek(
                chosen_v_tr, "V_TR",
                person=inferred_person, transitivity="tr", obj_person=obj_person_tr, env=repl
            ))
            repl.update(linguistics_en.render_verb_bundle_en(
                chosen_v_tr, "V_TR",
                person=inferred_person, imp_person=tmpl_imp_person
            ))
            repl.update(linguistics_es.render_verb_bundle_es(
                chosen_v_tr, "V_TR",
                person=inferred_person, imp_person=tmpl_imp_person
            ))

        if chosen_v_ditr:
            repl.update(linguistics_kek.render_verb_bundle_kek(
                chosen_v_ditr, "V_DITR",
                person=inferred_person, transitivity="ditr", obj_person=obj_person_ditr, env=repl
            ))
            repl.update(linguistics_en.render_verb_bundle_en(
                chosen_v_ditr, "V_DITR",
                person=inferred_person, imp_person=tmpl_imp_person
            ))
            repl.update(linguistics_es.render_verb_bundle_es(
                chosen_v_ditr, "V_DITR",
                person=inferred_person, imp_person=tmpl_imp_person
            ))

        # Central Spanish subject agreement info (reused for adjectives, occupations, etc.)
        subj_row_es, subj_gender_es, subj_is_plural_es = linguistics_es.subject_agreement_es(
            es_tmpl,
            np_noun,
            poss_noun,
            name_row_for_subject,
            inferred_person,
        )

        # --- Randomize gender for 1sg/2sg pronoun-only subjects in copular Spanish sentences ---
        #
        # Rationale:
        #   - EN/ES 1sg/2sg pronouns (I/you, yo/tú) are not morphologically gendered.
        #   - We still want roughly half of these copular sentences to behave as if the subject
        #     were masculine vs. feminine, so that Spanish predicative adjectives and occupations
        #     get both masculine and feminine forms.
        #
        # Conditions:
        #   * Spanish template uses a subject pronoun placeholder.
        #   * No explicit NP/NAME subject placeholder on the ES side.
        #   * Person is 1sg or 2sg.
        #   * Copular context (ser/estar via {COP_ES}/{COP_NEG_ES} or {V_SER_ES}).
        uses_subj_pron_es = ("{SUBJ_PRONOUN_ES}" in es_tmpl) or ("{SUBJ_PRON_ES}" in es_tmpl)
        has_es_subject_np = any(tag in es_tmpl for tag in (
            "{NP_ES}", "{NP_PL_ES}",
            "{POSS_NP_ES}", "{POSS_NP_PL_ES}",
            "{AGENT_NP_ES}", "{AGENT_NP_PL_ES}",
            "{NAME_ES}",
        ))
        is_copular_es = any(tag in es_tmpl for tag in ("{COP_ES}", "{COP_NEG_ES}", "{V_SER_ES}"))

        if (
            is_copular_es
            and uses_subj_pron_es
            and not has_es_subject_np
            and inferred_person in {"1sg", "2sg"}
        ):
            # Flip a coin for feminine vs. masculine subject.
            subj_gender_es = self.rng.choice(["m", "f"])

        ## OCCUPATION BLOCK ##

        # Sentences with an occupation
        if self._slot_present_any("OCCUPATION", tokens):
            occ_row = self._pick_noun_for_slot(
                "OCCUPATION",
                base_filters={"class": "occupation"},
                verb_arg=verb_arg,
                verb_meta=verb_row,
            )
            if occ_row is not None:
                _fill_np_variants("OCCUPATION", occ_row)

                # Determine a shared referential gender for predicative occupations
                # so EN and ES can both choose a feminine form when appropriate.
                #
                # Primary signal: subj_gender_es from subject_agreement_es
                # (based on NP gender_es, etc.).
                # Secondary signal: pure-pronoun Spanish subject 'ella' with no noun row.
                ref_gender = None
                if subj_gender_es == "f":
                    ref_gender = "f"
                elif repl.get("SUBJ_PRONOUN_ES", "").strip().lower() == "ella":
                    ref_gender = "f"

                # EN: may use gloss_en_f when ref_gender == 'f' and such a form exists.
                repl["OCCUPATION_EN"] = linguistics_en.build_np_en(
                    occ_row,
                    definite=False,
                    ref_gender=ref_gender,
                )

                # ES: already supports gloss_es_f + feminine article via ref_gender.
                repl["OCCUPATION_ES"] = linguistics_es.build_np_es(
                    occ_row,
                    definite=False,
                    ref_gender=ref_gender,
                )

        # Adjectives
        chosen_adj = None
        # Need an adjective if any of these slots are present (including comparatives)
        need_adj_slots = ("ADJ", "ADJ_EN", "ADJ_ES", "ADJ_COMP_EN")
        need_adj_for_any_slot = any("{" + k + "}" in tokens for k in need_adj_slots)
        
        # Recompute subject proxy for adjective/adverb semantic checks *after* NP/POSS/NAME selection
        # and after shared referential gender (REF_GENDER) has been established.
        subj_for_adj_adv_checks = np_noun or poss_noun or name_row_for_subject

        # Adjective-only pronoun templates (e.g., "{ADJ}{AFFIX_PRONOUN}.") have no NP/NAME/POSS_NP.
        # Use a generic "person" proxy, and for 3sg ensure gender matches the centralized REF_GENDER.
        if subj_for_adj_adv_checks is None and need_adj_for_any_slot:
            proxy = {"class": "person", "has_color": "0"}

            if inferred_person == "3sg":
                g = (repl.get("REF_GENDER") or "").strip().lower()
                if g not in {"m", "f"}:
                    g = self.rng.choice(["m", "f"])
                    repl["REF_GENDER"] = g  # central source of truth
                proxy["gender_es"] = g

            subj_for_adj_adv_checks = proxy

        if need_adj_for_any_slot and not self.adjs.empty:
            needs_comparative = self._slot_present_any("ADJ_COMP_EN", tokens)
            # Adjective types that should NOT be used in comparative constructions
            banned_comp_types = {"color", "shape"}

            for _ in range(20):
                cand = self._pick_adjective_for_subject(subj_for_adj_adv_checks)
                if cand is None:
                    continue

                # Basic semantic compatibility check
                if not linguistics_core.adj_compatible_with_noun(cand, subj_for_adj_adv_checks):
                    continue

                # If the template uses a comparative, filter out disallowed adj types
                if needs_comparative:
                    adj_type = utils._s(cand.get("adj_type")).strip().lower()
                    if adj_type in banned_comp_types:
                        continue

                chosen_adj = cand
                break

        # Copulas for simple noun-predicate templates
        # English: present BE and its negation should agree with the *clause subject*.
        be_person = inferred_person
        if repl.get("REF_IS_SUBJ_EN") == "1":
            be_person = "3pl" if repl.get("REF_IS_PL") == "1" else "3sg"

        if "BE_EN" not in repl:
            repl["BE_EN"] = linguistics_en.EN_BE.get(be_person, "is")
        if "BE_NEG_EN" not in repl:
            # Use contracted negatives (isn't / aren't) for consistency with DO_NEG_EN
            repl["BE_NEG_EN"] = linguistics_en.EN_BE_NEG_CT.get(be_person, "isn't")
        # English: future BE for future stative templates
        if "BE_FUT_EN" not in repl:
            repl["BE_FUT_EN"] = linguistics_en.EN_BE_FUT.get(be_person, "will be")
            
        # Spanish (present of 'ser')
        repl["V_SER_ES"] = linguistics_es.es_present("ser", inferred_person)

        # If the ES template carries {COP_ES}/{COP_NEG_ES} but we didn't set them
        # (because no adjective was realized), choose the copula:
        #   - use SER by default
        #   - but for location questions with {WH_LOC_ES}, pick via noun class.
        if (
            ("{COP_ES}" in es_tmpl) or ("{COP_NEG_ES}" in es_tmpl) or
            ("{COP_FUT_ES}" in es_tmpl) or ("{COP_NEG_FUT_ES}" in es_tmpl)
        ):
            # Default: SER (non-location predicates)
            cop_choice = "ser"

            if "{WH_LOC_ES}" in es_tmpl:
                # We are in a 'where' question. Try to figure out *what* is being located
                # by looking at which NP slot appears in the ES template.
                noun_row_for_cop = None

                # Prefer possessed NP for patterns like: ¿Dónde está mi casa?
                if "{POSS_NP_ES}" in es_tmpl and poss_noun is not None:
                    noun_row_for_cop = poss_noun
                # Then plain NP: ¿Dónde está el perro?
                elif "{NP_ES}" in es_tmpl and np_noun is not None:
                    noun_row_for_cop = np_noun
                # Finally, proper name subject: ¿Dónde está Yadira?
                elif "{NAME_ES}" in es_tmpl and name_row_for_subject is not None:
                    noun_row_for_cop = name_row_for_subject

                if noun_row_for_cop is not None:
                    noun_class = (noun_row_for_cop.get("class") or "").strip().lower()
                    cop_choice = linguistics_es.spanish_copula_for_location(noun_class)
                else:
                    # Fallback if we couldn't identify a noun row:
                    # treat as a generic physical entity → ESTAR is safer.
                    cop_choice = "estar"

            # Present copulas
            if "{COP_ES}" in es_tmpl:
                repl.setdefault("COP_ES", linguistics_es.es_present(cop_choice, inferred_person))
            if "{COP_NEG_ES}" in es_tmpl:
                repl.setdefault("COP_NEG_ES", f"no {linguistics_es.es_present(cop_choice, inferred_person)}")

            # Future copulas
            if "{COP_FUT_ES}" in es_tmpl:
                repl.setdefault("COP_FUT_ES", linguistics_es.es_future_periphrastic(cop_choice, inferred_person))
            if "{COP_NEG_FUT_ES}" in es_tmpl:
                repl.setdefault("COP_NEG_FUT_ES", f"no {linguistics_es.es_future_periphrastic(cop_choice, inferred_person)}")

        # POSSESSION
        # "Have" (EN/ES verbs; KEK existential + with-pronoun)

        # Detect possession needs via tokens (language-specific)
        needs_poss_en = any(
            tok.startswith("{V_POSS_TR_") and tok.endswith("_EN}")
            for tok in tokens
        )
        needs_poss_es = any(
            tok.startswith("{V_POSS_TR_") and tok.endswith("_ES}")
            for tok in tokens
        )
        # Trigger KEK possession logic if the template carries {EXIST} or {WITH_PRON}
        needs_poss_kek = (
            self._slot_present_any("EXIST", tokens) or
            self._slot_present_any("WITH_PRON", tokens)
        )

        if needs_poss_en or needs_poss_es or needs_poss_kek:
            # --- English 'have' for possession ---
            if needs_poss_en:
                p = inferred_person
                repl["V_POSS_TR_BASE_EN"] = "have"
                repl["V_POSS_TR_EN"]      = linguistics_en.EN_HAVE.get(p, "have")
                repl["V_POSS_TR_PST_EN"]  = linguistics_en.EN_HAVE_PST.get(p, "had")
                repl["V_POSS_TR_FUT_EN"]  = linguistics_en.EN_HAVE_FUT.get(p, "will have")
                # Avoid unidiomatic progressive for stative possession
                # If a template asks for a progressive form, reuse simple present.
                repl["V_POSS_TR_PRG_EN"]  = repl["V_POSS_TR_EN"]
                # Provide the participle for perfects: {HAVE_EN} {V_POSS_TR_PP_EN}
                repl["V_POSS_TR_PP_EN"]   = "had"

            # --- Spanish 'tener' for possession ---
            if needs_poss_es:
                repl["V_POSS_TR_ES"]        = linguistics_es.es_present("tener", inferred_person)
                repl["V_POSS_TR_PST_ES"]    = linguistics_es.es_past("tener", inferred_person)
                repl["V_POSS_TR_FUT_ES"]    = linguistics_es.es_future_periphrastic("tener", inferred_person)
                repl["V_POSS_TR_PRG_ES"]    = linguistics_es.es_progressive("tener", inferred_person)
                repl["V_POSS_TR_BASE_ES"]   = "tener"

            # --- KEK existential + with-pronoun ---
            if needs_poss_kek:
                # Decide plurality based on:
                #   1) Explicit plural NP markers in subject-side tokens
                #   2) Already-filled *_PL replacements
                #   3) NUM+NP cases where the numeral != 1
                def _num_is_plural(nrow: Optional[Dict]) -> bool:
                    if not nrow:
                        return False
                    en = (str(nrow.get("gloss_en") or "").strip().lower())
                    es = (str(nrow.get("gloss_es") or "").strip().lower())
                    return not (en in {"1","one"} or es in {"1","uno","un","una"})

                def _np_plural_in_this_template() -> bool:
                    # 1) direct subject-side plural markers among tokens
                    subject_pl_tokens = (
                        "{AGENT_NP_PL}", "{AGENT_NP_PL_EN}", "{AGENT_NP_PL_ES}",
                        "{NP_PL}", "{NP_PL_EN}", "{NP_PL_ES}",
                        "{NP_PL_INDEF}", "{NP_PL_INDEF_EN}", "{NP_PL_INDEF_EN}",
                        "{POSS_NP_PL}", "{POSS_NP_PL_EN}", "{POSS_NP_PL_ES}",
                    )
                    if any(tok in tokens for tok in subject_pl_tokens):
                        return True
                    # 2) any already-filled *_PL replacement
                    for k in repl:
                        if k.endswith("_PL") or "_PL_" in k:
                            return True
                    # 3) NUM+NP detected and numeral ≠ 1 → plural
                    if (
                        self._slot_present_any("NUM_NP", tokens) or
                        self._slot_present_any("NUM_THEME_NP", tokens) or
                        self._slot_present_any("NUM_GOAL_NP", tokens) or
                        "{NUM_EN}" in tokens or
                        "{NUM_ES}" in tokens
                    ):
                        if _num_is_plural(chosen_num):
                            return True
                    return False

                repl["EXIST"] = "wankeb’" if _np_plural_in_this_template() else "wan"
                repl["WITH_PRON"]  = linguistics_kek.kek_with_pron(inferred_person)

        # Auxiliary for negation (English: doesn't/don't)
        if self._slot_present_any("DO_NEG_EN", tokens):
            repl["DO_NEG_EN"] = linguistics_en.en_do_neg(inferred_person)

        # Affirmative auxiliary for questions/emphasis (English: do)
        if self._slot_present_any("DO_AFF_EN", tokens):
            repl["DO_AFF_EN"] = linguistics_en.en_do_aff(inferred_person)

        # Auxiliary for future (Spanish: ir)
        repl["IR_ES"]    = linguistics_es.es_ir_present(inferred_person)

        # --- Adjective ---

        # Ensure we have an adjective whenever the template needs {ADJ*} or {COP_ES}
        need_adj_for_template = (
            self._slot_present_any("ADJ", tokens) or
            self._slot_present_any("ADJ_EN", tokens) or
            self._slot_present_any("ADJ_ES", tokens) or
            self._slot_present_any("ADJ_COMP_EN", tokens)
        )
        if chosen_adj is None and need_adj_for_template:
            # STRICT: if no compatible adjective, abort this template instance
            return None

        if chosen_adj is not None:
            # pick_surface was removed in the refactor; just take the three fields directly
            kek_adj = utils._s(chosen_adj.get("lemma_kek"))
            en_adj  = utils._s(chosen_adj.get("gloss_en"))
            es_adj_base = utils._s(chosen_adj.get("gloss_es"))
            repl["ADJ"] = kek_adj
            repl["ADJ_EN"] = en_adj
            
            # English comparative form, if requested by the template
            if self._slot_present_any("ADJ_COMP_EN", tokens):
                repl["ADJ_COMP_EN"] = linguistics_en.build_adj_comparative_en(chosen_adj)

            # Spanish agreement & copula (reuse central subject-agreement helper)
            #
            # If REF_NP_DEF_ES is the clause subject (e.g., "... . {REF_NP_DEF_ES} {COP_ES} {ADJ_ES}"),
            # override agreement and copula conjugation to 3sg/3pl and use the referent noun's gender.
            cop_person_es = inferred_person
            if repl.get("REF_IS_SUBJ_ES") == "1":
                # number: prefer the REF plural flag (set earlier); fall back to non-plural
                subj_is_plural_es = (repl.get("REF_IS_PL") == "1")
                cop_person_es = "3pl" if subj_is_plural_es else "3sg"

                # gender: pull from the referent noun row if available
                if theme_row is not None:
                    g = utils._s(theme_row.get("gender_es")).strip().lower()
                    if g in {"m", "f"}:
                        subj_gender_es = g
                    else:
                        subj_gender_es = "m"
                else:
                    subj_gender_es = "m"

            # Adjective
            es_adj_base = (chosen_adj.get("gloss_es") or "").strip()
            agreed_adj_es = linguistics_es.adj_agree_es(
                chosen_adj,
                es_adj_base,
                subj_gender_es,
                subj_is_plural_es,
            )
            repl["ADJ_ES"] = agreed_adj_es

            # Decide copula type (ser vs estar) based on adjective metadata
            cop_type = linguistics_es.pick_copula_es(chosen_adj, subj_is_plural_es)
            # Conjugate copula by the *actual clause subject* (REF -> 3sg/3pl), otherwise inferred_person
            cop_form = linguistics_es.es_present(cop_type, cop_person_es)
            repl["COP_ES"] = cop_form
            repl["COP_NEG_ES"] = f"no {cop_form}"

        # --- Adverb ---
        if self._slot_present_any("ADV", tokens):
            # prefer ditransitive > transitive > intransitive as the "active" verb
            active_verb = chosen_v_ditr or chosen_v_tr or chosen_v_intr
            verb_category = utils._s(active_verb.get("category")).lower() if active_verb else ""

            tense_flags = self._compute_tense_flags(tokens)

            # Future stative templates (Q'eqchi stative future suffixes and/or EN/ES future copulas)
            is_future_stative = (
                ("ST_FUT_AQ" in tokens) or
                ("ST_FUT_IND" in tokens) or
                ("ST_FUT_OPT" in tokens) or
                ("ST_FUT_DBT" in tokens) or
                ("BE_FUT_EN" in tokens) or
                ("COP_FUT_ES" in tokens) or
                ("COP_NEG_FUT_ES" in tokens)
            )

            if is_future_stative and tense_flags.get("fut"):
                chosen_adv = self.pick_time_adverb_future()
            else:
                chosen_adv = self.pick_adverb(tense_flags, verb_category)

            if chosen_adv:
                repl["ADV"]    = utils._s(chosen_adv.get("lemma_kek"))
                repl["ADV_EN"] = utils._s(chosen_adv.get("gloss_en"))
                repl["ADV_ES"] = utils._s(chosen_adv.get("gloss_es"))

        # --- Numerals ---
        if chosen_num:
            repl["NUM"]    = utils._s(chosen_num.get("lemma_kek"))
            repl["NUM_EN"] = utils._s(chosen_num.get("gloss_en"))
            repl["NUM_ES"] = utils._s(chosen_num.get("gloss_es"))

        # --- Placeholders ---
        for k, v in linguistics_kek.DEFAULT_PLACEHOLDERS.items(): repl.setdefault(k, v)
        for k, v in linguistics_en.DEFAULT_PLACEHOLDERS_EN.items():  repl.setdefault(k, v)
        for k, v in linguistics_es.DEFAULT_PLACEHOLDERS_ES.items():  repl.setdefault(k, v)

        # Language-specific adjective placement (lives in linguistics_* modules)
        en_tmpl, repl = linguistics_en.embed_adj_into_np_in_template_en(en_tmpl, repl)
        es_tmpl, repl = linguistics_es.embed_adj_into_np_in_template_es(es_tmpl, repl)

        # --- Final Sentence Construction ---
        kek_sent = re.sub(r"\s+"," ", kek_tmpl.format_map(utils._SafeDict(repl))).strip()
        # Apply possessive morphology (li + Set A) to KEK
        kek_sent = linguistics_kek.style_kek_possessives(kek_sent)
        # Make sure affix/clitic pronouns (in/at/o/ex/eb') is attached DIRECTLY to the 
        # preceding token, e.g., "tz'iib' in." -> "tz'iib'in."
        kek_sent = linguistics_kek.style_kek_affix_pronouns(kek_sent)
        # Enforce 'moko ... ta' negation enclosure of a single word
        kek_sent = linguistics_kek.style_kek_moko_ta(kek_sent)
        # Clean up empty-affix artifacts (e.g. trailing '-') ---
        kek_sent = utils.cleanup_qeqchi_surface(kek_sent)
        en_sent  = re.sub(r"\s+"," ", en_tmpl.format_map(utils._SafeDict(repl))).strip()
        es_sent  = re.sub(r"\s+"," ", es_tmpl.format_map(utils._SafeDict(repl))).strip()

        # Clean-up
        # Sentence-case
        kek_sent = utils._sentence_case(kek_sent)
        kek_sent = utils.capitalize_sentences(kek_sent)

        en_sent  = utils._sentence_case(en_sent)
        en_sent  = utils.capitalize_sentences(en_sent)

        es_sent  = utils._sentence_case(es_sent)
        es_sent  = utils.capitalize_sentences(es_sent)

        # Remove unwanted space before punctuation
        kek_sent = re.sub(r"\s+([.?!,;:])", r"\1", kek_sent)
        en_sent  = re.sub(r"\s+([.?!,;:])", r"\1", en_sent)
        es_sent  = re.sub(r"\s+([.?!,;:])", r"\1", es_sent)

        # Final semantic validation
        chosen_for_check = {
            "adj": chosen_adj,
            "subj": subj_for_adj_adv_checks,
            "verb_tr":   chosen_v_tr,
            "verb_ditr": chosen_v_ditr,
            "template_has_theme": (
                self._slot_present_any("THEME_NP", tokens) or
                self._slot_present_any("THEME_POSS_NP", tokens)
            ),
            "template_has_goal":  self._slot_present_any("GOAL_NP", tokens),
            "theme_row": theme_row,
            "goal_row":  goal_row,
        }
        if not self._validate_semantics(int(t["id"]), chosen_for_check):
            return None

        return {
            "id": template_id,
            "kek": kek_sent,
            "en":  en_sent,
            "es":  es_sent,
        }

    def render_many(
            self,
            n: int,
            *,
            person: str = "3sg",
            allow_ids: Optional[List[int]] = None,
            exclude_en_set: Optional[set] = None, # <--- NEW ARGUMENT
        ) -> List[Dict[str, str]]:
            """
            Generate exactly n *attempted* sentences, distributing attempts over
            templates according to the p_value_norm column.

            - n: total desired number of *unique* sentences.
            - allow_ids: optional subset of template IDs to consider.
            - exclude_en_set: optional set of English sentences to strictly avoid (global history).
            """
            # 1) Select templates to use
            if allow_ids:
                df = self.templates[self.templates["id"].isin(allow_ids)].copy()
            else:
                df = self.templates.copy()

            if df.empty:
                return []

            # 2) Get probabilities for these templates
            if "p_value_norm" in df.columns:
                probs = [float(x) if x == x else 0.0  # x == x filters out NaN
                        for x in df["p_value_norm"].tolist()]
            else:
                # fallback: uniform probabilities
                probs = [1.0] * len(df)

            # 3) Compute how many times to *try* each template
            counts = _compute_template_counts(probs, total_n=n)

            template_ids = df["id"].tolist()

            out: List[Dict[str, str]] = []
            
            # Local seen set to prevent duplicates within this specific batch
            # (We rely on exclude_en_set for cross-batch duplicates)
            seen_local = set()
            
            tries = 0
            # Increase max_tries buffer to account for global collisions as dataset grows
            max_tries = n * 100 

            # 4) First pass: controlled template distribution
            for tid, c in zip(template_ids, counts):
                if len(out) >= n or tries >= max_tries:
                    break

                for _ in range(c):
                    if len(out) >= n or tries >= max_tries:
                        break

                    tries += 1
                    s = self.render(tid, person=person)
                    if s is None:
                        continue

                    # --- DEDUPLICATION CHECKS ---
                    
                    # 1. Check Global History (passed from main.py)
                    # We dedupe on English because that's what main.py tracks for API costs
                    if exclude_en_set is not None:
                        if s["en"] in exclude_en_set:
                            continue

                    # 2. Check Local History (triplet based)
                    key = (s["kek"], s["en"], s["es"])
                    if key in seen_local:
                        continue

                    seen_local.add(key)
                    out.append(s)

            # 5) Optional top-up: if we’re short because of dedupe / failed renders,
            #    fall back to the original random sampling logic.
            if len(out) < n and tries < max_tries:
                # Re-filter templates to valid ones only
                valid_template_ids = [t for t in template_ids if t is not None]
                
                while len(out) < n and tries < max_tries:
                    tries += 1
                    tid = self.rng.choice(valid_template_ids)
                    s = self.render(tid, person=person)
                    if s is None:
                        continue

                    # --- DEDUPLICATION CHECKS (Same as above) ---
                    if exclude_en_set is not None:
                        if s["en"] in exclude_en_set:
                            continue

                    key = (s["kek"], s["en"], s["es"])
                    if key in seen_local:
                        continue

                    seen_local.add(key)
                    out.append(s)

            return out