import pandas as pd
import json
import re
from typing import List, Dict, Tuple

"""Utility Functions - This module holds all small, reusable helper functions."""

_VOWELS = set(list("aeiouAEIOU") + ["ä","ë","ï","ö","ü","Ä","Ë","Ï","Ö","Ü"])
_IGNORABLE = set(["'", "’", "7", "-", " ", "\u200b"])

def _first_effective_char(word: str) -> str:
    for ch in str(word).lstrip():
        if ch in _IGNORABLE:
            continue
        return ch
    return ""

def _is_vowel_initial(word: str) -> bool:
    ch = _first_effective_char(word)
    return ch in _VOWELS

def _split_list(s: str) -> List[str]:
    if pd.isna(s) or s is None:
        return []
    return [x.strip() for x in str(s).split(";") if x.strip()]

def _to_bool(x):
    if isinstance(x, bool):
        return x
    try:
        if pd.isna(x):
            return None
    except Exception:
        pass
    if x is None or str(x).strip() == "":
        return None
    s = str(x).strip().lower()
    if s in ("true","1","yes","y"): return True
    if s in ("false","0","no","n"): return False
    return None

def _s(x) -> str:
    """Safe string: return empty for None/NaN, else str(x)."""
    try:
        if x is None or pd.isna(x):
            return ""
    except Exception:
        if x is None:
            return ""
    return str(x)

def _sentence_case(s: str) -> str:
    """
    Sentence-case with awareness of leading punctuation and digits.

    - Skip over any leading non-alphanumeric characters (quotes, ¿, ¡, parentheses, etc.).
    - If the first non-punctuation character is a digit → leave the string unchanged
      (e.g., '3 gatos corren.' stays as-is).
    - If the first non-punctuation character is a letter → uppercase that letter
      (e.g., '¿dónde estás?' → '¿Dónde estás?').
    """
    if not s:
        return s

    s = s.strip()
    if not s:
        return s

    n = len(s)
    idx = 0

    # Skip leading punctuation/whitespace, but NOT digits or letters
    while idx < n and not s[idx].isalpha() and not s[idx].isdigit():
        idx += 1

    if idx >= n:
        # String is all punctuation/whitespace
        return s

    # If the first non-punctuation char is a digit, don't touch the sentence
    if s[idx].isdigit():
        return s

    # Now we know s[idx] is alphabetic → capitalize it
    return s[:idx] + s[idx].upper() + s[idx + 1:]

def capitalize_sentences(text: str) -> str:
    """
    Capitalize the first alphabetic character after sentence boundaries.
    Handles ., ?, ! followed by whitespace.
    """
    def repl(match):
        punct = match.group(1)
        space = match.group(2)
        char  = match.group(3)
        return f"{punct}{space}{char.upper()}"

    # Capitalize first character of entire string
    text = text[:1].upper() + text[1:] if text else text

    # Capitalize after sentence boundaries
    return re.sub(r"([.!?])(\s+)([a-záéíóúüñ])", repl, text)

def cleanup_qeqchi_surface(text: str) -> str:
    """
    Cleans up Q'eqchi' surface artifacts caused by empty affixes.

    Specifically:
    - removes dangling hyphens at word boundaries (e.g. 'winaq-' -> 'winaq')
    - collapses accidental double hyphens
    """
    if not text:
        return text

    # Remove hyphen if it is the last character of a word
    text = re.sub(r"-\b", "", text)

    # Collapse any accidental multiple hyphens
    text = re.sub(r"--+", "-", text)

    return text

class _SafeDict(dict):
    """Return empty string when a format key is missing."""
    def __missing__(self, key):
        return ""

def save_as_jsonl(rows: List[Dict[str, str]], base_file_path: str, mode: str = 'w'):
    """
    Saves generated sentence pairs to two mirrored JSONL files.
    
    Args:
        rows: List of dictionary rows with keys 'kek', 'en', 'es'.
        base_file_path: The base path/filename (without extension).
        mode: 'w' for overwrite (default), 'a' for append.
    """
    
    # Define language codes
    kek_code = "kek_Latn"
    eng_code = "eng_Latn"
    spa_code = "spa_Latn" 

    # Define file paths
    en_path = f"{base_file_path}_kek_en.jsonl"
    es_path = f"{base_file_path}_kek_es.jsonl"

    # --- Process English-Q'eqchi' File ---
    with open(en_path, mode, encoding='utf-8') as f_en:
        for row in rows:
            kek_text = row.get("kek", "")
            eng_text = row.get("en", "")

            if not kek_text or not eng_text:
                continue 

            # 1. KEK -> ENG
            line1 = {"translation": {"src_lang_code": kek_code, "tgt_lang_code": eng_code, "src_text": kek_text, "tgt_text": eng_text, "type": "synthetic"}}
            f_en.write(json.dumps(line1, ensure_ascii=False) + '\n')
            
            # 2. ENG -> KEK
            line2 = {"translation": {"src_lang_code": eng_code, "tgt_lang_code": kek_code, "src_text": eng_text, "tgt_text": kek_text, "type": "synthetic"}}
            f_en.write(json.dumps(line2, ensure_ascii=False) + '\n')

    # --- Process Spanish-Q'eqchi' File ---
    with open(es_path, mode, encoding='utf-8') as f_es:
        for row in rows:
            kek_text = row.get("kek", "")
            spa_text = row.get("es", "") 

            if not kek_text or not spa_text:
                continue 
                
            # 1. KEK -> SPA
            line1 = {"translation": {"src_lang_code": kek_code, "tgt_lang_code": spa_code, "src_text": kek_text, "tgt_text": spa_text, "type": "synthetic"}}
            f_es.write(json.dumps(line1, ensure_ascii=False) + '\n')
            
            # 2. SPA -> KEK
            line2 = {"translation": {"src_lang_code": spa_code, "tgt_lang_code": kek_code, "src_text": spa_text, "tgt_text": kek_text, "type": "synthetic"}}
            f_es.write(json.dumps(line2, ensure_ascii=False) + '\n')

    # Only print confirmation if we are in write mode or it's the first batch, 
    # otherwise it might spam the console during 40+ batches.
    if mode == 'w':
        print(f"\nSuccessfully saved JSONL files to:")
        print(f"- {en_path}")
        print(f"- {es_path}")