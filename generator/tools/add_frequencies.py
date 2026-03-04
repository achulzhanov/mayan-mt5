import pandas as pd
from wordfreq import zipf_frequency
from pathlib import Path
import sys
import argparse

# =============================================================================
# STOPWORDS CONFIGURATION
# =============================================================================
STOPWORDS = {
    # ================= ENGLISH =================
    # Articles & Determiners
    "the", "a", "an", "this", "that", "these", "those", 
    "all", "some", "any", "no", "every", "each",
    
    # Prepositions
    "in", "on", "at", "by", "for", "from", "of", "with", "without", 
    "to", "into", "onto", "upon", "about", "over", "under", "through",
    
    # Pronouns
    "i", "me", "my", "mine", "myself",
    "you", "your", "yours", "yourself",
    "he", "him", "his", "himself",
    "she", "her", "hers", "herself",
    "it", "its", "itself",
    "we", "us", "our", "ours", "ourselves",
    "they", "them", "their", "theirs", "themselves",
    
    # Verbs
    "be", "is", "are", "was", "were", "been", "being",
    "do", "does", "did", "done", "doing",
    "have", "has", "had", "having",
    "get", "got", "gotten", "getting",
    "go", "went", "gone", "going",
    "will", "would", "can", "could", "should", "may", "might", "must",
    
    # Adverbs & Quantifiers
    "very", "more", "most", "less", "least", "too", "so", "just", "only", "quite",
    
    # Conjunctions
    "and", "or", "but", "if", "because", "as", "than", "while", "when",
    
    # Generic Gloss Fillers
    "something", "someone", "somebody", "anything", "anyone", "anybody", 
    "thing", "person", "one", "ones",

    # ================= SPANISH =================
    # Articles & Determiners
    "el", "la", "los", "las", "un", "una", "unos", "unas", "lo",
    "este", "esta", "estos", "estas", "ese", "esa", "esos", "esas", 
    "aquel", "aquella", "aquellos", "aquellas",
    "todo", "toda", "todos", "todas", "cada", "otro", "otra", "otros", "otras",
    
    # Prepositions & Contractions
    "de", "del", "a", "al", "en", "con", "sin", "por", "para", "ante", "bajo", "contra", 
    "desde", "hacia", "hasta", "sobre", "tras", "entre",
    
    # Pronouns & Possessives
    "yo", "tú", "él", "ella", "usted", 
    "nosotros", "nosotras", "ellos", "ellas", "ustedes",
    "mi", "mis", "tu", "tus", "su", "sus", 
    "nuestro", "nuestra", "nuestros", "nuestras",
    "vuestro", "vuestra",
    
    # Object Pronouns
    "me", "te", "le", "les", "nos", "os", "se", "lo", "la", "los", "las",
    
    # Verbs
    "ser", "soy", "eres", "es", "somos", "son", "fue", "fueron", "era", "eran", "sido", "siendo",
    "estar", "estoy", "estás", "está", "estamos", "están", "estaba", "estaban", "estado", "estando",
    "haber", "he", "has", "ha", "hemos", "han", "había", "habían", "habido",
    "hacer", "hago", "hace", "hizo", "hicieron", "hecho", "haciendo",
    "tener", "tengo", "tiene", "tienen", "tenía", "tenido",
    
    # Adverbs & Quantifiers
    "muy", "más", "menos", "tan", "tanto", "poco", "mucho", "mucha", "muchos", "muchas",
    
    # Conjunctions
    "y", "e", "o", "u", "pero", "si", "que", "como", "porque", "cuando", "donde",
    
    # Generic Gloss Fillers
    "algo", "alguien", "cosa", "persona", "gente"
}

def get_smart_frequency(text, lang_code):
    if not isinstance(text, str) or not text.strip():
        return 0.0
    
    clean_text = text.split('(')[0].split('/')[0].strip().lower()
    words = clean_text.split()
    if not words:
        return 0.0

    valid_freqs = []
    
    for w in words:
        if w in STOPWORDS:
            continue
            
        freq = zipf_frequency(w, lang_code)
        
        if freq == 0.0:
            freq = 1.5
            
        valid_freqs.append(freq)
    
    if not valid_freqs:
        return 5.0
        
    return max(valid_freqs)

def process_file(file_path, dry_run=False):
    path = Path(file_path)
    if not path.exists():
        print(f"Skipping {path}: File not found.")
        return

    mode_label = "DRY RUN" if dry_run else "WRITING"
    print(f"\nProcessing {path.name} [{mode_label}]...")
    
    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"  -> Error reading CSV: {e}")
        return
    
    col_en = 'gloss_en'
    col_es = 'gloss_es'
    
    if col_en not in df.columns and col_es not in df.columns:
        print(f"  -> WARNING: No '{col_en}' or '{col_es}' columns found. Skipping.")
        return

    weights = []
    T = 1.5
    
    # Print Header for High Values
    print(f"{'LEMMA':<20} | {'GLOSS EN':<20} | {'GLOSS ES':<20} | {'ZIPF':<5} | {'WEIGHT'}")
    print("-" * 90)

    for _, row in df.iterrows():
        score_en = 0.0
        score_es = 0.0
        
        if col_en in df.columns:
            score_en = get_smart_frequency(row[col_en], 'en')
        if col_es in df.columns:
            score_es = get_smart_frequency(row[col_es], 'es')
            
        base_zipf = max(score_en, score_es)
        adjusted_zipf = base_zipf / T
        linear_weight = 10 ** adjusted_zipf
        
        weights.append(linear_weight)

        # Monitor: Print High Values (Whether Dry Run or Not)
        if linear_weight > 1000:
            lemma = str(row.get('lemma_kek', ''))
            g_en = str(row.get(col_en, ""))[:18]
            g_es = str(row.get(col_es, ""))[:18]
            print(f"{lemma:<20} | {g_en:<20} | {g_es:<20} | {base_zipf:.2f}  | {linear_weight:.1f}")

    if not dry_run:
        df['p_weight'] = weights
        total = sum(weights)
        if total > 0:
            df['p_debug_percent'] = [(w / total) * 100 for w in weights]
        
        df.to_csv(path, index=False)
        print(f"\n  -> Success. Saved changes to {path.name}")
    else:
        print(f"\n  -> [DRY RUN COMPLETE] No changes saved to {path.name}")
        
    print("-" * 50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate frequency weights for Q'eqchi' dataset.")
    parser.add_argument(
        "--dry-run", 
        action="store_true", 
        help="Calculate and print high-frequency words without modifying the CSV files."
    )
    
    args = parser.parse_args()
    
    files_to_process = [
        "data/kek/kek_nouns.csv",
        "data/kek/kek_verbs.csv",
        "data/kek/kek_adjectives.csv",
        "data/kek/kek_adverbs.csv"
    ]
    
    print("Starting Frequency Injection...")
    print("-" * 50)
    
    for f in files_to_process:
        process_file(f, dry_run=args.dry_run)