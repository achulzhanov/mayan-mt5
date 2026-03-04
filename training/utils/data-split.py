import json
import random
import argparse
import numpy as np
from transformers import AutoTokenizer

# --- Configuration Defaults ---
INPUT_EN = 'train_kek_en.jsonl'
INPUT_ES = 'train_kek_es.jsonl'
OUTPUT_TRAIN = 'mT5_train_final.jsonl'
OUTPUT_VAL = 'mT5_val_final.jsonl'
OUTPUT_VAL_MINI = 'mT5_val_mini.jsonl'
SPLIT_RATIO = 0.1  # 10% Validation
MODEL_NAME = "google/mt5-base"
MAX_SEQ_LENGTH = 128

def load_and_group_concepts(file_path):
    """
    Reads a JSONL file and groups lines that belong to the same translation concept.
    Returns a list of lists: [ [line1, line2], [line3, line4], ... ]
    """
    print(f"Reading and grouping {file_path}...")
    concepts_map = {}
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    if "translation" in data:
                        t = data["translation"]
                        src_text = t.get("src_text", "").strip()
                        tgt_text = t.get("tgt_text", "").strip()
                        
                        # Unique Fingerprint: Sort texts so (A, B) and (B, A) are identical keys
                        pair_key = tuple(sorted((src_text, tgt_text)))
                        
                        if pair_key not in concepts_map:
                            concepts_map[pair_key] = []
                        concepts_map[pair_key].append(data)
                        
                except json.JSONDecodeError:
                    print(f"Skipping line {line_num}: Invalid JSON")
    except FileNotFoundError:
        print(f"ERROR: Could not find file {file_path}")
        return []
    
    return list(concepts_map.values())

def calculate_metrics(dataset, tokenizer, name="Dataset"):
    """
    Analyzes token lengths and language balance for a given dataset list.
    """
    print(f"   Analyzing {name} ({len(dataset)} lines)...")
    
    lengths = []
    en_count = 0
    es_count = 0
    over_limit_count = 0

    # Cache prefix tokens to avoid re-tokenizing static strings 1 million times
    # Note: This is an approximation. Real length varies slightly by tokenizer merging.
    # But for stats, this is fast and accurate enough.
    
    for obj in dataset:
        t = obj['translation']
        src_lang = t.get('src_lang_code', '')
        tgt_lang = t.get('tgt_lang_code', '')
        
        # Balance Check
        if "eng_Latn" in [src_lang, tgt_lang]:
            en_count += 1
        elif "spa_Latn" in [src_lang, tgt_lang]:
            es_count += 1
            
        # Length Check
        # We manually construct the prompt to match training script exactly
        # "translate synthetic {Lang} to {Lang}: {text}"
        # Simplified for speed: just tokenize the full strings
        
        src_text = t['src_text']
        tgt_text = t['tgt_text']
        
        # Approximate the prompt overhead (usually ~6-8 tokens)
        # "translate synthetic English to Q'eqchi': "
        prompt_overhead = 8 
        
        l_src = len(tokenizer(src_text).input_ids) + prompt_overhead
        l_tgt = len(tokenizer(tgt_text).input_ids)
        
        max_len = max(l_src, l_tgt)
        lengths.append(max_len)
        
        if max_len > MAX_SEQ_LENGTH:
            over_limit_count += 1

    # --- Print Report ---
    print(f"\n--- REPORT: {name} ---")
    print(f"   Total Samples:   {len(dataset):,}")
    print(f"   English Pairs:   {en_count:,} ({en_count/len(dataset):.1%})")
    print(f"   Spanish Pairs:   {es_count:,} ({es_count/len(dataset):.1%})")
    print(f"   ---------------- Length Stats (Tokens) ----------------")
    print(f"   Mean Length:     {int(np.mean(lengths))}")
    print(f"   Max Length:      {np.max(lengths)}")
    print(f"   95th Percentile: {int(np.percentile(lengths, 95))}")
    print(f"   99th Percentile: {int(np.percentile(lengths, 99))}")
    
    if over_limit_count > 0:
        print(f"   WARNING: {over_limit_count:,} samples ({over_limit_count/len(dataset):.2%}) exceed {MAX_SEQ_LENGTH} tokens!")
    else:
        print(f"   All samples within {MAX_SEQ_LENGTH} token limit.")
    print("   -------------------------------------------------------\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--en_input", default=INPUT_EN, help="English input JSONL")
    parser.add_argument("--es_input", default=INPUT_ES, help="Spanish input JSONL")
    parser.add_argument("--train_out", default=OUTPUT_TRAIN, help="Output Training JSONL")
    parser.add_argument("--val_out", default=OUTPUT_VAL, help="Output Validation JSONL")
    parser.add_argument("--val_mini_out", default=OUTPUT_VAL_MINI, help="Output Mini Val JSONL")
    args = parser.parse_args()

    # 1. Load Tokenizer (for metrics)
    print(f"Loading tokenizer ({MODEL_NAME})...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    except Exception as e:
        print(f"Could not load tokenizer: {e}")
        return

    # 2. Load and Group
    en_concepts = load_and_group_concepts(args.en_input)
    es_concepts = load_and_group_concepts(args.es_input)
    
    if not en_concepts and not es_concepts:
        print("No data found. Exiting.")
        return

    print(f"\nUnique Concepts Found:")
    print(f"English: {len(en_concepts):,} groups")
    print(f"Spanish: {len(es_concepts):,} groups")

    # 3. Shuffle Concepts (Safe Step)
    random.seed(42)
    random.shuffle(en_concepts)
    random.shuffle(es_concepts)
    
    # 4. Stratified Split Calculation
    split_idx_en = int(len(en_concepts) * (1 - SPLIT_RATIO))
    split_idx_es = int(len(es_concepts) * (1 - SPLIT_RATIO))
    
    # 5. Create Splits (still grouped)
    train_concepts = en_concepts[:split_idx_en] + es_concepts[:split_idx_es]
    val_concepts = en_concepts[split_idx_en:] + es_concepts[split_idx_es:]
    
    # 6. Flatten to Lines
    train_lines = [line for group in train_concepts for line in group]
    val_lines = [line for group in val_concepts for line in group]
    
    # 7. Final Shuffle (Training Order)
    random.shuffle(train_lines)
    random.shuffle(val_lines)
    
    # 8. Write to Disk
    print("\nWriting output files...")
    with open(args.train_out, 'w', encoding='utf-8') as f:
        for entry in train_lines:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
            
    with open(args.val_out, 'w', encoding='utf-8') as f:
        for entry in val_lines:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    mini_val_lines = val_lines[:1000]
    with open(args.val_mini_out, 'w', encoding='utf-8') as f:
        for entry in mini_val_lines:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
            
    print(f"Saved {args.train_out}")
    print(f"Saved {args.val_out}")
    print(f"Saved {args.val_mini_out} (1,000 samples)")

    # 9. Run Metrics Audit
    print("\nRunning final data audit...")
    calculate_metrics(train_lines, tokenizer, name="FINAL TRAINING SET")
    calculate_metrics(val_lines, tokenizer, name="FINAL VALIDATION SET")

if __name__ == "__main__":
    main()