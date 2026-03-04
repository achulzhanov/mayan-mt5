import argparse
import json
import re
from collections import Counter
from tqdm import tqdm

# Q'eqchi' tokenizer that preserves glottal stops (apostrophes)
# e.g., "q'eqchi'" stays as one token, not "q" and "eqchi"
TOKEN_PATTERN = r"[a-zA-Z0-9'’ʼ]+"

def analyze_file(file_path, target_lang_code="kek_Latn"):
    print(f"📊 Analyzing {file_path}...")
    
    word_counts = Counter()
    bigram_counts = Counter()
    sentence_counts = 0
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Scanning Corpus"):
            try:
                data = json.loads(line)
                entry = data['translation']
                
                # Determine which side is Q'eqchi'
                if entry['tgt_lang_code'] == target_lang_code:
                    text = entry['tgt_text']
                elif entry['src_lang_code'] == target_lang_code:
                    text = entry['src_text']
                else:
                    continue # Skip non-Q'eqchi' lines if mixed
                
                sentence_counts += 1
                
                # Normalize
                text_lower = text.lower()
                
                # 1. Tokenize
                tokens = re.findall(TOKEN_PATTERN, text_lower)
                
                # 2. Count Words
                word_counts.update(tokens)
                
                # 3. Count Bigrams (Pairs of words)
                # This catches phrases like "amaq’ tenamit" naturally
                if len(tokens) > 1:
                    bigrams = [f"{tokens[i]} {tokens[i+1]}" for i in range(len(tokens)-1)]
                    bigram_counts.update(bigrams)

            except Exception as e:
                continue

    return word_counts, bigram_counts, sentence_counts

def print_report(word_counts, bigram_counts, total_lines):
    print("\n" + "="*50)
    print(f"📄 DISTRIBUTION REPORT (Total Lines: {total_lines})")
    print("="*50)

    # 1. Top 40 Single Words
    print(f"\n🏆 TOP 40 WORDS:")
    print(f"{'Rank':<5} {'Word':<20} {'Count':<10} {'% Freq'}")
    print("-" * 45)
    
    # Updated to 40
    for rank, (word, count) in enumerate(word_counts.most_common(40), 1):
        percent = (count / total_lines) * 100 # Approx appearance per line
        print(f"{rank:<5} {word:<20} {count:<10} {percent:.2f}%")

    # 2. Top 40 Bigrams (Phrases)
    print(f"\n🔗 TOP 40 BIGRAMS (Phrases):")
    print(f"{'Rank':<5} {'Phrase':<25} {'Count':<10} {'% Freq'}")
    print("-" * 50)
    
    # Updated to 40
    for rank, (phrase, count) in enumerate(bigram_counts.most_common(40), 1):
        percent = (count / total_lines) * 100
        print(f"{rank:<5} {phrase:<25} {count:<10} {percent:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=True, help="Path to training .jsonl file")
    args = parser.parse_args()
    
    w, b, t = analyze_file(args.file)
    print_report(w, b, t)