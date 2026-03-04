import argparse
import json
import csv
import torch
import gc
import os
from tqdm import tqdm
from transformers import MT5Tokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel, PeftConfig
import evaluate
import traceback

# --- Configuration ---
LANG_CODE_MAP = {
    "eng_Latn": "English",
    "spa_Latn": "Spanish",
    "kek_Latn": "Q'eqchi'"
}

def construct_prompt(src_text, src_code, tgt_code):
    src_lang = LANG_CODE_MAP.get(src_code, src_code)
    tgt_lang = LANG_CODE_MAP.get(tgt_code, tgt_code)
    return f"translate synthetic {src_lang} to {tgt_lang}: {src_text}"

def load_metrics():
    print("Loading metrics (sacrebleu, chrf, ter)...")
    return {
        "bleu": evaluate.load("sacrebleu"),
        "chrf": evaluate.load("chrf"),
        "ter": evaluate.load("ter")
    }

def process_checkpoint(checkpoint_path, test_file, output_dir, batch_size, limit, metrics_tool, gen_kwargs):
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    checkpoint_name = os.path.basename(checkpoint_path)
    print(f"\nSTARTING EVALUATION: {checkpoint_name}")
    print(f"   Device: {device.upper()}")
    print(f"   Config: {gen_kwargs}")
    
    # 1. Load Model & Tokenizer
    # STRICT REQUIREMENT: Legacy tokenizer loading to match training exactly
    print(f"   Loading adapter and tokenizer...")
    config = PeftConfig.from_pretrained(checkpoint_path)
    
    # Load Base Model
    base_model = AutoModelForSeq2SeqLM.from_pretrained(config.base_model_name_or_path)
    base_model.config.use_cache = False
    
    # Load Tokenizer
    tokenizer = MT5Tokenizer.from_pretrained("google/mt5-base", use_fast=False)
    
    # Load Adapter
    model = PeftModel.from_pretrained(base_model, checkpoint_path)
    model.to(device)
    model.eval()

    # 2. Prepare Data
    sources = []
    targets = []
    print(f"   Loading test data from {test_file}...")
    with open(test_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if limit and i >= limit: break
            data = json.loads(line)['translation']
            prompt = construct_prompt(data['src_text'], data['src_lang_code'], data['tgt_lang_code'])
            sources.append(prompt)
            targets.append(data['tgt_text'])

    # 3. Setup Live CSV Writing
    pred_filename = os.path.join(output_dir, f"preds_{checkpoint_name}.csv")
    print(f"   Streaming predictions to {pred_filename}...")
    
    predictions = [] # Keep in memory for final metric calculation
    
    # Open file ONCE before the loop
    with open(pred_filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Prompt", "Target", "Prediction", "Match"]) # Header
        
        # 4. Inference Loop
        for i in tqdm(range(0, len(sources), batch_size), desc=f"   Inferencing {checkpoint_name}"):
            batch_src = sources[i : i + batch_size]
            batch_tgt = targets[i : i + batch_size]
            
            # Tokenize
            inputs = tokenizer(batch_src, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
            
            with torch.no_grad():
                generated_ids = model.generate(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_new_tokens=128, # Explicit limit to prevent runaway generation
                    **gen_kwargs
                )
            
            decoded_preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            
            # --- LIVE WRITE STEP ---
            for src, tgt, pred in zip(batch_src, batch_tgt, decoded_preds):
                writer.writerow([src, tgt, pred, tgt == pred])
                predictions.append(pred) 
            
            f.flush() # Force write to disk immediately

            # GC
            del inputs, generated_ids
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()

    # 5. Calculate Metrics (at the end)
    print("   Calculating metrics...")
    results = {}
    results['bleu'] = metrics_tool['bleu'].compute(predictions=predictions, references=[[t] for t in targets])['score']
    results['chrf'] = metrics_tool['chrf'].compute(predictions=predictions, references=[[t] for t in targets])['score']
    results['ter'] = metrics_tool['ter'].compute(predictions=predictions, references=[[t] for t in targets])['score']
    
    print(f"   RESULTS: BLEU: {results['bleu']:.2f} | chrF: {results['chrf']:.2f} | TER: {results['ter']:.2f}")

    # 6. Major Cleanup
    del model
    del base_model
    del tokenizer
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    gc.collect()
    
    return results

def main():
    parser = argparse.ArgumentParser()
    # Queue Arguments
    parser.add_argument("--checkpoints", nargs='+', required=True, help="List of checkpoint folders (space separated)")
    parser.add_argument("--test_file", type=str, required=True, help="Full Validation File (.jsonl)")
    parser.add_argument("--output_dir", type=str, default="inference_results", help="Folder to save CSVs")
    parser.add_argument("--summary_file", type=str, default="metrics_summary.csv", help="Summary CSV filename")
    
    # Inference Args
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--limit", type=int, default=None, help="Limit samples for testing (e.g. 100 for a quick check)")
    
    # Generation Config
    parser.add_argument("--repetition_penalty", type=float, default=1.0, help="Penalty for repeating words")
    parser.add_argument("--no_repeat_ngram_size", type=int, default=0, help="Prevent N-gram repetition")
    parser.add_argument("--num_beams", type=int, default=5, help="Beam search size")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    summary_path = os.path.join(args.output_dir, args.summary_file)
    
    # Load metric calculators once
    metrics_tool = load_metrics()

    # Generation Kwargs
    gen_kwargs = {
        "num_beams": args.num_beams,
        "early_stopping": True,
        "repetition_penalty": args.repetition_penalty,
        "no_repeat_ngram_size": args.no_repeat_ngram_size,
    }

    # Initialize Summary CSV header if it doesn't exist
    if not os.path.exists(summary_path):
        with open(summary_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["Checkpoint", "BLEU", "chrF", "TER", "Rep_Penalty", "No_Repeat_NGram"])

    # Processing Loop
    for ckpt in args.checkpoints:
        try:
            scores = process_checkpoint(
                ckpt, args.test_file, args.output_dir, args.batch_size, args.limit, metrics_tool, gen_kwargs
            )
            
            # Append to Summary
            with open(summary_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    os.path.basename(ckpt), 
                    f"{scores['bleu']:.4f}", 
                    f"{scores['chrf']:.4f}", 
                    f"{scores['ter']:.4f}",
                    args.repetition_penalty,
                    args.no_repeat_ngram_size
                ])
                
        except Exception as e:
            print(f"Error evaluating {ckpt}: {e}")
            traceback.print_exc()

    print(f"\nAll checkpoints processed. Summary saved to: {summary_path}")

if __name__ == "__main__":
    main()