import argparse
import pandas as pd
from pathlib import Path
import sys

# Import from custom modules
# We use try/except to handle potential import path issues if running directly vs as module
try:
    from .generator import QeqchiGenerator
    from .linguistics_core import PERSONS
    from .vertex_ai_filter import filter_with_gemini
    from .utils import save_as_jsonl, save_as_jsonl_mtl
    from .pos_tagger import build_kek_annotation
except ImportError:
    # Fallback for running locally without -m flag (not recommended but handled)
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from generators.generator import QeqchiGenerator
    from generators.linguistics_core import PERSONS
    from generators.vertex_ai_filter import filter_with_gemini
    from generators.utils import save_as_jsonl, save_as_jsonl_mtl
    from generators.pos_tagger import build_kek_annotation

"""
Entry Point - Q'eqchi' Sentence Generator
Runs the generator program with batching, global deduplication, and semantic filtering.
"""

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent # Adjusted to go up to project root
DEFAULT_DATA_DIR = ROOT_DIR / "data"

# --- ARGUMENT PARSING ---
parser = argparse.ArgumentParser(description="Generate Q'eqchi' sentences with semantic filtering.")

# Generator Control
parser.add_argument("--n", type=int, default=10, help="Total number of VALID sentences to generate.")
parser.add_argument("--batch_size", type=int, default=1000, help="Number of sentences to process per generation cycle (Save Interval).")
parser.add_argument("--person", type=str, choices=PERSONS, help="Force a specific person/number for possessed NPs.")
parser.add_argument("--data_dir", type=str, default=str(DEFAULT_DATA_DIR), help="Path to the root data directory.")

# Output Control
parser.add_argument("--jsonl_output", type=str, help="Base path for JSONL output (e.g., 'data/synthetic/train'). This is the recommended format.")
parser.add_argument("--out", type=str, help="Fallback output file path (.txt or .csv).")
parser.add_argument(
    "--mtl_output", type=str,
    help=(
        "Base path for MTL-annotated JSONL output (e.g., 'data/synthetic/train_mtl'). "
        "Generates additional _kek_en_mtl.jsonl and _kek_es_mtl.jsonl files alongside "
        "the regular translation-only JSONL. Both datasets contain the exact same "
        "generated sentences, enabling apples-to-apples ablation comparisons."
    ),
)

# Filter / API Control
parser.add_argument("--use_gemini_filter", action="store_true", help="Enable semantic filtering with Gemini API.")
parser.add_argument("--project_id", type=str, help="Your Google Cloud Project ID (required for Gemini filter).")
parser.add_argument("--location", type=str, default="us-central1", help="Vertex AI Region (e.g., us-central1, europe-west4).")
parser.add_argument("--api_batch_size", type=int, default=50, help="Number of sentences sent to Gemini in a single prompt.")

args = parser.parse_args()

# --- VALIDATION ---
if args.use_gemini_filter and not args.project_id:
    parser.error("--project_id is required when --use_gemini_filter is enabled.")

# MTL output requires a base translation output path so both datasets are co-located.
if args.mtl_output and not args.jsonl_output:
    parser.error("--mtl_output requires --jsonl_output to also be set.")

# --- INITIALIZATION ---
data_root = Path(args.data_dir).resolve()
print(f"Loading data from: {data_root}")

try:
    gen = QeqchiGenerator(data_root=data_root)
except Exception as e:
    print(f"Error loading generator: {e}")
    sys.exit(1)

# Whether to build annotation data during generation (only when --mtl_output is set).
_annotate = bool(args.mtl_output)

# --- CLEAN SLATE SETUP ---
# If using JSONL output, remove old files to prevent mixed runs
if args.jsonl_output:
    out_path_base = Path(args.jsonl_output).resolve()
    out_path_base.parent.mkdir(parents=True, exist_ok=True)

    files_cleaned = 0
    for suffix in ["_kek_en.jsonl", "_kek_es.jsonl"]:
        p = Path(f"{str(out_path_base)}{suffix}")
        if p.exists():
            print(f"Removing existing file to start fresh: {p}")
            p.unlink()
            files_cleaned += 1

# Clean up any existing MTL output files for a fresh run.
if args.mtl_output:
    mtl_path_base = Path(args.mtl_output).resolve()
    mtl_path_base.parent.mkdir(parents=True, exist_ok=True)

    for suffix in ["_kek_en_mtl.jsonl", "_kek_es_mtl.jsonl"]:
        p = Path(f"{str(mtl_path_base)}{suffix}")
        if p.exists():
            print(f"Removing existing MTL file to start fresh: {p}")
            p.unlink()

# --- GLOBAL DEDUPLICATION STATE ---
# We track unique English sentences to prevent duplicates across batches.
# This ensures we don't pay for the same sentence twice in the API.
global_seen_sentences = set()

total_valid_saved = 0
batch_num = 1

# Saturation Protection
# If the generator returns empty batches repeatedly, we stop to avoid infinite loops.
MAX_CONSECUTIVE_EMPTY_BATCHES = 5
empty_batches_count = 0

print(f"\n--- STARTING GENERATION ---")
print(f"Target Goal: {args.n} valid sentences")
print(f"Batch Size:  {args.batch_size}")
if args.use_gemini_filter:
    print(f"Filter:      ENABLED (Project: {args.project_id}, Region: {args.location})")
else:
    print(f"Filter:      DISABLED")
if _annotate:
    print(f"MTL Output:  ENABLED → {args.mtl_output}")

# --- MAIN LOOP ---
while total_valid_saved < args.n:
    # 1. Calculate how many we still need
    remaining_needed = args.n - total_valid_saved

    # We request the full batch_size to keep the pipeline efficient
    # (The loop will cut off naturally when total_valid_saved >= n)
    count_to_request = args.batch_size

    print(f"\n--- Batch {batch_num} ---")
    print(f"Generating {count_to_request} candidates (Total Progress: {total_valid_saved}/{args.n})...")

    # 2. GENERATE with Global Deduplication
    # We pass 'global_seen_sentences' so render_many knows what to skip immediately.
    # When --mtl_output is active, annotate=True attaches _annotation_info to each row.
    raw_rows = gen.render_many(
        count_to_request,
        person=args.person,
        exclude_en_set=global_seen_sentences,
        annotate=_annotate,
    )

    # Check for Saturation
    if not raw_rows:
        print("Warning: Generator returned 0 unique sentences.")
        empty_batches_count += 1
        if empty_batches_count >= MAX_CONSECUTIVE_EMPTY_BATCHES:
            print("CRITICAL: Generator is saturated. Max consecutive empty batches reached.")
            print("Stopping generation to prevent infinite loop.")
            break
        continue
    else:
        empty_batches_count = 0 # Reset counter on success

    # 3. UPDATE GLOBAL HISTORY
    # Mark these sentences as seen so we never generate them again
    # (Even if they are rejected by the filter, we don't want to retry them)
    new_uniques = 0
    for r in raw_rows:
        en_text = r.get('en', '')
        if en_text not in global_seen_sentences:
            global_seen_sentences.add(en_text)
            new_uniques += 1

    print(f"Generated {len(raw_rows)} unique candidates. (Global History: {len(global_seen_sentences)})")

    # 4. FILTER (Vertex AI)
    if args.use_gemini_filter:
        print(f"Filtering with Gemini (Batch Size: {args.api_batch_size})...")
        final_rows = filter_with_gemini(
            raw_rows,
            project_id=args.project_id,
            location=args.location,
            batch_size=args.api_batch_size
        )

        if raw_rows:
            rejection_count = len(raw_rows) - len(final_rows)
            rejection_rate = rejection_count / len(raw_rows)
            print(f"Kept {len(final_rows)}/{len(raw_rows)} ({rejection_rate:.1%} rejected)")
    else:
        final_rows = raw_rows

    if not final_rows:
        print("Batch resulted in 0 valid sentences after filtering. Continuing...")
        continue

    # 5a. Compute POS / semantic annotations for MTL output.
    #     This runs AFTER filtering so we only annotate sentences we actually save.
    #     The _annotation_info key is then stripped before saving to keep output clean.
    if _annotate:
        for row in final_rows:
            ann = row.pop("_annotation_info", None)
            if ann:
                try:
                    pos_kek, sem_kek = build_kek_annotation(row["kek"], ann)
                except Exception:
                    pos_kek, sem_kek = "", ""
                row["pos_kek"]      = pos_kek
                row["semantic_kek"] = sem_kek
    else:
        # Ensure no _annotation_info leaks into plain JSONL output.
        for row in final_rows:
            row.pop("_annotation_info", None)

    # 5b. SAVE (Append Mode)
    if args.jsonl_output:
        # Note: We use mode='a' to append this batch to the file
        save_as_jsonl(final_rows, str(out_path_base), mode='a')
        print(f"Appended {len(final_rows)} sentences to JSONL.")

        if _annotate:
            save_as_jsonl_mtl(final_rows, str(mtl_path_base), mode='a')
            print(f"Appended {len(final_rows)} sentences to MTL JSONL.")
    else:
        # Fallback TXT/CSV logic
        out_path = Path(args.out) if args.out else (ROOT_DIR / "examples" / "generated_output.txt")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        mode = 'a' if total_valid_saved > 0 else 'w'
        header = (total_valid_saved == 0)

        if out_path.suffix.lower() == ".csv":
            pd.DataFrame(final_rows).to_csv(out_path, mode=mode, header=header, index=False, encoding="utf-8")
        else:
            with open(out_path, mode, encoding="utf-8") as f:
                for r in final_rows:
                   f.write(f"[{r.get('id', '')}] KEK: {r.get('kek', '')} | ES: {r.get('es', '')} | EN: {r.get('en', '')}\n")
        print(f"Appended to {out_path}")

    # 6. Update Progress
    total_valid_saved += len(final_rows)
    batch_num += 1

    # Safety break (should be caught by while loop, but just in case)
    if total_valid_saved >= args.n:
        break

print(f"\n--- GENERATION COMPLETE ---")
print(f"Successfully saved {total_valid_saved} sentences.")
