import os
import gc
import json
import argparse
import warnings
import logging
import csv
import numpy as np
import torch
import torch.nn.functional as F
import sacrebleu
from datasets import Dataset, load_dataset
from transformers import (
    MT5ForConditionalGeneration,
    MT5Tokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    TrainerCallback
)
from peft import (
    get_peft_model,
    LoraConfig,
    TaskType,
    PeftModel,
)

# --- ENVIRONMENT & LOGGING SETUP ---
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", message="Trainer.tokenizer is now deprecated")

logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
logging.getLogger("transformers.trainer").setLevel(logging.ERROR)

# --- CONFIGURATION ---
LANGUAGE_MAP = {
    "eng_Latn": "English",
    "kek_Latn": "Q'eqchi'",
    "spa_Latn": "Spanish"
}

DEFAULT_TARGET_MODULES = ["q", "v", "k", "o", "wi_0", "wi_1", "wo"]

# --- CUSTOM COMPONENTS ---

class CSVLogCallback(TrainerCallback):
    """Writes training metrics to a CSV file in real-time."""
    def __init__(self, output_dir, hyperparams=None):
        self.output_dir = output_dir
        self.csv_path = os.path.join(output_dir, "training_logs.csv")
        self.hyperparams = hyperparams or {}
        self.header_written = False

    def on_train_begin(self, args, state, control, **kwargs):
        os.makedirs(self.output_dir, exist_ok=True)
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, "w", newline="", encoding='utf-8') as f:
                f.write("# --- Training Hyperparameters ---\n")
                for k, v in self.hyperparams.items():
                    f.write(f"# {k}: {v}\n")
                f.write("# --------------------------------\n")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        logs_to_save = {k: v for k, v in logs.items() if not k.startswith("total_")}
        logs_to_save["step"] = state.global_step
        logs_to_save["epoch"] = state.epoch

        write_header = not self.header_written
        if write_header and os.path.exists(self.csv_path) and os.path.getsize(self.csv_path) > 500:
            write_header = False
            self.header_written = True

        with open(self.csv_path, "a", newline="", encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=logs_to_save.keys())
            if write_header:
                writer.writeheader()
                self.header_written = True
            writer.writerow(logs_to_save)


class MemoryCallback(TrainerCallback):
    """
    Flushes the MPS allocator cache and runs the GC every N training steps.
    On Apple Silicon, the MPS allocator pools memory aggressively and won't
    return it to the OS unless explicitly told to. Without periodic flushes,
    the process memory grows continuously over a long training run.
    """
    def __init__(self, flush_every_n_steps=100):
        self.flush_every_n_steps = flush_every_n_steps

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.flush_every_n_steps == 0:
            gc.collect()
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()


class MTLDataCollator(DataCollatorForSeq2Seq):
    """
    Extends DataCollatorForSeq2Seq to pass task_weight through as a float tensor.
    The weight is popped from features before the parent collator sees them
    (which only handles standard seq2seq fields), then re-attached to the batch.
    """
    def __call__(self, features):
        task_weights = [float(f.pop("task_weight", 1.0)) for f in features]
        batch = super().__call__(features)
        batch["task_weight"] = torch.tensor(task_weights, dtype=torch.float)
        return batch


class MTLFocalLossTrainer(Seq2SeqTrainer):
    """
    Custom Trainer combining Focal Loss with per-example task weighting.

    Each batch example carries a task_weight (translation=1.0, POS=0.2, semantic=0.5
    by default) that scales its contribution to the mean batch loss. This lets
    auxiliary tasks (POS, semantic tagging) participate in training without
    overwhelming the primary translation objective.
    """
    def __init__(self, focal_gamma=2.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.focal_gamma = focal_gamma

    def log(self, logs):
        for k, v in logs.items():
            logs[k] = round(v, 6) if k == "learning_rate" else round(v, 4)
        super().log(logs)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        # Pop task_weight before the forward pass — the model doesn't expect it.
        task_weights = inputs.pop("task_weight", None)

        outputs = model(**inputs)
        logits = outputs.get("logits")

        loss = None
        if labels is not None:
            batch_size, seq_len = labels.shape

            logits_flat = logits.view(-1, logits.size(-1))
            labels_flat = labels.view(-1)

            # Focal loss, per token, no reduction
            ce_loss = F.cross_entropy(logits_flat, labels_flat, reduction='none', ignore_index=-100)
            pt = torch.exp(-ce_loss)
            focal_loss = ((1 - pt) ** self.focal_gamma) * ce_loss

            # Average per example over non-padding tokens → [batch_size]
            focal_loss = focal_loss.view(batch_size, seq_len)
            valid_mask = (labels != -100).float()
            per_example_loss = (focal_loss * valid_mask).sum(dim=1) / valid_mask.sum(dim=1).clamp(min=1)

            # Scale each example by its task weight
            if task_weights is not None:
                per_example_loss = per_example_loss * task_weights.to(per_example_loss.device)

            loss = per_example_loss.mean()

        return (loss, outputs) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        # task_weight must be removed before the standard eval path, which passes
        # inputs directly to model.generate() and model(**inputs).
        inputs.pop("task_weight", None)
        return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)


class MTLSampleCallback(TrainerCallback):
    """
    At each evaluation step, runs a fixed set of POS and semantic tagging
    examples through the model (greedy decoding) and prints the predictions
    alongside their targets. This is purely for training monitoring — it does
    not affect saved metrics or checkpoints.
    """
    def __init__(self, model, tokenizer, sample_file, device, n_samples=3):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.pos_samples = []   # [(input_str, target_str), ...]
        self.sem_samples = []
        self._load_samples(sample_file, n_samples)

    def _load_samples(self, sample_file, n):
        if not sample_file or not os.path.exists(sample_file):
            print(f"  [MTLSampleCallback] Warning: sample file not found: {sample_file}")
            return
        seen_kek = set()
        with open(sample_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)['translation']
                # Only take kek-source rows to avoid duplicate tagging for both directions
                if obj.get('src_lang_code') != 'kek_Latn':
                    continue
                kek_text = obj['src_text']
                if kek_text in seen_kek:
                    continue
                seen_kek.add(kek_text)

                pos_kek = obj.get('pos_kek', '').strip()
                sem_kek = obj.get('semantic_kek', '').strip()
                if pos_kek and len(self.pos_samples) < n:
                    self.pos_samples.append((f"tag POS Q'eqchi': {kek_text}", pos_kek))
                if sem_kek and len(self.sem_samples) < n:
                    self.sem_samples.append((f"tag semantic Q'eqchi': {kek_text}", sem_kek))
                if len(self.pos_samples) >= n and len(self.sem_samples) >= n:
                    break

        print(f"  [MTLSampleCallback] Loaded {len(self.pos_samples)} POS and "
              f"{len(self.sem_samples)} semantic samples for eval logging.")

    def on_evaluate(self, args, state, control, **kwargs):  # args/control/kwargs: required by TrainerCallback interface
        if not self.pos_samples and not self.sem_samples:
            return

        self.model.eval()
        print("\n" + "="*60)
        print(f"MTL TASK SAMPLES  (Step {state.global_step})")
        print("="*60)

        for task_label, samples in [("POS Tagging", self.pos_samples), ("Semantic Tagging", self.sem_samples)]:
            if not samples:
                continue
            print(f"--- {task_label} ---")
            for inp, tgt in samples:
                enc = self.tokenizer(
                    inp, return_tensors='pt', max_length=128, truncation=True
                ).to(self.device)
                with torch.no_grad():
                    ids = self.model.generate(
                        input_ids=enc.input_ids,
                        attention_mask=enc.attention_mask,
                        max_new_tokens=128,
                        num_beams=1,  # Greedy — speed matters here, this is for monitoring only
                    )
                pred = self.tokenizer.decode(ids[0], skip_special_tokens=True)
                kek_text = inp.split(': ', 1)[1] if ': ' in inp else inp
                print(f"  Source:  {kek_text}")
                print(f"  Target:  {tgt}")
                print(f"  Model:   {pred}")
            print()

        print("="*60 + "\n")


# --- HELPER FUNCTIONS ---

def setup_clean_model(model_name, args):
    """
    Loads mT5-base and tokenizer in strict legacy mode.
    Applies a fresh LoRA adapter unless --adapter_path is given.
    For the MTL ablation, always start from scratch (no --adapter_path).
    """
    print("Loading Tokenizer...")
    tokenizer = MT5Tokenizer.from_pretrained(model_name, legacy=True, use_fast=False)

    print(f"Loading Base Model: {model_name}")
    model = MT5ForConditionalGeneration.from_pretrained(model_name)
    model.config.use_cache = False

    if args.adapter_path:
        print(f"RESUMING: Loading adapter from {args.adapter_path}")
        model = PeftModel.from_pretrained(model, args.adapter_path, is_trainable=True)
    else:
        print(f"Applying LoRA: Rank={args.lora_r}, Alpha={args.lora_alpha}, Targets={args.target_modules}")
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            inference_mode=False,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=args.target_modules,
        )
        model = get_peft_model(model, peft_config)

    model.print_trainable_parameters()
    return tokenizer, model


def expand_mtl_dataset(file_path, w_trans, w_pos, w_sem):
    """
    Reads an MTL JSONL file and expands each row into up to 3 training records:

      1. Translation  (always, for both kek→other and other→kek rows)
         Prefix: "translate {src_lang} to {tgt_lang}: {src_text}"
         Target: {tgt_text}

      2. POS tagging  (only when src_lang_code == 'kek_Latn')
         Prefix: "tag POS Q'eqchi': {kek_text}"
         Target: {pos_kek}

      3. Semantic tagging  (only when src_lang_code == 'kek_Latn')
         Prefix: "tag semantic Q'eqchi': {kek_text}"
         Target: {semantic_kek}

    Tagging examples are only generated from kek-source rows to avoid emitting
    duplicate (identical) tagging examples for both directions of the same sentence.
    This gives a training ratio of 2:1:1 (translation:POS:semantic) relative to
    unique Q'eqchi' sentences, i.e. effective loss shares of ~74%:7%:19% at
    default weights (1.0 / 0.2 / 0.5).

    Returns a HuggingFace Dataset with columns: input_text, target_text, task_weight.
    """
    # Count records in a first pass so we can report totals without holding
    # the entire expanded list in memory.
    n_trans = n_pos = n_sem = 0

    def _generator():
        nonlocal n_trans, n_pos, n_sem
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)['translation']
                src_lang = obj.get('src_lang_code', '')
                tgt_lang = obj.get('tgt_lang_code', '')
                src_text = obj.get('src_text', '')
                tgt_text = obj.get('tgt_text', '')
                pos_kek  = obj.get('pos_kek', '').strip()
                sem_kek  = obj.get('semantic_kek', '').strip()

                src_name = LANGUAGE_MAP.get(src_lang, src_lang)
                tgt_name = LANGUAGE_MAP.get(tgt_lang, tgt_lang)

                # 1. Translation — always
                n_trans += 1
                yield {
                    "input_text":  f"translate {src_name} to {tgt_name}: {src_text}",
                    "target_text": tgt_text,
                    "task_weight": w_trans,
                }

                # 2 & 3. Tagging — only from kek-source rows with annotations
                if src_lang == 'kek_Latn' and pos_kek and sem_kek:
                    n_pos += 1
                    yield {
                        "input_text":  f"tag POS Q'eqchi': {src_text}",
                        "target_text": pos_kek,
                        "task_weight": w_pos,
                    }
                    n_sem += 1
                    yield {
                        "input_text":  f"tag semantic Q'eqchi': {src_text}",
                        "target_text": sem_kek,
                        "task_weight": w_sem,
                    }

    print(f"Expanding MTL dataset from {file_path}...")
    # from_generator streams records directly into an Arrow cache on disk,
    # avoiding an intermediate Python list of 1.3M dicts in RAM.
    dataset = Dataset.from_generator(_generator)
    print(f"  Total: {len(dataset):,} records  "
          f"(translation: {n_trans:,}  POS: {n_pos:,}  semantic: {n_sem:,})")
    return dataset


def preprocess_function_mtl(examples, tokenizer, max_src, max_tgt):
    """
    Tokenizes pre-expanded MTL records (output of expand_mtl_dataset).
    task_weight is passed through as a raw list; MTLDataCollator will tensor-ify it.
    """
    model_inputs = tokenizer(
        examples["input_text"],
        max_length=max_src,
        padding=False,
        truncation=True,
    )
    labels = tokenizer(
        text_target=examples["target_text"],
        max_length=max_tgt,
        padding=False,
        truncation=True,
    ).input_ids
    labels = [
        [(l if l != tokenizer.pad_token_id else -100) for l in label]
        for label in labels
    ]
    model_inputs["labels"] = labels
    model_inputs["task_weight"] = examples["task_weight"]
    return model_inputs


def preprocess_function_val(examples, tokenizer, max_src, max_tgt):
    """
    Preprocesses translation-only validation data for the MTL model.
    Uses "translate {lang} to {lang}:" prefix (no 'synthetic') to match
    the MTL model's training convention.
    The validation file may have a 'type' field — it is intentionally ignored here.
    """
    source_texts = [ex.get("src_text", "") for ex in examples["translation"]]
    target_texts = [ex.get("tgt_text", "") for ex in examples["translation"]]
    source_langs = [ex.get("src_lang_code", "") for ex in examples["translation"]]
    target_langs = [ex.get("tgt_lang_code", "") for ex in examples["translation"]]

    inputs = [
        f"translate {LANGUAGE_MAP.get(src, src)} to {LANGUAGE_MAP.get(tgt, tgt)}: {text}"
        for src, tgt, text in zip(source_langs, target_langs, source_texts)
    ]

    model_inputs = tokenizer(inputs, max_length=max_src, padding=False, truncation=True)
    labels = tokenizer(
        text_target=target_texts, max_length=max_tgt, padding=False, truncation=True
    ).input_ids
    labels = [
        [(l if l != tokenizer.pad_token_id else -100) for l in label]
        for label in labels
    ]
    model_inputs["labels"] = labels
    return model_inputs


def compute_metrics(eval_preds, tokenizer):
    """
    Calculates BLEU, chrF, and TER on translation-only validation data,
    and prints 1 sample per translation direction for monitoring.
    """
    preds, labels, inputs = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]

    inputs = np.where(inputs != -100, inputs, tokenizer.pad_token_id)
    decoded_inputs = tokenizer.batch_decode(inputs, skip_special_tokens=True)

    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_inputs = [s.strip() for s in decoded_inputs]
    decoded_preds  = [s.strip() for s in decoded_preds]
    decoded_labels = [s.strip() for s in decoded_labels]

    # --- DIRECTIONAL SAMPLES ---
    targets = {"Kek -> Eng": None, "Eng -> Kek": None, "Kek -> Spa": None, "Spa -> Kek": None}
    for i, src in enumerate(decoded_inputs):
        sl = src.lower()
        key = None
        if "q'eqchi' to english" in sl:   key = "Kek -> Eng"
        elif "english to q'eqchi'" in sl:  key = "Eng -> Kek"
        elif "q'eqchi' to spanish" in sl:  key = "Kek -> Spa"
        elif "spanish to q'eqchi'" in sl:  key = "Spa -> Kek"
        if key and targets[key] is None:
            targets[key] = {"label": decoded_labels[i], "pred": decoded_preds[i]}
        if all(targets.values()):
            break

    print("\n" + "="*60)
    print("VALIDATION SAMPLES")
    print("="*60)
    for direction, sample in targets.items():
        print(f"--- {direction} ---")
        if sample:
            print(f"Target: {sample['label']}")
            print(f"Model:  {sample['pred']}")
        else:
            print("(No samples found in this batch)")
    print("="*60 + "\n")

    decoded_labels_list = [[l] for l in decoded_labels]
    bleu = sacrebleu.corpus_bleu(decoded_preds, decoded_labels_list)
    chrf = sacrebleu.corpus_chrf(decoded_preds, decoded_labels_list)
    ter  = sacrebleu.corpus_ter(decoded_preds, decoded_labels_list)

    return {
        "bleu": round(bleu.score, 4),
        "chrf": round(chrf.score, 4),
        "ter":  round(ter.score, 4),
    }


def main(args):
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device.upper()}")
    print(f"Task weights — translation: {args.weight_translation}, "
          f"POS: {args.weight_pos}, semantic: {args.weight_semantic}")

    # 1. Build expanded MTL training dataset
    raw_train = expand_mtl_dataset(
        args.train_file,
        w_trans=args.weight_translation,
        w_pos=args.weight_pos,
        w_sem=args.weight_semantic,
    )

    # 2. Load translation-only validation dataset(s)
    print(f"Loading validation file(s): {args.valid_file}")
    val_files = [vf.strip() for vf in args.valid_file.split(",")]
    eval_datasets_raw = {}
    for vf in val_files:
        name = os.path.basename(vf).replace(".jsonl", "").replace("val_", "")
        eval_datasets_raw[name] = load_dataset("json", data_files=vf)["train"]

    # 3. Model & Tokenizer
    tokenizer, model = setup_clean_model(args.model_name, args)

    # 4. Tokenize
    print("Tokenizing datasets...")
    fn_kwargs = {"tokenizer": tokenizer, "max_src": args.max_source_length, "max_tgt": args.max_target_length}

    # Remove all original columns — preprocess_function_mtl re-emits task_weight
    tokenized_train = raw_train.map(
        preprocess_function_mtl,
        batched=True,
        fn_kwargs=fn_kwargs,
        remove_columns=raw_train.column_names,
    )
    # raw_train is no longer needed — drop it before loading the model
    # so its Arrow-backed memory can be reclaimed.
    del raw_train
    gc.collect()

    tokenized_evals = {
        k: v.map(
            preprocess_function_val,
            batched=True,
            fn_kwargs=fn_kwargs,
            remove_columns=v.column_names,
        )
        for k, v in eval_datasets_raw.items()
    }

    # 5. Callbacks
    csv_callback = CSVLogCallback(args.output_dir, hyperparams=vars(args))
    mem_callback = MemoryCallback(flush_every_n_steps=args.mem_flush_steps)
    mtl_callback = MTLSampleCallback(
        model=model,
        tokenizer=tokenizer,
        sample_file=args.mtl_sample_file,
        device=device,
        n_samples=args.n_mtl_samples,
    )

    metric_name = f"eval_{list(tokenized_evals.keys())[0]}_loss"

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        eval_accumulation_steps=args.eval_accum_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        optim=args.optim,
        label_smoothing_factor=args.label_smoothing_factor,
        max_grad_norm=args.max_grad_norm,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,

        # Logging & Saving
        logging_strategy="steps",
        logging_steps=20,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,

        # Best Model
        load_best_model_at_end=True,
        metric_for_best_model=metric_name,
        greater_is_better=False,

        # Generation (validation uses generate)
        predict_with_generate=True,
        generation_max_length=args.max_target_length,
        include_for_metrics=["inputs"],

        # Hardware / MPS
        use_cpu=False,
        fp16=False,
        bf16=False,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        group_by_length=True,
        report_to="none",
        ddp_find_unused_parameters=False,
    )

    # 6. Trainer
    data_collator = MTLDataCollator(tokenizer, model=model, pad_to_multiple_of=8)
    trainer = MTLFocalLossTrainer(
        focal_gamma=args.focal_gamma,
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_evals,
        data_collator=data_collator,
        processing_class=tokenizer,
        compute_metrics=lambda p: compute_metrics(p, tokenizer),
        callbacks=[csv_callback, mem_callback, mtl_callback],
    )

    print("Starting MTL Training...")
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\nInterrupt detected. Saving state...")
        trainer.save_model(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        exit(0)

    print(f"Saving Final Model & Tokenizer to {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MTL training for mT5-base LoRA: translation + POS tagging + semantic tagging."
    )

    # Data & Paths
    parser.add_argument("--train_file", type=str, required=True,
                        help="MTL annotated training JSONL (e.g. train_mtl_kek_en_mtl.jsonl).")
    parser.add_argument("--valid_file", type=str, required=True,
                        help="Translation-only validation JSONL. Comma-separate for multiple files.")
    parser.add_argument("--mtl_sample_file", type=str, default=None,
                        help="MTL JSONL used to draw POS/semantic logging samples at each eval step. "
                             "Can be the same as --train_file.")
    parser.add_argument("--n_mtl_samples", type=int, default=3,
                        help="Number of POS and semantic samples to print at each eval step.")
    parser.add_argument("--output_dir", type=str, default="lora_output_mtl")
    parser.add_argument("--model_name", type=str, default="google/mt5-base")
    parser.add_argument("--adapter_path", type=str, default=None,
                        help="Resume from existing LoRA adapter. Leave unset for fresh MTL training.")
    parser.add_argument("--save_total_limit", type=int, default=50)

    # Hyperparameters — same defaults as baseline
    parser.add_argument("--optim", type=str, default="adamw_torch")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--train_batch_size", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--grad_accum_steps", type=int, default=8)
    parser.add_argument("--eval_accum_steps", type=int, default=1)
    parser.add_argument("--lr_scheduler_type", type=str, default="linear")
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--label_smoothing_factor", type=float, default=0.1)
    parser.add_argument("--max_grad_norm", type=float, default=10.0)
    parser.add_argument("--max_source_length", type=int, default=128)
    parser.add_argument("--max_target_length", type=int, default=128)

    # Logging
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--mem_flush_steps", type=int, default=100,
                        help="Flush MPS allocator cache every N steps to prevent memory creep. (default: 100)")

    # LoRA
    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.2)
    parser.add_argument("--target_modules", nargs="+",
                        default=["q", "v", "k", "o", "wi_0", "wi_1", "wo"])

    # Loss
    parser.add_argument("--focal_gamma", type=float, default=2.0,
                        help="Focal loss gamma. 0.0 = standard cross-entropy.")
    parser.add_argument("--weight_translation", type=float, default=1.0,
                        help="Loss weight for translation examples. (default: 1.0)")
    parser.add_argument("--weight_pos", type=float, default=0.4,
                        help="Loss weight for POS tagging examples. (default: 0.4)")
    parser.add_argument("--weight_semantic", type=float, default=0.8,
                        help="Loss weight for semantic tagging examples. (default: 0.8)")

    args = parser.parse_args()
    main(args)
