import os
import json
import argparse
import warnings
import logging
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sacrebleu
from datasets import load_dataset
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

# Mute heavy loggers, keep script-level info active
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
    """
    Writes training metrics to a CSV file in real-time.
    """
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
        if not logs: return
        
        # Filter internal Trainer keys
        logs_to_save = {k: v for k, v in logs.items() if not k.startswith("total_")}
        logs_to_save["step"] = state.global_step
        logs_to_save["epoch"] = state.epoch
        
        # Determine if header is needed
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

class FocalLossTrainer(Seq2SeqTrainer):
    """
    Custom Trainer that implements Focal Loss instead of CrossEntropy.
    Focal Loss focuses training on hard-to-classify examples.
    """
    def __init__(self, focal_gamma=2.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.focal_gamma = focal_gamma

    def log(self, logs):
        """
        Round all float values to 4 decimals to avoid console clutter.
        """
        for k, v in logs.items():
            if k == "learning_rate":
                logs[k] = round(v, 6)
            else:
                logs[k] = round(v, 4)
        super().log(logs)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        loss = None
        if labels is not None:
            # Flatten to [batch * seq_len, vocab_size]
            logits_flat = logits.view(-1, logits.size(-1))
            labels_flat = labels.view(-1)
            
            # Standard Cross Entropy (reduction='none' so we can apply weights)
            # ignore_index=-100 handles the padding tokens
            ce_loss = F.cross_entropy(logits_flat, labels_flat, reduction='none', ignore_index=-100)
            
            # Calculate Probabilities (pt)
            pt = torch.exp(-ce_loss)
            
            # Focal Loss Formula: (1 - pt)^gamma * log(pt)
            focal_loss = ((1 - pt) ** self.focal_gamma) * ce_loss
            
            # Mean over non-ignored tokens
            loss = focal_loss.mean()

        return (loss, outputs) if return_outputs else loss

# --- HELPER FUNCTIONS ---

def setup_clean_model(model_name, args):
    """
    Loads mT5 base and tokenizer in strict 'Slow' mode.
    Applies LoRA with expanded target modules.
    """
    print(f"Loading Tokenizer...")
    # STRICT MODE: Legacy=True, Use_Fast=False. 
    # This prevents the 'Frankenstein' tokenizer creation.
    tokenizer = MT5Tokenizer.from_pretrained(model_name, legacy=True, use_fast=False)
    
    print(f"Loading Base Model: {model_name}")
    model = MT5ForConditionalGeneration.from_pretrained(model_name)
    
    # Optimization: Disable cache for training (saves VRAM)
    model.config.use_cache = False
    
    # LoRA Configuration
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
            target_modules=args.target_modules
        )
        model = get_peft_model(model, peft_config)
    
    model.print_trainable_parameters()
    return tokenizer, model

def preprocess_function(examples, tokenizer, max_src, max_tgt):
    types = examples.get("type", ["synthetic"] * len(examples["translation"]))
    source_texts = [ex.get("src_text", "") for ex in examples["translation"]]
    target_texts = [ex.get("tgt_text", "") for ex in examples["translation"]]
    source_langs = [ex.get("src_lang_code", "") for ex in examples["translation"]]
    target_langs = [ex.get("tgt_lang_code", "") for ex in examples["translation"]]
    
    prefixes = []
    for typ, src, tgt in zip(types, source_langs, target_langs):
        src_name = LANGUAGE_MAP.get(src, src)
        tgt_name = LANGUAGE_MAP.get(tgt, tgt)
        prefixes.append(f"translate {typ} {src_name} to {tgt_name}: ")
        
    inputs = [prefix + text for prefix, text in zip(prefixes, source_texts)]
    
    model_inputs = tokenizer(inputs, max_length=max_src, padding=False, truncation=True)
    labels = tokenizer(text_target=target_texts, max_length=max_tgt, padding=False, truncation=True).input_ids
    
    # Replace padding token id with -100 to ignore in loss
    labels = [
        [(l if l != tokenizer.pad_token_id else -100) for l in label]
        for label in labels
    ]
    model_inputs["labels"] = labels
    return model_inputs

def compute_metrics(eval_preds, tokenizer):
    """
    Calculates BLEU, chrF, and TER scores during evaluation,
    and prints 1 sample per translation direction.
    """
    preds, labels, inputs = eval_preds
    if isinstance(preds, tuple): preds = preds[0]
    
    # 1. Decode Inputs (Source) to find the translation direction
    #    (Skip special tokens to see the prefix "translate X to Y")
    inputs = np.where(inputs != -100, inputs, tokenizer.pad_token_id)
    decoded_inputs = tokenizer.batch_decode(inputs, skip_special_tokens=True)

    # 2. Decode Preds (Model Output)
    #    Ensure -100 is replaced with pad_token_id to prevent crashes
    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    
    # 3. Decode Labels (Correct Answer)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Clean up whitespace
    decoded_inputs = [i.strip() for i in decoded_inputs]
    decoded_preds = [p.strip() for p in decoded_preds]
    decoded_labels = [l.strip() for l in decoded_labels]
    
    # --- DIRECTIONAL SAMPLES ---
    # We want 1 sample for each of these 4 scenarios
    targets = {
        "Kek -> Eng": None,
        "Eng -> Kek": None,
        "Kek -> Spa": None,
        "Spa -> Kek": None
    }
    
    # Scan through the batch to fill the slots
    for i, src in enumerate(decoded_inputs):
        src_lower = src.lower()
        found_key = None
        
        # Identify direction based on the prefix we added in preprocessing
        if "q'eqchi' to english" in src_lower:
            found_key = "Kek -> Eng"
        elif "english to q'eqchi'" in src_lower:
            found_key = "Eng -> Kek"
        elif "q'eqchi' to spanish" in src_lower:
            found_key = "Kek -> Spa"
        elif "spanish to q'eqchi'" in src_lower:
            found_key = "Spa -> Kek"
            
        # If we found a direction and haven't saved a sample for it yet, save it
        if found_key and targets[found_key] is None:
            targets[found_key] = {
                "input": src,
                "label": decoded_labels[i],
                "pred": decoded_preds[i]
            }
        
        # Stop early if we found all 4
        if all(targets.values()):
            break

    print("\n" + "="*60)
    print("VALIDATION SAMPLES")
    print("="*60)
    for direction, sample in targets.items():
        if sample:
            print(f"--- {direction} ---")
            # print(f"Input:  {sample['input']}") # Optional: Uncomment to see source
            print(f"Target: {sample['label']}")
            print(f"Model:  {sample['pred']}")
        else:
            print(f"--- {direction} ---")
            print("(No samples found in this evaluation batch)")
    print("="*60 + "\n")
    # ---------------------------

    decoded_labels_list = [[label] for label in decoded_labels]
    
    bleu = sacrebleu.corpus_bleu(decoded_preds, decoded_labels_list)
    chrf = sacrebleu.corpus_chrf(decoded_preds, decoded_labels_list)
    ter = sacrebleu.corpus_ter(decoded_preds, decoded_labels_list)
    
    return {
        "bleu": round(bleu.score, 4),
        "chrf": round(chrf.score, 4),
        "ter": round(ter.score, 4)
    }

def main(args):
    # 1. Load Data
    print(f"Loading Train: {args.train_file}")
    train_dataset = load_dataset("json", data_files=args.train_file)["train"]
    
    print(f"Loading Valid: {args.valid_file}")
    # Handling comma-separated validation files if needed
    val_files = [f.strip() for f in args.valid_file.split(",")]
    eval_datasets = {}
    for vf in val_files:
        name = os.path.basename(vf).replace(".jsonl", "").replace("val_", "")
        eval_datasets[name] = load_dataset("json", data_files=vf)["train"]

    # 2. Setup Clean Model & Tokenizer
    tokenizer, model = setup_clean_model(args.model_name, args)
    
    # 3. Preprocess
    print("Tokenizing datasets...")
    fn_kwargs = {"tokenizer": tokenizer, "max_src": args.max_source_length, "max_tgt": args.max_target_length}
    tokenized_train = train_dataset.map(preprocess_function, batched=True, fn_kwargs=fn_kwargs, remove_columns=train_dataset.column_names)
    tokenized_evals = {k: v.map(preprocess_function, batched=True, fn_kwargs=fn_kwargs, remove_columns=v.column_names) for k, v in eval_datasets.items()}

    # 4. Callbacks & Trainer
    # Gather params for CSV header
    all_params = vars(args)
    csv_callback = CSVLogCallback(args.output_dir, hyperparams=all_params)
    
    # Use the first validation set for metric tracking
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
        
        # Generation
        predict_with_generate=True,
        generation_max_length=args.max_target_length,
        include_for_metrics=["inputs"],
        
        # Hardware / MPS Optimizations
        use_cpu=False, # Use MPS if available
        fp16=False,    # MPS doesn't like fp16
        bf16=False,    # MPS support varies, False is safer
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        group_by_length=True,
        report_to="none",
        ddp_find_unused_parameters=False,
    )

    # Use FocalLossTrainer instead of standard Seq2SeqTrainer
    trainer = FocalLossTrainer(
        focal_gamma=args.focal_gamma,
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_evals,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model, pad_to_multiple_of=8),
        processing_class=tokenizer,
        compute_metrics=lambda p: compute_metrics(p, tokenizer),
        callbacks=[csv_callback]
    )

    print("Starting Training...")
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\nInterrupt detected. Saving state...")
        trainer.save_model(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        exit(0)

    # 5. Final Safe Save
    print(f"Saving Final Model & Tokenizer to {args.output_dir}")
    trainer.save_model(args.output_dir)
    # Explicitly save the clean tokenizer
    tokenizer.save_pretrained(args.output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Data & Paths
    parser.add_argument("--train_file", type=str, required=True)
    parser.add_argument("--valid_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="lora_output")
    parser.add_argument("--model_name", type=str, default="google/mt5-base")
    parser.add_argument("--adapter_path", type=str, default=None, help="Path to resume existing LoRA")
    parser.add_argument("--save_total_limit", type=int, default=50)
    
    # Hyperparameters
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
    
    # Advanced LoRA Settings
    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.2)
    # Default includes Feed Forward layers (wi_0, wi_1, wo) for mT5
    parser.add_argument("--target_modules", nargs="+", default=["q", "v", "k", "o", "wi_0", "wi_1", "wo"])
    
    # Advanced Loss
    parser.add_argument("--focal_gamma", type=float, default=2.0, help="Gamma for Focal Loss. 0.0 = Cross Entropy")

    args = parser.parse_args()
    main(args)