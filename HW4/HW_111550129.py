"""
HW4 — LLM Tool Calling Agent
SFT with QLoRA (Qwen3-14B) + Hard Negative Sampling

Runs all stages sequentially:
  1. Prepare training data (full + structural, hard negative distractors)
  2. Train: full-info configuration
  3. Train: structural-only configuration
  4. Inference: full
  5. Inference: structural

Output: submission_sft_full.csv, submission_sft_structural.csv

Usage:
  python HW_111550129.py
"""

import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import csv
import inspect
import json
import random
import re
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel, get_peft_model
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTConfig, SFTTrainer

# ── Config ────────────────────────────────────────────────────────────────────

DATA_DIR    = Path(__file__).parent
BASE_MODEL  = "Qwen/Qwen3-14B"
MAX_SEQ_LEN = 1024
TARGET_N    = 8
LABELS      = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
EVAL_RATIO  = 0.05
SEED        = 42
HARD_RATIO  = 0.5   # fraction of distractors chosen by name-token overlap


# ── Prompt utilities ──────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are an AI assistant that selects the correct tool to call "
    "at each step of a task. /no_think"
)

INSTRUCTION = (
    "Based on the task context and current step, which tool should be called?\n"
    "Reply with ONLY a single uppercase letter (e.g. A, B, C, D, E, F, G, or H). "
    "No explanation."
)


def format_option_full(label: str, tool: dict) -> str:
    lines = [f"{label}. {tool['name']}"]
    if tool.get('description'):
        lines.append(f"   Description: {tool['description']}")
    args = tool.get('arguments', {}).get('properties', {})
    if args:
        lines.append("   Arguments:")
        for k, v in args.items():
            lines.append(f"     - {k} ({v.get('type','?')}): {v.get('description','')}")
    results = tool.get('results', {}).get('properties', {})
    if results:
        lines.append("   Returns:")
        for k, v in results.items():
            lines.append(f"     - {k} ({v.get('type','?')}): {v.get('description','')}")
    return '\n'.join(lines)


def format_option_structural(label: str, tool: dict) -> str:
    lines = [f"{label}. [tool]"]
    args = tool.get('arguments', {}).get('properties', {})
    if args:
        arg_parts = ', '.join(f"{k}: {v.get('type','?')}" for k, v in args.items())
        lines.append(f"   Arguments: {arg_parts}")
    results = tool.get('results', {}).get('properties', {})
    if results:
        res_parts = ', '.join(f"{k}: {v.get('type','?')}" for k, v in results.items())
        lines.append(f"   Returns: {res_parts}")
    return '\n'.join(lines)


def build_user_message(sample: dict, mode: str) -> str:
    fmt = format_option_full if mode == 'full' else format_option_structural
    options_text = '\n\n'.join(
        fmt(label, tool) for label, tool in sample['options'].items()
    )
    return (
        f"## Task Context\n{sample['full_context']}\n\n"
        f"## Current Step\n{sample['current_step']}\n\n"
        f"## Candidate Tools\n{options_text}\n\n"
        f"## Instruction\n{INSTRUCTION}"
    )


def build_messages(sample: dict, mode: str, answer: str | None = None) -> list[dict]:
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": build_user_message(sample, mode)},
    ]
    if answer is not None:
        msgs.append({"role": "assistant", "content": answer})
    return msgs


# ── Stage 1: Data preparation ─────────────────────────────────────────────────

def build_distractor_pool(all_samples: list[dict]) -> list[dict]:
    seen, pool = set(), []
    for s in all_samples:
        for tool in s['options'].values():
            if tool['name'] not in seen:
                seen.add(tool['name'])
                pool.append(tool)
    return pool


def name_overlap(a: dict, b: dict) -> float:
    pa = set(a['name'].lower().split('_'))
    pb = set(b['name'].lower().split('_'))
    return len(pa & pb) / max(len(pa | pb), 1)


def pad_to_n_hard(sample: dict, distractor_pool: list[dict],
                  n: int = TARGET_N, hard_ratio: float = HARD_RATIO) -> dict:
    correct_label = sample['answer']
    correct_tool  = sample['options'][correct_label]
    other_tools   = [v for k, v in sample['options'].items() if k != correct_label]

    existing_names = {t['name'] for t in sample['options'].values()}
    candidates     = [t for t in distractor_pool if t['name'] not in existing_names]

    needed = n - len(sample['options'])
    if needed > 0 and candidates:
        scored    = sorted(candidates, key=lambda t: name_overlap(correct_tool, t), reverse=True)
        n_hard    = max(1, int(needed * hard_ratio))
        n_random  = needed - n_hard
        hard_pool = [t for t in scored if name_overlap(correct_tool, t) > 0]
        n_hard    = min(n_hard, len(hard_pool))
        hard_negs = hard_pool[:n_hard]
        hard_names  = {t['name'] for t in hard_negs}
        random_pool = [t for t in candidates if t['name'] not in hard_names]
        rand_negs   = random.sample(random_pool, min(n_random, len(random_pool)))
        other_tools = other_tools + hard_negs + rand_negs

    random.shuffle(other_tools)
    correct_pos = random.randint(0, min(n - 1, len(other_tools)))
    all_tools   = other_tools[:correct_pos] + [correct_tool] + other_tools[correct_pos:]
    all_tools   = all_tools[:n]

    new_options = {LABELS[i]: all_tools[i] for i in range(len(all_tools))}
    new_answer  = LABELS[correct_pos]
    return {**sample, 'options': new_options, 'answer': new_answer}


def prepare_data():
    print("\n" + "="*60)
    print("STAGE 1: Prepare training data")
    print("="*60)

    train_jsonl = DATA_DIR / 'train.jsonl'
    if not train_jsonl.exists():
        train_jsonl = DATA_DIR.parent / 'train.jsonl'
    if not train_jsonl.exists():
        raise FileNotFoundError("train.jsonl not found")

    with open(train_jsonl, 'r', encoding='utf-8') as f:
        all_samples = [json.loads(line) for line in f]

    distractor_pool = build_distractor_pool(all_samples)
    print(f"Distractor pool: {len(distractor_pool)} unique tools")

    for mode in ['full', 'structural']:
        random.seed(SEED)
        valid = [s for s in all_samples if s['answer'] in s['options']]
        random.shuffle(valid)
        split     = int(len(valid) * (1 - EVAL_RATIO))
        train_raw = valid[:split]
        eval_raw  = valid[split:]

        def convert(samples):
            out = []
            for s in samples:
                s    = pad_to_n_hard(s, distractor_pool, TARGET_N)
                msgs = build_messages(s, mode, answer=s['answer'])
                out.append({"messages": msgs})
            return out

        train_data = convert(train_raw)
        eval_data  = convert(eval_raw)

        train_path = DATA_DIR / f"sft_train_{mode}.jsonl"
        eval_path  = DATA_DIR / f"sft_eval_{mode}.jsonl"

        with open(train_path, 'w', encoding='utf-8') as f:
            for item in train_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        with open(eval_path, 'w', encoding='utf-8') as f:
            for item in eval_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

        print(f"[{mode}] Train: {len(train_data)}  Eval: {len(eval_data)}")


# ── Stage 2: Training ─────────────────────────────────────────────────────────

def train(mode: str):
    print("\n" + "="*60)
    print(f"STAGE 2: Train — mode={mode}")
    print("="*60)

    output_dir = DATA_DIR / f"sft_model_{mode}"
    train_path = DATA_DIR / f"sft_train_{mode}.jsonl"
    eval_path  = DATA_DIR / f"sft_eval_{mode}.jsonl"

    if not train_path.exists():
        raise FileNotFoundError(f"{train_path} not found — run prepare_data() first.")

    print(f"Base model : {BASE_MODEL}")
    print(f"GPU        : {torch.cuda.get_device_name(0)}")
    print(f"VRAM       : {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    tokenizer.padding_side = 'right'
    tokenizer.model_max_length = MAX_SEQ_LEN

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    torch.cuda.empty_cache()
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    lora_config = LoraConfig(
        r=16, lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    def load_jsonl(path):
        with open(path, 'r', encoding='utf-8') as f:
            return [json.loads(line) for line in f]

    train_dataset = Dataset.from_list(load_jsonl(train_path))
    eval_dataset  = Dataset.from_list(load_jsonl(eval_path))
    print(f"Train: {len(train_dataset)} | Eval: {len(eval_dataset)}")

    _sft_valid     = inspect.signature(SFTConfig.__init__).parameters
    _trainer_valid = inspect.signature(SFTTrainer.__init__).parameters

    sft_kwargs = dict(
        output_dir=str(output_dir),
        num_train_epochs=2,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=16,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        bf16=True,
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=200,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="none",
    )
    if "max_seq_length" in _sft_valid:
        sft_kwargs["max_seq_length"] = MAX_SEQ_LEN
    if "dataset_text_field" in _sft_valid:
        sft_kwargs["dataset_text_field"] = None

    sft_config = SFTConfig(**sft_kwargs)

    trainer_kwargs = dict(
        model=model, args=sft_config,
        train_dataset=train_dataset, eval_dataset=eval_dataset,
    )
    if "processing_class" in _trainer_valid:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer

    SFTTrainer(**trainer_kwargs).train()

    final_dir = output_dir / "final"
    model.save_pretrained(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"Saved to {final_dir}")

    # Free VRAM before next run
    del model
    torch.cuda.empty_cache()


# ── Stage 3: Inference ────────────────────────────────────────────────────────

def parse_answer(text: str) -> str:
    for pattern in [r'^\s*([A-H])\s*$', r'\bAnswer[:\s]+([A-H])\b', r'\b([A-H])\b']:
        m = re.search(pattern, text, re.MULTILINE | re.IGNORECASE)
        if m:
            return m.group(1).upper()
    return 'A'


def infer(mode: str):
    print("\n" + "="*60)
    print(f"STAGE 3: Inference — mode={mode}")
    print("="*60)

    adapter_dir = DATA_DIR / f"sft_model_{mode}" / "final"
    output_path = DATA_DIR / f"submission_sft_{mode}.csv"
    cache_path  = DATA_DIR / f"cache_sft_{mode}.jsonl"

    if not adapter_dir.exists():
        raise FileNotFoundError(f"{adapter_dir} not found — run train() first.")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(str(adapter_dir), trust_remote_code=True)
    base  = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, quantization_config=bnb_config, device_map="auto", trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, str(adapter_dir))
    model.eval()

    test_jsonl = DATA_DIR / 'test.jsonl'
    if not test_jsonl.exists():
        test_jsonl = DATA_DIR.parent / 'test.jsonl'
    with open(test_jsonl, 'r', encoding='utf-8') as f:
        test_data = [json.loads(line) for line in f]

    done = {}
    if cache_path.exists():
        with open(cache_path, 'r', encoding='utf-8') as f:
            for line in f:
                rec = json.loads(line)
                done[rec['id']] = rec['answer']
        print(f"Resuming: {len(done)} already done")

    cache_file = open(cache_path, 'a', encoding='utf-8')
    errors = 0
    todo   = [s for s in test_data if s['id'] not in done]

    pbar = tqdm(todo, desc=f"[{mode}]", unit="sample",
                initial=len(done), total=len(test_data), dynamic_ncols=True)

    for sample in pbar:
        sid  = sample['id']
        msgs = build_messages(sample, mode)

        try:
            text = tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
        except TypeError:
            text = tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            )

        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        try:
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=8,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    pad_token_id=tokenizer.eos_token_id,
                )
            new_tokens = out[0][inputs['input_ids'].shape[1]:]
            raw    = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            answer = parse_answer(raw)
        except Exception as e:
            tqdm.write(f"[ERROR] id={sid}: {e}")
            answer = 'A'
            errors += 1

        done[sid] = answer
        cache_file.write(json.dumps({"id": sid, "answer": answer}) + '\n')
        cache_file.flush()
        pbar.set_postfix(errors=errors, ans=answer)

    pbar.close()
    cache_file.close()

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'answer'])
        for sample in test_data:
            writer.writerow([sample['id'], done.get(sample['id'], 'A')])

    print(f"Saved to {output_path}  (errors: {errors})")

    del model
    torch.cuda.empty_cache()


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    prepare_data()

    for mode in ['full', 'structural']:
        train(mode)

    for mode in ['full', 'structural']:
        infer(mode)

    print("\n" + "="*60)
    print("All done. Output files:")
    print("  submission_sft_full.csv")
    print("  submission_sft_structural.csv")
    print("="*60)
