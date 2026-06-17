#!/usr/bin/env python3
"""
MMLU Evaluation — BASE MODELS — HPC batch version
======================================================
Mirror of mmlu_instruct.py, with chat templates stripped so prompts
are sent as raw text. Prompt SUBSTANCE (wording, structure, seed strings) is
identical to the instruct version, so base-vs-instruct results are comparable.

Usage:
    python mmlu_base.py <model_id> <quant> <subject_group>

Examples:
    python mmlu_base.py meta-llama/Llama-3.2-1B False smoke_test
    python mmlu_base.py meta-llama/Llama-3.1-8B True  math
    python mmlu_base.py meta-llama/Llama-3.1-70B True  humanities_a
"""

# =============================================================================
# MODELS EVALUATED
# =============================================================================
# This study (cross-sectional analysis) evaluated three open-weight model
# families across a range of parameter sizes, each in BASE and INSTRUCT
# variants, so that model scale could serve as a within-family proxy for
# "competence" when testing for a Dunning-Kruger-style pattern.
#
# Pass any HuggingFace model id as the first CLI argument. Use a BASE id with
# the *_base_* scripts and an INSTRUCT id with the *_instruct_* scripts.
#
# Note the instruct-suffix convention differs by family:
#     * Llama / Qwen  -> instruct ids end in "-Instruct"
#     * Gemma         -> instruct ids end in "-it"
#
#   Family    Base id (examples)          Instruct id (examples)
#   --------  --------------------------  ----------------------------------
#   Llama 3   meta-llama/Llama-3.2-1B     meta-llama/Llama-3.2-1B-Instruct
#             meta-llama/Llama-3.2-3B     meta-llama/Llama-3.2-3B-Instruct
#             meta-llama/Llama-3.1-8B     meta-llama/Llama-3.1-8B-Instruct
#             meta-llama/Llama-3.1-70B    meta-llama/Llama-3.1-70B-Instruct
#   Qwen2.5   Qwen/Qwen2.5-0.5B           Qwen/Qwen2.5-0.5B-Instruct
#             Qwen/Qwen2.5-3B             Qwen/Qwen2.5-3B-Instruct
#             Qwen/Qwen2.5-7B             Qwen/Qwen2.5-7B-Instruct
#             Qwen/Qwen2.5-14B            Qwen/Qwen2.5-14B-Instruct
#             Qwen/Qwen2.5-32B            Qwen/Qwen2.5-32B-Instruct
#             Qwen/Qwen2.5-72B            Qwen/Qwen2.5-72B-Instruct
#   Gemma 2   google/gemma-2-2b           google/gemma-2-2b-it
#             google/gemma-2-9b           google/gemma-2-9b-it
#             google/gemma-2-27b          google/gemma-2-27b-it
#
# Loading precision: larger models (roughly >=13B) were loaded with NF4 4-bit
# quantization (pass quant=True); smaller models were run in bfloat16
# (quant=False). See the model-loading block in __main__.
#
# The separate DEVELOPMENTAL analysis (a different script, not this one) instead
# sweeps training checkpoints of a single model -- OLMo-2
# (allenai/OLMo-2-1124-13B) -- with Pythia (EleutherAI/pythia-12b) used as an
# exploratory comparison.
# =============================================================================

# ── CACHE SETUP — must happen before any HuggingFace imports ─────────────────
import os
# HuggingFace cache location. On a shared cluster you usually want this on
# fast scratch storage rather than your home directory. Point the HF_CACHE
# environment variable wherever you have space; defaults to ~/hf_cache:
#     export HF_CACHE=/path/to/scratch/hf_cache
HF_CACHE = os.environ.get("HF_CACHE", os.path.expanduser("~/hf_cache"))
os.environ["HF_HOME"]            = HF_CACHE
os.environ["HF_DATASETS_CACHE"]  = f"{HF_CACHE}/datasets"
os.environ["TRANSFORMERS_CACHE"] = f"{HF_CACHE}/transformers"
os.environ["HF_HUB_CACHE"]       = f"{HF_CACHE}/hub"
os.makedirs(HF_CACHE, exist_ok=True)
print("Cache set to:", os.environ["HF_HOME"])

# ── Imports ───────────────────────────────────────────────────────────────────
import argparse
import sys
import math
import re
import random
from string import ascii_uppercase

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset
from dotenv import load_dotenv

# ── SUBJECT GROUPS ────────────────────────────────────────────────────────────
SUBJECT_GROUPS = {
    # ── ALREADY DONE ─────────────────────────────────────────────────────────
    "math": [
        ("abstract_algebra",        100),
        ("college_mathematics",     100),
        ("high_school_mathematics", 270),
        ("elementary_mathematics",  378),
    ],
    "biology": [
        ("college_biology",         144),
        ("high_school_biology",     310),
        ("professional_medicine",   272),
    ],
    "social_sciences_a": [
        ("high_school_psychology",  545),
    ],

    # ── TO RUN ───────────────────────────────────────────────────────────────
    "physics_cs_a": [
        ("college_physics",    102),
        ("conceptual_physics", 235),
    ],
    "physics_cs_b": [
        ("high_school_physics",          151),
        ("college_computer_science",     100),
        ("high_school_computer_science", 100),
    ],
    "social_sciences_b": [
        ("professional_psychology", 612),
    ],
    "social_sciences_c": [
        ("econometrics",          114),
        ("sociology",             201),
        ("public_relations",      110),
        ("security_studies",      245),
        ("high_school_geography", 198),
    ],
    "humanities_a": [
        ("philosophy",                   311),
        ("prehistory",                   324),
    ],
    "humanities_b": [
        ("high_school_world_history",    237),
        ("high_school_european_history", 165),
        ("moral_disputes",               346),
    ],
    "humanities_c": [
        ("logical_fallacies",  163),
        ("international_law",  121),
        ("jurisprudence",      108),
        ("world_religions",    171),
    ],

    # ── SMOKE TEST ───────────────────────────────────────────────────────────
    "smoke_test": [
        ("abstract_algebra", 10),
    ],
}

# ── BOOLEAN ARG ───────────────────────────────────────────────────────────────
def str2bool(v):
    if v.lower() == "true":  return True
    if v.lower() == "false": return False
    raise argparse.ArgumentTypeError("Boolean value expected: True or False")

# ── CONSTANTS ─────────────────────────────────────────────────────────────────
LETTER_OPTIONS = ['A', 'B', 'C', 'D']
IDX_TO_LETTER  = {i: L for i, L in enumerate(LETTER_OPTIONS)}

# ── TOKEN MAPS ────────────────────────────────────────────────────────────────
def build_letter_token_map(tokenizer) -> dict:
    """
    Maps each uppercase letter A-Z to ALL token IDs that decode to that letter
    (covering surface forms like 'A', ' A', '▁A', 'ĠA').
    Pooling variants ensures we capture probability mass regardless of
    whether the model emits a space-prefixed or bare letter token.
    """
    mapping = {L: [] for L in ascii_uppercase}
    for tok_id in range(tokenizer.vocab_size):
        decoded = tokenizer.decode(tok_id).replace('▁', '').replace('Ġ', '').strip()
        if decoded in mapping:
            mapping[decoded].append(tok_id)
    return mapping


def build_binary_token_map(tokenizer) -> dict:
    """
    Maps '1' and '0' to ALL token IDs that decode to those digits.
    Space+number is two tokens so we mostly capture bare digit tokens only.
    """
    mapping = {'1': [], '0': []}
    for tok_id in range(tokenizer.vocab_size):
        decoded = tokenizer.decode(tok_id).replace('▁', '').replace('Ġ', '').strip()
        if decoded in mapping:
            mapping[decoded].append(tok_id)
    return mapping

# ── PROMPTS ───────────────────────────────────────────────────────────────────
# NOTE: Prompt SUBSTANCE is identical to the instruct version.
# The only difference is that we don't wrap the message in a chat template.
# The same seed strings ('Answer:', 'Answer: (', 'My confidence: ', 'Confidence: ')
# are appended so base models complete the pattern at the same anchor point,
# making instruct-vs-base results directly comparable.

def build_base_prompt(question: str, choices: list, tokenizer) -> str:
    """
    Pass 1 prompt for base models.
    'Answer:' cue at the end — base model pattern-completes with a choice letter.
    """
    opts = "\n".join(f"{L}. {c}" for L, c in zip(LETTER_OPTIONS, choices))
    return (
        "Answer the following multiple choice question with only the letter "
        "of the correct answer — no explanation.\n\n"
        f"Question: {question}\n"
        f"{opts}\n"
        "Answer:"
    )


def letter_to_text(letter: str, choices: list) -> str:
    """Resolve predicted letter to 'A. Rome' format. Falls back gracefully."""
    if letter in LETTER_OPTIONS:
        return f"{letter}. {choices[LETTER_OPTIONS.index(letter)]}"
    return "(no valid answer produced)"


def build_p_true_prompt(question: str, choices: list, predicted_letter: str, tokenizer) -> str:
    """P(True) prompt. Seeded with 'Answer: (' → next token is 'A' or 'B'."""
    choice_lines = "\n".join(f"{L}. {choices[i]}" for i, L in enumerate(LETTER_OPTIONS))
    return (
        f"Question: {question}\n"
        f"{choice_lines}\n"
        f"Proposed answer: {letter_to_text(predicted_letter, choices)}\n"
        f"Is the proposed answer correct?\n"
        f"(A) True\n"
        f"(B) False\n"
        f"Answer: ("
    )


def build_p_one_prompt(question: str, choices: list, predicted_letter: str, tokenizer) -> str:
    """P(1) prompt. Seeded with 'My confidence: ' → next token is '1' or '0'."""
    choice_lines = "\n".join(f"{L}. {choices[i]}" for i, L in enumerate(LETTER_OPTIONS))
    return (
        f"Question: {question}\n"
        f"{choice_lines}\n"
        f"I answered: {letter_to_text(predicted_letter, choices)}\n\n"
        f"Now I will rate my confidence that my answer was correct.\n"
        f"1 = I believe my answer is correct.\n"
        f"0 = I believe my answer is incorrect.\n"
        f"My confidence: "
    )


def build_verbal_prompt(question: str, choices: list, predicted_letter: str, tokenizer) -> str:
    """Verbal prompt. Seeded with 'Confidence: ' → model generates a number."""
    choice_lines = "\n".join(f"{L}. {choices[i]}" for i, L in enumerate(LETTER_OPTIONS))
    return (
        f"Question: {question}\n"
        f"{choice_lines}\n"
        f"Your previous proposed answer was: {letter_to_text(predicted_letter, choices)}\n"
        f"On a scale of 0-100, how confident are you that this answer is correct?\n"
        f"Confidence: "
    )

# ── CONFIDENCE SCORERS ────────────────────────────────────────────────────────
def compute_p_true(probs, letter_map):
    p_a = probs[letter_map['A']].sum().item()
    p_b = probs[letter_map['B']].sum().item()
    return p_a / (p_a + p_b)

def compute_p_one(probs, binary_token_map):
    p_one  = probs[binary_token_map['1']].sum().item()
    p_zero = probs[binary_token_map['0']].sum().item()
    return p_one / (p_one + p_zero)

def extract_confidence_MSP(probs, letter_map):
    raw   = {L: probs[letter_map[L]].sum().item() for L in LETTER_OPTIONS}
    total = sum(raw.values())
    return {L: v / total for L, v in raw.items()}

def compute_entropy_metrics(renorm_probs):
    prob_tensor = torch.tensor(list(renorm_probs.values()))
    raw_entropy = torch.distributions.Categorical(probs=prob_tensor).entropy().item()
    H_max       = math.log(len(renorm_probs))
    return raw_entropy, 1.0 - (raw_entropy / H_max)

def compute_margin_renorm(renorm_probs):
    vals = sorted(renorm_probs.values(), reverse=True)
    return vals[0] - vals[1]

def parse_verbal_confidence(text):
    matches = re.findall(r"\b(\d{1,3})\b", text)
    return max(0, min(100, int(matches[0]))) if matches else None

# ── PER-QUESTION PIPELINE ─────────────────────────────────────────────────────
def run_question_all(question, choices, model, tokenizer, letter_map, binary_token_map):
    """Four independent forward passes per question (Passes 1-4)."""

    # Pass 1
    prompt = build_base_prompt(question, choices, tokenizer)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    n_prompt_pass1 = inputs["input_ids"].shape[1]
    with torch.no_grad():
        gen_out = model.generate(
            **inputs, max_new_tokens=10, do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True, output_scores=True,
        )
    logits = gen_out.scores[0][0].float()
    pass1_10_token_generation = tokenizer.decode(
        gen_out.sequences[0, n_prompt_pass1:], skip_special_tokens=True
    ).strip()
    raw_output    = tokenizer.decode(logits.argmax().item())
    letter_greedy = next((c for c in raw_output if c in LETTER_OPTIONS), None)
    probs_pass1   = torch.softmax(logits, dim=-1)
    renorm_MSP    = extract_confidence_MSP(probs_pass1, letter_map)
    letter_MSP    = max(renorm_MSP, key=renorm_MSP.get)
    raw_entropy, confidence_entropy = compute_entropy_metrics(renorm_MSP)
    margin_renorm = compute_margin_renorm(renorm_MSP)
    top_probs, top_ids = torch.topk(probs_pass1, k=2)
    margin_full  = (top_probs[0] - top_probs[1]).item()
    top_2_tokens = [tokenizer.decode(top_ids[0]), tokenizer.decode(top_ids[1])]

    # Pass 2: P(True)
    pt_inputs = tokenizer(
        build_p_true_prompt(question, choices, letter_greedy, tokenizer),
        return_tensors="pt"
    ).to(model.device)
    with torch.no_grad():
        pt_logits = model(**pt_inputs).logits[0, -1, :].float()
    pt_probs       = torch.softmax(pt_logits, dim=-1)
    p_true_score   = compute_p_true(pt_probs, letter_map)
    p_true_raw_out = tokenizer.decode(pt_logits.argmax().item())

    # Pass 3: P(1)
    p1_inputs = tokenizer(
        build_p_one_prompt(question, choices, letter_greedy, tokenizer),
        return_tensors="pt"
    ).to(model.device)
    with torch.no_grad():
        p1_logits = model(**p1_inputs).logits[0, -1, :].float()
    p1_probs    = torch.softmax(p1_logits, dim=-1)
    p_one_score = compute_p_one(p1_probs, binary_token_map)
    p1_raw_out  = tokenizer.decode(p1_logits.argmax().item())

    # Pass 4: verbal
    verbal_inputs = tokenizer(
        build_verbal_prompt(question, choices, letter_greedy, tokenizer),
        return_tensors="pt"
    ).to(model.device)
    with torch.no_grad():
        verbal_out = model.generate(
            **verbal_inputs, max_new_tokens=10, do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    n_prompt  = verbal_inputs["input_ids"].shape[1]
    generated = tokenizer.decode(verbal_out[0, n_prompt:], skip_special_tokens=True).strip()
    verbal_parsed = parse_verbal_confidence(generated)

    return {
        "raw_output"                : raw_output,
        "pass1_10_token_generation" : pass1_10_token_generation,
        "letter_greedy"             : letter_greedy,
        "letter_MSP"                : letter_MSP,
        "renorm_MSP"                : renorm_MSP,
        "msp_confidence"            : renorm_MSP[letter_MSP],
        "greedy_matches_msp"        : letter_greedy == letter_MSP,
        "raw_entropy"               : raw_entropy,
        "entropy_conf"              : confidence_entropy,
        "margin_renorm"             : margin_renorm,
        "margin_full"               : margin_full,
        "top_2_tokens"              : top_2_tokens,
        "p_true_raw_output"         : p_true_raw_out,
        "p_true_probability"        : p_true_score,
        "p1_raw_output"             : p1_raw_out,
        "p1_probability"            : p_one_score,
        "verbal_raw"                : generated,
        "verbal_parsed"             : verbal_parsed,
    }

# ── EVALUATE FUNCTION ─────────────────────────────────────────────────────────
def evaluate_mmlu(model, tokenizer, model_id, subject, split="test", num_questions=None):
    """Run all confidence metrics on one MMLU subject → save CSV."""
    letter_map_local       = build_letter_token_map(tokenizer)
    binary_token_map_local = build_binary_token_map(tokenizer)
    dataset = load_dataset("cais/mmlu", subject, split=split)
    if num_questions is not None:
        dataset = dataset.select(range(num_questions))
    model_short    = model_id.split('/')[-1]
    parse_failures = 0
    records        = []
    print(f"[All metrics] {model_short} | {subject} | {len(dataset)} questions", flush=True)

    for i, row in enumerate(dataset):
        correct_letter = IDX_TO_LETTER[row["answer"]]
        result = run_question_all(
            row["question"], row["choices"],
            model, tokenizer,
            letter_map_local, binary_token_map_local,
        )
        if result["verbal_parsed"] is None:
            parse_failures += 1
        records.append({
            "Question"                           : row["question"],
            "Choices"                            : str(row["choices"]),
            "correct_letter"                     : correct_letter,
            "llm_first_token_output"             : result["raw_output"],
            "letter_llm_outputted_detected"      : result["letter_greedy"],
            "pass1_10_token_generation"          : result["pass1_10_token_generation"],
            "Letter_MSP"                         : result["letter_MSP"],
            "Letter_MSP_confidence"              : result["msp_confidence"],
            "is_correct"                         : result["letter_greedy"] == correct_letter,
            "check_letter_output_is_highest_MSP" : result["greedy_matches_msp"],
            "prob_A"                             : result["renorm_MSP"]["A"],
            "prob_B"                             : result["renorm_MSP"]["B"],
            "prob_C"                             : result["renorm_MSP"]["C"],
            "prob_D"                             : result["renorm_MSP"]["D"],
            "Raw_Entropy"                        : result["raw_entropy"],
            "Entropy_as_confidence"              : result["entropy_conf"],
            "Margin_softmax_full_vocab"          : result["margin_full"],
            "Margin_renorm_ABCD"                 : result["margin_renorm"],
            "Top_2_Tokens"                       : result["top_2_tokens"],
            "p_true_raw_output"                  : result["p_true_raw_output"],
            "p_true_probability"                 : result["p_true_probability"],
            "p1_raw_output"                      : result["p1_raw_output"],
            "p1_probability"                     : result["p1_probability"],
            "verbal_full_text_generated"         : result["verbal_raw"],
            "verbal_integer_found"               : result["verbal_parsed"],
        })
        if (i + 1) % 10 == 0:
            acc    = np.mean([r["is_correct"] for r in records])
            avg_pt = np.mean([r["p_true_probability"] for r in records])
            avg_p1 = np.mean([r["p1_probability"] for r in records])
            agree  = np.mean([r["check_letter_output_is_highest_MSP"] for r in records])
            valid  = sum(1 for r in records if r["verbal_integer_found"] is not None)
            print(f"  [{i+1:>4}/{len(dataset)}]  acc={acc:.3f}  "
                  f"avg_p_true={avg_pt:.3f}  avg_p1={avg_p1:.3f}  "
                  f"agree={agree:.3f}  verbal_valid={valid}/{i+1}", flush=True)

    df = pd.DataFrame(records)
    out_csv = f"mmlu_{subject}_{model_short}_base_all.csv"
    df.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}", flush=True)
    print(f"Verbal parse failures: {parse_failures}/{len(dataset)}", flush=True)
    return df

# ── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_id",      help="HF model id")
    parser.add_argument("quant",         type=str2bool, help="True or False")
    parser.add_argument("subject_group", help=f"One of: {list(SUBJECT_GROUPS.keys())}")
    args = parser.parse_args()

    print(f"model_id      : {args.model_id}")
    print(f"quant         : {args.quant}")
    print(f"subject_group : {args.subject_group}")

    if args.subject_group not in SUBJECT_GROUPS:
        print(f"Error: '{args.subject_group}' not in {list(SUBJECT_GROUPS.keys())}")
        sys.exit(1)
    subjects_to_run = SUBJECT_GROUPS[args.subject_group]

    # Load the HuggingFace token from a local .env file containing the line:
    #     HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    # (Some gated models, e.g. Llama, require accepting a license on the Hub
    # first.) Point DOTENV_PATH at your .env, or drop a .env in the working
    # directory. NEVER commit your .env -- it is gitignored in this repo.
    # Alternative: run `huggingface-cli login` instead of using a .env file.
    env_path = os.environ.get("DOTENV_PATH", ".env")
    load_dotenv(dotenv_path=env_path)
    token = os.getenv("HF_TOKEN")

    print(f"PyTorch : {torch.__version__}")
    print(f"CUDA    : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU     : {torch.cuda.get_device_name(0)}")

    SEED = 42
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, token=token)
    tokenizer.pad_token = tokenizer.eos_token
    print(f"Tokenizer loaded | vocab size: {tokenizer.vocab_size:,}")

    if args.quant:
        print("Loading with NF4 4-bit quantization")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id, device_map="auto",
            quantization_config=bnb_config, token=token,
        )
    else:
        print("Loading in bfloat16")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id, device_map="auto",
            dtype=torch.bfloat16, token=token,
        )
    model.eval()
    print(f"Model loaded | device: {next(model.parameters()).device} | dtype: {next(model.parameters()).dtype}")

    for subject, n in subjects_to_run:
        print(f"\n{'='*60}\nSUBJECT: {subject}  ({n} questions)\n{'='*60}")
        evaluate_mmlu(
            model=model, tokenizer=tokenizer, model_id=args.model_id,
            subject=subject, split="test", num_questions=n,
        )

    del model
    torch.cuda.empty_cache()
    print("\nDone.")
