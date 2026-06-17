#!/usr/bin/env python3
"""
MMLU-Pro Evaluation — INSTRUCT MODELS — HPC batch version
==========================================================
Notebook-equivalent: Insutrct_MMLUPRO.ipynb

Usage:
    python mmlupro_instruct.py <model_id> <quant> <subject_group>

Examples:
    python mmlupro_instruct.py meta-llama/Llama-3.2-1B-Instruct False smoke_test
    python mmlupro_instruct.py meta-llama/Llama-3.1-8B-Instruct True  stem
    python mmlupro_instruct.py Qwen/Qwen2.5-72B-Instruct       True  all
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
# MMLU-Pro stratified sampling plan — 3,000 questions across 9 categories.
#
# CONTAMINATION-FILTERED VERSION
# ------------------------------
# This plan only includes MMLU-Pro categories that contain a non-trivial
# number of NEWLY-ADDED questions (i.e. not carried over from the original
# MMLU).  In addition, `evaluate_mmlu_pro` filters out any remaining
# original-MMLU questions on a per-row basis using the `src` field, so that
# every question evaluated here is one that did NOT appear in the prior
# MMLU run.
#
# MMLU-Pro per-category newly-added question counts (from the official
# MMLU-Pro paper, Table 5 / dataset card):
#
#   Discipline         Total   From MMLU   Newly Added
#   ─────────────────  ─────   ─────────   ───────────
#   Math               1351    846         505
#   Physics            1299    411         888
#   Chemistry          1132    178         954
#   Law                1101    1101          0   ← DROPPED
#   Engineering         969     67         902
#   Other               924    924           0   ← DROPPED
#   Economics           844    444         400
#   Health              818    818           0   ← DROPPED
#   Psychology          798    493         305
#   Business            789    155         634
#   Biology             717    219         498
#   Philosophy          499    499           0   ← DROPPED
#   Computer Science    410    274         136
#   History             381    381           0   ← DROPPED
#
# Categories with zero non-MMLU questions are dropped entirely (law, health,
# philosophy, history, other).  Humanities is therefore not represented in
# this run — an explicit, documented limitation of using MMLU-Pro for a
# contamination-controlled evaluation.
#
# Allocation
# ----------
# STEM cluster — 1,661 questions across 6 categories:
#   * Computer Science only has 136 newly-added questions, so we use all of
#     them and split the remaining 1,525 evenly across the other 5 STEM
#     categories (305 each).
#
# Social Sciences cluster — 1,339 questions across 3 categories:
#   * Uses ALL available newly-added questions in each category
#     (economics 400, business 634, psychology 305).
#
# Grand total: 3,000 questions.
#
# HPC sub-batches
# ---------------
# To allow smaller jobs that queue faster on the HPC, the full plan is
# split into 5 sub-batches that each load the model once and evaluate
# 1–2 categories.  Each sub-batch is between ~440 and ~705 questions.
#
#   stem_a  = math (305)        + biology (305)            = 610
#   stem_b  = physics (305)     + chemistry (305)          = 610
#   stem_c  = engineering (305) + computer science (136)   = 441
#   ss_a    = economics (400)   + psychology (305)         = 705
#   ss_b    = business (634)                                = 634
#
# Per-category groups (e.g. "math_only", "business_only") are also provided
# for the finest-grained HPC submission.
SUBJECT_GROUPS = {
    # ── Full clusters ───────────────────────────────────────────────────────
    "stem": [
        ("math",              305),
        ("physics",           305),
        ("chemistry",         305),
        ("engineering",       305),
        ("biology",           305),
        ("computer science",  136),   # all available newly-added CS questions
    ],

    "social_sciences": [
        ("economics",         400),   # all available newly-added economics questions
        ("business",          634),   # all available newly-added business questions
        ("psychology",        305),   # all available newly-added psychology questions
    ],

    # ── HPC sub-batches: STEM split into 3 ──────────────────────────────────
    "stem_a": [
        ("math",              305),
        ("biology",           305),
    ],
    "stem_b": [
        ("physics",           305),
        ("chemistry",         305),
    ],
    "stem_c": [
        ("engineering",       305),
        ("computer science",  136),
    ],

    # ── HPC sub-batches: Social Sciences split into 2 ───────────────────────
    "ss_a": [
        ("economics",         400),
        ("psychology",        305),
    ],
    "ss_b": [
        ("business",          634),
    ],

    # ── Per-category groups (one category per HPC job) ──────────────────────
    "math_only":             [("math",              305)],
    "physics_only":          [("physics",           305)],
    "chemistry_only":        [("chemistry",         305)],
    "engineering_only":      [("engineering",       305)],
    "biology_only":          [("biology",           305)],
    "computer_science_only": [("computer science",  136)],
    "economics_only":        [("economics",         400)],
    "business_only":         [("business",          634)],
    "psychology_only":       [("psychology",        305)],

    # ── Convenience: full 3,000-question run (STEM + Social Sciences) ───────
    "all": [
        ("math",              305),
        ("physics",           305),
        ("chemistry",         305),
        ("engineering",       305),
        ("biology",           305),
        ("computer science",  136),
        ("economics",         400),
        ("business",          634),
        ("psychology",        305),
    ],

    # ── Smoke test — quick sanity check ─────────────────────────────────────
    "smoke_test": [
        ("math", 10),
    ],
}

# ── BOOLEAN ARG ───────────────────────────────────────────────────────────────
def str2bool(v):
    if v.lower() == "true":  return True
    if v.lower() == "false": return False
    raise argparse.ArgumentTypeError("Boolean value expected: True or False")

# ── CONSTANTS ─────────────────────────────────────────────────────────────────
# CHANGED FOR MMLU-Pro: 10 options (A–J) instead of 4 (A–D).
# Some MMLU-Pro questions have fewer than 10 options; the pipeline handles
# that per-question by only using the letters that actually exist for the
# current question.  LETTER_OPTIONS is the *maximum* set.
LETTER_OPTIONS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
IDX_TO_LETTER  = {i: L for i, L in enumerate(LETTER_OPTIONS)}

# ── TOKEN MAPS ────────────────────────────────────────────────────────────────
def build_letter_token_map(tokenizer) -> dict:
    """
    Maps each uppercase letter A-Z to ALL token IDs that decode to that letter
    (covering surface forms like 'A', ' A', '▁A', 'ĠA').

    Why pool variants?
    Tokenizers represent ' A' (space+A) and 'A' as different token IDs.
    After 'Answer:' the model may assign probability to either form.
    Summing both gives the true probability of the model intending letter A.
    """
    mapping = {L: [] for L in ascii_uppercase}
    for tok_id in range(tokenizer.vocab_size):
        decoded = tokenizer.decode(tok_id).replace('▁', '').replace('Ġ', '').strip()
        if decoded in mapping:
            mapping[decoded].append(tok_id)
    return mapping


def build_binary_token_map(tokenizer) -> dict:
    """
    UPDATED: Maps '1' and '0' to ALL token IDs that decode to those digits,
    including variants with spaces or special prefix characters (e.g., ' 1', 'Ġ1').
    """
    mapping = {'1': [], '0': []}
    for tok_id in range(tokenizer.vocab_size):
        # We now apply the same cleaning logic as the letter map
        decoded = tokenizer.decode(tok_id).replace('▁', '').replace('Ġ', '').strip()
        if decoded in mapping:
            mapping[decoded].append(tok_id)
    return mapping

# ── PROMPTS ───────────────────────────────────────────────────────────────────
def build_instruct_prompt(question: str, choices: list, tokenizer) -> str:
    """
    Chat-template prompt for Pass 1 (getting the initial answer letter).
    Appends the 'Answer:' cue AFTER the template to ensure the first
    predicted token is the choice letter.
    """
    opts = "\n".join(f"{L}. {c}" for L, c in zip(LETTER_OPTIONS, choices))

    # We removed "Answer:" from here
    user_msg = (
        "Answer the following multiple choice question with only the letter "
        "of the correct answer — no explanation.\n\n"
        f"Question: {question}\n"
        f"{opts}\n"
    )

    chat = [{"role": "user", "content": user_msg}]

    # Generate the base template
    base = tokenizer.apply_chat_template(
        chat, tokenize=False, add_generation_prompt=True
    )

    # Append the cue to the assistant turn for a "forced" start
    return base + "Answer:"


def letter_to_text(letter: str, choices: list) -> str:
    """
    Resolve a predicted letter to 'A. London' format for use in confidence prompts.
    Falls back to a safe placeholder if Pass 1 failed to produce a valid letter
    (e.g., model emitted whitespace, quote, or a non-A-J token as the first token).

    CHANGED FOR MMLU-Pro: now also guards against `letter` being a valid letter
    (e.g., 'J') that is *out of range* for this particular question (which may
    have fewer than 10 options).
    """
    if letter in LETTER_OPTIONS:
        idx = LETTER_OPTIONS.index(letter)
        if idx < len(choices):                         # NEW guard
            return f"{letter}. {choices[idx]}"
    return "(no valid answer produced)"


def build_p_true_prompt(question: str, choices: list, predicted_letter: str, tokenizer) -> str:
    """
    P(True) prompt — chat-template version.
    User message contains the full context; the assistant turn is seeded with
    'Answer: (' so the very next token is 'A' (True) or 'B' (False).
    apply_chat_template handles model-specific special tokens automatically.
    """
    choice_lines = "\n".join(
        f"{L}. {choices[i]}" for i, L in enumerate(LETTER_OPTIONS) if i < len(choices)
    )
    user_msg = (
        f"Question: {question}\n"
        f"{choice_lines}\n"
        f"Proposed answer: {letter_to_text(predicted_letter, choices)}\n"
        f"Is the proposed answer correct?\n"
        f"(A) True\n"
        f"(B) False\n"
    )
    chat = [{"role": "user", "content": user_msg}]
    # add_generation_prompt=True appends the assistant-turn opening tokens.
    # We then append our cue so the next predicted token is 'A' or 'B'.
    base = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    return base + "Answer: ("


def build_p_one_prompt(question: str, choices: list, predicted_letter: str, tokenizer) -> str:
    """
    P(1) prompt — chat-template version.
    User message contains the full context; the assistant turn is seeded with
    'My confidence: ' so the next token is '1' or '0'.
    """
    choice_lines = "\n".join(
        f"{L}. {choices[i]}" for i, L in enumerate(LETTER_OPTIONS) if i < len(choices)
    )
    user_msg = (
        f"Question: {question}\n"
        f"{choice_lines}\n"
        f"I answered: {letter_to_text(predicted_letter, choices)}\n\n"
        f"Now I will rate my confidence that my answer was correct.\n"
        f"1 = I believe my answer is correct.\n"
        f"0 = I believe my answer is incorrect.\n"
    )
    chat = [{"role": "user", "content": user_msg}]
    base = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    return base + "My confidence: "


def build_verbal_prompt(question: str, choices: list, predicted_letter: str, tokenizer) -> str:
    """
    Verbal confidence prompt — chat-template version.
    User message contains the full context; the assistant turn is seeded with
    'Confidence: ' and model.generate continues from there.
    """
    choice_lines = "\n".join(
        f"{L}. {choices[i]}" for i, L in enumerate(LETTER_OPTIONS) if i < len(choices)
    )
    user_msg = (
        f"Question: {question}\n"
        f"{choice_lines}\n"
        f"Your previous proposed answer was: {letter_to_text(predicted_letter, choices)}\n"
        f"On a scale of 0-100, how confident are you that this answer is correct?\n"
    )
    chat = [{"role": "user", "content": user_msg}]
    base = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    return base + "Confidence: "

# ── CONFIDENCE SCORERS ────────────────────────────────────────────────────────
def compute_p_true(probs, letter_map):
    """
    Renormalised P(A) over {A, B} from a full-vocabulary softmax tensor.
    A = True, B = False.  Returns value in [0, 1].
    Higher = model believes its answer is correct.
    """
    p_a = probs[letter_map['A']].sum().item()
    p_b = probs[letter_map['B']].sum().item()
    return p_a / (p_a + p_b)


def compute_p_one(probs, binary_token_map):
    """
    Renormalised P('1') over {'1', '0'} from a full-vocabulary softmax tensor.
    Returns value in [0, 1].  Higher = model believes its answer is correct.
    """
    p_one  = probs[binary_token_map['1']].sum().item()
    p_zero = probs[binary_token_map['0']].sum().item()
    return p_one / (p_one + p_zero)


def extract_confidence_MSP(probs, letter_map, letters):
    """
    Maximum Softmax Probability renormalised over the option letters that exist
    for THIS question (e.g., A–D for a 4-option question, A–J for a 10-option
    question).  Returns a dict {letter: renorm_prob} that sums to 1.
    """
    raw   = {L: probs[letter_map[L]].sum().item() for L in letters}
    total = sum(raw.values())
    if total == 0:
        # Pathological case — model assigned zero mass to every option letter.
        # Return a uniform distribution so downstream metrics are still defined.
        print("Assigned zero to every option")
        return {L: 1.0 / len(letters) for L in letters}
    return {L: v / total for L, v in raw.items()}


def compute_entropy_metrics(renorm_probs):
    """
    Returns (raw_entropy, confidence_entropy) from a renormalised probability dict.
    confidence_entropy = 1 - H/H_max, so higher = more confident.
    H_max scales with the number of options (so 4-option and 10-option questions
    are both normalised to [0, 1]).
    """
    prob_tensor = torch.tensor(list(renorm_probs.values()))
    raw_entropy = torch.distributions.Categorical(probs=prob_tensor).entropy().item()
    H_max       = math.log(len(renorm_probs))
    return raw_entropy, 1.0 - (raw_entropy / H_max)


def compute_margin_renorm(renorm_probs):
    """Top-1 minus Top-2 over the renormalised options distribution."""
    vals = sorted(renorm_probs.values(), reverse=True)
    return vals[0] - vals[1]


def parse_verbal_confidence(text):
    """Extract the first integer in 0-100 from a generated string, or None."""
    matches = re.findall(r"\b(\d{1,3})\b", text)
    return max(0, min(100, int(matches[0]))) if matches else None

# ── PER-QUESTION PIPELINE ─────────────────────────────────────────────────────
def run_question_all(
    question: str,
    choices: list,
    model,
    tokenizer,
    letter_map: dict,
    binary_token_map: dict,
) -> dict:
    """
    Four independent forward passes per question.
    No memory is shared between passes — each receives only the static
    question context plus the predicted letter string (not any hidden state).
    """

    # Letters available for THIS question.
    letters = LETTER_OPTIONS[:len(choices)]

    # ── Pass 1: instruct-formatted answer + implicit confidence metrics ───────
    # We use generate() with output_scores=True so we get the first-token logits
    # (for MSP/entropy/margin) AND the full 10-token generation (for diagnostics
    # when Pass 1 fails to produce a valid letter as the first token).
    prompt = build_instruct_prompt(question, choices, tokenizer)
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    n_prompt_pass1 = inputs["input_ids"].shape[1]

    with torch.no_grad():
        gen_out = model.generate(
            **inputs,
            max_new_tokens=10,
            do_sample=False,                    # Greedy — deterministic
            pad_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=True,                 # First-token logits via .scores[0]
        )

    # First-token logits — mathematically identical to the old
    # model(**inputs).logits[0, -1, :] approach, because greedy generation
    # computes the next-token distribution the same way.
    logits = gen_out.scores[0][0].float()

    # Full generated text (up to 10 tokens) — diagnostic view of Pass 1
    pass1_10_token_generation = tokenizer.decode(
        gen_out.sequences[0, n_prompt_pass1:], skip_special_tokens=True
    ).strip()

    raw_output    = tokenizer.decode(logits.argmax().item())
    # CHANGED: only accept letters that are valid options for THIS question.
    letter_greedy = next((c for c in raw_output if c in letters), None)

    probs_pass1  = torch.softmax(logits, dim=-1)
    renorm_MSP   = extract_confidence_MSP(probs_pass1, letter_map, letters)
    letter_MSP   = max(renorm_MSP, key=renorm_MSP.get)

    raw_entropy, confidence_entropy = compute_entropy_metrics(renorm_MSP)
    margin_renorm = compute_margin_renorm(renorm_MSP)

    top_probs, top_ids = torch.topk(probs_pass1, k=2)
    margin_full    = (top_probs[0] - top_probs[1]).item()
    top_2_tokens   = [tokenizer.decode(top_ids[0]), tokenizer.decode(top_ids[1])]

    # ── Pass 2: P(True) — chat-template prompt, independent call ─────────────
    pt_inputs = tokenizer(
        build_p_true_prompt(question, choices, letter_greedy, tokenizer),
        return_tensors="pt", add_special_tokens=False
    ).to(model.device)
    with torch.no_grad():
        pt_logits = model(**pt_inputs).logits[0, -1, :].float()
    pt_probs       = torch.softmax(pt_logits, dim=-1)
    p_true_score   = compute_p_true(pt_probs, letter_map)
    p_true_raw_out = tokenizer.decode(pt_logits.argmax().item())

    # ── Pass 3: P(1) — chat-template prompt, independent call ────────────────
    p1_inputs = tokenizer(
        build_p_one_prompt(question, choices, letter_greedy, tokenizer),
        return_tensors="pt", add_special_tokens=False
    ).to(model.device)
    with torch.no_grad():
        p1_logits = model(**p1_inputs).logits[0, -1, :].float()
    p1_probs    = torch.softmax(p1_logits, dim=-1)
    p_one_score = compute_p_one(p1_probs, binary_token_map)
    p1_raw_out  = tokenizer.decode(p1_logits.argmax().item())

    # ── Pass 4: verbal — chat-template prompt, independent call ───────────────
    verbal_inputs = tokenizer(
        build_verbal_prompt(question, choices, letter_greedy, tokenizer),
        return_tensors="pt", add_special_tokens=False
    ).to(model.device)
    with torch.no_grad():
        verbal_out = model.generate(
            **verbal_inputs,
            max_new_tokens=10,
            do_sample=False,          # Greedy — deterministic
            pad_token_id=tokenizer.eos_token_id,
        )
    n_prompt      = verbal_inputs["input_ids"].shape[1]
    generated     = tokenizer.decode(verbal_out[0, n_prompt:], skip_special_tokens=True).strip()
    verbal_parsed = parse_verbal_confidence(generated)

    return {
        "raw_output"                : raw_output,
        "pass1_10_token_generation" : pass1_10_token_generation,
        "letter_greedy"             : letter_greedy,
        "letter_MSP"                : letter_MSP,
        "renorm_MSP"                : renorm_MSP,           # may have 4–10 keys
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
        "n_options"                 : len(choices),         # NEW for MMLU-Pro
    }

# ── EVALUATE FUNCTION ─────────────────────────────────────────────────────────
# Lazy global cache so we only load the 12,032-question file once per process.
_MMLU_PRO_CACHE = None

def _get_mmlu_pro():
    """
    Lazily load the MMLU-Pro test split and DROP any rows whose `src` field
    indicates the question was carried over from the original MMLU.

    Original-MMLU rows in MMLU-Pro use `src` values prefixed with either
    `cot_lib-` (most categories) or `ori_mmlu-` (some categories).  All
    newly-added questions use prefixes from the new sources: `stemez-`,
    `theoremQA-`, `scibench-`.  By filtering out the two MMLU prefixes we
    keep only questions that did NOT appear in the original MMLU dataset,
    which is exactly the contamination-controlled subset we want to test.

    The filter is applied once at load time, so every downstream call to
    `evaluate_mmlu_pro` automatically sees only the filtered dataset.
    """
    global _MMLU_PRO_CACHE
    if _MMLU_PRO_CACHE is None:
        print("Loading TIGER-Lab/MMLU-Pro (test split, 12,032 questions)...", flush=True)
        full = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
        n_before = len(full)

        # CONTAMINATION FILTER: drop any row whose src indicates original-MMLU origin.
        def _is_new(ex):
            s = (ex.get("src") or "").lower()
            return not (s.startswith("cot_lib-") or s.startswith("ori_mmlu-"))

        _MMLU_PRO_CACHE = full.filter(_is_new)
        n_after = len(_MMLU_PRO_CACHE)
        print(f"Loaded.  Kept {n_after} of {n_before} questions after dropping "
              f"original-MMLU rows ({n_before - n_after} removed).", flush=True)

        # Sanity-check report: show how many newly-added questions remain per
        # category so the user can confirm the allocation in SUBJECT_GROUPS
        # is feasible.
        from collections import Counter
        per_cat = Counter(_MMLU_PRO_CACHE["category"])
        print("Non-MMLU questions available per category:", flush=True)
        for cat in sorted(per_cat):
            print(f"  {cat:<22} {per_cat[cat]}", flush=True)
    return _MMLU_PRO_CACHE


def evaluate_mmlu_pro(
    model,
    tokenizer,
    model_id: str,
    subject: str,
    split: str          = "test",   # kept for API parity with old evaluate_mmlu
    num_questions: int  = None,
) -> pd.DataFrame:
    """
    Run all confidence metrics (MSP, Entropy, Margin, P(True), P(1), Verbal)
    on one MMLU-Pro category and save a single CSV:
        mmlupro_<category>_<model>_instruct_all.csv
    """
    letter_map_local       = build_letter_token_map(tokenizer)
    binary_token_map_local = build_binary_token_map(tokenizer)

    # CHANGED: filter the (already contamination-filtered) global MMLU-Pro
    # test set by category.  The src-based contamination filter is applied
    # in _get_mmlu_pro() at load time; the assertion below is a defensive
    # double-check so a future change to _get_mmlu_pro can't silently
    # reintroduce original-MMLU rows.
    full_ds = _get_mmlu_pro()
    dataset = full_ds.filter(lambda ex: ex["category"] == subject)
    for _ex in dataset:
        _src = (_ex.get("src") or "").lower()
        assert not (_src.startswith("cot_lib-") or _src.startswith("ori_mmlu-")), (
            f"Contamination check failed: row with src={_ex.get('src')!r} "
            f"slipped past the filter."
        )
        break  # one row is enough to verify the filter is active
    if num_questions is not None:
        dataset = dataset.select(range(min(num_questions, len(dataset))))

    model_short    = model_id.split('/')[-1]
    parse_failures = 0
    records        = []
    print(f"[All metrics] {model_short} | {subject} | {len(dataset)} questions", flush=True)

    for i, row in enumerate(dataset):
        # CHANGED: use answer_index (int) instead of answer (which is a letter
        # in MMLU-Pro).  IDX_TO_LETTER converts 0–9 to 'A'–'J'.
        correct_letter = IDX_TO_LETTER[row["answer_index"]]
        choices        = row["options"]                    # list of up to 10 strings

        result = run_question_all(
            row["question"], choices,
            model, tokenizer,
            letter_map_local, binary_token_map_local,
        )
        if result["verbal_parsed"] is None:
            parse_failures += 1

        # CHANGED: write prob_A … prob_J columns; NaN for letters not present
        # in this question's options.
        prob_cols = {f"prob_{L}": result["renorm_MSP"].get(L, float("nan"))
                     for L in LETTER_OPTIONS}

        records.append({
            # ── question metadata ──────────────────────────────────────────
            "Question"                           : row["question"],
            "Choices"                            : str(choices),
            "n_options"                          : result["n_options"],     # NEW
            "category"                           : row["category"],         # NEW
            "question_id"                        : row.get("question_id"),  # NEW
            "src"                                : row.get("src"),          # NEW: provenance (e.g. stemez-*, theoremQA-*, scibench-*)
            "correct_letter"                     : correct_letter,
            # ── Pass 1: answer letter ──────────────────────────────────────
            "llm_first_token_output"             : result["raw_output"],
            "letter_llm_outputted_detected"      : result["letter_greedy"],
            "pass1_10_token_generation"          : result["pass1_10_token_generation"],
            "Letter_MSP"                         : result["letter_MSP"],
            "Letter_MSP_confidence"              : result["msp_confidence"],
            "is_correct"                         : result["letter_greedy"] == correct_letter,
            "check_letter_output_is_highest_MSP" : result["greedy_matches_msp"],
            # ── Pass 1: implicit metrics ───────────────────────────────────
            **prob_cols,                                                    # CHANGED: prob_A..prob_J
            "Raw_Entropy"                        : result["raw_entropy"],
            "Entropy_as_confidence"              : result["entropy_conf"],
            "Margin_softmax_full_vocab"          : result["margin_full"],
            "Margin_renorm_options"              : result["margin_renorm"], # renamed
            "Top_2_Tokens"                       : result["top_2_tokens"],
            # ── Pass 2: P(True) ────────────────────────────────────────────
            "p_true_raw_output"                  : result["p_true_raw_output"],
            "p_true_probability"                 : result["p_true_probability"],
            # ── Pass 3: P(1) ───────────────────────────────────────────────
            "p1_raw_output"                      : result["p1_raw_output"],
            "p1_probability"                     : result["p1_probability"],
            # ── Pass 4: verbal ─────────────────────────────────────────────
            "verbal_full_text_generated"         : result["verbal_raw"],
            "verbal_integer_found"               : result["verbal_parsed"],
        })

        if (i + 1) % 10 == 0:
            acc     = np.mean([r["is_correct"] for r in records])
            avg_pt  = np.mean([r["p_true_probability"] for r in records])
            avg_p1  = np.mean([r["p1_probability"] for r in records])
            agree   = np.mean([r["check_letter_output_is_highest_MSP"] for r in records])
            valid   = sum(1 for r in records if r["verbal_integer_found"] is not None)
            print(f"  [{i+1:>4}/{len(dataset)}]  acc={acc:.3f}  "
                  f"avg_p_true={avg_pt:.3f}  avg_p1={avg_p1:.3f}  "
                  f"agree={agree:.3f}  verbal_valid={valid}/{i+1}", flush=True)

    df = pd.DataFrame(records)
    safe_subject = subject.replace(" ", "_")
    out_csv = f"mmlupro_{safe_subject}_{model_short}_instruct_all.csv"
    df.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}", flush=True)
    print(f"Verbal parse failures: {parse_failures}/{len(dataset)}", flush=True)
    return df


# Backwards-compatible alias
evaluate_mmlu = evaluate_mmlu_pro

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
        print(f"\n{'='*60}\nCATEGORY: {subject}  ({n} questions)\n{'='*60}")
        evaluate_mmlu_pro(
            model=model, tokenizer=tokenizer, model_id=args.model_id,
            subject=subject, split="test", num_questions=n,
        )

    del model
    torch.cuda.empty_cache()
    print("\nDone.")
