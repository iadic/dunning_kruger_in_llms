# Do Large Language Models Have the Dunning–Kruger Effect?
This project tests whether AI language models (LLMs) show a **Dunning–Kruger–style** pattern: do the *weaker* models act **over-confident** — sure of answers they get wrong — and does that over-confidence fade as the models get bigger and better?

In humans, the Dunning–Kruger effect is the tendency for less-skilled people to *over-estimate* how good they are. Here we ask the same question of LLMs by comparing **how confident a model says it is** against **how often it is actually right**.

---

## The idea in one picture

For every multiple-choice question we record two things:

- **Was the model right?** (its accuracy)
- **How confident was it?** (measured six different ways)

If confidence sits **above** accuracy — especially for the smaller or less-trained models — that's the over-confidence. We then check whether that gap **shrinks as models grow**.

We look at this from three angles:

1. **Across model sizes** — small vs. large models in the same family (LLaMA, Qwen, Gemma).
2. **Base vs. instruction-tuned** — does the chat/instruction fine-tuning step change the picture?
3. **Across training time** — a single model (OLMo-2) measured at many points during its training, to see when over-confidence appears.

We do all of this on an **easy** test (MMLU) and a **hard** test (MMLU-Pro).

---

## What's in this repository

```
.
├── README.md
│
├── 1_evaluation_scripts/              ← run the models (command line, for a GPU cluster)
│   ├── mmlu_base.py                   ←   MMLU       · base models
│   ├── mmlu_instruct.py               ←   MMLU       · instruction-tuned models
│   ├── mmlupro_base.py                ←   MMLU-Pro   · base models
│   ├── mmlupro_instruct.py            ←   MMLU-Pro   · instruction-tuned models
│   └── run_example.slurm              ←   example job script for a SLURM scheduler
│
├── 2_evaluation_notebooks/            ← the exact same code, in notebook form
│   ├── base_MMLU.ipynb                ←   (easier to read, step through, or run on Colab)
│   ├── base_MMLUPro.ipynb
│   ├── Instruct_MMLU.ipynb
│   └── Instruct_MMLUPro.ipynb
│
├── 3_developmental_checkpoints/       ← measure ONE model across its training checkpoints
│   ├── checkpoint_analysis_mmlu.ipynb     ←   easy test (MMLU)
│   └── checkpoint_analysis_mmlupro.ipynb  ←   hard test (MMLU-Pro)
│
└── 4_analysis_and_plots/              ← turn the result files into figures
    ├── DKE_analysis_and_plotting.ipynb        ←   main analysis + headline figures
    ├── checkpoint_easy_vs_hard_plots.ipynb    ←   training-checkpoint figures
    └── individual_family_analysis_example.ipynb ←  worked example for one model family
```

**In plain terms:**

- **Folder "JupterNotebook verisions" & "python verisions"** do the same job (collect the data) in two formats. The **scripts** are for running many models on a powerful computer or cluster; the **notebooks** are the same thing, but easier to read and tinker with. Use whichever suits you.
-  **Folder Code for plotting and analysis** takes the result files produced by folders "JupterNotebook verisions" & "python verisions" and makes the charts and tables (analysises the results).
  
- **Folder checkpoint_analysis** is a special case: instead of comparing different models, it takes *one* model (OLMo-2) and measures it at many stages of its training, so you can watch over-confidence develop over time. Also included is the Pythia model checkpoints and plotting of those results.

There are four evaluation scripts because **base** and **instruction-tuned** models need different prompting, and the **easy** and **hard** tests load different question sets.

---

## How a model is tested

Each model answers the question, then is asked about its confidence in several ways. We record **six confidence measures** per question:

| Measure | What it means (simply) |
|---|---|
| **MSP** | How much probability the model put on its chosen answer letter |
| **Entropy** | How "spread out" its answer probabilities were (spread = unsure) |
| **Margin** | The gap between its top two answer choices |
| **P(True)** | The model's own estimate that its answer is correct |
| **P(1)** | A yes/no "is my answer right?" self-rating |
| **Verbal** | A 0–100 confidence the model states in words |

Each run saves one CSV file per subject, with the model's answer, whether it was correct, and these six numbers — that's the raw data the analysis notebooks read.

---

## Setup

You need Python 3.10+ and (for running the models) a machine with a GPU.

```bash
# 1. Install the libraries
pip install torch transformers datasets bitsandbytes accelerate \
            python-dotenv numpy pandas matplotlib scikit-learn

# 2. Add your HuggingFace token
cp .env.example .env
#    then open .env and paste in your token:  HF_TOKEN=hf_xxxxxxxx
```

**Why a token?** The models and test questions are downloaded from the HuggingFace Hub, and some models (such as LLaMA) require you to accept a licence on their page first. Get a free token at <https://huggingface.co/settings/tokens>.

Your `.env` file is private and is ignored by Git, so your token is never uploaded.

*(Optional)* If you're on a shared cluster, send the large model downloads to scratch storage:

```bash
export HF_CACHE=/path/to/scratch/hf_cache
```

---

## How to run it

**Step 1 — Collect the data.** Each script takes three arguments: the model, whether to use 4-bit quantization (`True` for big models, `False` for small ones), and which group of subjects to run.

```bash
# quick test
python 1_evaluation_scripts/mmlu_instruct.py meta-llama/Llama-3.2-1B-Instruct False smoke_test

# a real run
python 1_evaluation_scripts/mmlu_instruct.py meta-llama/Llama-3.1-8B-Instruct False math
```

On a cluster, edit and submit `run_example.slurm` instead.

**Step 2 *(optional)* — Training-checkpoint runs.** Open a notebook in `3_developmental_checkpoints/`, set the checkpoint name in the CONFIG cell, and run it once per checkpoint.

**Step 3 — Make the figures.** Open `4_analysis_and_plots/DKE_analysis_and_plotting.ipynb`, point the folder paths at your result CSVs, and run it top to bottom. It produces the confidence-vs-accuracy figures, the over-confidence charts, calibration curves, and summary heatmaps.

---

## Models tested

Three open families, each in base and instruction-tuned versions, at several sizes:

- **LLaMA 3** — e.g. `meta-llama/Llama-3.2-1B`, `Llama-3.1-8B`, `Llama-3.1-70B`
- **Qwen 2.5** — e.g. `Qwen/Qwen2.5-0.5B`, `Qwen2.5-7B`, `Qwen2.5-72B`
- **Gemma 2** — e.g. `google/gemma-2-2b`, `gemma-2-9b`, `gemma-2-27b`

Naming tip: LLaMA and Qwen use a `-Instruct` suffix for their tuned versions; Gemma uses `-it`.

The training-checkpoint analysis uses **OLMo-2** (`allenai/OLMo-2-1124-13B`), with **Pythia** (`EleutherAI/pythia-12b`) as an extra comparison.

---

## Notes

- **Reproducibility.** A fixed random seed is set and decoding is greedy, so runs are repeatable on the same hardware.
- **No data is stored here.** Questions and model weights are downloaded at run time; result CSVs are generated on your own machine.
- **The two tests.** MMLU is the easier benchmark; MMLU-Pro is harder and has more answer options. The MMLU-Pro code also removes any questions that overlap with the original MMLU, so the two tests stay separate.

---

*This code accompanies an MPhil dissertation investigating the Dunning–Kruger effect in large language models.*
