# Terminal Assistant

A fine-tuned 2B parameter LLM that generates and executes bash commands, running entirely on a MacBook.

**Base model:** [Qwen3.5-2B-OptiQ-4bit](https://huggingface.co/mlx-community/Qwen3.5-2B-OptiQ-4bit) (~1.2GB, auto-downloaded on first run)
**Adapter:** 11MB LoRA adapter trained with MLX on Apple Silicon
**Training data:** 2,072 examples distilled from Claude Sonnet
**Result:** 60% → 89% functional correctness on command tasks

### Requirements

- **Apple Silicon Mac** (M1/M2/M3/M4) — MLX is Apple Silicon only
- **Python 3.11+**
- **Docker** (optional, for sandbox mode)

## Quick Start

```bash
# Setup
git clone https://github.com/zackedds/terminal-agent.git
cd terminal-agent
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Interactive mode (approve each command)
# First run downloads the base model (~1.2GB) from HuggingFace
python3 cli.py

# Auto-execute safe commands
python3 cli.py --auto

# Run in Docker sandbox (safe for testing)
cd sandbox && docker compose up -d && cd ..
python3 cli.py --sandbox

# Dev mode (show tokens/sec, model thinking, raw output)
python3 cli.py --dev

# One-shot
python3 cli.py "find all python files with TODO comments"
```

## How It Works

```
> how many lines of code are in each python file
  $ find . -name "*.py" | xargs wc -l
Execute? [y/N] y
   11 ./tests/test_utils.py
   34 ./src/utils.py
   30 ./src/api.py
   96 total
```

The model generates bash commands using Qwen's native tool-calling format. The CLI parses the command, checks it against safety rules, and either executes or asks for approval.

### Safety Layers

1. **Blacklist** — `rm -rf /`, fork bombs, `dd` to disk, `curl|bash` → blocked, never executed
2. **Approval required** — `sudo`, `rm -rf`, `git push`, `kill` → always prompts, even in `--auto` mode
3. **Default approval** — all commands require `y/n` unless `--auto` is set

## Training

### Reproduce from scratch

```bash
# Train (~28 min on M2 Pro, requires mlx-lm)
bash training/train.sh training/lora_config_best.yaml adapters

# Evaluate (requires Docker)
cd sandbox && docker compose up -d && cd ..
python3 eval/run_baseline.py --adapter adapters --trials 3 --output eval/results.json
```

### Key training decisions

- **QLoRA with MLX** — 4-bit quantized base, LoRA rank 8, 8/24 layers, dropout 0.05
- **Cosine LR decay** with 100-step warmup, 2000 iterations (~1 epoch)
- **Command-only training data** — safety/refusal examples were removed after discovering LoRA task interference (contradictory signals in a low-rank subspace degrade both tasks)
- **Knowledge distillation** — Claude Sonnet generated all training data

### macOS Tahoe GPU Bug

Training on macOS 26.x crashes with `kIOGPUCommandBufferCallbackErrorImpactingInteractivity` — the GPU watchdog kills MLX because command buffers block the display compositor.

**Fix:** `AGX_RELAX_CDM_CTXSTORE_TIMEOUT=1` (already set in `training/train.sh`)

Details: [ml-explore/mlx#3267](https://github.com/ml-explore/mlx/issues/3267)

## Results

| Metric | Base 2B | Fine-tuned 2B | Improvement |
|---|---|---|---|
| Format Compliance | 92.3% | **100.0%** | +7.7% |
| Functional Correctness | 60.0% | **88.9%** | +28.9% |
| Consistency | 80.0% | **98.5%** | +18.5% |

Evaluated on 65 test cases across 7 categories (file ops, text processing, git, system info, process management, compound tasks, edge cases), each run 3 times in a Docker sandbox.

## Project Structure

```
├── cli.py                      # Terminal assistant CLI
├── adapters/                   # Best LoRA adapter (command-only, v7)
├── adapters_best_overall/      # Best overall adapter (v5, includes safety)
├── data/
│   ├── train.jsonl             # Training split (2,072 examples)
│   ├── valid.jsonl             # Validation split (259 examples)
│   ├── test.jsonl              # Test split (260 examples)
│   └── source_batches/         # Original generation batches
├── eval/
│   ├── run_baseline.py         # Evaluation runner
│   ├── model_serve.py          # MLX model wrapper
│   ├── sandbox_exec.py         # Docker sandbox execution
│   ├── score.py                # Scoring logic
│   ├── test_cases.json         # 65 test cases
│   ├── results/                # All eval results (JSON)
│   ├── BASELINE_RESULTS.md     # Full methodology write-up
│   └── RESULTS_SNAPSHOT.md     # Latest results summary
├── training/
│   ├── lora_config_best.yaml   # Best training config
│   ├── lora_config.yaml        # Original config
│   └── train.sh                # Training script (includes AGX fix)
├── sandbox/
│   ├── Dockerfile              # Ubuntu 22.04 eval sandbox
│   ├── docker-compose.yml      # Network-isolated container
│   └── setup_workspace.sh      # Populates sandbox with test project
├── BLOG_POST.md                # Full project write-up
└── RESEARCH_NOTES.md           # LoRA capacity and task interference research
```

## Blog Post

See [BLOG_POST.md](BLOG_POST.md) for the full story — model selection, eval methodology, the GPU bug investigation, task interference discovery, and lessons learned.
