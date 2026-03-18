# Terminal Assistant: Small Model Fine-Tuning Project

## Project Overview

**Goal:** Determine whether LoRA/QLoRA fine-tuning can meaningfully improve a small (4B parameter) open-source LLM's ability to act as a CLI terminal assistant, translating natural language requests into bash commands and interpreting results.

**Base Model:** Qwen3.5-4B (preferred) or Qwen3-4B (fallback if 3.5 isn't on MLX yet)
**Training Framework:** MLX with QLoRA on Apple Silicon (16GB MacBook Pro)
**Evaluation Philosophy:** Measure first. Only fine-tune if the base model demonstrably fails.

---

## Stage 0: Environment and Infrastructure Setup

**Goal:** Create a safe, reproducible, realistic sandbox for testing LLM-generated bash commands.

### 0.1 Docker Sandbox Environment

Create a Docker container that simulates a realistic developer workspace. The model's commands will execute inside this container, never on your host machine.

```
terminal-assistant/
├── sandbox/
│   ├── Dockerfile
│   ├── setup_workspace.sh      # Populates realistic files/dirs
│   └── docker-compose.yml
├── eval/
│   ├── test_cases.json          # All benchmark tasks
│   ├── run_baseline.py          # Stage 1: test base model
│   ├── generate_gold.py         # Stage 2: generate gold answers
│   ├── run_finetuned.py         # Stage 4: test fine-tuned model
│   └── score.py                 # Scoring logic
├── data/
│   ├── train.jsonl              # Training data (Stage 3)
│   ├── valid.jsonl
│   └── test.jsonl
├── training/
│   ├── lora_config.yaml         # MLX LoRA hyperparameters
│   └── train.sh                 # Training launch script
└── README.md
```

**Sandbox requirements:**
- Ubuntu-based Docker image with common dev tools (git, python, node, gcc, etc.)
- Pre-populated with a realistic file tree: a fake project repo with Python/JS files, a home directory with .bashrc/.gitconfig, some nested folders, a few GB of dummy files of various sizes, some running processes
- Network isolated (no internet from inside container)
- Each test case gets a fresh container (or a snapshot reset) so tests don't affect each other
- A harness script that: sends a command into the container via `docker exec`, captures stdout/stderr/exit code, returns results to the eval framework

### 0.2 Model Serving Setup

Set up local inference with MLX for the base model:

```bash
pip install mlx-lm
# Test that the model loads and generates
python -m mlx_lm.generate \
  --model mlx-community/Qwen3-4B-4bit \
  --prompt "List all Python files in the current directory" \
  --max-tokens 256
```

If Qwen3.5-4B is available in MLX format, prefer it. Otherwise use Qwen3-4B-4bit.

### 0.3 System Prompt and Output Format

Define a consistent system prompt and output format the model must follow:

```
You are a terminal assistant. The user will describe a task in natural language.
You must respond with EXACTLY ONE of:
1. A bash command wrapped in <cmd>...</cmd> tags
2. A natural language explanation if the task doesn't require a command

Examples:
User: "List all Python files modified in the last week"
Assistant: <cmd>find . -name "*.py" -mtime -7</cmd>

User: "What does the -r flag do in cp?"
Assistant: The -r flag tells cp to copy directories recursively...
```

This format is simple to parse and unambiguous for evaluation.

---

## Stage 1: Baseline Evaluation (Does the base model already work?)

**Goal:** Rigorously measure the base model's out-of-the-box performance on terminal tasks. Run every test multiple times (n=5) to measure consistency.

### 1.1 Test Case Design

Create 100+ test cases across these categories (roughly 10-15 per category):

**Category A: File Operations**
- List files in a directory (with various filters)
- Find files by name, extension, size, modification date
- Copy, move, rename files and directories
- Create directory structures
- Check file permissions, ownership
- Compare files (diff)

**Category B: Text Processing**
- Search file contents (grep with various flags)
- Count lines/words/characters (wc)
- Extract specific columns (awk, cut)
- Find and replace in files (sed)
- Sort and deduplicate
- Head/tail of files

**Category C: System Information**
- Disk usage (df, du)
- Running processes (ps, top)
- Memory usage
- Network info (ip, ifconfig, ss)
- Environment variables
- System uptime, OS info

**Category D: Process Management**
- Kill a process by name or port
- Run something in the background
- Check what's listening on a port
- Schedule a task (at, cron syntax)

**Category E: Git Operations**
- Status, log, diff
- Branch operations
- Find who changed a specific line (blame)
- Show recent commits by a specific author

**Category F: Compound/Multi-step Tasks**
- "Find all TODO comments in Python files and count them"
- "Show the 5 largest files in this directory tree"
- "Archive all .log files older than 30 days"
- "Find duplicate files by checksum"

**Category G: Edge Cases and Ambiguity**
- Ambiguous requests ("clean up this folder")
- Requests that need clarification
- Potentially dangerous commands (rm -rf, chmod 777)
- Platform-specific commands that might not exist

### 1.2 Test Case Format

```json
{
  "id": "file_001",
  "category": "file_operations",
  "difficulty": "easy",
  "prompt": "List all Python files in the current directory and subdirectories",
  "acceptable_commands": [
    "find . -name '*.py'",
    "find . -name '*.py' -type f",
    "find . -type f -name '*.py'"
  ],
  "validation_type": "output_match",
  "expected_output_contains": ["setup.py", "main.py", "utils/helper.py"],
  "expected_output_excludes": ["node_modules"],
  "dangerous": false
}
```

### 1.3 Evaluation Metrics

For each test case, run the model 5 times and measure:

1. **Format Compliance Rate:** Did the model output a parseable `<cmd>...</cmd>` block? (If it rambles or outputs multiple commands when one was expected, this fails.)

2. **Syntax Validity:** Does the command parse without bash syntax errors? (Run `bash -n` on it.)

3. **Execution Success:** Does the command run without error (exit code 0) in the sandbox?

4. **Functional Correctness:** Does the command's output match expected results? This is the hardest metric. Use a combination of:
   - Output contains expected substrings
   - Output does NOT contain excluded substrings
   - File system state changes match expectations (for mutation commands)
   - For commands with deterministic output, exact match

5. **Safety:** Did the model ever suggest something destructive without warning?

6. **Consistency (across 5 runs):** What percentage of the 5 runs produced the same (correct) answer? A model that gets it right 2/5 times is worse than one that gets it right 5/5.

### 1.4 Baseline Run Protocol

```
For each test case (100+):
    For each trial (5):
        1. Spin up fresh Docker sandbox
        2. Send system prompt + test prompt to base model
        3. Parse model output for <cmd> tags
        4. If command found:
           a. Execute in sandbox via docker exec
           b. Capture stdout, stderr, exit code
           c. Score against expected results
        5. Record all raw outputs and scores
    Aggregate per-case and per-category metrics
```

### 1.5 Decision Point

After Stage 1, you'll have a clear picture:

| Baseline Result | Action |
|---|---|
| >80% functional correctness overall | Model is already good. Fine-tuning may not be worth it. Pivot to a harder task or focus on edge cases only. |
| 40-80% correctness | Sweet spot. Clear room for improvement, and fine-tuning should show measurable gains. Proceed to Stage 2. |
| <40% correctness | Model may be too small for this task, OR the test cases are too hard. Analyze failure modes before proceeding. Consider whether 8B would be a better target. |

Also look at WHERE it fails. If it's great at simple file ops but terrible at compound tasks, you can focus fine-tuning data on the weak spots.

---

## Stage 2: Gold Standard Generation

**Goal:** Generate verified correct answers for every test case using a strong model (Claude via Claude Code, or the Anthropic API), creating the ground truth dataset AND the seed for training data.

### 2.1 Generating Gold Answers

For each of the 100+ test cases:

1. Send the same system prompt + test prompt to Claude (via API or Claude Code)
2. Parse the suggested command
3. Execute it in the Docker sandbox
4. Verify it produces the expected output
5. If it fails, iterate with Claude or manually correct
6. Store the verified (prompt, correct_command, expected_output) triple

### 2.2 Expanding to Training Data

The 100 test cases are your eval set. You need separate training data. Use Claude to generate 1,500-2,000 additional examples:

- Same categories as the test set, but different specific tasks
- Vary the phrasing: casual ("yo whats eating my disk space"), technical ("display inode usage for mounted filesystems"), terse ("python files, last week, modified")
- Include multi-turn examples where the model sees the output of a command and needs to suggest a follow-up
- Include "should NOT run a command" examples (questions about what a flag does, requests for explanation)

### 2.3 Data Format for MLX

Convert to the chat format MLX expects:

```jsonl
{"messages": [{"role": "system", "content": "You are a terminal assistant..."}, {"role": "user", "content": "Find all Python files modified in the last week"}, {"role": "assistant", "content": "<cmd>find . -name '*.py' -mtime -7</cmd>"}]}
```

Split: 80% train.jsonl, 10% valid.jsonl, 10% test.jsonl (keep the original 100+ cases as a separate held-out eval set that never appears in training).

---

## Stage 3: Fine-Tuning

**Goal:** QLoRA fine-tune the base model on the curated terminal assistant dataset.

### 3.1 Training Configuration

Starting point for 16GB MacBook Pro with MLX:

```yaml
# lora_config.yaml
model: "mlx-community/Qwen3-4B-4bit"  # or Qwen3.5-4B equivalent
fine_tune_type: "lora"
batch_size: 1                    # Conservative for 16GB
num_layers: 8                    # Start with 8, try 16 if memory allows
lora_parameters:
  rank: 8                        # Start here, try 16 if results are mediocre
  scale: 20.0
  dropout: 0.0
iters: 1000                      # With ~1500 training examples
steps_per_eval: 50
learning_rate: 1e-5
grad_checkpoint: true            # Helps with memory
mask_prompt: true                # Only compute loss on the assistant response
```

### 3.2 Training Command

```bash
python -m mlx_lm.lora \
  --model mlx-community/Qwen3-4B-4bit \
  --train \
  --data ./data \
  --config ./training/lora_config.yaml \
  --adapter-path ./adapters
```

### 3.3 Training Monitoring

Watch for:
- Validation loss should decrease and then plateau
- If val loss increases while train loss decreases = overfitting (reduce iters, increase dropout, reduce rank)
- Training should take 1-3 hours on M-series with 16GB

### 3.4 Hyperparameter Experiments

If time permits, try a few variations:

| Experiment | Change | Why |
|---|---|---|
| Higher rank | rank: 16 | More capacity for the adapter |
| More layers | num_layers: 16 | Adapts more of the model |
| DPO/DoRA | fine_tune_type: "dora" | Sometimes outperforms basic LoRA |
| More data | 3000+ examples | See if more data helps |

### 3.5 Optional: Cloud Training for Comparison

If you want to also try the 8B model or full fine-tuning:
- RunPod or Lambda Labs: ~$1-2/hr for an A100
- Estimated cost for a 4B QLoRA run: < $5
- Estimated cost for an 8B QLoRA run: < $15
- Use Unsloth or HuggingFace PEFT on the cloud, compare results to MLX local

---

## Stage 4: Post-Fine-Tuning Evaluation

**Goal:** Run the exact same evaluation from Stage 1 on the fine-tuned model. Compare head-to-head.

### 4.1 Inference with Adapter

```bash
python -m mlx_lm.generate \
  --model mlx-community/Qwen3-4B-4bit \
  --adapter-path ./adapters \
  --prompt "List all Python files modified in the last week" \
  --max-tokens 256
```

### 4.2 Evaluation Protocol

Identical to Stage 1:
- Same 100+ test cases
- Same 5 runs per case
- Same Docker sandbox with same file tree
- Same scoring metrics

### 4.3 Comparison Report

Generate a comparison showing:

```
                          Base Model    Fine-Tuned    Delta
Format Compliance         XX%           XX%           +X%
Syntax Validity           XX%           XX%           +X%
Execution Success         XX%           XX%           +X%
Functional Correctness    XX%           XX%           +X%
Consistency (5/5 agree)   XX%           XX%           +X%

By Category:
  File Operations         XX%           XX%           +X%
  Text Processing         XX%           XX%           +X%
  System Info             XX%           XX%           +X%
  Process Management      XX%           XX%           +X%
  Git Operations          XX%           XX%           +X%
  Compound Tasks          XX%           XX%           +X%
  Edge Cases              XX%           XX%           +X%
```

### 4.4 Failure Analysis

For cases where fine-tuning didn't help (or made things worse):
- Categorize failure modes
- Check for regressions (things the base model got right but fine-tuned got wrong)
- Identify whether more data in specific categories would help

---

## Stage 5: Integration Demo (Optional but High Impact)

**Goal:** Wire the fine-tuned model into a working CLI tool that you can actually use.

### 5.1 CLI Wrapper

Build a simple Python CLI tool that:
1. Takes natural language input from the user
2. Sends it to the local MLX model (with adapter)
3. Parses the `<cmd>` output
4. Shows the proposed command and asks for confirmation
5. Executes it
6. Shows the output
7. Optionally follows up ("want me to do anything with these results?")

### 5.2 Integration with Live SWE Agent

Since your team already has Live SWE Agent as a minimal bash-access scaffold:
- Swap in the fine-tuned small model as the backend
- Compare its performance to using a large API model
- Document the latency/quality tradeoff

### 5.3 Demo and Write-up

For the GitHub repo README and potential blog post:
- Before/after comparison charts
- Example interactions showing improvement
- Latency numbers (tokens/sec on M-series)
- Total training cost and time
- Lessons learned

---

## Timeline Estimate

| Stage | Time | Notes |
|---|---|---|
| Stage 0: Setup | 2-3 hours | Docker sandbox + model serving |
| Stage 1: Baseline | 4-6 hours | Writing test cases (2-3 hrs) + running eval (1-2 hrs) |
| Decision point | 30 min | Analyze results, decide whether to proceed |
| Stage 2: Gold data | 4-6 hours | Generating + validating training data |
| Stage 3: Training | 2-4 hours | Including hyperparameter experiments |
| Stage 4: Eval | 2-3 hours | Mostly automated, reusing Stage 1 infra |
| Stage 5: Demo | 3-4 hours | CLI wrapper + write-up |
| **Total** | **~2-3 weekends** | Can be compressed if you skip Stage 5 |

---

## Key Risks and Mitigations

**Risk: Base model is already good enough.**
Mitigation: This is actually a fine outcome! You've still built an eval framework and gained experience. Pivot the story to "I evaluated whether fine-tuning was necessary and determined the base model was sufficient" which is an equally valid engineering conclusion.

**Risk: 4B model is fundamentally too small.**
Mitigation: If Stage 1 shows <30% accuracy, try the Qwen3-4B in thinking mode (which uses more inference compute but might improve reasoning). If still bad, pivot to 8B and use cloud training.

**Risk: Fine-tuning overfits to training data style.**
Mitigation: Diverse phrasing in training data. Separate held-out eval set that never appears in training. Watch val loss carefully.

**Risk: Model generates dangerous commands.**
Mitigation: Docker sandbox for all eval. The CLI wrapper (Stage 5) always asks for confirmation before executing. Add explicit "refuse dangerous operations" examples to training data.

---

## What This Project Demonstrates

For interviews and your portfolio, this project shows:
1. **Evaluation-first mindset** (you measured before optimizing)
2. **Hands-on LoRA/QLoRA fine-tuning** with MLX on Apple Silicon
3. **Dataset curation** for a specific task
4. **Quantitative comparison** with clear metrics
5. **Systems thinking** (Docker sandbox, safety, reproducibility)
6. **Practical application** tied to real SWE tooling

For Apple specifically: you trained an on-device model using their framework for a task directly relevant to developer tools.
