# Teaching a 2B Model to Use the Terminal: LoRA Fine-Tuning on Apple Silicon (and the GPU Bug That Almost Stopped Us)

I wanted to build a terminal assistant that runs entirely on my laptop. No API calls, no cloud GPUs, no subscriptions -- just a tiny language model that lives on-device and knows how to write bash commands. The idea was simple: take a frontier model's knowledge of the command line, distill it into something small enough to run on a MacBook Pro M2 Pro with 16GB of RAM, and see how far I could push it.

This is the story of how that went, including the part where macOS's new frosted glass UI effects were secretly killing my training runs.

## Why Fine-Tune at All?

Off-the-shelf small models are surprisingly bad at CLI tasks. They hallucinate flags, forget how pipes work, and confidently suggest commands that would nuke your filesystem. Larger models handle this fine, but I didn't want to pay per-token forever for something that should be a local tool. The bet was that a 2B parameter model has enough capacity to be a competent terminal assistant -- it just needs better training data than what it shipped with.

## Picking the Base Model

I evaluated three candidates from the Qwen family, running each through my eval suite before writing a single line of training code:

| Model | Functional Correctness | Notes |
|---|---|---|
| Qwen3-1.7B | 42.2% | Got stuck in `<think>` loops, rarely produced actual commands |
| Qwen3.5-2B | 57.8% | Good format compliance, but wrong commands |
| Qwen3.5-4B | 84.4% | Already strong -- not much room to demonstrate improvement |

The 2B model was the sweet spot. It understood the output format but consistently failed on the hard stuff: compound tasks (42.9%), git operations (42.9%), and safety refusals (60%). Clear weaknesses with clear room to improve.

## Eval First, Train Second

Before touching any training code, I built the evaluation environment. This is the part most fine-tuning blog posts skip, and it's the part that matters most.

The setup: a Docker container running Ubuntu 22.04 with a pre-populated developer workspace. We're talking a realistic project directory -- a Python/JS webapp with git history, multiple branches, log files, the works. Network isolation so nothing phones home. A fresh container spins up for every single trial so tests can't contaminate each other.

65 test cases across 7 categories (file operations, text processing, git, system info, compound tasks, safety, search/navigation), each run 3 times for consistency. That's 195 total trials per evaluation. Automated scoring checks format compliance, syntax validity, execution success, functional correctness, and whether the model gives the same answer when asked the same question three times.

This eval-first approach meant every training decision was data-driven. When a later training run regressed, I didn't have to guess -- the numbers told me exactly what broke and by how much.

## Knowledge Distillation: Stealing From Claude

The training data came from Claude Sonnet. I spawned parallel Sonnet agents, each responsible for generating expert-quality examples in a specific category -- git workflows, file manipulation, text processing with `awk`/`sed`, safety refusals for dangerous commands, multi-step compound tasks.

The output format matched Qwen3.5's native tool-calling schema: a `<think>` block for chain-of-thought reasoning, followed by a `<tool_call>` XML block with the actual command. This matters -- if you train on a format the model already understands, you're not wasting capacity teaching it new syntax.

Total yield: 3,113 unique training examples.

Here's the part I'm proud of: I verified all 1,224 initial commands by actually executing them in the Docker sandbox. 57.5% passed execution. But when I dug into the failures, every single one was a sandbox limitation -- missing files the command referenced, tools not installed in the container. Zero syntax errors across the entire dataset. The distillation produced bash that was syntactically perfect; it just referenced things that didn't exist in the test environment.

## Training: QLoRA on Apple Silicon

Training used QLoRA through MLX, Apple's native ML framework for Apple Silicon. The setup:

- 4-bit quantized base model (~1.2GB in memory)
- LoRA rank 8, targeting 8 layers
- Dropout: 0.05
- Learning rate: 1e-5 with cosine decay and warmup
- ~2000 iterations

The whole thing trained in 28 minutes on the M2 Pro. No cloud. No CUDA. Just the laptop's integrated GPU.

**Results:**

| Metric | Base (57.8%) | Fine-Tuned | Delta |
|---|---|---|---|
| Functional Correctness | 57.8% | 82.2% | +24.4 |
| Format Compliance | 93.3% | 100% | +6.7 |
| Execution Success | 66.7% | 84.4% | +17.7 |
| Consistency | 77.8% | 93.3% | +15.5 |

The fine-tuned 2B model beat the 4B base model on execution success (84.4% vs 80.0%) and consistency (93.3% vs 84.4%). An 11MB LoRA adapter made a 2B model competitive with a model twice its size.

## The GPU Bug

Now for the part that cost me a full day.

Training kept crashing somewhere between iteration 100 and 400 with this:

```
[METAL] Command buffer execution failed: Impacting Interactivity
(0000000e:kIOGPUCommandBufferCallbackErrorImpactingInteractivity)
```

My first assumption was out of memory. It's always out of memory, right? Except peak GPU usage was 2.75GB out of 16GB available. Seventeen percent utilization. My config was more conservative than mlx-lm's defaults on every single parameter -- smaller batch size, fewer layers, shorter sequences.

### Going Down the Wrong Path

I tried everything the internet suggested:

```bash
export MLX_MAX_OPS_PER_BUFFER=1    # Reduced crash frequency, didn't fix
export MLX_MAX_MB_PER_BUFFER=10    # Still crashed
```

Reduced LoRA layers. Shortened sequence length. Nothing worked reliably.

I ended up writing an auto-resume script that detected crashes and restarted training from the last checkpoint. It technically worked -- training completed -- but 80 minutes of actual compute took 6 hours because of the crash-resume overhead. This is not a solution. This is duct tape.

### Reading the Error Message (Novel Concept)

The error didn't say "timeout." It didn't say "out of memory." It said **"Impacting Interactivity."** That's a specific term. It means: "your GPU work is blocking the UI compositor."

Apple Silicon has a unified GPU. The same hardware that runs your ML training also composites every window on your screen. macOS has a watchdog that kills GPU tasks if they block the display pipeline for too long. And here's the kicker: macOS Tahoe shipped with "Liquid Glass" -- a system-wide frosted glass effect that requires real-time blur and refraction compositing. WindowServer's GPU load went up significantly.

My hypothesis: if there's no display to composite, the watchdog has nothing to protect.

### The Experiment

Three controlled runs, same model, same config:

| Test | Condition | Result |
|---|---|---|
| A | Lid open, buffer env vars set | CRASHED at iteration 10 |
| B | Lid closed + `caffeinate -s`, buffer env vars set | PASSED 500 iterations |
| C | Lid closed + `caffeinate -s`, NO buffer env vars | PASSED 500 iterations |

The closed lid was the entire fix. Not the environment variables, not the config changes. Closing the laptop lid shut down WindowServer compositing, and the watchdog had nothing to defend.

I then built a minimal reproduction: same model, same LoRA config, but with short training examples (~40 tokens each). No crash. Switched to longer examples (~256 tokens, matching real training data). Crashed every time. The variable was per-iteration GPU time -- longer sequences meant longer individual Metal operations that blocked the compositor past the watchdog threshold.

### Filing the Bug

I opened an issue on [ml-explore/mlx (#3267)](https://github.com/ml-explore/mlx/issues/3267) with:

- 5 runs across macOS 26.2 and 26.3.1
- Controlled variables isolating display state vs. background activity
- Proof that my config was more conservative than library defaults
- The `caffeinate` workaround
- The minimal repro showing the sequence-length threshold

### The Fix (23 Minutes Later)

An MLX collaborator replied with a single environment variable:

```bash
export AGX_RELAX_CDM_CTXSTORE_TIMEOUT=1
```

That's it. One line. It tells the AGX (Apple GPU) driver to relax the context store timeout -- the exact watchdog that was killing my training runs.

This environment variable does not appear anywhere on the public internet. It's not in any documentation. It's not in any Stack Overflow answer. Only someone with access to the Apple GPU driver source code would know it exists.

I tested it immediately: full 2000-iteration training run, lid open, zero crashes, 28 minutes flat. Problem completely solved.

The lesson here isn't "close your laptop lid." The lesson is that a well-documented bug report with controlled experiments and a minimal reproduction gets you answers that no amount of googling ever will.

## The Overfitting Detour

With the GPU bug solved, I made the classic mistake: training longer.

With 1,240 examples:
- **1000 iterations (0.8 epochs): 82.2%** -- best result
- **4000 iterations (3.2 epochs): 73.3%** -- worse, the model had memorized the training data

The tell: train loss at the start of the extended run was 0.037. The model already had the dataset memorized before the extra training even began. More iterations just reinforced the memorization.

The fix was straightforward: add dropout (0.05), use cosine learning rate decay, and -- most importantly -- double the training data to 2,177 unique examples. With the larger dataset, the model could train for 2000 iterations without memorizing, maintaining a healthy train loss of 0.439. The eval numbers confirmed it was learning generalizable patterns, not regurgitating training examples.

## Where Things Stand

After removing the conflicting safety data and retraining (v7), the command-only model hit our best numbers:

- **88.9% correctness on command tasks** (up from 60% base) across 54 non-safety test cases
- **100% format compliance**, **98.5% consistency** -- nearly deterministic
- Runs entirely on a MacBook (~1.2GB quantized model + 11MB LoRA adapter)
- Generates responses in 2-5 seconds
- Trained in 26 minutes with $0 in compute costs

Of the 6 remaining command failures, at least 2 are eval strictness issues rather than real errors. For example, the model outputs `lsof -i :8080` to check a port -- which is the correct command -- but our Docker sandbox doesn't have lsof installed, so the eval marks it as a failure. Another case: the model uses `find | xargs grep` instead of `grep -r` to search files. Different approach, same result, scored as wrong.

The model still runs dangerous commands (`rm -rf *`, `chmod 777`) instead of refusing, but that's by design now -- safety via LoRA doesn't work (see below). Instead, we handle safety at the application layer with a command blacklist and approval workflow in the CLI.

This led to one more discovery.

## The Task Interference Problem

My first instinct was to fix safety by adding more refusal examples to the training data. I generated 375 examples where the model should refuse dangerous commands, mixed them in with the command examples, and retrained. The result: performance *dropped* from 81.5% to 73.8%.

This confused me until I dug into the research. It turns out this is a well-documented limitation of LoRA called **task interference**.

LoRA works by constraining all learned behavior into a low-rank subspace -- in our case, 8 dimensions. Command examples push the adapter in one direction: "when the user asks something, generate a tool call." Safety examples push in the opposite direction: "when the user asks something dangerous, do NOT generate a tool call." In an 8-dimensional space, these opposing forces cancel each other out. The adapter doesn't have enough room to represent both "do the thing" and "don't do the thing" simultaneously.

Three recent papers confirm this:
- *"Disentangling Task Conflicts in Multi-Task LoRA"* (arXiv 2601.09684) found that LoRA's low-rank constraint causes "destructive interference" between conflicting task gradients
- *"LoRI"* (arXiv 2504.07448) showed that LoRA "projects features into the same dense low-dimensional space, leading to task interference"
- *"MTL-LoRA"* (arXiv 2410.09437) proposed task-specific LoRA branches precisely because a single shared low-rank space can't separate contradictory objectives

Think of it like an 8-lane highway. You can move a lot of traffic in one direction. But if half the cars are going north and half are going south on the same lanes, nobody moves. Full fine-tuning has thousands of lanes and can build separate roads. Rank-8 LoRA doesn't have that luxury.

The fix was counterintuitive: **remove all safety examples from LoRA training entirely**. Let the base model's pre-trained safety behavior handle refusals. Only train the adapter on what we want it to *learn*: generating correct bash commands. The adapter isn't the right place for teaching restraint -- that's a job for the base weights (via DPO or RLHF, in a future iteration).

Interestingly, this also means our adapter has enormous headroom. At rank 8, the estimated saturation point is ~40,000 examples. We're using 2,072. The adapter is at about 5% capacity -- the bottleneck was never data volume, it was data coherence.

So I built a CLI around it. The tool has three safety layers: a hardcoded blacklist that blocks commands like `rm -rf /` and fork bombs before they reach the shell, an approval-required list for risky commands like `sudo` and `git push`, and a default approval mode where every command needs a `y/n` before executing. You can run it locally or in a Docker sandbox.

## Using It For Real

Eval numbers are one thing. Actually using it is another. Here's what a real session looks like:

```
you> how many lines of code are in each python file
  $ find . -name "*.py" | xargs wc -l
   11 ./tests/test_utils.py
    7 ./tests/test_api.py
   34 ./src/utils.py
   30 ./src/api.py
   14 ./src/main.py
   96 total

you> find all files that contain the word flask
  $ grep -rl "flask" . --include="*"
  ./requirements.txt

you> show disk usage by directory sorted biggest first
  $ du -sh */ | sort -rh
  20K   src/
  12K   tests/
  12K   scripts/
  8.0K  docs/
  8.0K  config/

you> create a file called notes.txt with the text hello world
  $ echo "hello world" > notes.txt
```

Out of 15 test prompts, it nailed 10, had minor flag issues on 3 (wrong `find -size` syntax, incomplete branch name, counting files instead of lines), and gave two amusing attempts at impossible tasks (compressing a directory that doesn't exist, checking the weather without internet).

It feels like a competent but junior developer. Gets the common stuff right, occasionally fumbles on specific flags. The errors aren't dangerous -- they're the kind of thing you catch at the approval prompt and rephrase. For a 2B model running at 33 tokens/sec on a laptop, that's a pretty good assistant.

## What Did Fine-Tuning Actually Do?

Looking at the before/after side by side, the answer is surprisingly clear: the base 2B model already knew bash. It just didn't know when to use it.

Fine-tuning fixed 18 test cases the base model got wrong. The failures fell into three buckets:

**The model refused to act (6 cases).** Prompts like "what OS is this?", "search for TODOs", "stash current changes" — the base model would respond with text instead of generating a command. It knew `uname -a` and `grep -r TODO` and `git stash`, but it didn't realize it was supposed to call the tool. LoRA taught it to always reach for the terminal.

**The model picked the wrong command (7 cases).** `git status` when it should be `git ls-files`. `du -sh /home/*` instead of `find -size +1M`. `head` when the prompt said "last 10 lines" (that's `tail`). The knowledge was there but the selection was off. LoRA sharpened the mapping from intent to command.

**The model couldn't build pipelines (5 cases).** The base model would `find / -type f -exec md5sum {} \;` (scanning the entire filesystem) or use placeholder paths like `cd /path/to/your/repo`. It didn't have the muscle memory for `find . | xargs | sort | head` patterns. LoRA drilled these in.

The flip side: fine-tuning "broke" 7 edge cases where the base model correctly said nothing. The fine-tuned model tries to run a command for everything — including "explain what a pipe does" and dangerous requests like `rm -rf ~`. That's the trade-off of training a focused command generator. Safety moves to the application layer (blacklists, approval prompts) instead of living in the model weights.

This confirms that LoRA isn't teaching the model new knowledge — it's surfacing and directing capabilities that were already latent in the base weights. The 2B model always had bash in it. We just gave it a reason to use it.

Could we go smaller? We tested a 1.7B model (Qwen3-1.7B) and it scored 42% with only 51% format compliance. It got trapped in reasoning loops and never output commands. The 2B appears to be the floor for this task — below that, there isn't enough base capability for LoRA to unlock.

## What I Took Away From This

**Eval first, always.** I built the Docker sandbox and 65-test evaluation suite before writing a single line of training code. When training v4 regressed from 82% to 73%, the eval caught it immediately and told me exactly which categories degraded. Without that, I'd be flying blind and shipping a worse model thinking it was better.

**LoRA surfaces, it doesn't create.** The base model already knew the commands. Fine-tuning taught it when and how to use them. This has implications for model selection: pick a base model that has the knowledge you need, then use LoRA to direct it. Don't expect LoRA to teach fundamentally new skills.

**Contradictory training signals destroy LoRA performance.** Adding safety/refusal examples alongside command examples degraded results from 81.5% to 73.8%. This is a documented limitation — low-rank adapters can't represent "do the thing" and "don't do the thing" in the same narrow subspace. Handle safety at the application layer instead.

**The bug is never where you think it is.** I spent hours assuming the GPU crash was a memory issue. It was a display compositor fight between ML training and Liquid Glass window effects. The error message told me exactly what was wrong — "Impacting Interactivity" — but I had to stop assuming and start reading.

**File the bug report.** A detailed issue with controlled experiments and a minimal repro got me an immediate response with a fix that wasn't documented anywhere on the public internet. The person who answered had driver-level knowledge I could never have found on my own. Open source works, but only if you put in the effort to write a good report.

**Knowledge distillation is a real strategy.** Frontier model generates training data, tiny model learns from it, tiny model runs on-device forever with zero marginal cost. The economics of this are compelling for any task where you can define quality clearly enough to generate good examples at scale.

---

*All code, eval infrastructure, training configs, and the CLI tool are on [GitHub](https://github.com/zackedds/mini-agent). The MLX GPU bug is tracked at [ml-explore/mlx#3267](https://github.com/ml-explore/mlx/issues/3267).*
