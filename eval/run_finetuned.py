"""Stage 4: Run evaluation on fine-tuned model. Reuses baseline infrastructure."""

import sys
sys.argv += ["--adapter", "adapters", "--output", "eval/results_finetuned.json"]

# Simply re-run the baseline eval with the adapter flag
from run_baseline import main
main()
