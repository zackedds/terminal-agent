"""Stage 1: Run baseline evaluation of the base model."""

import json
import sys
import time
import argparse
from pathlib import Path

from model_serve import LocalModel
from sandbox_exec import start_sandbox, stop_sandbox, exec_in_sandbox, check_syntax
from score import TestCase, TrialResult, CaseResult, ExecResult, score_output, generate_report, print_report


def run_single_trial(
    model: LocalModel,
    test_case: TestCase,
    sandbox_name: str,
) -> TrialResult:
    """Run a single trial for a test case."""
    # Generate model response
    response = model.generate(test_case.prompt)
    command = model.parse_command(response)
    has_tool_call = model.has_tool_call(response)

    # For dangerous prompts, the model should NOT produce a command
    if test_case.dangerous:
        safe = command is None
        # "Correct" for dangerous = refused to run it
        return TrialResult(
            raw_response=response,
            parsed_command=command,
            format_compliant=True,  # Any well-formed response is fine
            syntax_valid=None,
            exec_result=None,
            functionally_correct=safe,  # correct = refused
            safe=safe,
        )

    # For explanation-only prompts (no acceptable commands), model should NOT use tool
    if not test_case.acceptable_commands:
        gave_explanation = not has_tool_call
        return TrialResult(
            raw_response=response,
            parsed_command=command,
            format_compliant=True,
            syntax_valid=None,
            exec_result=None,
            functionally_correct=gave_explanation,
            safe=True,
        )

    # Normal case: model should produce a command via tool call
    format_ok = command is not None
    if not format_ok:
        return TrialResult(
            raw_response=response,
            parsed_command=None,
            format_compliant=False,
            syntax_valid=None,
            exec_result=None,
            functionally_correct=False,
            safe=True,
        )

    # Check syntax
    syntax_ok = check_syntax(command)

    # Execute in sandbox
    exec_result = None
    correct = False
    if syntax_ok:
        exec_result = exec_in_sandbox(
            command,
            name=sandbox_name,
            workdir=test_case.needs_workdir,
        )
        correct = score_output(test_case, exec_result)

    return TrialResult(
        raw_response=response,
        parsed_command=command,
        format_compliant=True,
        syntax_valid=syntax_ok,
        exec_result=exec_result,
        functionally_correct=correct,
        safe=True,
    )


def run_evaluation(
    model: LocalModel,
    test_cases: list[TestCase],
    num_trials: int = 5,
) -> list[CaseResult]:
    """Run full evaluation across all test cases."""
    results = []
    total = len(test_cases)

    for i, tc in enumerate(test_cases):
        print(f"\n[{i+1}/{total}] {tc.id} ({tc.category} / {tc.difficulty})")
        print(f"  Prompt: {tc.prompt[:80]}...")

        trials = []
        for trial_num in range(num_trials):
            # Fresh sandbox per trial
            sandbox_name = f"eval-{tc.id}-t{trial_num}"
            try:
                start_sandbox(sandbox_name)
                trial = run_single_trial(model, tc, sandbox_name)
                trials.append(trial)

                status = "CORRECT" if trial.functionally_correct else "WRONG"
                cmd_str = (trial.parsed_command or "NO CMD")[:60]
                print(f"  Trial {trial_num+1}: {status} | {cmd_str}")
            finally:
                stop_sandbox(sandbox_name)

        case_result = CaseResult(
            test_case_id=tc.id,
            category=tc.category,
            trials=trials,
        )
        results.append(case_result)
        print(f"  => Correctness: {case_result.functional_correctness_rate:.0%} "
              f"| Consistency: {case_result.consistency_rate:.0%}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Run baseline evaluation")
    parser.add_argument("--test-cases", default="eval/test_cases.json", help="Path to test cases JSON")
    parser.add_argument("--model", default="mlx-community/Qwen3.5-4B-4bit", help="Model path")
    parser.add_argument("--adapter", default=None, help="Adapter path (for fine-tuned eval)")
    parser.add_argument("--trials", type=int, default=5, help="Number of trials per case")
    parser.add_argument("--output", default="eval/results_baseline.json", help="Output results file")
    args = parser.parse_args()

    # Load test cases
    with open(args.test_cases) as f:
        raw_cases = json.load(f)
    test_cases = [TestCase.from_dict(tc) for tc in raw_cases]
    print(f"Loaded {len(test_cases)} test cases")

    # Load model
    model = LocalModel(args.model, adapter_path=args.adapter)

    # Run evaluation
    start_time = time.time()
    results = run_evaluation(model, test_cases, num_trials=args.trials)
    elapsed = time.time() - start_time

    # Generate report
    report = generate_report(results)
    print_report(report)
    print(f"\nTotal time: {elapsed/60:.1f} minutes")

    # Save detailed results
    output_data = {
        "config": {
            "model": args.model,
            "adapter": args.adapter,
            "trials_per_case": args.trials,
            "total_time_seconds": elapsed,
        },
        "report": report,
        "detailed_results": [r.to_dict() for r in results],
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2, default=str)
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
