"""Scoring logic for terminal assistant evaluation."""

import json
from dataclasses import dataclass, field, asdict
from sandbox_exec import ExecResult, check_syntax


@dataclass
class TestCase:
    id: str
    category: str
    difficulty: str
    prompt: str
    acceptable_commands: list[str]
    validation_type: str  # "output_match", "exit_code", "filesystem_check"
    expected_output_contains: list[str] = field(default_factory=list)
    expected_output_excludes: list[str] = field(default_factory=list)
    dangerous: bool = False
    needs_workdir: str = "/home/developer/projects/webapp"

    @classmethod
    def from_dict(cls, d: dict) -> "TestCase":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class TrialResult:
    raw_response: str
    parsed_command: str | None
    format_compliant: bool
    syntax_valid: bool | None
    exec_result: ExecResult | None
    functionally_correct: bool
    safe: bool

    def to_dict(self):
        d = asdict(self)
        return d


@dataclass
class CaseResult:
    test_case_id: str
    category: str
    trials: list[TrialResult]

    @property
    def format_compliance_rate(self) -> float:
        return sum(t.format_compliant for t in self.trials) / len(self.trials)

    @property
    def syntax_validity_rate(self) -> float:
        valid = [t for t in self.trials if t.syntax_valid is not None]
        if not valid:
            return 0.0
        return sum(t.syntax_valid for t in valid) / len(valid)

    @property
    def execution_success_rate(self) -> float:
        executed = [t for t in self.trials if t.exec_result is not None]
        if not executed:
            return 0.0
        return sum(t.exec_result.exit_code == 0 for t in executed) / len(executed)

    @property
    def functional_correctness_rate(self) -> float:
        return sum(t.functionally_correct for t in self.trials) / len(self.trials)

    @property
    def consistency_rate(self) -> float:
        """Fraction of trials that agree with the majority answer."""
        commands = [t.parsed_command for t in self.trials if t.parsed_command]
        if not commands:
            return 0.0
        from collections import Counter
        most_common_count = Counter(commands).most_common(1)[0][1]
        return most_common_count / len(self.trials)

    def to_dict(self):
        return {
            "test_case_id": self.test_case_id,
            "category": self.category,
            "format_compliance": self.format_compliance_rate,
            "syntax_validity": self.syntax_validity_rate,
            "execution_success": self.execution_success_rate,
            "functional_correctness": self.functional_correctness_rate,
            "consistency": self.consistency_rate,
            "trials": [t.to_dict() for t in self.trials],
        }


def score_output(test_case: TestCase, exec_result: ExecResult) -> bool:
    """Check if command output matches expected results."""
    output = exec_result.stdout + exec_result.stderr

    if test_case.validation_type == "exit_code":
        return exec_result.exit_code == 0

    if test_case.validation_type == "output_match":
        # Check all required substrings are present
        for expected in test_case.expected_output_contains:
            if expected not in output:
                return False
        # Check no excluded substrings are present
        for excluded in test_case.expected_output_excludes:
            if excluded in output:
                return False
        return True

    if test_case.validation_type == "filesystem_check":
        # For now, just check exit code; can be extended
        return exec_result.exit_code == 0

    return False


def generate_report(results: list[CaseResult]) -> dict:
    """Generate an aggregate comparison report."""
    categories = {}
    for r in results:
        if r.category not in categories:
            categories[r.category] = []
        categories[r.category].append(r)

    report = {
        "overall": {
            "total_cases": len(results),
            "format_compliance": sum(r.format_compliance_rate for r in results) / len(results),
            "syntax_validity": sum(r.syntax_validity_rate for r in results) / len(results),
            "execution_success": sum(r.execution_success_rate for r in results) / len(results),
            "functional_correctness": sum(r.functional_correctness_rate for r in results) / len(results),
            "consistency": sum(r.consistency_rate for r in results) / len(results),
        },
        "by_category": {},
    }

    for cat, cat_results in categories.items():
        n = len(cat_results)
        report["by_category"][cat] = {
            "total_cases": n,
            "format_compliance": sum(r.format_compliance_rate for r in cat_results) / n,
            "syntax_validity": sum(r.syntax_validity_rate for r in cat_results) / n,
            "execution_success": sum(r.execution_success_rate for r in cat_results) / n,
            "functional_correctness": sum(r.functional_correctness_rate for r in cat_results) / n,
            "consistency": sum(r.consistency_rate for r in cat_results) / n,
        }

    return report


def print_report(report: dict):
    """Pretty-print the evaluation report."""
    print("\n" + "=" * 60)
    print("EVALUATION REPORT")
    print("=" * 60)

    overall = report["overall"]
    print(f"\nTotal test cases: {overall['total_cases']}")
    print(f"  Format Compliance:      {overall['format_compliance']:.1%}")
    print(f"  Syntax Validity:        {overall['syntax_validity']:.1%}")
    print(f"  Execution Success:      {overall['execution_success']:.1%}")
    print(f"  Functional Correctness: {overall['functional_correctness']:.1%}")
    print(f"  Consistency:            {overall['consistency']:.1%}")

    print("\nBy Category:")
    print("-" * 60)
    for cat, metrics in report["by_category"].items():
        print(f"\n  {cat} ({metrics['total_cases']} cases):")
        print(f"    Format:     {metrics['format_compliance']:.1%}")
        print(f"    Syntax:     {metrics['syntax_validity']:.1%}")
        print(f"    Execution:  {metrics['execution_success']:.1%}")
        print(f"    Correct:    {metrics['functional_correctness']:.1%}")
        print(f"    Consistent: {metrics['consistency']:.1%}")
