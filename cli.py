#!/usr/bin/env python3
"""
Terminal Assistant CLI — a fine-tuned 2B model that generates and runs bash commands.

Usage:
    python cli.py                     # Interactive mode, approve each command
    python cli.py --auto              # Auto-execute (skip approval)
    python cli.py --dev               # Dev mode: show model internals
    python cli.py --sandbox           # Run commands in Docker sandbox
    python cli.py "find all python files"  # One-shot mode
"""

import argparse
import os
import re
import sys
import time
import subprocess
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Command safety
# ---------------------------------------------------------------------------

BLACKLIST_PATTERNS = [
    # Destructive file operations
    r"\brm\s+(-[a-zA-Z]*)?.*\s+/\s*$",      # rm -rf /
    r"\brm\s+(-[a-zA-Z]*)?.*\s+/\*",          # rm -rf /*
    r"\brm\s+(-[a-zA-Z]*)?.*\s+~/?$",         # rm -rf ~
    r"\bmkfs\b",                                # format filesystem
    r"\bdd\b.*\bof=/dev/[sh]d",                # write to raw disk
    r">\s*/dev/[sh]d",                          # redirect to raw disk
    # Permission escalation
    r"\bchmod\s+777\s+.*(/|~)",                # chmod 777 on system dirs
    r"\bchmod\s+-R\s+777\b",                   # recursive chmod 777
    # System destruction
    r":\(\)\s*\{\s*:\|:\s*&\s*\}\s*;?\s*:",   # fork bomb
    r"\bkill\s+-9\s+(-1|1)\b",                # kill init / all processes
    # Dangerous overwrites
    r">\s*/etc/(passwd|shadow|sudoers)",        # overwrite auth files
    r"\bcurl\b.*\|\s*(sudo\s+)?bash",          # pipe untrusted script to bash
    r"\bwget\b.*\|\s*(sudo\s+)?bash",          # pipe untrusted script to bash
]

BLACKLIST_COMPILED = [re.compile(p) for p in BLACKLIST_PATTERNS]

REQUIRES_APPROVAL_PATTERNS = [
    r"\brm\s+-[a-zA-Z]*r",        # any recursive rm
    r"\brm\s+-[a-zA-Z]*f",        # any force rm
    r"\bsudo\b",                    # anything with sudo
    r"\bchmod\b",                   # permission changes
    r"\bchown\b",                   # ownership changes
    r"\bkill\b",                    # killing processes
    r"\breboot\b",                  # reboot
    r"\bshutdown\b",               # shutdown
    r"\bgit\s+push\b",            # pushing to remote
    r"\bgit\s+reset\s+--hard\b",  # destructive git
    r"\bdrop\s+database\b",       # SQL drops
    r"\bdrop\s+table\b",
]

REQUIRES_APPROVAL_COMPILED = [re.compile(p, re.IGNORECASE) for p in REQUIRES_APPROVAL_PATTERNS]


@dataclass
class SafetyCheck:
    allowed: bool
    blocked: bool
    needs_approval: bool
    reason: str


def check_command_safety(command: str) -> SafetyCheck:
    """Check a command against blacklist and approval-required patterns."""
    for pattern in BLACKLIST_COMPILED:
        if pattern.search(command):
            return SafetyCheck(
                allowed=False, blocked=True, needs_approval=False,
                reason=f"BLOCKED: matches dangerous pattern",
            )
    for pattern in REQUIRES_APPROVAL_COMPILED:
        if pattern.search(command):
            return SafetyCheck(
                allowed=True, blocked=False, needs_approval=True,
                reason=f"Requires approval: potentially destructive command",
            )
    return SafetyCheck(allowed=True, blocked=False, needs_approval=False, reason="")


# ---------------------------------------------------------------------------
# Model wrapper
# ---------------------------------------------------------------------------

class TerminalAssistant:
    def __init__(self, model_path: str, adapter_path: str | None = None, dev: bool = False):
        self.dev = dev
        self._log("Loading model...")
        import logging
        logging.disable(logging.WARNING)
        from mlx_lm import load
        import os, contextlib
        with contextlib.redirect_stderr(open(os.devnull, 'w')) if not dev else contextlib.nullcontext():
            self.model, self.tokenizer = load(model_path, adapter_path=adapter_path)
        logging.disable(logging.NOTSET)
        self._log("Model loaded.")

        self.system_prompt = (
            "You are a terminal assistant. Translate user requests into bash commands. "
            "If a request is ambiguous or potentially dangerous, explain why instead of "
            "running it. If the request is a question that doesn't need a command, answer "
            "it directly without calling any tools."
        )
        self.tools = [{
            "type": "function",
            "function": {
                "name": "run_bash",
                "description": "Execute a bash command in the terminal",
                "parameters": {
                    "type": "object",
                    "properties": {"command": {"type": "string", "description": "The bash command to run"}},
                    "required": ["command"],
                },
            },
        }]

    def _log(self, msg: str):
        if self.dev:
            print(f"\033[90m[dev] {msg}\033[0m", file=sys.stderr)

    def ask(self, user_prompt: str, max_tokens: int = 512, temperature: float = 0.0) -> dict:
        """Generate a response. Returns dict with 'command', 'explanation', 'raw', and timing info."""
        from mlx_lm import generate
        from mlx_lm.sample_utils import make_sampler

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        prompt = self.tokenizer.apply_chat_template(messages, tools=self.tools, add_generation_prompt=True)
        if isinstance(prompt, list):
            input_tokens = len(prompt)
            prompt = self.tokenizer.decode(prompt)
        else:
            input_tokens = len(self.tokenizer.encode(prompt))

        self._log(f"Input tokens: {input_tokens}")

        sampler = make_sampler(temp=temperature)
        t0 = time.time()
        response = generate(self.model, self.tokenizer, prompt=prompt, max_tokens=max_tokens, sampler=sampler)
        elapsed = time.time() - t0

        output_tokens = len(self.tokenizer.encode(response))
        tokens_per_sec = output_tokens / elapsed if elapsed > 0 else 0

        self._log(f"Output tokens: {output_tokens}")
        self._log(f"Generation time: {elapsed:.2f}s ({tokens_per_sec:.1f} tok/s)")
        self._log(f"Raw response:\n{response}")

        command = self._parse_command(response)
        explanation = self._get_explanation(response) if not command else None
        thinking = self._get_thinking(response)

        return {
            "command": command,
            "explanation": explanation,
            "thinking": thinking,
            "raw": response,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "elapsed_s": elapsed,
            "tokens_per_sec": tokens_per_sec,
        }

    @staticmethod
    def _parse_command(response: str) -> str | None:
        match = re.search(
            r"<function=run_bash>\s*<parameter=command>\s*(.*?)\s*</parameter>",
            response, re.DOTALL,
        )
        if match:
            return match.group(1).strip()
        match = re.search(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", response, re.DOTALL)
        if match:
            try:
                import json
                data = json.loads(match.group(1))
                if data.get("name") == "run_bash":
                    return data["arguments"]["command"].strip()
            except (json.JSONDecodeError, KeyError):
                pass
        return None

    @staticmethod
    def _get_explanation(response: str) -> str | None:
        if "<tool_call>" in response:
            return None
        text = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL)
        return text.strip() or None

    @staticmethod
    def _get_thinking(response: str) -> str | None:
        match = re.search(r"<think>(.*?)</think>", response, re.DOTALL)
        return match.group(1).strip() if match else None


# ---------------------------------------------------------------------------
# Command execution
# ---------------------------------------------------------------------------

def exec_local(command: str, timeout: int = 30) -> tuple[str, str, int]:
    """Execute a command locally."""
    try:
        result = subprocess.run(
            ["bash", "-c", command],
            capture_output=True, text=True, timeout=timeout,
        )
        return result.stdout, result.stderr, result.returncode
    except subprocess.TimeoutExpired:
        return "", f"Command timed out after {timeout}s", -1


def exec_sandbox(command: str, timeout: int = 30) -> tuple[str, str, int]:
    """Execute a command in the Docker sandbox."""
    try:
        result = subprocess.run(
            ["docker", "exec", "-w", "/home/developer/projects/webapp",
             "-u", "developer", "terminal-assistant-sandbox",
             "bash", "-c", command],
            capture_output=True, text=True, timeout=timeout,
        )
        return result.stdout, result.stderr, result.returncode
    except subprocess.TimeoutExpired:
        return "", f"Command timed out after {timeout}s", -1


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

def print_command(cmd: str):
    print(f"\033[1;36m$ {cmd}\033[0m")


def print_output(stdout: str, stderr: str, exit_code: int):
    if stdout:
        print(stdout, end="" if stdout.endswith("\n") else "\n")
    if stderr:
        print(f"\033[33m{stderr}\033[0m", end="" if stderr.endswith("\n") else "\n")
    if exit_code != 0:
        print(f"\033[31m(exit code {exit_code})\033[0m")
    print()  # blank line after output, before next prompt


def print_thinking(thinking: str):
    print(f"\033[90m{thinking}\033[0m")


def print_explanation(explanation: str):
    print(f"\n{explanation}")


def prompt_approval(command: str, reason: str = "") -> bool:
    """Ask user to approve a command."""
    if reason:
        print(f"\033[33m  {reason}\033[0m")
    try:
        answer = input("\033[1mExecute? [y/N] \033[0m").strip().lower()
        return answer in ("y", "yes")
    except (EOFError, KeyboardInterrupt):
        print()
        return False


def print_dev_stats(result: dict):
    print(f"\033[90m  {result['input_tokens']} in / {result['output_tokens']} out | "
          f"{result['elapsed_s']:.2f}s | {result['tokens_per_sec']:.1f} tok/s\033[0m")


class Spinner:
    """Animated spinner for model generation."""
    FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

    def __init__(self):
        self._thread = None
        self._stop = False

    def start(self):
        import threading
        self._stop = False
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()

    def _spin(self):
        import time
        i = 0
        while not self._stop:
            frame = self.FRAMES[i % len(self.FRAMES)]
            print(f"\r\033[36m{frame}\033[0m \033[90mthinking...\033[0m", end="", flush=True)
            time.sleep(0.08)
            i += 1
        print(f"\r{' ' * 20}\r", end="", flush=True)

    def stop(self):
        self._stop = True
        if self._thread:
            self._thread.join()


def print_banner(mode: str, auto: bool, dev: bool, model_name: str = "Qwen 3.5 2B"):
    tags = [mode]
    if auto: tags.append("auto")
    if dev: tags.append("dev")
    tag_str = " | ".join(tags)

    print()
    print(f"  \033[1;36m> terminal-agent\033[0m")
    print(f"  \033[90m{model_name} + LoRA | {tag_str}\033[0m")
    print(f"  \033[90mType a request. Ctrl+C to exit.\033[0m")
    print()


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run_interactive(assistant: TerminalAssistant, auto: bool = False, sandbox: bool = False, dev: bool = False):
    executor = exec_sandbox if sandbox else exec_local
    mode = "sandbox" if sandbox else "local"
    print_banner(mode, auto, dev)

    while True:
        try:
            user_input = input("\033[1;32m> \033[0m").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            break

        spinner = Spinner()
        spinner.start()
        result = assistant.ask(user_input)
        spinner.stop()

        if dev:
            print_dev_stats(result)

        if result["thinking"] and dev:
            print_thinking(result["thinking"])

        if result["explanation"]:
            print_explanation(result["explanation"])
            continue

        if not result["command"]:
            print("\033[33mModel didn't generate a command.\033[0m")
            if dev:
                print(f"\033[90mRaw: {result['raw'][:200]}\033[0m")
            continue

        command = result["command"]
        safety = check_command_safety(command)

        if safety.blocked:
            print_command(command)
            print(f"\033[1;31m  BLOCKED: This command matches a dangerous pattern and will not be executed.\033[0m")
            continue

        print_command(command)

        should_execute = False
        if auto and not safety.needs_approval:
            should_execute = True
        elif auto and safety.needs_approval:
            should_execute = prompt_approval(command, safety.reason)
        else:
            should_execute = prompt_approval(command)

        if should_execute:
            stdout, stderr, exit_code = executor(command)
            print_output(stdout, stderr, exit_code)
        else:
            print("\033[90m  Skipped.\033[0m")


def run_oneshot(assistant: TerminalAssistant, prompt: str, auto: bool = False, sandbox: bool = False, dev: bool = False):
    executor = exec_sandbox if sandbox else exec_local
    result = assistant.ask(prompt)

    if dev:
        print_dev_stats(result)

    if result["thinking"] and dev:
        print_thinking(result["thinking"])

    if result["explanation"]:
        print_explanation(result["explanation"])
        return

    if not result["command"]:
        print("Model didn't generate a command.")
        return

    command = result["command"]
    safety = check_command_safety(command)

    if safety.blocked:
        print_command(command)
        print(f"\033[1;31m  BLOCKED: Dangerous command.\033[0m")
        return

    print_command(command)

    if auto and not safety.needs_approval:
        stdout, stderr, exit_code = executor(command)
        print_output(stdout, stderr, exit_code)
    else:
        if prompt_approval(command, safety.reason if safety.needs_approval else ""):
            stdout, stderr, exit_code = executor(command)
            print_output(stdout, stderr, exit_code)
        else:
            print("\033[90m  Skipped.\033[0m")


def main():
    parser = argparse.ArgumentParser(description="Terminal Assistant CLI")
    parser.add_argument("prompt", nargs="?", help="One-shot prompt (omit for interactive mode)")
    parser.add_argument("--model", default="mlx-community/Qwen3.5-2B-OptiQ-4bit", help="Model path")
    parser.add_argument("--adapter", default="adapters", help="LoRA adapter path")
    parser.add_argument("--auto", action="store_true", help="Auto-execute commands (skip approval for safe commands)")
    parser.add_argument("--sandbox", action="store_true", help="Execute in Docker sandbox instead of locally")
    parser.add_argument("--dev", action="store_true", help="Dev mode: show tokens/sec, raw output, thinking")
    parser.add_argument("--no-adapter", action="store_true", help="Run base model without adapter")
    args = parser.parse_args()

    adapter = None if args.no_adapter else args.adapter
    assistant = TerminalAssistant(args.model, adapter_path=adapter, dev=args.dev)

    if args.prompt:
        run_oneshot(assistant, args.prompt, auto=args.auto, sandbox=args.sandbox, dev=args.dev)
    else:
        run_interactive(assistant, auto=args.auto, sandbox=args.sandbox, dev=args.dev)


if __name__ == "__main__":
    main()
