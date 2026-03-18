"""Execute commands in the Docker sandbox and capture results."""

import subprocess
import json
import time
from dataclasses import dataclass, asdict


@dataclass
class ExecResult:
    stdout: str
    stderr: str
    exit_code: int
    duration_ms: float

    def to_dict(self):
        return asdict(self)


CONTAINER_IMAGE = "sandbox-sandbox"
DOCKER_COMPOSE_DIR = "sandbox"


def start_sandbox(name: str = "eval-sandbox") -> str:
    """Start a fresh sandbox container. Returns container ID."""
    # Remove any existing container with the same name
    subprocess.run(
        ["docker", "rm", "-f", name],
        capture_output=True,
    )
    result = subprocess.run(
        [
            "docker", "run", "-d",
            "--name", name,
            "--network", "none",
            "--memory", "512m",
            "--cpus", "1.0",
            CONTAINER_IMAGE,
            "sleep", "3600",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Failed to start sandbox: {result.stderr}")
    return result.stdout.strip()


def stop_sandbox(name: str = "eval-sandbox"):
    """Stop and remove the sandbox container."""
    subprocess.run(["docker", "rm", "-f", name], capture_output=True)


def exec_in_sandbox(
    command: str,
    name: str = "eval-sandbox",
    timeout: int = 30,
    workdir: str = "/home/developer",
) -> ExecResult:
    """Execute a command in the sandbox and return results."""
    start = time.time()
    try:
        result = subprocess.run(
            [
                "docker", "exec",
                "-w", workdir,
                "-u", "developer",
                name,
                "bash", "-c", command,
            ],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        duration_ms = (time.time() - start) * 1000
        return ExecResult(
            stdout=result.stdout,
            stderr=result.stderr,
            exit_code=result.returncode,
            duration_ms=duration_ms,
        )
    except subprocess.TimeoutExpired:
        duration_ms = (time.time() - start) * 1000
        return ExecResult(
            stdout="",
            stderr=f"Command timed out after {timeout}s",
            exit_code=-1,
            duration_ms=duration_ms,
        )


def check_syntax(command: str) -> bool:
    """Check if a bash command has valid syntax."""
    result = subprocess.run(
        ["bash", "-n", "-c", command],
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


if __name__ == "__main__":
    # Quick self-test
    print("Starting sandbox...")
    start_sandbox()
    print("Running test command...")
    result = exec_in_sandbox("ls /home/developer/projects/webapp/src/")
    print(f"Exit code: {result.exit_code}")
    print(f"Output: {result.stdout}")
    print(f"Duration: {result.duration_ms:.0f}ms")
    stop_sandbox()
    print("Done.")
