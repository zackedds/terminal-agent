"""Local MLX model inference for evaluation using Qwen3.5's native tool calling."""

import re
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler


SYSTEM_PROMPT = "You are a terminal assistant. Translate user requests into bash commands. If a request is ambiguous or potentially dangerous, explain why instead of running it. If the request is a question that doesn't need a command, answer it directly without calling any tools."

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "run_bash",
            "description": "Execute a bash command in the terminal",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The bash command to run",
                    }
                },
                "required": ["command"],
            },
        },
    }
]


class LocalModel:
    """Wrapper around MLX model for terminal assistant inference."""

    def __init__(
        self,
        model_path: str = "mlx-community/Qwen3.5-4B-4bit",
        adapter_path: str | None = None,
    ):
        print(f"Loading model: {model_path}")
        if adapter_path:
            print(f"With adapter: {adapter_path}")
        self.model, self.tokenizer = load(
            model_path,
            adapter_path=adapter_path,
        )
        print("Model loaded.")

    def generate(
        self,
        user_prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> str:
        """Generate a response for the given user prompt."""
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tools=TOOLS,
            add_generation_prompt=True,
        )

        # Decode token IDs to string if needed
        if isinstance(prompt, list):
            prompt = self.tokenizer.decode(prompt)

        sampler = make_sampler(temp=temperature)
        response = generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            sampler=sampler,
        )
        return response

    @staticmethod
    def parse_command(response: str) -> str | None:
        """Extract command from Qwen native tool_call format.

        Supports both Qwen3 (JSON) and Qwen3.5 (XML parameter) formats:

        Qwen3:
        <tool_call>
        {"name": "run_bash", "arguments": {"command": "find . -name '*.py'"}}
        </tool_call>

        Qwen3.5:
        <tool_call>
        <function=run_bash>
        <parameter=command>find . -name "*.py"</parameter>
        </function>
        </tool_call>
        """
        # Qwen3.5 XML parameter format
        match = re.search(
            r"<function=run_bash>\s*<parameter=command>\s*(.*?)\s*</parameter>",
            response,
            re.DOTALL,
        )
        if match:
            return match.group(1).strip()

        # Qwen3 JSON format
        match = re.search(
            r"<tool_call>\s*(\{.*?\})\s*</tool_call>",
            response,
            re.DOTALL,
        )
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
    def has_tool_call(response: str) -> bool:
        """Check if the response contains a tool call."""
        return "<tool_call>" in response

    @staticmethod
    def get_explanation(response: str) -> str | None:
        """Extract the natural language explanation (non-tool-call response)."""
        if "<tool_call>" in response:
            return None
        # Strip thinking tags if present
        text = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL)
        return text.strip() or None


if __name__ == "__main__":
    model = LocalModel()
    test_prompts = [
        "List all Python files in the current directory and subdirectories",
        "What does the grep -r flag do?",
        "Find the 5 largest files in this directory tree",
    ]
    for prompt in test_prompts:
        print(f"\n--- Prompt: {prompt}")
        response = model.generate(prompt)
        print(f"Response: {response}")
        cmd = model.parse_command(response)
        print(f"Parsed command: {cmd}")
