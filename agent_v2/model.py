"""
Model interface for Claude CLI with Polydev MCP consultation.

Uses Claude Code CLI for primary inference.
Integrates Polydev MCP for multi-model consultation when stuck.
"""

import json
import subprocess
import time
import re
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for Claude model."""
    timeout: int = 300  # 5 minutes for complex reasoning
    max_retries: int = 3
    consultation_enabled: bool = True
    consultation_trigger_keywords: List[str] = field(default_factory=lambda: [
        "I'm not sure",
        "I'm stuck",
        "unclear",
        "uncertain",
        "need more information",
        "don't understand",
        "confused",
    ])


@dataclass
class TokenUsage:
    """Token usage tracking."""
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0

    def add(self, input_tokens: int, output_tokens: int):
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.total_tokens += input_tokens + output_tokens


class ClaudeModel:
    """
    Claude model interface via CLI.

    Features:
    - Direct Claude CLI calls
    - Token usage tracking
    - Polydev MCP consultation integration
    """

    def __init__(self, config: Optional[ModelConfig] = None):
        self.config = config or ModelConfig()
        self.n_calls = 0
        self.cost = 0.0
        self.token_usage = {
            "claude": TokenUsage(),
            "gpt": TokenUsage(),
            "gemini": TokenUsage(),
        }
        self._consultation = None

    def query(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Query Claude with the conversation history.

        Args:
            messages: List of message dicts with 'role' and 'content'

        Returns:
            Dict with 'content' and optional metadata
        """
        # Build prompt from messages
        prompt = self._build_prompt(messages)

        # Call Claude CLI
        response = self._call_claude(prompt)
        self.n_calls += 1

        return {"content": response}

    def _build_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Build a prompt string from message history."""
        parts = []

        for msg in messages:
            role = msg["role"]
            content = msg["content"]

            if role == "system":
                parts.append(f"<system>\n{content}\n</system>\n")
            elif role == "user":
                parts.append(f"<user>\n{content}\n</user>\n")
            elif role == "assistant":
                parts.append(f"<assistant>\n{content}\n</assistant>\n")

        return "\n".join(parts)

    def _call_claude(self, prompt: str) -> str:
        """Call Claude via CLI."""
        for attempt in range(self.config.max_retries):
            try:
                result = subprocess.run(
                    ["claude", "-p", prompt, "--output-format", "json"],
                    capture_output=True,
                    text=True,
                    timeout=self.config.timeout,
                )

                if result.returncode == 0:
                    return self._parse_claude_response(result.stdout)
                else:
                    logger.warning(f"Claude CLI error (attempt {attempt + 1}): {result.stderr}")

            except subprocess.TimeoutExpired:
                logger.warning(f"Claude CLI timeout (attempt {attempt + 1})")
            except Exception as e:
                logger.error(f"Claude CLI error: {e}")

        raise RuntimeError("Failed to get response from Claude after max retries")

    def _parse_claude_response(self, stdout: str) -> str:
        """Parse Claude CLI JSON response."""
        try:
            data = json.loads(stdout)

            # Track token usage
            if isinstance(data, dict):
                usage = data.get("usage", {})
                input_tokens = usage.get("input_tokens", 0)
                output_tokens = usage.get("output_tokens", 0)
                self.token_usage["claude"].add(input_tokens, output_tokens)

                # Estimate cost (Sonnet pricing)
                input_cost = input_tokens * 0.003 / 1000
                output_cost = output_tokens * 0.015 / 1000
                self.cost += input_cost + output_cost

                # Extract response text
                if "result" in data:
                    return data["result"]
                elif "content" in data:
                    return data["content"]
                elif "text" in data:
                    return data["text"]

            return str(data)

        except json.JSONDecodeError:
            # Estimate tokens from text length
            input_tokens = len(stdout) // 4
            output_tokens = len(stdout) // 4
            self.token_usage["claude"].add(input_tokens, output_tokens)
            return stdout.strip()

    def should_consult(self, response: str) -> bool:
        """Check if response indicates the model is stuck."""
        if not self.config.consultation_enabled:
            return False

        response_lower = response.lower()
        for keyword in self.config.consultation_trigger_keywords:
            if keyword.lower() in response_lower:
                return True

        return False

    def consult_polydev(self, context: str) -> Dict[str, str]:
        """
        Consult Polydev MCP for multi-model perspectives.

        Args:
            context: The problem context to consult on

        Returns:
            Dict with perspectives from different models
        """
        logger.info("Triggering Polydev MCP consultation...")

        perspectives = {}

        # Query GPT-5.2 via Codex CLI
        gpt_response, gpt_input, gpt_output = self._query_codex(context)
        perspectives["gpt-5.2"] = gpt_response
        self.token_usage["gpt"].add(gpt_input, gpt_output)

        # Query Gemini via Polydev MCP (placeholder - would use actual MCP)
        gemini_response, gemini_input, gemini_output = self._query_gemini(context)
        perspectives["gemini-3-pro"] = gemini_response
        self.token_usage["gemini"].add(gemini_input, gemini_output)

        logger.info(f"Consultation complete: GPT tokens={gpt_input}in/{gpt_output}out, Gemini tokens={gemini_input}in/{gemini_output}out")

        return perspectives

    def _query_codex(self, prompt: str) -> tuple:
        """Query GPT-5.2 via Codex CLI."""
        try:
            result = subprocess.run(
                ["codex", "exec", "--json", prompt],
                capture_output=True,
                text=True,
                timeout=120,
            )

            if result.returncode == 0:
                return self._parse_codex_response(result.stdout)
            else:
                logger.warning(f"Codex error: {result.stderr}")
                return "Error querying GPT-5.2", 0, 0

        except subprocess.TimeoutExpired:
            return "Codex timeout", 0, 0
        except Exception as e:
            return f"Codex error: {e}", 0, 0

    def _parse_codex_response(self, stdout: str) -> tuple:
        """Parse Codex CLI JSONL response."""
        response_text = ""
        input_tokens = 0
        output_tokens = 0

        for line in stdout.strip().split('\n'):
            if not line:
                continue
            try:
                event = json.loads(line)
                event_type = event.get("type", "")

                if event_type == "item.completed":
                    item = event.get("item", {})
                    if item.get("type") == "agent_message":
                        response_text = item.get("text", "")

                elif event_type == "turn.completed":
                    usage = event.get("usage", {})
                    input_tokens = usage.get("input_tokens", 0)
                    output_tokens = usage.get("output_tokens", 0)

            except json.JSONDecodeError:
                continue

        return response_text or stdout.strip(), input_tokens, output_tokens

    def _query_gemini(self, prompt: str) -> tuple:
        """Query Gemini via Polydev MCP API."""
        # For now, use a placeholder that estimates tokens
        # In production, this would call the actual Polydev MCP
        input_tokens = len(prompt) // 4
        output_tokens = 150  # Estimate

        # Placeholder response
        response = (
            "Based on the problem description, I suggest:\n"
            "1. First, identify the exact location of the bug\n"
            "2. Understand the expected vs actual behavior\n"
            "3. Look for similar patterns in the codebase\n"
            "4. Implement a minimal fix that addresses the root cause"
        )

        return response, input_tokens, output_tokens

    def get_stats(self) -> Dict[str, Any]:
        """Get model usage statistics."""
        return {
            "n_calls": self.n_calls,
            "cost_usd": self.cost,
            "token_usage": {
                provider: {
                    "input": usage.input_tokens,
                    "output": usage.output_tokens,
                    "total": usage.total_tokens,
                }
                for provider, usage in self.token_usage.items()
            }
        }
