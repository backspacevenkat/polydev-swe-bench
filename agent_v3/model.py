"""
Model interface for Claude CLI with REAL Polydev MCP consultation.

Uses Claude Code CLI for primary inference.
Integrates actual Polydev MCP for multi-model consultation.
"""

import json
import subprocess
import time
import os
import logging
import re
import requests  # Add HTTP client
from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

# Polydev HTTP API endpoint
POLYDEV_HTTP_URL = os.environ.get("POLYDEV_HTTP_URL", "http://localhost:3847")

# Path to the consultation script
CONSULTATION_SCRIPT = Path(__file__).parent / "polydev_consultation.mjs"

# Full paths to CLIs (to avoid PATH issues in subprocesses)
CLAUDE_CLI = "/Users/venkat/.nvm/versions/node/v22.20.0/bin/claude"
NODE_CLI = "/Users/venkat/.nvm/versions/node/v22.20.0/bin/node"


@dataclass
class ModelConfig:
    """Configuration for Claude model."""
    timeout: int = 300  # 5 minutes for complex reasoning
    max_retries: int = 3
    consultation_enabled: bool = True
    consultation_timeout: int = 180  # 3 minutes for multi-model consultation
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
    Claude model interface via CLI with REAL Polydev MCP consultation.
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

    def query(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """Query Claude with the conversation history."""
        prompt = self._build_prompt(messages)
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
                    [CLAUDE_CLI, "-p", prompt, "--output-format", "json"],
                    capture_output=True,
                    text=True,
                    timeout=self.config.timeout,
                )

                if result.returncode == 0:
                    return self._parse_claude_response(result.stdout)
                else:
                    logger.warning(f"Claude CLI error (attempt {attempt + 1}): {result.stderr[:200]}")

            except subprocess.TimeoutExpired:
                logger.warning(f"Claude CLI timeout (attempt {attempt + 1})")
            except Exception as e:
                logger.error(f"Claude CLI error: {e}")

        raise RuntimeError("Failed to get response from Claude after max retries")

    def _parse_claude_response(self, stdout: str) -> str:
        """Parse Claude CLI JSON response."""
        try:
            # Handle potential multiple JSON objects or wrapped output
            # First try to find a JSON object
            stdout = stdout.strip()

            # Try parsing directly
            try:
                data = json.loads(stdout)
            except json.JSONDecodeError:
                # Try to extract JSON from the output
                import re
                json_match = re.search(r'\{.*\}', stdout, re.DOTALL)
                if json_match:
                    data = json.loads(json_match.group())
                else:
                    # No JSON found, return as text
                    self.token_usage["claude"].add(len(stdout) // 4, len(stdout) // 4)
                    return stdout

            # Track token usage
            if isinstance(data, dict):
                usage = data.get("usage", {})
                input_tokens = usage.get("input_tokens", 0)
                output_tokens = usage.get("output_tokens", 0)

                # If no usage info, estimate from content
                if input_tokens == 0 and output_tokens == 0:
                    content = str(data.get("result", data.get("content", data.get("text", ""))))
                    input_tokens = len(stdout) // 4
                    output_tokens = len(content) // 4

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

        except Exception as e:
            logger.warning(f"Failed to parse Claude response: {e}")
            # Estimate tokens from text length
            self.token_usage["claude"].add(len(stdout) // 4, len(stdout) // 4)
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
        Consult Polydev MCP for REAL multi-model perspectives via HTTP API.

        Uses the Polydev HTTP server (polydev_http_server.mjs) which wraps
        the actual Polydev MCP server for proper token tracking.
        """
        logger.info("Triggering REAL Polydev MCP consultation via HTTP...")
        start_time = time.time()

        perspectives = {}

        try:
            # Call Polydev HTTP API
            response = requests.post(
                f"{POLYDEV_HTTP_URL}/perspectives",
                json={
                    "prompt": context,
                    "models": ["gpt-4o", "claude-3-5-sonnet-20241022", "gemini-2.0-flash-exp"],
                },
                timeout=self.config.consultation_timeout,
            )

            if response.status_code == 200:
                data = response.json()

                if data.get("success"):
                    result = data.get("result", {})

                    # Parse perspectives from MCP response
                    # The result structure depends on Polydev's response format
                    if isinstance(result, dict):
                        content = result.get("content", [])
                        if isinstance(content, list):
                            for item in content:
                                if isinstance(item, dict) and item.get("type") == "text":
                                    # Parse the text content for model responses
                                    text = item.get("text", "")
                                    perspectives["polydev"] = text

                                    # Estimate tokens
                                    tokens = len(text) // 4
                                    self.token_usage["gpt"].add(tokens // 3, tokens // 3)
                                    self.token_usage["claude"].add(tokens // 3, tokens // 3)
                                    self.token_usage["gemini"].add(tokens // 3, tokens // 3)

                    latency = data.get("latency_ms", (time.time() - start_time) * 1000)
                    logger.info(f"Consultation complete in {latency/1000:.1f}s via Polydev MCP")

                else:
                    error_msg = data.get("error", "Unknown error")
                    logger.warning(f"Polydev API error: {error_msg}")
                    perspectives["error"] = f"Polydev API error: {error_msg}"

            else:
                logger.warning(f"Polydev HTTP error {response.status_code}: {response.text[:200]}")
                perspectives["error"] = f"HTTP {response.status_code}: {response.text[:200]}"

        except requests.Timeout:
            logger.warning("Polydev consultation timed out")
            perspectives["error"] = "Consultation timed out"
        except requests.ConnectionError:
            logger.warning("Cannot connect to Polydev HTTP server - is it running?")
            perspectives["error"] = "Polydev server not running. Start with: node polydev_http_server.mjs"
        except Exception as e:
            logger.error(f"Consultation error: {e}")
            perspectives["error"] = str(e)

        return perspectives

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
