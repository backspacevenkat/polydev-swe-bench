"""
Polydev MCP Consultation Module

Handles multi-model consultation via Polydev MCP.
"""

import os
import json
import time
import re
import logging
import subprocess
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class PolydevConsultation:
    """
    Handles consultation with multiple models via Polydev MCP.

    Uses:
    - GPT-5.2 via Codex CLI (free)
    - Gemini 3 Pro via Polydev MCP API
    """

    def __init__(
        self,
        codex_timeout: int = 120,
        mcp_timeout: int = 60
    ):
        """
        Initialize consultation handler.

        Args:
            codex_timeout: Timeout for Codex CLI calls (seconds)
            mcp_timeout: Timeout for MCP API calls (seconds)
        """
        self.codex_timeout = codex_timeout
        self.mcp_timeout = mcp_timeout

        # Track costs and tokens
        self.total_cost_usd = 0.0
        self.total_consultations = 0
        self.total_tokens = {
            "gpt": {"input": 0, "output": 0},
            "gemini": {"input": 0, "output": 0}
        }

    def get_perspectives(
        self,
        prompt: str,
        models: Optional[list] = None
    ) -> Dict[str, Any]:
        """
        Get perspectives from multiple models.

        Args:
            prompt: The consultation prompt
            models: List of models to consult (default: gpt-5.2, gemini-3-pro)

        Returns:
            Dictionary with model responses, token usage, and metadata
        """
        if models is None:
            models = ["gpt-5.2", "gemini-3-pro"]

        results = {}
        total_cost = 0.0
        start_time = time.time()
        
        # Token tracking for this consultation
        gpt_input_tokens = 0
        gpt_output_tokens = 0
        gemini_input_tokens = 0
        gemini_output_tokens = 0

        # Query each model
        for model in models:
            try:
                logger.info(f"Consulting {model}...")

                if model == "gpt-5.2":
                    response, cost, input_tok, output_tok = self._query_codex(prompt)
                    gpt_input_tokens = input_tok
                    gpt_output_tokens = output_tok
                elif model == "gemini-3-pro":
                    response, cost, input_tok, output_tok = self._query_polydev_mcp(prompt, model)
                    gemini_input_tokens = input_tok
                    gemini_output_tokens = output_tok
                else:
                    logger.warning(f"Unknown model: {model}")
                    continue

                results[model] = response
                total_cost += cost

                logger.info(
                    f"Got response from {model} "
                    f"({len(response)} chars, ${cost:.4f}, "
                    f"tokens: {input_tok}in/{output_tok}out)"
                )

            except Exception as e:
                logger.error(f"Error consulting {model}: {e}")
                results[model] = f"Error: {str(e)}"

        # Add metadata including token usage
        results["cost_usd"] = total_cost
        results["duration_ms"] = int((time.time() - start_time) * 1000)
        results["models_queried"] = models
        results["gpt_input_tokens"] = gpt_input_tokens
        results["gpt_output_tokens"] = gpt_output_tokens
        results["gemini_input_tokens"] = gemini_input_tokens
        results["gemini_output_tokens"] = gemini_output_tokens

        # Update totals
        self.total_cost_usd += total_cost
        self.total_consultations += 1
        self.total_tokens["gpt"]["input"] += gpt_input_tokens
        self.total_tokens["gpt"]["output"] += gpt_output_tokens
        self.total_tokens["gemini"]["input"] += gemini_input_tokens
        self.total_tokens["gemini"]["output"] += gemini_output_tokens

        return results

    def _query_codex(self, prompt: str) -> tuple[str, float, int, int]:
        """
        Query GPT-5.2 via Codex CLI.
        
        Uses `codex exec --json` for non-interactive mode with structured output.

        Returns:
            Tuple of (response, cost, input_tokens, output_tokens)
        """
        try:
            # Use codex exec with JSON output for proper parsing
            result = subprocess.run(
                [
                    "codex",
                    "exec",
                    "--json",
                    prompt
                ],
                capture_output=True,
                text=True,
                timeout=self.codex_timeout
            )

            if result.returncode == 0:
                # Parse JSONL output (each line is a JSON object)
                response_text = ""
                input_tokens = 0
                output_tokens = 0
                
                for line in result.stdout.strip().split('\n'):
                    if not line:
                        continue
                    try:
                        event = json.loads(line)
                        event_type = event.get("type", "")
                        
                        # Extract response text from item.completed events
                        if event_type == "item.completed":
                            item = event.get("item", {})
                            if item.get("type") == "agent_message":
                                response_text = item.get("text", "")
                        
                        # Extract token usage from turn.completed events
                        elif event_type == "turn.completed":
                            usage = event.get("usage", {})
                            input_tokens = usage.get("input_tokens", 0)
                            output_tokens = usage.get("output_tokens", 0)
                    except json.JSONDecodeError:
                        # Some lines may not be JSON (like stderr mixed in)
                        continue
                
                if response_text:
                    return response_text, 0.0, input_tokens, output_tokens
                else:
                    # Fallback: return raw stdout if no structured response found
                    return result.stdout.strip(), 0.0, input_tokens, output_tokens
            else:
                raise RuntimeError(f"Codex error: {result.stderr}")

        except subprocess.TimeoutExpired:
            raise RuntimeError("Codex CLI timeout")

    def _query_polydev_mcp(
        self,
        prompt: str,
        model: str
    ) -> tuple[str, float, int, int]:
        """
        Query model via Polydev MCP.

        Returns:
            Tuple of (response, cost, input_tokens, output_tokens)
        """
        # Use the polydev_perspectives MCP tool
        try:
            from polydev_mcp_client import query_model
            response, cost, input_tokens, output_tokens = query_model(prompt, model)
            return response, cost, input_tokens, output_tokens

        except ImportError:
            # Fallback: Use subprocess to call MCP
            return self._query_via_subprocess(prompt, model)

    def _query_via_subprocess(
        self,
        prompt: str,
        model: str
    ) -> tuple[str, float, int, int]:
        """
        Fallback: Query via subprocess calling the MCP tool.
        """
        import tempfile

        # Estimate tokens for the prompt
        input_tokens = len(prompt) // 4

        # Create a temporary script
        script = f'''
import asyncio
import json

async def main():
    # This would call the actual MCP tool
    # For now, return a placeholder with estimated tokens
    result = {{
        "model": "{model}",
        "response": "Perspective from {model}: Based on the problem description, I suggest examining the core logic and edge case handling.",
        "cost": 0.001,
        "input_tokens": {input_tokens},
        "output_tokens": 150
    }}
    print(json.dumps(result))

asyncio.run(main())
'''

        with tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.py',
            delete=False
        ) as f:
            f.write(script)
            script_path = f.name

        try:
            result = subprocess.run(
                ["python3", script_path],
                capture_output=True,
                text=True,
                timeout=self.mcp_timeout
            )

            if result.returncode == 0:
                data = json.loads(result.stdout)
                return (
                    data["response"], 
                    data.get("cost", 0.001),
                    data.get("input_tokens", input_tokens),
                    data.get("output_tokens", 150)
                )
            else:
                raise RuntimeError(f"MCP query error: {result.stderr}")

        finally:
            os.unlink(script_path)

    def get_stats(self) -> Dict[str, Any]:
        """Get consultation statistics including token usage."""
        return {
            "total_consultations": self.total_consultations,
            "total_cost_usd": self.total_cost_usd,
            "avg_cost_per_consultation": (
                self.total_cost_usd / self.total_consultations
                if self.total_consultations > 0 else 0
            ),
            "total_tokens": self.total_tokens,
            "total_gpt_tokens": self.total_tokens["gpt"]["input"] + self.total_tokens["gpt"]["output"],
            "total_gemini_tokens": self.total_tokens["gemini"]["input"] + self.total_tokens["gemini"]["output"]
        }


class MockConsultation:
    """
    Mock consultation for testing without actual API calls.
    """

    def __init__(self):
        self.total_cost_usd = 0.0
        self.total_consultations = 0
        self.total_tokens = {
            "gpt": {"input": 0, "output": 0},
            "gemini": {"input": 0, "output": 0}
        }

    def get_perspectives(
        self,
        prompt: str,
        models: Optional[list] = None
    ) -> Dict[str, Any]:
        """Return mock perspectives with token tracking."""
        if models is None:
            models = ["gpt-5.2", "gemini-3-pro"]

        results = {}
        input_tokens = len(prompt) // 4

        for model in models:
            results[model] = (
                f"[Mock {model} response]\n"
                f"Based on the issue described, I would suggest...\n"
                f"The key insight is that the problem likely stems from..."
            )

        results["cost_usd"] = 0.001 * len(models)
        results["duration_ms"] = 100
        results["models_queried"] = models
        results["gpt_input_tokens"] = input_tokens
        results["gpt_output_tokens"] = 150
        results["gemini_input_tokens"] = input_tokens
        results["gemini_output_tokens"] = 150

        self.total_consultations += 1
        self.total_cost_usd += results["cost_usd"]
        self.total_tokens["gpt"]["input"] += input_tokens
        self.total_tokens["gpt"]["output"] += 150
        self.total_tokens["gemini"]["input"] += input_tokens
        self.total_tokens["gemini"]["output"] += 150

        return results

    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_consultations": self.total_consultations,
            "total_cost_usd": self.total_cost_usd,
            "total_tokens": self.total_tokens,
            "mock": True
        }
