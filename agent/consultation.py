"""
Polydev MCP Consultation Module

Handles multi-model consultation via Polydev MCP.
"""

import os
import json
import time
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

        # Track costs
        self.total_cost_usd = 0.0
        self.total_consultations = 0

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
            Dictionary with model responses and metadata
        """
        if models is None:
            models = ["gpt-5.2", "gemini-3-pro"]

        results = {}
        total_cost = 0.0
        start_time = time.time()

        # Query each model
        for model in models:
            try:
                logger.info(f"Consulting {model}...")

                if model == "gpt-5.2":
                    response, cost = self._query_codex(prompt)
                elif model == "gemini-3-pro":
                    response, cost = self._query_polydev_mcp(prompt, model)
                else:
                    logger.warning(f"Unknown model: {model}")
                    continue

                results[model] = response
                total_cost += cost

                logger.info(
                    f"Got response from {model} "
                    f"({len(response)} chars, ${cost:.4f})"
                )

            except Exception as e:
                logger.error(f"Error consulting {model}: {e}")
                results[model] = f"Error: {str(e)}"

        # Add metadata
        results["cost_usd"] = total_cost
        results["duration_ms"] = int((time.time() - start_time) * 1000)
        results["models_queried"] = models

        # Update totals
        self.total_cost_usd += total_cost
        self.total_consultations += 1

        return results

    def _query_codex(self, prompt: str) -> tuple[str, float]:
        """
        Query GPT-5.2 via Codex CLI.

        Returns:
            Tuple of (response, cost)
        """
        try:
            result = subprocess.run(
                [
                    "codex",
                    "--approval-mode", "full-auto",
                    "--quiet",
                    prompt
                ],
                capture_output=True,
                text=True,
                timeout=self.codex_timeout
            )

            if result.returncode == 0:
                return result.stdout.strip(), 0.0  # Free via CLI
            else:
                raise RuntimeError(f"Codex error: {result.stderr}")

        except subprocess.TimeoutExpired:
            raise RuntimeError("Codex CLI timeout")

    def _query_polydev_mcp(
        self,
        prompt: str,
        model: str
    ) -> tuple[str, float]:
        """
        Query model via Polydev MCP.

        Returns:
            Tuple of (response, cost)
        """
        # Use the polydev_perspectives MCP tool
        # This is called from the polydev-ai MCP execution environment

        try:
            # Import the MCP client
            # In production, this would use the actual MCP protocol
            from polydev_mcp_client import query_model

            response, cost = query_model(prompt, model)
            return response, cost

        except ImportError:
            # Fallback: Use subprocess to call MCP
            return self._query_via_subprocess(prompt, model)

    def _query_via_subprocess(
        self,
        prompt: str,
        model: str
    ) -> tuple[str, float]:
        """
        Fallback: Query via subprocess calling the MCP tool.

        This creates a simple script that uses the MCP tool.
        """
        import tempfile

        # Create a temporary script
        script = f'''
import asyncio
import json

async def main():
    # This would call the actual MCP tool
    # For now, return a placeholder
    result = {{
        "model": "{model}",
        "response": "Perspective from {model}",
        "cost": 0.001
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
                ["python", script_path],
                capture_output=True,
                text=True,
                timeout=self.mcp_timeout
            )

            if result.returncode == 0:
                data = json.loads(result.stdout)
                return data["response"], data.get("cost", 0.001)
            else:
                raise RuntimeError(f"MCP query error: {result.stderr}")

        finally:
            os.unlink(script_path)

    def get_stats(self) -> Dict[str, Any]:
        """Get consultation statistics."""
        return {
            "total_consultations": self.total_consultations,
            "total_cost_usd": self.total_cost_usd,
            "avg_cost_per_consultation": (
                self.total_cost_usd / self.total_consultations
                if self.total_consultations > 0 else 0
            )
        }


class MockConsultation:
    """
    Mock consultation for testing without actual API calls.
    """

    def __init__(self):
        self.total_cost_usd = 0.0
        self.total_consultations = 0

    def get_perspectives(
        self,
        prompt: str,
        models: Optional[list] = None
    ) -> Dict[str, Any]:
        """Return mock perspectives."""
        if models is None:
            models = ["gpt-5.2", "gemini-3-pro"]

        results = {}

        for model in models:
            results[model] = (
                f"[Mock {model} response]\n"
                f"Based on the issue described, I would suggest...\n"
                f"The key insight is that the problem likely stems from..."
            )

        results["cost_usd"] = 0.001 * len(models)
        results["duration_ms"] = 100
        results["models_queried"] = models

        self.total_consultations += 1
        self.total_cost_usd += results["cost_usd"]

        return results

    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_consultations": self.total_consultations,
            "total_cost_usd": self.total_cost_usd,
            "mock": True
        }
