"""
Local Environment for bash command execution.

Based on mini-swe-agent's design - stateless execution via subprocess.
"""

import os
import subprocess
import platform
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class EnvironmentConfig:
    """Configuration for local environment."""
    cwd: str = ""
    timeout: int = 120  # 2 minutes default
    env_vars: Dict[str, str] = field(default_factory=lambda: {
        "PAGER": "cat",
        "MANPAGER": "cat",
        "LESS": "-R",
        "PIP_PROGRESS_BAR": "off",
        "TQDM_DISABLE": "1",
    })


class LocalEnvironment:
    """
    Executes bash commands locally via subprocess.

    Each command runs in a fresh subprocess - stateless execution.
    """

    def __init__(self, config: Optional[EnvironmentConfig] = None):
        self.config = config or EnvironmentConfig()

    def execute(self, command: str, cwd: Optional[str] = None, timeout: Optional[int] = None) -> Dict[str, Any]:
        """
        Execute a bash command and return the result.

        Args:
            command: Bash command to execute
            cwd: Working directory (defaults to config.cwd or current dir)
            timeout: Command timeout in seconds

        Returns:
            Dict with 'output', 'returncode', and 'timed_out' keys
        """
        cwd = cwd or self.config.cwd or os.getcwd()
        timeout = timeout or self.config.timeout

        # Merge environment variables
        env = os.environ.copy()
        env.update(self.config.env_vars)

        logger.debug(f"Executing: {command[:100]}...")

        try:
            result = subprocess.run(
                command,
                shell=True,
                text=True,
                cwd=cwd,
                env=env,
                timeout=timeout,
                encoding="utf-8",
                errors="replace",
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )

            return {
                "output": result.stdout,
                "returncode": result.returncode,
                "timed_out": False,
            }

        except subprocess.TimeoutExpired as e:
            output = e.stdout.decode("utf-8", errors="replace") if e.stdout else ""
            logger.warning(f"Command timed out after {timeout}s")
            return {
                "output": output,
                "returncode": -1,
                "timed_out": True,
            }

        except Exception as e:
            logger.error(f"Command execution error: {e}")
            return {
                "output": str(e),
                "returncode": -1,
                "timed_out": False,
            }

    def get_system_info(self) -> Dict[str, str]:
        """Get system information for template rendering."""
        uname = platform.uname()
        return {
            "system": uname.system,
            "release": uname.release,
            "version": uname.version,
            "machine": uname.machine,
        }


class DockerEnvironment:
    """
    Executes commands in a Docker container for isolation.

    Used for SWE-bench evaluation where we need repository isolation.
    """

    def __init__(self, image: str = "python:3.11", container_name: Optional[str] = None):
        self.image = image
        self.container_name = container_name
        self.container_id: Optional[str] = None

    def start(self, repo_path: str) -> bool:
        """Start a Docker container with the repository mounted."""
        import uuid

        self.container_name = self.container_name or f"polydev-swe-{uuid.uuid4().hex[:8]}"

        cmd = [
            "docker", "run", "-d",
            "--name", self.container_name,
            "-v", f"{repo_path}:/workspace",
            "-w", "/workspace",
            self.image,
            "sleep", "infinity"
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            self.container_id = result.stdout.strip()
            logger.info(f"Started container: {self.container_id[:12]}")
            return True
        else:
            logger.error(f"Failed to start container: {result.stderr}")
            return False

    def execute(self, command: str, timeout: int = 120) -> Dict[str, Any]:
        """Execute command in the Docker container."""
        if not self.container_id:
            return {"output": "Container not started", "returncode": -1, "timed_out": False}

        cmd = ["docker", "exec", self.container_id, "bash", "-c", command]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
            )

            return {
                "output": result.stdout + result.stderr,
                "returncode": result.returncode,
                "timed_out": False,
            }

        except subprocess.TimeoutExpired:
            return {
                "output": f"Command timed out after {timeout}s",
                "returncode": -1,
                "timed_out": True,
            }

    def stop(self):
        """Stop and remove the container."""
        if self.container_id:
            subprocess.run(["docker", "rm", "-f", self.container_id], capture_output=True)
            logger.info(f"Stopped container: {self.container_id[:12]}")
            self.container_id = None
