"""
Patch Generator Module

Extracts and validates unified diff patches from model responses.
"""

import re
import logging
from typing import Optional, List

logger = logging.getLogger(__name__)


class PatchGenerator:
    """
    Extracts unified diff patches from model responses.

    Handles various formats:
    - Code blocks with ```diff
    - Raw diff output
    - Mixed text and diff content
    """

    def extract_patch(self, response: str) -> str:
        """
        Extract unified diff patch from model response.

        Args:
            response: Model response potentially containing a patch

        Returns:
            Extracted patch string or empty string
        """
        # Try different extraction methods
        patch = self._extract_from_code_block(response)

        if not patch:
            patch = self._extract_raw_diff(response)

        if not patch:
            patch = self._extract_inline_diff(response)

        if patch:
            patch = self._clean_patch(patch)
            if self._validate_patch(patch):
                return patch
            else:
                logger.warning("Extracted patch failed validation")

        logger.warning("Could not extract valid patch from response")
        return ""

    def _extract_from_code_block(self, response: str) -> Optional[str]:
        """Extract patch from markdown code block."""
        # Look for ```diff ... ``` blocks
        pattern = r"```diff\n(.*?)```"
        match = re.search(pattern, response, re.DOTALL)

        if match:
            return match.group(1)

        # Also try generic code blocks that look like diffs
        pattern = r"```\n(diff --git.*?)```"
        match = re.search(pattern, response, re.DOTALL)

        if match:
            return match.group(1)

        return None

    def _extract_raw_diff(self, response: str) -> Optional[str]:
        """Extract raw diff from response."""
        # Look for diff starting marker
        diff_start = response.find("diff --git")

        if diff_start == -1:
            return None

        # Find the end (next non-diff section or end of string)
        diff_text = response[diff_start:]

        # Split on double newline followed by non-diff content
        parts = re.split(r"\n\n(?![-+@ ])", diff_text, maxsplit=1)

        return parts[0] if parts else None

    def _extract_inline_diff(self, response: str) -> Optional[str]:
        """Extract diff from inline content."""
        lines = response.split("\n")
        diff_lines = []
        in_diff = False

        for line in lines:
            if line.startswith("diff --git") or line.startswith("---"):
                in_diff = True

            if in_diff:
                # Stop at clear non-diff content
                if (line and not line[0] in " +-@" and
                    not line.startswith("diff") and
                    not line.startswith("---") and
                    not line.startswith("+++") and
                    not line.startswith("index")):
                    break

                diff_lines.append(line)

        return "\n".join(diff_lines) if diff_lines else None

    def _clean_patch(self, patch: str) -> str:
        """Clean up extracted patch."""
        # Remove leading/trailing whitespace
        patch = patch.strip()

        # Ensure proper line endings
        patch = patch.replace("\r\n", "\n")

        # Remove any trailing whitespace on lines
        lines = [line.rstrip() for line in patch.split("\n")]

        # Ensure patch ends with newline
        return "\n".join(lines) + "\n"

    def _validate_patch(self, patch: str) -> bool:
        """
        Validate that patch is well-formed.

        Checks:
        - Has diff header
        - Has file markers (--- and +++)
        - Has at least one hunk (@@ ... @@)
        """
        if not patch:
            return False

        # Must have diff header or file markers
        has_header = "diff --git" in patch or "---" in patch

        # Must have hunk markers
        has_hunks = "@@" in patch

        # Must have some actual changes (+ or - lines)
        has_changes = re.search(r"^[+-][^+-]", patch, re.MULTILINE)

        if not has_header:
            logger.debug("Patch missing header")
            return False

        if not has_hunks:
            logger.debug("Patch missing hunk markers")
            return False

        if not has_changes:
            logger.debug("Patch has no actual changes")
            return False

        return True

    def format_patch(
        self,
        file_path: str,
        original: str,
        modified: str
    ) -> str:
        """
        Generate a unified diff from original and modified content.

        Args:
            file_path: Path to the file
            original: Original file content
            modified: Modified file content

        Returns:
            Unified diff string
        """
        import difflib

        original_lines = original.splitlines(keepends=True)
        modified_lines = modified.splitlines(keepends=True)

        diff = difflib.unified_diff(
            original_lines,
            modified_lines,
            fromfile=f"a/{file_path}",
            tofile=f"b/{file_path}"
        )

        return "".join(diff)


def parse_patch_files(patch: str) -> List[str]:
    """
    Extract list of files modified by a patch.

    Args:
        patch: Unified diff patch

    Returns:
        List of file paths
    """
    files = []

    # Look for diff --git headers
    for match in re.finditer(r"diff --git a/(.*?) b/", patch):
        files.append(match.group(1))

    # Also look for --- headers
    for match in re.finditer(r"--- a/(.*)", patch):
        file_path = match.group(1).strip()
        if file_path not in files:
            files.append(file_path)

    return files


def count_patch_changes(patch: str) -> dict:
    """
    Count additions and deletions in a patch.

    Args:
        patch: Unified diff patch

    Returns:
        Dict with 'additions' and 'deletions' counts
    """
    additions = 0
    deletions = 0

    for line in patch.split("\n"):
        if line.startswith("+") and not line.startswith("+++"):
            additions += 1
        elif line.startswith("-") and not line.startswith("---"):
            deletions += 1

    return {
        "additions": additions,
        "deletions": deletions,
        "total": additions + deletions
    }
