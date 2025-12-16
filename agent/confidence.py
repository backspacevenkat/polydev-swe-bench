"""
Confidence Assessment Module

Extracts and validates confidence scores from model responses.
"""

import re
import logging
from typing import Tuple, List

logger = logging.getLogger(__name__)


class ConfidenceAssessor:
    """
    Extracts confidence scores from model responses.

    Looks for patterns like:
    - <score>7</score>
    - Confidence: 7/10
    - confidence score: 7
    """

    def __init__(self, default_score: int = 5):
        """
        Initialize assessor.

        Args:
            default_score: Default confidence if extraction fails
        """
        self.default_score = default_score

        # Patterns to look for confidence scores
        self.patterns = [
            # XML-style tags
            r"<score>(\d+)</score>",
            r"<confidence.*?>(\d+)</confidence>",

            # Natural language patterns
            r"confidence[:\s]+(\d+)\s*/\s*10",
            r"confidence[:\s]+(\d+)\b",
            r"confidence score[:\s]+(\d+)",

            # Simple number in context
            r"(\d+)\s*/\s*10\s*confidence",
            r"rating[:\s]+(\d+)",
        ]

    def assess(self, response: str) -> Tuple[int, str, List[str]]:
        """
        Extract confidence score from model response.

        Args:
            response: Model response containing confidence assessment

        Returns:
            Tuple of (score, reasoning, uncertainties)
        """
        score = self._extract_score(response)
        reasoning = self._extract_reasoning(response)
        uncertainties = self._extract_uncertainties(response)

        logger.debug(
            f"Confidence assessment: score={score}, "
            f"uncertainties={len(uncertainties)}"
        )

        return score, reasoning, uncertainties

    def _extract_score(self, response: str) -> int:
        """Extract numeric confidence score."""
        response_lower = response.lower()

        for pattern in self.patterns:
            match = re.search(pattern, response_lower)
            if match:
                try:
                    score = int(match.group(1))
                    # Clamp to valid range
                    score = max(1, min(10, score))
                    return score
                except ValueError:
                    continue

        logger.warning(
            "Could not extract confidence score, using default"
        )
        return self.default_score

    def _extract_reasoning(self, response: str) -> str:
        """Extract confidence reasoning."""
        # Try XML tag first
        reasoning_match = re.search(
            r"<reasoning>(.*?)</reasoning>",
            response,
            re.DOTALL | re.IGNORECASE
        )

        if reasoning_match:
            return reasoning_match.group(1).strip()

        # Try natural language patterns
        patterns = [
            r"reasoning[:\s]+(.*?)(?=\n\n|\Z)",
            r"because[:\s]+(.*?)(?=\n\n|\Z)",
            r"confidence is .+ because (.*?)(?=\n\n|\Z)"
        ]

        for pattern in patterns:
            match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
            if match:
                return match.group(1).strip()

        return ""

    def _extract_uncertainties(self, response: str) -> List[str]:
        """Extract list of uncertainties."""
        uncertainties = []

        # Try XML tag first
        uncertainties_match = re.search(
            r"<uncertainties>(.*?)</uncertainties>",
            response,
            re.DOTALL | re.IGNORECASE
        )

        if uncertainties_match:
            content = uncertainties_match.group(1)
            for line in content.split("\n"):
                line = line.strip()
                if line.startswith("- "):
                    uncertainties.append(line[2:].strip())
                elif line and not line.startswith("<"):
                    uncertainties.append(line)

        # Also look for uncertainty markers in general text
        uncertainty_markers = [
            r"uncertain about (.*?)(?:\.|,|\n)",
            r"not sure about (.*?)(?:\.|,|\n)",
            r"unclear (.*?)(?:\.|,|\n)",
            r"might be wrong about (.*?)(?:\.|,|\n)",
            r"possible that (.*?)(?:\.|,|\n)"
        ]

        for pattern in uncertainty_markers:
            for match in re.finditer(pattern, response, re.IGNORECASE):
                uncertainty = match.group(1).strip()
                if uncertainty and uncertainty not in uncertainties:
                    uncertainties.append(uncertainty)

        return uncertainties

    def should_consult(self, score: int, threshold: int = 8) -> bool:
        """
        Determine if consultation should be triggered.

        Args:
            score: Confidence score (1-10)
            threshold: Consultation threshold

        Returns:
            True if consultation should be triggered
        """
        return score < threshold


def detect_uncertainty_markers(text: str) -> List[str]:
    """
    Detect phrases that indicate uncertainty in text.

    Returns list of detected uncertainty markers.
    """
    markers = []

    patterns = [
        "i'm not sure",
        "i'm uncertain",
        "multiple approaches",
        "could be done in several ways",
        "not familiar with",
        "might be wrong",
        "unclear",
        "ambiguous",
        "several possibilities",
        "trade-off",
        "depends on"
    ]

    text_lower = text.lower()

    for pattern in patterns:
        if pattern in text_lower:
            markers.append(pattern)

    return markers
