"""
Polydev SWE-bench Agent

A lightweight agent for evaluating multi-model consultation on SWE-bench.
"""

from .polydev_agent import PolydevAgent
from .confidence import ConfidenceAssessor
from .consultation import PolydevConsultation
from .patch_generator import PatchGenerator

__version__ = "0.1.0"
__all__ = ["PolydevAgent", "ConfidenceAssessor", "PolydevConsultation", "PatchGenerator"]
