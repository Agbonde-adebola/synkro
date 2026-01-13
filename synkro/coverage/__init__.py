"""Coverage tracking for scenario diversity analysis.

This module provides tools for tracking and improving scenario coverage
across policy sub-categories, similar to code coverage for tests.

Components:
- TaxonomyExtractor: Extract sub-category taxonomy from policy
- ScenarioTagger: Tag scenarios with sub-category IDs
- CoverageCalculator: Calculate coverage metrics
- CoverageImprover: Generate scenarios to fill coverage gaps
"""

from synkro.coverage.taxonomy_extractor import TaxonomyExtractor
from synkro.coverage.scenario_tagger import ScenarioTagger
from synkro.coverage.calculator import CoverageCalculator
from synkro.coverage.improver import CoverageImprover

__all__ = [
    "TaxonomyExtractor",
    "ScenarioTagger",
    "CoverageCalculator",
    "CoverageImprover",
]
