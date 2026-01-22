"""Taxonomy Editor - LLM-powered interactive refinement of category/sub-category structure."""

from __future__ import annotations

from typing import TYPE_CHECKING

from synkro.llm.client import LLM
from synkro.prompts.interactive_templates import TAXONOMY_REFINEMENT_PROMPT
from synkro.schemas import TaxonomyRefinementOutput
from synkro.types.coverage import SubCategory, SubCategoryTaxonomy

if TYPE_CHECKING:
    from synkro.types.logic_map import LogicMap


class TaxonomyEditor:
    """
    LLM-powered taxonomy editor that interprets natural language feedback.

    The editor takes user feedback in natural language (e.g., "add a category for...",
    "add a sub-category for...", "rename X to Y") and uses an LLM to interpret and apply
    the changes to the taxonomy.

    Examples:
        >>> editor = TaxonomyEditor(llm=grading_llm)
        >>> new_taxonomy, summary = await editor.refine(
        ...     taxonomy=current_taxonomy,
        ...     feedback="Add a category for travel expenses",
        ...     policy_text=policy.text,
        ...     logic_map=logic_map
        ... )
    """

    def __init__(self, llm: LLM):
        """
        Initialize the Taxonomy Editor.

        Args:
            llm: LLM client to use for editing (typically the grading model)
        """
        self.llm = llm

    async def refine(
        self,
        taxonomy: SubCategoryTaxonomy,
        feedback: str,
        policy_text: str,
        logic_map: LogicMap,
    ) -> tuple[SubCategoryTaxonomy, str]:
        """
        Refine the taxonomy based on natural language feedback.

        Args:
            taxonomy: Current taxonomy to refine
            feedback: Natural language instruction from user
            policy_text: Original policy text for context
            logic_map: Logic Map for rule references

        Returns:
            Tuple of (refined SubCategoryTaxonomy, changes summary string)
        """
        # Format current taxonomy as string for prompt
        taxonomy_str = self._format_taxonomy_for_prompt(taxonomy)
        logic_map_str = self._format_logic_map_for_prompt(logic_map)

        # Format the prompt
        prompt = TAXONOMY_REFINEMENT_PROMPT.format(
            taxonomy_formatted=taxonomy_str,
            logic_map_formatted=logic_map_str,
            policy_text=policy_text[:4000] if len(policy_text) > 4000 else policy_text,
            feedback=feedback,
        )

        # Generate structured output with temperature=0 for strict instruction following
        result = await self.llm.generate_structured(prompt, TaxonomyRefinementOutput, temperature=0)

        # Convert to domain model
        refined_taxonomy = self._convert_to_taxonomy(result, taxonomy)

        return refined_taxonomy, result.changes_summary

    def _format_taxonomy_for_prompt(self, taxonomy: SubCategoryTaxonomy) -> str:
        """Format a taxonomy as a string for the LLM prompt."""
        lines = []
        lines.append(f"Total Sub-Categories: {len(taxonomy.sub_categories)}")

        # Group by parent category
        by_category: dict[str, list[SubCategory]] = {}
        for sc in taxonomy.sub_categories:
            cat = sc.parent_category
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append(sc)

        lines.append(f"Categories: {', '.join(by_category.keys())}")
        lines.append("")

        for cat_name, sub_cats in by_category.items():
            lines.append(f"[{cat_name}] ({len(sub_cats)} sub-categories)")
            for sc in sub_cats:
                lines.append(f"  {sc.id}: {sc.name}")
                lines.append(f"    Description: {sc.description}")
                lines.append(f"    Priority: {sc.priority}")
                lines.append(f"    Related Rules: {', '.join(sc.related_rule_ids) or 'None'}")
                lines.append("")

        return "\n".join(lines)

    def _format_logic_map_for_prompt(self, logic_map: LogicMap) -> str:
        """Format Logic Map rules as string for reference."""
        lines = []
        lines.append(f"Total Rules: {len(logic_map.rules)}")
        lines.append("")
        for rule in logic_map.rules:
            lines.append(f"{rule.rule_id}: {rule.text[:100]}...")
        return "\n".join(lines)

    def _convert_to_taxonomy(
        self,
        output: TaxonomyRefinementOutput,
        original: SubCategoryTaxonomy,
    ) -> SubCategoryTaxonomy:
        """Convert schema output to domain model."""
        sub_categories = []
        for sc_out in output.sub_categories:
            sc = SubCategory(
                id=sc_out.id,
                name=sc_out.name,
                description=sc_out.description,
                parent_category=sc_out.parent_category,
                related_rule_ids=sc_out.related_rule_ids,
                priority=sc_out.priority,
            )
            sub_categories.append(sc)

        return SubCategoryTaxonomy(
            sub_categories=sub_categories,
            reasoning=f"Refined from user feedback. Original had {len(original.sub_categories)} sub-categories.",
        )

    def validate_refinement(
        self,
        original: SubCategoryTaxonomy,
        refined: SubCategoryTaxonomy,
    ) -> tuple[bool, list[str]]:
        """
        Validate that refinement is sensible.

        Args:
            original: Original taxonomy
            refined: Refined taxonomy

        Returns:
            Tuple of (is_valid, list of issue descriptions)
        """
        issues = []

        # Check for unique IDs
        ids = [sc.id for sc in refined.sub_categories]
        if len(ids) != len(set(ids)):
            duplicates = [id for id in ids if ids.count(id) > 1]
            issues.append(f"Duplicate sub-category IDs: {', '.join(set(duplicates))}")

        # Check that all sub-categories have non-empty required fields
        for sc in refined.sub_categories:
            if not sc.id:
                issues.append("Found sub-category with empty ID")
            if not sc.name:
                issues.append(f"Sub-category {sc.id} has empty name")
            if not sc.parent_category:
                issues.append(f"Sub-category {sc.id} has no parent category")

        # Check that we have at least one sub-category
        if not refined.sub_categories:
            issues.append("Taxonomy must have at least one sub-category")

        return len(issues) == 0, issues
