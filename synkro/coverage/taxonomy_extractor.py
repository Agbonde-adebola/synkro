"""Taxonomy Extractor - Extract sub-category taxonomy from policy.

Extracts a hierarchical taxonomy of sub-categories from a policy document,
enabling coverage tracking across different testable aspects.
"""

from synkro.llm.client import LLM
from synkro.models import Model, OpenAI
from synkro.schemas import TaxonomyOutput
from synkro.types.coverage import SubCategory, SubCategoryTaxonomy
from synkro.types.logic_map import LogicMap
from synkro.types.core import Category
from synkro.prompts.coverage_templates import SUBCATEGORY_EXTRACTION_PROMPT


class TaxonomyExtractor:
    """
    Extract sub-category taxonomy from policy documents.

    The taxonomy organizes a policy into testable sub-categories within
    each planning category, enabling coverage tracking and gap analysis.

    Examples:
        >>> extractor = TaxonomyExtractor(llm=LLM(model=OpenAI.GPT_4O))
        >>> taxonomy = await extractor.extract(
        ...     policy_text=policy_text,
        ...     logic_map=logic_map,
        ...     categories=[cat1, cat2],
        ... )
        >>> print(f"Extracted {len(taxonomy.sub_categories)} sub-categories")
    """

    def __init__(
        self,
        llm: LLM | None = None,
        model: Model = OpenAI.GPT_4O,
    ):
        """
        Initialize the Taxonomy Extractor.

        Args:
            llm: LLM client to use (creates one if not provided)
            model: Model to use if creating LLM (default: GPT-4O for accuracy)
        """
        self.llm = llm or LLM(model=model, temperature=0.3)

    async def extract(
        self,
        policy_text: str,
        logic_map: LogicMap,
        categories: list[Category | str],
    ) -> SubCategoryTaxonomy:
        """
        Extract a sub-category taxonomy from a policy document.

        Args:
            policy_text: The policy document text
            logic_map: Extracted Logic Map with rules
            categories: Planning categories to organize sub-categories under

        Returns:
            SubCategoryTaxonomy with extracted sub-categories

        Raises:
            ValueError: If extraction fails or produces invalid taxonomy
        """
        # Format inputs for the prompt
        logic_map_str = self._format_logic_map(logic_map)
        categories_str = self._format_categories(categories)

        # Format the prompt
        prompt = SUBCATEGORY_EXTRACTION_PROMPT.format(
            policy_text=policy_text,
            logic_map=logic_map_str,
            categories=categories_str,
        )

        # Generate structured output
        result = await self.llm.generate_structured(prompt, TaxonomyOutput)

        # Convert to domain model
        taxonomy = self._convert_to_taxonomy(result)

        # Validate taxonomy
        self._validate_taxonomy(taxonomy, categories)

        return taxonomy

    def _format_logic_map(self, logic_map: LogicMap) -> str:
        """Format Logic Map for the prompt."""
        lines = []
        for rule in logic_map.rules:
            deps = f" (depends on: {', '.join(rule.dependencies)})" if rule.dependencies else ""
            lines.append(f"- {rule.rule_id}: {rule.text}{deps}")
            lines.append(f"  Category: {rule.category.value}")
            lines.append(f"  Condition: {rule.condition}")
            lines.append(f"  Action: {rule.action}")
        return "\n".join(lines)

    def _format_categories(self, categories: list[Category | str]) -> str:
        """Format categories for the prompt."""
        lines = []
        for cat in categories:
            if hasattr(cat, 'name'):
                lines.append(f"- {cat.name}: {cat.description}")
            else:
                lines.append(f"- {cat}")
        return "\n".join(lines)

    def _convert_to_taxonomy(self, output: TaxonomyOutput) -> SubCategoryTaxonomy:
        """Convert schema output to domain model."""
        sub_categories = []
        for sc_out in output.sub_categories:
            sub_category = SubCategory(
                id=sc_out.id,
                name=sc_out.name,
                description=sc_out.description,
                parent_category=sc_out.parent_category,
                related_rule_ids=sc_out.related_rule_ids,
                priority=sc_out.priority,
            )
            sub_categories.append(sub_category)

        return SubCategoryTaxonomy(
            sub_categories=sub_categories,
            reasoning=output.reasoning,
        )

    def _validate_taxonomy(
        self,
        taxonomy: SubCategoryTaxonomy,
        categories: list[Category | str],
    ) -> None:
        """Validate the extracted taxonomy."""
        # Handle both Category objects and strings
        category_names = {
            cat.name if hasattr(cat, 'name') else str(cat)
            for cat in categories
        }

        # Check that sub-categories reference valid parent categories
        for sc in taxonomy.sub_categories:
            if sc.parent_category not in category_names:
                # Try to match case-insensitively
                matched = False
                for cat_name in category_names:
                    if sc.parent_category.lower() == cat_name.lower():
                        sc.parent_category = cat_name
                        matched = True
                        break
                if not matched:
                    # Assign to first category if no match
                    first_cat = categories[0]
                    sc.parent_category = first_cat.name if hasattr(first_cat, 'name') else str(first_cat)

        # Ensure unique IDs
        seen_ids = set()
        for i, sc in enumerate(taxonomy.sub_categories):
            if sc.id in seen_ids:
                sc.id = f"SC{i+1:03d}"
            seen_ids.add(sc.id)
