"""Fast Logic Extractor - Parallel chunked extraction.

Optimized for large documents by:
1. Splitting document into chunks
2. Extracting rules from each chunk in parallel
3. Merging and deduplicating results
4. Building global dependency graph
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass

from synkro.llm.client import LLM
from synkro.models import Model, OpenAI
from synkro.types.logic_map import LogicMap, Rule, RuleCategory

# Simplified schema for faster extraction
CHUNK_EXTRACTION_PROMPT = """Extract rules from this SECTION of a policy document.

SECTION:
{chunk_text}

Extract ONLY rules found in this section. For each rule provide:
- text: The exact rule statement
- category: One of: constraint, permission, procedure, exception
- condition: The "if" part (when this rule applies)
- action: The "then" part (what happens)

Be thorough - extract EVERY distinct rule, requirement, or condition.
Look for: "must", "shall", "should", "can", "cannot", "if...then", "unless", "except"

Output as a JSON array of rules."""


@dataclass
class ChunkRule:
    """Rule extracted from a single chunk."""

    text: str
    category: str
    condition: str
    action: str
    chunk_index: int


class FastLogicExtractor:
    """
    Fast parallel Logic Map extractor for large documents.

    Splits document into chunks, extracts in parallel, then merges.
    """

    def __init__(
        self,
        llm: LLM | None = None,
        model: Model = OpenAI.GPT_4O_MINI,  # Faster model by default
        chunk_size: int = 3000,  # Sweet spot for chunk size
        chunk_overlap: int = 200,  # Small overlap for context
        max_concurrent: int = 10,  # Balanced concurrency (avoid rate limits)
        on_progress: callable | None = None,  # Progress callback(completed, total, rules_so_far)
    ):
        self.llm = llm or LLM(model=model, temperature=0.2)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.max_concurrent = max_concurrent
        self.on_progress = on_progress

    def get_chunk_count(self, policy_text: str) -> int:
        """Get the number of chunks that will be created for progress tracking."""
        return len(self._split_into_chunks(policy_text))

    async def extract(self, policy_text: str) -> LogicMap:
        """Extract Logic Map using parallel chunked extraction."""
        # Split into chunks
        chunks = self._split_into_chunks(policy_text)

        if len(chunks) == 1:
            # Small document - use standard extraction
            return await self._extract_single(policy_text)

        # Extract from all chunks in parallel with progress tracking
        semaphore = asyncio.Semaphore(self.max_concurrent)
        completed = 0
        total_rules = 0
        all_rules: list[ChunkRule] = []
        lock = asyncio.Lock()

        async def extract_chunk(i: int, chunk: str) -> list[ChunkRule]:
            nonlocal completed, total_rules
            async with semaphore:
                rules = await self._extract_from_chunk(chunk, i)
                async with lock:
                    completed += 1
                    total_rules += len(rules)
                    all_rules.extend(rules)
                    if self.on_progress:
                        self.on_progress(completed, len(chunks), total_rules)
                return rules

        tasks = [extract_chunk(i, chunk) for i, chunk in enumerate(chunks)]
        await asyncio.gather(*tasks, return_exceptions=True)

        # Deduplicate and merge
        unique_rules = self._deduplicate_rules(all_rules)

        # Convert to LogicMap with IDs and dependencies
        logic_map = self._build_logic_map(unique_rules, policy_text)

        return logic_map

    def _split_into_chunks(self, text: str) -> list[str]:
        """Split document into overlapping chunks at paragraph boundaries."""
        # Split by double newlines (paragraphs) or section headers
        paragraphs = re.split(r"\n\s*\n|\n(?=[A-Z][A-Z\s]+:?\n)", text)

        chunks = []
        current_chunk = ""

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            # If adding this paragraph exceeds chunk size, start new chunk
            if len(current_chunk) + len(para) > self.chunk_size and current_chunk:
                chunks.append(current_chunk)
                # Start new chunk with overlap from end of previous
                overlap_text = (
                    current_chunk[-self.chunk_overlap :]
                    if len(current_chunk) > self.chunk_overlap
                    else current_chunk
                )
                current_chunk = overlap_text + "\n\n" + para
            else:
                current_chunk = current_chunk + "\n\n" + para if current_chunk else para

        if current_chunk:
            chunks.append(current_chunk)

        return chunks if chunks else [text]

    async def _extract_from_chunk(self, chunk: str, chunk_index: int) -> list[ChunkRule]:
        """Extract rules from a single chunk."""
        from typing import Literal

        from pydantic import BaseModel, Field

        class RuleOutput(BaseModel):
            text: str
            category: Literal["constraint", "permission", "procedure", "exception"]
            condition: str = ""
            action: str = ""

        class ChunkRulesOutput(BaseModel):
            rules: list[RuleOutput] = Field(default_factory=list)

        prompt = CHUNK_EXTRACTION_PROMPT.format(chunk_text=chunk)

        try:
            result = await self.llm.generate_structured(prompt, ChunkRulesOutput)
            return [
                ChunkRule(
                    text=r.text,
                    category=r.category,
                    condition=r.condition,
                    action=r.action,
                    chunk_index=chunk_index,
                )
                for r in result.rules
            ]
        except Exception:
            return []

    async def _extract_single(self, policy_text: str) -> LogicMap:
        """Standard extraction for small documents."""
        from synkro.generation.logic_extractor import LogicExtractor

        extractor = LogicExtractor(llm=self.llm)
        return await extractor.extract(policy_text)

    def _deduplicate_rules(self, rules: list[ChunkRule]) -> list[ChunkRule]:
        """Remove duplicate rules - fast O(n) exact match."""
        seen_texts: set[str] = set()
        unique: list[ChunkRule] = []

        for rule in rules:
            # Normalize: lowercase, collapse whitespace, first 100 chars
            normalized = re.sub(r"\s+", " ", rule.text.lower().strip())[:100]

            if normalized not in seen_texts:
                seen_texts.add(normalized)
                unique.append(rule)

        return unique

    def _build_logic_map(self, rules: list[ChunkRule], policy_text: str) -> LogicMap:
        """Build LogicMap with IDs and inferred dependencies."""
        # Assign sequential IDs
        logic_rules: list[Rule] = []
        root_rules: list[str] = []

        for i, chunk_rule in enumerate(rules):
            rule_id = f"R{i+1:03d}"

            # Infer dependencies based on text references
            dependencies = self._infer_dependencies(chunk_rule, rules, i)

            rule = Rule(
                rule_id=rule_id,
                text=chunk_rule.text,
                condition=chunk_rule.condition,
                action=chunk_rule.action,
                dependencies=dependencies,
                category=RuleCategory(chunk_rule.category),
            )
            logic_rules.append(rule)

            if not dependencies:
                root_rules.append(rule_id)

        return LogicMap(rules=logic_rules, root_rules=root_rules)

    def _infer_dependencies(
        self, rule: ChunkRule, all_rules: list[ChunkRule], current_index: int
    ) -> list[str]:
        """Infer dependencies based on text analysis."""
        dependencies = []
        rule_text_lower = rule.text.lower()

        # Look for references to other rules
        dependency_patterns = [
            r"if .*(?:approved|granted|met)",
            r"after .*(?:completing|submitting)",
            r"subject to",
            r"in accordance with",
            r"as specified in",
            r"per (?:the )?(?:above|previous)",
        ]

        # Check if this rule seems to depend on earlier rules
        for i, other_rule in enumerate(all_rules):
            if i >= current_index:
                continue

            other_rule.text.lower()

            # Check for explicit references
            for pattern in dependency_patterns:
                if re.search(pattern, rule_text_lower):
                    # This rule has conditional language, might depend on earlier rules
                    # Check if the other rule's action relates to this rule's condition
                    if self._rules_related(other_rule, rule):
                        dependencies.append(f"R{i+1:03d}")
                        break

        return dependencies[:3]  # Limit to avoid over-connecting

    def _rules_related(self, rule_a: ChunkRule, rule_b: ChunkRule) -> bool:
        """Check if two rules are semantically related."""
        # Simple heuristic: check for shared key terms
        key_terms_a = set(re.findall(r"\b[a-z]{4,}\b", rule_a.text.lower()))
        key_terms_b = set(re.findall(r"\b[a-z]{4,}\b", rule_b.text.lower()))

        # Remove common words
        common_words = {
            "must",
            "shall",
            "should",
            "will",
            "have",
            "been",
            "with",
            "from",
            "that",
            "this",
            "they",
            "their",
        }
        key_terms_a -= common_words
        key_terms_b -= common_words

        overlap = len(key_terms_a & key_terms_b)
        return overlap >= 2


__all__ = ["FastLogicExtractor"]
