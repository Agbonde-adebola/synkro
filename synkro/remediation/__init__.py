"""Policy violation detection and remediation.

Detect policy violations in traces and generate compliant golden traces
for finetuning.

Example:
    >>> from synkro import ingest, PolicyDetector, PolicyRefiner, ViolationStore
    >>>
    >>> # 1. Process policy
    >>> config = ingest("./policy.pdf")
    >>>
    >>> # 2. Detect violations
    >>> detector = PolicyDetector(config.logic_map)
    >>> violations = await detector.detect("./bad_traces.jsonl")
    >>>
    >>> # 3. Store violations (hashmap, LangSmith/Langfuse/Datadog compatible)
    >>> store = ViolationStore()
    >>> store.add_batch(violations)
    >>>
    >>> # 4. Generate golden traces
    >>> refiner = PolicyRefiner(config.logic_map)
    >>> golden = await refiner.refine(store, traces_per_violation=3)
    >>>
    >>> # 5. Save for finetuning
    >>> golden.save("./golden_traces.jsonl")
"""

from synkro.remediation.detector import PolicyDetector
from synkro.remediation.refiner import PolicyRefiner
from synkro.remediation.store import ViolationStore, ViolationStoreProtocol
from synkro.remediation.types import Violation

__all__ = [
    "PolicyDetector",
    "PolicyRefiner",
    "Violation",
    "ViolationStore",
    "ViolationStoreProtocol",
]
