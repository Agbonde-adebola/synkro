# Session API Tools Reference

Complete reference for all Session API methods. Use these to build HITL workflows.

---

## Session Management

### Create Session

```python
from synkro import Session
from synkro.models import Google

session = await Session.create(
    policy="Your policy text here...",
    session_id="my-session",  # Optional - auto-generated if not provided
    db_url=None,  # Optional - defaults to SQLite at ~/.synkro/sessions.db
)

# Configure models
session.model = Google.GEMINI_25_FLASH
session.grading_model = Google.GEMINI_25_PRO
```

### Load Session

```python
session = await Session.load_from_db("my-session")

# With custom database
session = await Session.load_from_db("my-session", db_url="postgresql://...")
```

### List Sessions

```python
sessions = await Session.list_sessions()

for s in sessions:
    print(f"{s['session_id']} - {s['updated_at']}")
```

### Delete Session

```python
deleted = await session.delete()
# Returns: True
```

### Undo Last Change

```python
result = await session.undo()
# Returns: "Restored: Before edit: add rule..."
```

### Get Status

```python
status = session.status()
# Returns: "Rules: ✓ (18) | Taxonomy: ✓ (12) | Scenarios: ✓ (30) | Traces: ✓ (30) | Verified: ✓ (28/30)"
```

---

## Pipeline Methods

### Extract Rules

```python
result = await session.extract_rules(session.policy)

# Access extracted rules
for rule in session.logic_map.rules:
    print(f"{rule.rule_id}: {rule.text}")
```

### Generate Taxonomy

```python
await session.generate_taxonomy()

# Access taxonomy
taxonomy = session.taxonomy
for cat in taxonomy.sub_categories:
    print(f"{cat.id}: {cat.name}")
```

### Generate Scenarios

```python
result = await session.generate_scenarios(count=30)

# Access scenarios
for scenario in session.scenarios:
    print(f"{scenario.description} [{scenario.scenario_type}]")
```

### Synthesize Traces

```python
result = await session.synthesize_traces(turns=3)

# Access traces
for trace in session.traces:
    print(f"Messages: {len(trace.messages)}")
```

### Verify Traces

```python
result = await session.verify_traces(max_iterations=3)

# Access verified traces
for trace in session.verified_traces:
    print(f"Passed: {trace.grade.passed}")
```

### Done (Complete Pipeline)

```python
# Run remaining steps + export
dataset = await session.done(output="training.jsonl")

# Or without file export
dataset = await session.done()
```

---

## HITL Refinement Methods

### Refine Rules

```python
# Add rules
summary = await session.refine_rules("add rule: meals capped at $75/day")

# Remove rules
summary = await session.refine_rules("remove R005")

# Merge rules
summary = await session.refine_rules("merge R002 and R003")

# Modify rules
summary = await session.refine_rules("change R001 threshold from $50 to $100")
```

### Refine Scenarios

```python
# Add scenarios
summary = await session.refine_scenarios("add 5 edge cases for meal limits")

# Remove scenarios
summary = await session.refine_scenarios("delete S3")

# Target specific rules
summary = await session.refine_scenarios("add more negative cases for R002")
```

### Refine Taxonomy

```python
# Add categories
summary = await session.refine_taxonomy("add category for travel expenses")

# Rename categories
summary = await session.refine_taxonomy("rename SC001 to Approval Thresholds")
```

### Refine Trace

```python
# Refine individual trace by index
result = await session.refine_trace(0, "mention the receipt requirement")
# Returns: "Trace #0 refined - now passes"
```

---

## Show / Inspection Methods

### Show Rules

```python
output = session.show_rules(limit=5)
# Returns formatted string:
# Rules (18 total):
#   R001: Expenses under $50: No approval required
#   R002: Expenses $50-$500: Manager approval required
#   ...
```

### Show Scenarios

```python
# Show all
output = session.show_scenarios(limit=10)

# Filter by type
output = session.show_scenarios(filter="edge_case", limit=5)
```

### Show Distribution

```python
output = session.show_distribution()
# Returns:
# Distribution:
#   positive       12 ( 40.0%) ████████
#   negative        9 ( 30.0%) ██████
#   edge_case       6 ( 20.0%) ████
#   irrelevant      3 ( 10.0%) ██
```

### Show Taxonomy

```python
output = session.show_taxonomy(limit=10)
# Returns formatted taxonomy tree
```

### Show Passed Traces

```python
output = session.show_passed()
# Returns:
# Pass Rate: 28/30 (93.3%) ██████████████████░░
```

### Show Failed Traces

```python
output = session.show_failed()
# Returns list of failed traces with issues
```

### Show Individual Trace

```python
output = session.show_trace(0)
# Returns:
# Trace #0:
# Scenario: I need to submit an expense...
# Type: positive
# Grade: ✓ PASSED
# Conversation:
#   [USER] I need to submit an expense...
#   [ASSISTANT] I can help you with that...
```

---

## Export Methods

### Get Dataset from Session

```python
# Get dataset object (from DB, no file needed)
dataset = session.to_dataset()

# Access traces
print(f"Traces: {len(dataset.traces)}")
print(f"Pass rate: {dataset.passing_rate:.1%}")

# Save to file
dataset.save("output.jsonl")

# Save with specific format
dataset.save("output.jsonl", format="chatml")
dataset.save("output.jsonl", format="langsmith")
dataset.save("output.jsonl", format="langfuse")
```

---

## Properties

| Property | Type | Description |
|----------|------|-------------|
| `session.session_id` | `str` | Session identifier |
| `session.policy` | `Policy` | Policy document |
| `session.logic_map` | `LogicMap` | Extracted rules |
| `session.taxonomy` | `Taxonomy` | Sub-category taxonomy |
| `session.scenarios` | `list[GoldenScenario]` | Generated scenarios |
| `session.distribution` | `dict[str, int]` | Scenario type counts |
| `session.traces` | `list[Trace]` | Generated traces |
| `session.verified_traces` | `list[Trace]` | Verified traces |
| `session.model` | `Model` | Generation model |
| `session.grading_model` | `Model` | Grading model |
| `session.dataset_type` | `DatasetType` | Output format type |

---

## Complete Example

```python
from synkro import Session
from synkro.models import Google

async def main():
    # Create session
    session = await Session.create(
        policy="Expenses over $50 need manager approval...",
        session_id="demo"
    )
    session.model = Google.GEMINI_25_FLASH
    session.grading_model = Google.GEMINI_25_PRO

    # Extract and refine rules
    await session.extract_rules(session.policy)
    print(session.show_rules())
    await session.refine_rules("add rule: receipts required over $25")

    # Generate taxonomy
    await session.generate_taxonomy()
    print(session.show_taxonomy())

    # Generate and refine scenarios
    await session.generate_scenarios(count=30)
    print(session.show_distribution())
    await session.refine_scenarios("add 5 edge cases")

    # Complete pipeline
    dataset = await session.done(output="training.jsonl")
    print(session.show_passed())

    # Check status
    print(session.status())

    # Later: reload and get dataset
    session = await Session.load_from_db("demo")
    dataset = session.to_dataset()
    print(f"Loaded {len(dataset.traces)} traces from DB")
```
