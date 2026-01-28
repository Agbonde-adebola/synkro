"""Tests for the streaming API and new types.

These tests cover:
1. Event types and serialization
2. Metrics tracking
3. Result types
4. Session class functionality

Integration tests (requiring API keys) are marked with @pytest.mark.integration
"""

import os

import pytest

_HAS_LLM_API_KEY = any(
    os.getenv(k)
    for k in (
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
        "GOOGLE_API_KEY",
        "GEMINI_API_KEY",
    )
)

# =============================================================================
# EVENT TYPE TESTS (No API calls needed)
# =============================================================================


def test_progress_event():
    """Test ProgressEvent creation and serialization."""
    from synkro.types.events import ProgressEvent

    event = ProgressEvent(
        phase="extraction",
        message="Processing rules...",
        progress=0.5,
        completed=5,
        total=10,
    )

    assert event.type == "progress"
    assert event.progress == 0.5
    assert event.completed == 5
    assert event.total == 10

    # Test serialization
    data = event.to_dict()
    assert data["type"] == "progress"
    assert data["phase"] == "extraction"
    assert data["progress"] == 0.5


def test_rule_found_event():
    """Test RuleFoundEvent creation."""
    from synkro.types.events import RuleFoundEvent

    event = RuleFoundEvent(index=0)
    assert event.type == "rule_found"
    assert event.index == 0


def test_complete_event():
    """Test CompleteEvent creation."""
    from synkro.types.events import CompleteEvent

    event = CompleteEvent(result={"test": "data"})
    assert event.type == "complete"
    assert event.result == {"test": "data"}


def test_error_event():
    """Test ErrorEvent creation."""
    from synkro.types.events import ErrorEvent

    error = ValueError("Test error")
    event = ErrorEvent(error=error, message="Test error", phase="extraction")

    assert event.type == "error"
    assert event.message == "Test error"
    assert event.phase == "extraction"


# =============================================================================
# METRICS TESTS (No API calls needed)
# =============================================================================


def test_phase_metrics():
    """Test PhaseMetrics tracking."""
    from synkro.types.metrics import PhaseMetrics

    metrics = PhaseMetrics(phase="extraction")
    metrics.start()
    metrics.add_call(0.001, "gpt-4o")
    metrics.add_call(0.002, "gpt-4o")
    metrics.complete()

    assert metrics.calls == 2
    assert metrics.cost == 0.003
    assert metrics.model == "gpt-4o"
    assert metrics.duration_seconds >= 0


def test_metrics_aggregation():
    """Test Metrics aggregation across phases."""
    from synkro.types.metrics import Metrics

    metrics = Metrics()

    # Simulate extraction phase
    metrics.start_phase("extraction", model="gpt-4o")
    metrics.add_call("extraction", 0.01)
    metrics.add_call("extraction", 0.02)
    metrics.end_phase("extraction", cost=0.03, calls=2)

    # Simulate scenarios phase
    metrics.start_phase("scenarios", model="gpt-4o-mini")
    metrics.add_call("scenarios", 0.005)
    metrics.end_phase("scenarios", cost=0.005, calls=1)

    assert metrics.total_cost == pytest.approx(0.035)
    assert metrics.total_calls == 3
    assert metrics.breakdown == {"extraction": 0.03, "scenarios": 0.005}


def test_metrics_serialization():
    """Test Metrics serialization."""
    from synkro.types.metrics import Metrics

    metrics = Metrics()
    metrics.start_phase("extraction")
    metrics.end_phase("extraction", cost=0.05, calls=3)

    # Serialize
    data = metrics.to_dict()
    assert "phases" in data
    assert "extraction" in data["phases"]
    assert data["total_cost"] == 0.05

    # Deserialize
    restored = Metrics.from_dict(data)
    assert restored.total_cost == 0.05
    assert restored.total_calls == 3


def test_metrics_format_summary():
    """Test Metrics formatting."""
    from synkro.types.metrics import Metrics

    metrics = Metrics()
    metrics.start_phase("extraction")
    metrics.end_phase("extraction", cost=0.42, calls=91)

    summary = metrics.format_summary()
    assert "$0.42" in summary
    assert "91" in summary


# =============================================================================
# RESULT TYPE TESTS (No API calls needed)
# =============================================================================


def test_extraction_result():
    """Test ExtractionResult creation."""
    from synkro.types.logic_map import LogicMap, Rule, RuleCategory
    from synkro.types.metrics import PhaseMetrics
    from synkro.types.results import ExtractionResult

    # Create a simple logic map
    rules = [
        Rule(
            rule_id="R001",
            text="All expenses require approval",
            condition="expense submitted",
            action="require approval",
            dependencies=[],
            category=RuleCategory.CONSTRAINT,
        )
    ]
    logic_map = LogicMap(rules=rules, root_rules=["R001"])

    metrics = PhaseMetrics(phase="extraction")
    result = ExtractionResult(logic_map=logic_map, metrics=metrics)

    assert len(result.logic_map.rules) == 1
    assert result.format_summary() == "Extracted 1 rules (1 root, 0 dependent)"


def test_scenarios_result():
    """Test ScenariosResult creation."""
    from synkro.types.logic_map import GoldenScenario, LogicMap, ScenarioType
    from synkro.types.metrics import PhaseMetrics
    from synkro.types.results import ScenariosResult

    logic_map = LogicMap(rules=[], root_rules=[])
    scenarios = [
        GoldenScenario(
            description="Test scenario",
            context="Test context",
            expected_outcome="Approved",
            target_rule_ids=["R001"],
            scenario_type=ScenarioType.POSITIVE,
        )
    ]
    distribution = {"positive": 1}

    result = ScenariosResult(
        scenarios=scenarios,
        logic_map=logic_map,
        distribution=distribution,
        metrics=PhaseMetrics(phase="scenarios"),
    )

    assert len(result) == 1
    assert result.distribution["positive"] == 1
    assert "positive: 1" in result.format_summary()


# =============================================================================
# PIPELINE STATE TESTS (No API calls needed)
# =============================================================================


def test_pipeline_phase_enum():
    """Test PipelinePhase enum."""
    from synkro.types.state import PipelinePhase

    assert PipelinePhase.EXTRACTION.value == "extraction"
    assert PipelinePhase.EXTRACTION.display_name == "Extracting Rules"
    assert PipelinePhase.COMPLETE.weight == 1.0


def test_pipeline_state_transitions():
    """Test PipelineState transitions."""
    from synkro.types.state import PipelinePhase, PipelineState

    state = PipelineState()
    assert state.current_phase == PipelinePhase.IDLE
    assert not state.is_running

    state.transition_to(PipelinePhase.EXTRACTION, "Extracting rules...")
    assert state.current_phase == PipelinePhase.EXTRACTION
    assert state.is_running
    assert state.phase_message == "Extracting rules..."

    state.update_progress(0.5, "Halfway done")
    assert state.phase_progress == 0.5
    assert state.total_progress > 0

    state.complete_phase()
    assert state.phase_progress == 1.0


def test_pipeline_state_serialization():
    """Test PipelineState serialization."""
    from synkro.types.state import PipelinePhase, PipelineState

    state = PipelineState()
    state.transition_to(PipelinePhase.EXTRACTION)
    state.update_progress(0.5)

    data = state.to_dict()
    assert data["current_phase"] == "extraction"
    assert data["phase_progress"] == 0.5
    assert data["is_running"] is True


# =============================================================================
# SESSION TESTS (No API calls needed for basic functionality)
# =============================================================================


def test_session_creation():
    """Test Session creation."""
    from synkro.session import Session

    session = Session()
    assert session.policy is None
    assert session.logic_map is None
    assert session.scenarios is None


def test_session_format_status():
    """Test Session status formatting."""
    from synkro.session import Session

    session = Session()
    status = session.format_status()

    assert "Session Status:" in status
    assert "Not set" in status or "Not extracted" in status


def test_session_persistence(tmp_path):
    """Test Session save/load."""
    from synkro.core.policy import Policy
    from synkro.session import Session

    # Create session with some state
    session = Session()
    session.policy = Policy(text="Test policy")
    session.model = "gpt-4o-mini"

    # Save
    path = tmp_path / "test_session.json"
    session.save(path)

    # Load
    restored = Session.load(path)
    assert restored.policy.text == "Test policy"
    assert restored.model == "gpt-4o-mini"


# =============================================================================
# TOOL DEFINITIONS TESTS (No API calls needed)
# =============================================================================


def test_tool_definitions_structure():
    """Test TOOL_DEFINITIONS structure."""
    from synkro.tools import TOOL_DEFINITIONS

    assert isinstance(TOOL_DEFINITIONS, list)
    assert len(TOOL_DEFINITIONS) > 0

    # Check first tool has correct structure
    tool = TOOL_DEFINITIONS[0]
    assert tool["type"] == "function"
    assert "function" in tool
    assert "name" in tool["function"]
    assert "description" in tool["function"]
    assert "parameters" in tool["function"]


def test_tool_definitions_names():
    """Test expected tool names exist."""
    from synkro.tools import get_tool_names

    names = get_tool_names()

    expected = [
        "extract_rules",
        "edit_rules",
        "generate_scenarios",
        "edit_scenarios",
        "synthesize_traces",
        "verify_traces",
    ]

    for name in expected:
        assert name in names, f"Missing tool: {name}"


def test_get_tool_by_name():
    """Test getting tool by name."""
    from synkro.tools import get_tool_by_name

    tool = get_tool_by_name("extract_rules")
    assert tool is not None
    assert tool["function"]["name"] == "extract_rules"

    # Test non-existent tool
    assert get_tool_by_name("nonexistent") is None


# =============================================================================
# INTEGRATION TESTS (Require API keys)
# =============================================================================


@pytest.mark.integration
@pytest.mark.skipif(
    not _HAS_LLM_API_KEY,
    reason="Integration test requires an LLM API key (ANTHROPIC_API_KEY / OPENAI_API_KEY / GOOGLE_API_KEY / GEMINI_API_KEY)",
)
@pytest.mark.asyncio
async def test_extract_rules_stream():
    """Test streaming rule extraction.

    Run with: pytest -m integration tests/test_streaming_api.py
    Requires one of: ANTHROPIC_API_KEY, OPENAI_API_KEY, or GOOGLE_API_KEY.
    The API will auto-detect which provider is available.
    """
    from synkro.api import extract_rules_stream

    policy = "All expenses over $50 require manager approval. Expenses over $100 require director approval."

    events = []
    async for event in extract_rules_stream(policy):
        events.append(event)
        print(f"Event: {event.type}")

    # Should have progress, rule_found, and complete events
    event_types = [e.type for e in events]
    assert "progress" in event_types
    assert "complete" in event_types

    # Last event should be complete with result
    complete_event = events[-1]
    assert complete_event.type == "complete"
    assert complete_event.result is not None
    assert len(complete_event.result.logic_map.rules) > 0


@pytest.mark.integration
@pytest.mark.skipif(
    not _HAS_LLM_API_KEY,
    reason="Integration test requires an LLM API key (ANTHROPIC_API_KEY / OPENAI_API_KEY / GOOGLE_API_KEY / GEMINI_API_KEY)",
)
@pytest.mark.asyncio
async def test_session_full_workflow():
    """Test Session full workflow.

    Run with: pytest -m integration tests/test_streaming_api.py
    Requires one of: ANTHROPIC_API_KEY, OPENAI_API_KEY, or GOOGLE_API_KEY.
    """
    from synkro.session import Session

    session = Session()

    # Extract rules
    await session.extract_rules("Expenses over $50 require approval.")
    assert session.logic_map is not None
    assert len(session.logic_map.rules) > 0

    # Generate scenarios
    await session.generate_scenarios(count=3)
    assert session.scenarios is not None
    assert len(session.scenarios) > 0

    # Check metrics accumulated
    assert session.metrics.total_cost > 0
    assert session.metrics.total_calls > 0

    print(f"Session status:\n{session.format_status()}")


@pytest.mark.integration
@pytest.mark.skipif(
    not _HAS_LLM_API_KEY,
    reason="Integration test requires an LLM API key (ANTHROPIC_API_KEY / OPENAI_API_KEY / GOOGLE_API_KEY / GEMINI_API_KEY)",
)
@pytest.mark.asyncio
async def test_streaming_scenario_generation():
    """Test streaming scenario generation."""
    from synkro.api import extract_rules_async, generate_scenarios_stream

    # First extract rules
    policy = "All expenses over $50 require approval."
    extraction = await extract_rules_async(policy)

    # Then stream scenario generation
    events = []
    async for event in generate_scenarios_stream(policy, logic_map=extraction.logic_map, count=3):
        events.append(event)
        if event.type == "scenario_generated":
            print(f"Generated scenario: {event.scenario.description[:50]}...")

    event_types = [e.type for e in events]
    assert "scenario_generated" in event_types
    assert "complete" in event_types


# =============================================================================
# MOCK-BASED TESTS (No API calls, uses mocks)
# =============================================================================


@pytest.mark.asyncio
async def test_extract_rules_stream_events_structure():
    """Test that streaming yields proper event structure (mocked)."""
    from synkro.types.events import CompleteEvent, ErrorEvent, ProgressEvent

    # Test event creation patterns used in streaming
    progress = ProgressEvent(
        phase="extraction",
        message="Analyzing policy...",
        progress=0.5,
        completed=2,
        total=4,
    )

    assert progress.type == "progress"
    assert 0 <= progress.progress <= 1

    complete = CompleteEvent(result={"logic_map": {}})
    assert complete.type == "complete"

    error = ErrorEvent(error=Exception("test"), phase="extraction")
    assert error.type == "error"
