"""Test 20: Tests for session persistence edge cases."""

import json

from synkro.types.state import PipelinePhase, PipelineState


def test_pipeline_state_to_dict_serializable():
    """to_dict output is JSON serializable."""
    state = PipelineState()
    state.transition_to(PipelinePhase.EXTRACTION)
    state.update_progress(0.75, "Processing")
    data = state.to_dict()
    # Should not raise
    json_str = json.dumps(data)
    assert len(json_str) > 0


def test_pipeline_state_artifacts_tracking():
    """State tracks artifact presence in to_dict."""
    state = PipelineState()
    data = state.to_dict()
    assert data["artifacts"]["has_logic_map"] is False
    assert data["artifacts"]["scenario_count"] == 0
    assert data["artifacts"]["trace_count"] == 0


def test_pipeline_state_multiple_transitions():
    """State handles multiple phase transitions correctly."""
    state = PipelineState()
    phases = [
        PipelinePhase.PLANNING,
        PipelinePhase.EXTRACTION,
        PipelinePhase.SCENARIOS,
        PipelinePhase.TRACES,
        PipelinePhase.VERIFICATION,
        PipelinePhase.COMPLETE,
    ]
    for phase in phases:
        state.transition_to(phase)
    assert state.current_phase == PipelinePhase.COMPLETE
    assert state.total_progress == 1.0


def test_pipeline_state_error_preserves_history():
    """Error state preserves phase history."""
    state = PipelineState()
    state.transition_to(PipelinePhase.EXTRACTION)
    state.transition_to(PipelinePhase.SCENARIOS)
    state.set_error(RuntimeError("Failed"))
    assert len(state.phase_history) >= 2
    assert state.is_error is True


def test_pipeline_state_metrics_included():
    """State includes metrics in to_dict."""
    state = PipelineState()
    data = state.to_dict()
    assert "metrics" in data


def test_pipeline_state_error_message_from_exception():
    """set_error uses exception string if no message provided."""
    state = PipelineState()
    state.set_error(ValueError("Specific error message"))
    assert "Specific error message" in state.error_message


def test_pipeline_state_set_artifact_logic_map():
    """set_artifact sets logic_map."""
    state = PipelineState()
    # We'll use a mock object here
    mock_logic_map = {"rules": []}
    state.set_artifact(logic_map=mock_logic_map)
    assert state.logic_map is not None


def test_pipeline_state_set_artifact_scenarios():
    """set_artifact sets scenarios."""
    state = PipelineState()
    mock_scenarios = [{"description": "Test"}]
    state.set_artifact(scenarios=mock_scenarios)
    assert state.scenarios is not None
    assert len(state.scenarios) == 1


def test_pipeline_state_set_artifact_traces():
    """set_artifact sets traces."""
    state = PipelineState()
    mock_traces = [{"messages": []}, {"messages": []}]
    state.set_artifact(traces=mock_traces)
    assert state.traces is not None
    assert len(state.traces) == 2


def test_pipeline_state_format_status_without_message():
    """format_status works without phase message."""
    state = PipelineState()
    state.transition_to(PipelinePhase.TRACES)
    state.update_progress(0.5)
    state.phase_message = ""  # Clear message
    status = state.format_status()
    assert "50%" in status


def test_pipeline_state_is_error_false_by_default():
    """is_error is False by default."""
    state = PipelineState()
    assert state.is_error is False


def test_pipeline_phase_all_phases_have_weight():
    """All pipeline phases have defined weights."""
    for phase in PipelinePhase:
        assert phase.weight >= 0.0
        assert phase.weight <= 1.0


def test_pipeline_phase_all_phases_have_display_name():
    """All pipeline phases have display names."""
    for phase in PipelinePhase:
        assert len(phase.display_name) > 0
