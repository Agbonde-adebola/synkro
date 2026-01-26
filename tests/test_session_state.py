"""Test 19: Tests for session/pipeline state transitions."""

from synkro.types.state import PipelinePhase, PipelineState


def test_pipeline_phase_values():
    """PipelinePhase has expected phase values."""
    assert PipelinePhase.IDLE.value == "idle"
    assert PipelinePhase.EXTRACTION.value == "extraction"
    assert PipelinePhase.COMPLETE.value == "complete"
    assert PipelinePhase.ERROR.value == "error"


def test_pipeline_phase_display_names():
    """PipelinePhase has human-readable display names."""
    assert PipelinePhase.IDLE.display_name == "Idle"
    assert PipelinePhase.EXTRACTION.display_name == "Extracting Rules"
    assert PipelinePhase.VERIFICATION.display_name == "Verifying Traces"


def test_pipeline_phase_weights():
    """PipelinePhase has progress weights."""
    assert PipelinePhase.IDLE.weight == 0.0
    assert PipelinePhase.COMPLETE.weight == 1.0
    assert 0 < PipelinePhase.EXTRACTION.weight < 1


def test_pipeline_state_initial():
    """PipelineState starts in IDLE phase."""
    state = PipelineState()
    assert state.current_phase == PipelinePhase.IDLE
    assert state.phase_progress == 0.0
    assert state.is_running is False


def test_pipeline_state_transition():
    """transition_to changes current phase."""
    state = PipelineState()
    state.transition_to(PipelinePhase.EXTRACTION)
    assert state.current_phase == PipelinePhase.EXTRACTION
    assert state.phase_progress == 0.0  # Resets on transition


def test_pipeline_state_transition_with_message():
    """transition_to sets phase message."""
    state = PipelineState()
    state.transition_to(PipelinePhase.SCENARIOS, "Starting scenario generation")
    assert "Starting" in state.phase_message


def test_pipeline_state_update_progress():
    """update_progress sets progress within phase."""
    state = PipelineState()
    state.transition_to(PipelinePhase.TRACES)
    state.update_progress(0.5, "Halfway done")
    assert state.phase_progress == 0.5
    assert state.phase_message == "Halfway done"


def test_pipeline_state_progress_clamped():
    """update_progress clamps values to 0-1 range."""
    state = PipelineState()
    state.update_progress(1.5)
    assert state.phase_progress == 1.0
    state.update_progress(-0.5)
    assert state.phase_progress == 0.0


def test_pipeline_state_complete_phase():
    """complete_phase sets progress to 1.0."""
    state = PipelineState()
    state.transition_to(PipelinePhase.EXTRACTION)
    state.update_progress(0.5)
    state.complete_phase()
    assert state.phase_progress == 1.0


def test_pipeline_state_is_running():
    """is_running is True during active phases."""
    state = PipelineState()
    assert state.is_running is False  # IDLE

    state.transition_to(PipelinePhase.EXTRACTION)
    assert state.is_running is True

    state.transition_to(PipelinePhase.COMPLETE)
    assert state.is_running is False


def test_pipeline_state_is_complete():
    """is_complete is True for COMPLETE and ERROR phases."""
    state = PipelineState()
    assert state.is_complete is False

    state.transition_to(PipelinePhase.COMPLETE)
    assert state.is_complete is True

    state2 = PipelineState()
    state2.transition_to(PipelinePhase.ERROR)
    assert state2.is_complete is True


def test_pipeline_state_set_error():
    """set_error transitions to ERROR phase."""
    state = PipelineState()
    state.transition_to(PipelinePhase.EXTRACTION)
    state.set_error(ValueError("Test error"), "Something went wrong")
    assert state.is_error is True
    assert state.current_phase == PipelinePhase.ERROR
    assert "wrong" in state.error_message


def test_pipeline_state_total_progress():
    """total_progress reflects overall pipeline progress."""
    state = PipelineState()
    assert state.total_progress == 0.0

    state.transition_to(PipelinePhase.COMPLETE)
    assert state.total_progress == 1.0


def test_pipeline_state_format_status():
    """format_status returns readable status string."""
    state = PipelineState()
    state.transition_to(PipelinePhase.TRACES)
    state.update_progress(0.5, "Generating trace 5 of 10")
    status = state.format_status()
    assert "Traces" in status or "traces" in status.lower()
    assert "50%" in status


def test_pipeline_state_to_dict():
    """to_dict returns serializable dictionary."""
    state = PipelineState()
    state.transition_to(PipelinePhase.SCENARIOS)
    data = state.to_dict()
    assert data["current_phase"] == "scenarios"
    assert "total_progress" in data
    assert "is_running" in data
    assert "artifacts" in data


def test_pipeline_state_phase_history():
    """Phase transitions are recorded in history."""
    state = PipelineState()
    state.transition_to(PipelinePhase.EXTRACTION)
    state.transition_to(PipelinePhase.SCENARIOS)
    assert len(state.phase_history) >= 1
