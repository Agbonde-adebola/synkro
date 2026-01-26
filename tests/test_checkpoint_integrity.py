"""Test 8: Tests for checkpoint save/load integrity."""

import tempfile
from pathlib import Path

from synkro.core.checkpoint import CheckpointData, CheckpointManager, hash_policy


def test_checkpoint_manager_creates_directory():
    """CheckpointManager creates directory if it doesn't exist."""
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_dir = Path(tmpdir) / "nested" / "checkpoints"
        CheckpointManager(checkpoint_dir)  # Constructor creates directory
        assert checkpoint_dir.exists()


def test_checkpoint_has_checkpoint_false_initially():
    """has_checkpoint returns False when no checkpoint exists."""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = CheckpointManager(tmpdir)
        assert not manager.has_checkpoint()


def test_checkpoint_load_returns_none_when_empty():
    """load() returns None when no checkpoint exists."""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = CheckpointManager(tmpdir)
        assert manager.load() is None


def test_checkpoint_stage_starts_at_start():
    """Initial stage is 'start' before any data is saved."""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = CheckpointManager(tmpdir)
        assert manager.stage == "start"


def test_checkpoint_data_round_trip():
    """CheckpointData can be serialized and deserialized."""
    data = CheckpointData(
        policy_hash="abc123",
        target_traces=50,
        dataset_type="messages",
    )
    json_str = data.model_dump_json()
    restored = CheckpointData.model_validate_json(json_str)
    assert restored.policy_hash == "abc123"
    assert restored.target_traces == 50
    assert restored.dataset_type == "messages"


def test_checkpoint_matches_config():
    """matches_config returns True when config matches."""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = CheckpointManager(tmpdir)
        # Create checkpoint with specific config
        data = manager._load_or_create()
        data.policy_hash = "test_hash"
        data.target_traces = 100
        data.dataset_type = "messages"
        manager._save()

        # Verify matching config
        assert manager.matches_config("test_hash", 100, "messages")
        assert not manager.matches_config("other_hash", 100, "messages")
        assert not manager.matches_config("test_hash", 50, "messages")
        assert not manager.matches_config("test_hash", 100, "qa")


def test_checkpoint_clear():
    """clear() removes checkpoint file and resets state."""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = CheckpointManager(tmpdir)
        # Create some data
        data = manager._load_or_create()
        data.policy_hash = "test"
        manager._save()
        assert manager.has_checkpoint()

        # Clear and verify
        manager.clear()
        assert not manager.has_checkpoint()
        assert manager.load() is None


def test_hash_policy_consistent():
    """hash_policy returns consistent hash for same input."""
    text = "This is a test policy document"
    hash1 = hash_policy(text)
    hash2 = hash_policy(text)
    assert hash1 == hash2
    assert len(hash1) == 16  # HASH_LENGTH


def test_hash_policy_different_for_different_text():
    """hash_policy returns different hash for different input."""
    hash1 = hash_policy("Policy A")
    hash2 = hash_policy("Policy B")
    assert hash1 != hash2


def test_checkpoint_summary_format():
    """summary() returns formatted status string."""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = CheckpointManager(tmpdir)
        summary = manager.summary()
        assert "Stage:" in summary
        assert "Target traces:" in summary
        assert "Logic Map:" in summary


def test_checkpoint_pending_indices():
    """get_pending_scenario_indices returns incomplete indices."""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = CheckpointManager(tmpdir)
        data = manager._load_or_create()
        data.completed_scenario_indices = [0, 2, 4]
        manager._save()

        pending = manager.get_pending_scenario_indices(total=5)
        assert pending == [1, 3]
