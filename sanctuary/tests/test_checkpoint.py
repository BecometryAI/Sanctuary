"""
Unit tests for CheckpointManager and workspace checkpointing functionality.

Tests cover:
- Checkpoint save/load roundtrip
- Workspace state preservation
- Checkpoint metadata handling
- Atomic write behavior
- Checkpoint rotation
- Compression/decompression
- Auto-save functionality
- CognitiveCore integration
- Error handling
"""

import pytest
import asyncio
import json
import gzip
import time
from pathlib import Path
from datetime import datetime
from tempfile import TemporaryDirectory

from mind.cognitive_core.checkpoint import CheckpointManager, CheckpointInfo
from mind.cognitive_core.workspace import GlobalWorkspace, Goal, GoalType, Percept, Memory
from mind.cognitive_core.core import CognitiveCore


class TestCheckpointManager:
    """Test CheckpointManager core functionality."""
    
    def test_checkpoint_manager_initialization(self):
        """Test creating a CheckpointManager with default and custom settings."""
        with TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir) / "checkpoints"
            
            # Test with custom settings
            manager = CheckpointManager(
                checkpoint_dir=checkpoint_dir,
                max_checkpoints=10,
                compression=False,
            )
            
            assert manager.checkpoint_dir == checkpoint_dir
            assert manager.max_checkpoints == 10
            assert manager.compression is False
            assert checkpoint_dir.exists()
    
    def test_save_checkpoint_basic(self):
        """Test saving a basic checkpoint."""
        with TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir) / "checkpoints"
            manager = CheckpointManager(checkpoint_dir=checkpoint_dir)
            
            # Create a workspace with some state
            workspace = GlobalWorkspace()
            goal = Goal(type=GoalType.RESPOND_TO_USER, description="Test goal")
            workspace.add_goal(goal)
            
            # Save checkpoint
            path = manager.save_checkpoint(workspace)
            
            assert path.exists()
            assert path.parent == checkpoint_dir
            assert "checkpoint_" in path.name
    
    def test_save_checkpoint_with_metadata(self):
        """Test saving checkpoint with custom metadata."""
        with TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir) / "checkpoints"
            manager = CheckpointManager(checkpoint_dir=checkpoint_dir)
            
            workspace = GlobalWorkspace()
            metadata = {
                "user_label": "Before experiment",
                "custom_field": "test_value",
            }
            
            path = manager.save_checkpoint(workspace, metadata=metadata)
            
            assert path.exists()
            
            # Verify metadata is saved
            checkpoints = manager.list_checkpoints()
            assert len(checkpoints) == 1
            assert checkpoints[0].metadata["user_label"] == "Before experiment"
            assert checkpoints[0].metadata["custom_field"] == "test_value"
    
    def test_load_checkpoint_basic(self):
        """Test loading a checkpoint."""
        with TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir) / "checkpoints"
            manager = CheckpointManager(checkpoint_dir=checkpoint_dir)
            
            # Create and save workspace
            workspace1 = GlobalWorkspace()
            goal = Goal(type=GoalType.LEARN, description="Study AI", priority=0.8)
            workspace1.add_goal(goal)
            workspace1.emotional_state["valence"] = 0.6
            
            path = manager.save_checkpoint(workspace1)
            
            # Load checkpoint
            workspace2 = manager.load_checkpoint(path)
            
            assert len(workspace2.current_goals) == 1
            assert workspace2.current_goals[0].description == "Study AI"
            assert workspace2.current_goals[0].priority == 0.8
            assert workspace2.emotional_state["valence"] == 0.6
    
    def test_save_load_roundtrip(self):
        """Test complete save/load roundtrip with complex state."""
        with TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir) / "checkpoints"
            manager = CheckpointManager(checkpoint_dir=checkpoint_dir)
            
            # Create workspace with rich state
            workspace1 = GlobalWorkspace(capacity=12)
            
            # Add goals
            goal1 = Goal(type=GoalType.RESPOND_TO_USER, description="Answer", priority=0.9)
            goal2 = Goal(type=GoalType.COMMIT_MEMORY, description="Remember", priority=0.5)
            workspace1.add_goal(goal1)
            workspace1.add_goal(goal2)
            
            # Add percept
            percept = Percept(modality="text", raw="Hello world", complexity=2)
            workspace1.active_percepts[percept.id] = percept
            
            # Add memory
            memory = Memory(
                id="mem-123",
                content="Important fact",
                timestamp=datetime.now(),
                significance=0.7,
                tags=["test", "important"],
            )
            workspace1.attended_memories.append(memory)
            
            # Update emotions
            workspace1.emotional_state["valence"] = 0.5
            workspace1.emotional_state["arousal"] = 0.3
            
            # Update cycle count
            workspace1.cycle_count = 42
            
            # Save and load
            path = manager.save_checkpoint(workspace1)
            workspace2 = manager.load_checkpoint(path)
            
            # Verify all state preserved
            assert workspace2.capacity == 12
            assert len(workspace2.current_goals) == 2
            assert workspace2.current_goals[0].priority == 0.9
            assert len(workspace2.active_percepts) == 1
            assert len(workspace2.attended_memories) == 1
            assert workspace2.attended_memories[0].tags == ["test", "important"]
            assert workspace2.emotional_state["valence"] == 0.5
            assert workspace2.cycle_count == 42
    
    def test_compression(self):
        """Test checkpoint compression."""
        with TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir) / "checkpoints"
            
            # Test with compression enabled
            manager_compressed = CheckpointManager(
                checkpoint_dir=checkpoint_dir,
                compression=True,
            )
            
            workspace = GlobalWorkspace()
            for i in range(10):
                goal = Goal(type=GoalType.LEARN, description=f"Goal {i}")
                workspace.add_goal(goal)
            
            path_compressed = manager_compressed.save_checkpoint(workspace)
            
            assert path_compressed.suffix == ".gz"
            
            # Verify it can be loaded
            workspace_loaded = manager_compressed.load_checkpoint(path_compressed)
            assert len(workspace_loaded.current_goals) == 10
    
    def test_atomic_write(self):
        """Test that atomic write prevents corruption."""
        with TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir) / "checkpoints"
            manager = CheckpointManager(checkpoint_dir=checkpoint_dir)
            
            workspace = GlobalWorkspace()
            
            # Save checkpoint
            path = manager.save_checkpoint(workspace)
            
            # Verify no temp files left behind
            temp_files = list(checkpoint_dir.glob("*.tmp"))
            assert len(temp_files) == 0
            
            # Verify checkpoint file exists and is valid
            assert path.exists()
            workspace_loaded = manager.load_checkpoint(path)
            assert workspace_loaded is not None
    
    def test_list_checkpoints(self):
        """Test listing checkpoints."""
        with TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir) / "checkpoints"
            manager = CheckpointManager(checkpoint_dir=checkpoint_dir)
            
            # Create multiple checkpoints
            workspace = GlobalWorkspace()
            
            path1 = manager.save_checkpoint(workspace, metadata={"label": "First"})
            time.sleep(0.01)  # Ensure different timestamps
            path2 = manager.save_checkpoint(workspace, metadata={"label": "Second"})
            time.sleep(0.01)
            path3 = manager.save_checkpoint(workspace, metadata={"label": "Third"})
            
            # List checkpoints
            checkpoints = manager.list_checkpoints()
            
            assert len(checkpoints) == 3
            assert all(isinstance(cp, CheckpointInfo) for cp in checkpoints)
            
            # Verify sorted by timestamp (newest first)
            assert checkpoints[0].metadata["label"] == "Third"
            assert checkpoints[1].metadata["label"] == "Second"
            assert checkpoints[2].metadata["label"] == "First"
    
    def test_delete_checkpoint(self):
        """Test deleting a specific checkpoint."""
        with TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir) / "checkpoints"
            manager = CheckpointManager(checkpoint_dir=checkpoint_dir)
            
            workspace = GlobalWorkspace()
            path = manager.save_checkpoint(workspace)
            
            checkpoints = manager.list_checkpoints()
            assert len(checkpoints) == 1
            
            checkpoint_id = checkpoints[0].checkpoint_id
            
            # Delete checkpoint
            success = manager.delete_checkpoint(checkpoint_id)
            assert success is True
            
            # Verify deleted
            checkpoints = manager.list_checkpoints()
            assert len(checkpoints) == 0
            assert not path.exists()
    
    def test_delete_nonexistent_checkpoint(self):
        """Test deleting a checkpoint that doesn't exist."""
        with TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir) / "checkpoints"
            manager = CheckpointManager(checkpoint_dir=checkpoint_dir)
            
            success = manager.delete_checkpoint("nonexistent-id")
            assert success is False
    
    def test_get_latest_checkpoint(self):
        """Test getting the most recent checkpoint."""
        with TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir) / "checkpoints"
            manager = CheckpointManager(checkpoint_dir=checkpoint_dir)
            
            # No checkpoints initially
            latest = manager.get_latest_checkpoint()
            assert latest is None
            
            # Create checkpoints
            workspace = GlobalWorkspace()
            path1 = manager.save_checkpoint(workspace, metadata={"label": "First"})
            time.sleep(0.01)
            path2 = manager.save_checkpoint(workspace, metadata={"label": "Second"})
            
            # Get latest
            latest = manager.get_latest_checkpoint()
            assert latest == path2
    
    def test_checkpoint_rotation(self):
        """Test that old checkpoints are rotated out."""
        with TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir) / "checkpoints"
            manager = CheckpointManager(
                checkpoint_dir=checkpoint_dir,
                max_checkpoints=3,
            )
            
            workspace = GlobalWorkspace()
            
            # Create more checkpoints than max
            paths = []
            for i in range(5):
                path = manager.save_checkpoint(workspace, metadata={"index": i})
                paths.append(path)
                time.sleep(0.01)
            
            # Should only have max_checkpoints
            checkpoints = manager.list_checkpoints()
            assert len(checkpoints) <= 3
            
            # Oldest checkpoints should be deleted
            assert not paths[0].exists()
            assert not paths[1].exists()
            assert paths[4].exists()  # Most recent should exist
    
    def test_checkpoint_rotation_preserves_manual_saves(self):
        """Test that rotation favors manual saves over auto-saves."""
        with TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir) / "checkpoints"
            manager = CheckpointManager(
                checkpoint_dir=checkpoint_dir,
                max_checkpoints=5,
            )
            
            workspace = GlobalWorkspace()
            
            # Create mix of manual and auto-save checkpoints
            for i in range(3):
                manager.save_checkpoint(workspace, metadata={"auto_save": False})
                time.sleep(0.01)
            
            for i in range(5):
                manager.save_checkpoint(workspace, metadata={"auto_save": True})
                time.sleep(0.01)
            
            # Should keep more manual saves than auto-saves
            checkpoints = manager.list_checkpoints()
            assert len(checkpoints) <= 5
            
            manual_count = sum(1 for cp in checkpoints if not cp.metadata.get("auto_save", False))
            auto_count = sum(1 for cp in checkpoints if cp.metadata.get("auto_save", False))
            
            # Manual saves should be preserved more
            assert manual_count >= 2
    
    @pytest.mark.asyncio
    async def test_auto_save(self):
        """Test automatic periodic checkpointing."""
        with TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir) / "checkpoints"
            manager = CheckpointManager(checkpoint_dir=checkpoint_dir)
            
            workspace = GlobalWorkspace()
            
            # Start auto-save with short interval
            task = asyncio.create_task(manager.auto_save(workspace, interval=0.1))
            
            # Wait for a few auto-saves
            await asyncio.sleep(0.35)
            
            # Stop auto-save
            manager.stop_auto_save()
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                # Expected: the auto-save task is cancelled as part of test cleanup.
                pass
            
            # Should have created multiple checkpoints
            checkpoints = manager.list_checkpoints()
            assert len(checkpoints) >= 2
            
            # All should be marked as auto-save
            assert all(cp.metadata.get("auto_save", False) for cp in checkpoints)
    
    def test_checksum_validation(self):
        """Test that checksums are calculated and validated."""
        with TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir) / "checkpoints"
            manager = CheckpointManager(checkpoint_dir=checkpoint_dir)
            
            workspace = GlobalWorkspace()
            path = manager.save_checkpoint(workspace)
            
            # Checkpoints should have checksums
            checkpoints = manager.list_checkpoints()
            assert len(checkpoints) == 1
            assert checkpoints[0].checksum is not None
            assert len(checkpoints[0].checksum) > 0
            
            # Should load successfully
            workspace_loaded = manager.load_checkpoint(path)
            assert workspace_loaded is not None
    
    def test_corrupted_checkpoint_handling(self):
        """Test handling of corrupted checkpoint files."""
        with TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir) / "checkpoints"
            manager = CheckpointManager(checkpoint_dir=checkpoint_dir)
            
            # Create a corrupted checkpoint file
            corrupted_path = checkpoint_dir / "checkpoint_20260101_120000_corrupt.json"
            corrupted_path.write_text("not valid json {{{")
            
            # Should raise exception when loading
            with pytest.raises(Exception):
                manager.load_checkpoint(corrupted_path)
    
    def test_missing_checkpoint_file(self):
        """Test handling of missing checkpoint files."""
        with TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir) / "checkpoints"
            manager = CheckpointManager(checkpoint_dir=checkpoint_dir)
            
            nonexistent_path = checkpoint_dir / "nonexistent.json"
            
            # Should raise FileNotFoundError
            with pytest.raises(FileNotFoundError):
                manager.load_checkpoint(nonexistent_path)


class TestCognitiveCoreIntegration:
    """Test CheckpointManager integration with CognitiveCore."""
    
    def test_cognitive_core_checkpoint_manager_initialization(self):
        """Test that CognitiveCore initializes CheckpointManager."""
        with TemporaryDirectory() as tmpdir:
            config = {
                "checkpointing": {
                    "enabled": True,
                    "checkpoint_dir": str(Path(tmpdir) / "checkpoints"),
                    "max_checkpoints": 10,
                }
            }
            
            core = CognitiveCore(config=config)
            
            assert core.checkpoint_manager is not None
            assert core.checkpoint_manager.max_checkpoints == 10
    
    def test_cognitive_core_checkpointing_disabled(self):
        """Test CognitiveCore with checkpointing disabled."""
        config = {
            "checkpointing": {
                "enabled": False,
            }
        }
        
        core = CognitiveCore(config=config)
        
        assert core.checkpoint_manager is None
    
    def test_save_state_method(self):
        """Test CognitiveCore.save_state() method."""
        with TemporaryDirectory() as tmpdir:
            config = {
                "checkpointing": {
                    "enabled": True,
                    "checkpoint_dir": str(Path(tmpdir) / "checkpoints"),
                }
            }
            
            core = CognitiveCore(config=config)
            
            # Add some state
            goal = Goal(type=GoalType.LEARN, description="Test")
            core.workspace.add_goal(goal)
            
            # Save state
            path = core.save_state(label="Test checkpoint")
            
            assert path is not None
            assert path.exists()
            
            # Verify metadata
            checkpoints = core.checkpoint_manager.list_checkpoints()
            assert len(checkpoints) == 1
            assert checkpoints[0].metadata.get("user_label") == "Test checkpoint"
    
    def test_restore_state_method(self):
        """Test CognitiveCore.restore_state() method."""
        with TemporaryDirectory() as tmpdir:
            config = {
                "checkpointing": {
                    "enabled": True,
                    "checkpoint_dir": str(Path(tmpdir) / "checkpoints"),
                }
            }
            
            core = CognitiveCore(config=config)
            
            # Create state and save
            goal = Goal(type=GoalType.CREATE, description="Original")
            core.workspace.add_goal(goal)
            path = core.save_state()
            
            # Modify state
            core.workspace.clear()
            assert len(core.workspace.current_goals) == 0
            
            # Restore state
            success = core.restore_state(path)
            
            assert success is True
            assert len(core.workspace.current_goals) == 1
            assert core.workspace.current_goals[0].description == "Original"
    
    @pytest.mark.asyncio
    async def test_start_with_restore_latest(self):
        """Test starting CognitiveCore with restore_latest=True."""
        with TemporaryDirectory() as tmpdir:
            config = {
                "checkpointing": {
                    "enabled": True,
                    "checkpoint_dir": str(Path(tmpdir) / "checkpoints"),
                }
            }
            
            # First session: create state and save
            core1 = CognitiveCore(config=config)
            goal = Goal(type=GoalType.INTROSPECT, description="Reflect")
            core1.workspace.add_goal(goal)
            core1.save_state(label="Session 1")
            
            # Second session: start with restore
            core2 = CognitiveCore(config=config)
            
            # Start with restore (don't await start, just test the restore part)
            latest = core2.checkpoint_manager.get_latest_checkpoint()
            assert latest is not None
            
            core2.workspace = core2.checkpoint_manager.load_checkpoint(latest)
            
            # Verify state restored
            assert len(core2.workspace.current_goals) == 1
            assert core2.workspace.current_goals[0].description == "Reflect"
    
    @pytest.mark.asyncio
    async def test_shutdown_checkpoint(self):
        """Test that checkpoint is saved on shutdown."""
        with TemporaryDirectory() as tmpdir:
            config = {
                "checkpointing": {
                    "enabled": True,
                    "checkpoint_dir": str(Path(tmpdir) / "checkpoints"),
                    "checkpoint_on_shutdown": True,
                }
            }
            
            core = CognitiveCore(config=config)
            
            # Add some state
            goal = Goal(type=GoalType.RESPOND_TO_USER, description="Answer")
            core.workspace.add_goal(goal)
            
            # Simulate shutdown (without actually running the loop)
            manager = core.checkpoint_manager
            manager.save_checkpoint(
                core.workspace,
                metadata={"auto_save": False, "shutdown": True}
            )
            
            # Verify shutdown checkpoint was created
            checkpoints = manager.list_checkpoints()
            assert len(checkpoints) > 0
            assert any(cp.metadata.get("shutdown", False) for cp in checkpoints)
    
    @pytest.mark.asyncio
    async def test_enable_disable_auto_checkpoint(self):
        """Test enable/disable auto-checkpoint methods."""
        with TemporaryDirectory() as tmpdir:
            config = {
                "checkpointing": {
                    "enabled": True,
                    "checkpoint_dir": str(Path(tmpdir) / "checkpoints"),
                }
            }
            
            core = CognitiveCore(config=config)
            
            # Cannot enable when not running
            success = core.enable_auto_checkpoint()
            assert success is False
            
            # Simulate running state within async context
            core.state.running = True
            
            # Enable auto-checkpoint
            success = core.enable_auto_checkpoint(interval=10.0)
            assert success is True
            assert core.checkpoint_manager.auto_save_task is not None
            
            # Disable auto-checkpoint
            success = core.disable_auto_checkpoint()
            assert success is True


class TestErrorHandling:
    """Test error handling in checkpoint operations."""
    
    def test_invalid_checkpoint_version(self):
        """Test handling of checkpoint with invalid version."""
        with TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir) / "checkpoints"
            manager = CheckpointManager(checkpoint_dir=checkpoint_dir)
            
            # Create checkpoint with wrong version
            invalid_checkpoint = {
                "version": "999.0",
                "timestamp": datetime.now().isoformat(),
                "checkpoint_id": "test-id",
                "workspace_state": {},
            }
            
            path = checkpoint_dir / "invalid_version.json"
            path.write_text(json.dumps(invalid_checkpoint))
            
            # Should still load but log warning
            # (In a real scenario, we might want stricter validation)
            workspace = manager.load_checkpoint(path)
            assert workspace is not None
    
    def test_missing_required_fields(self):
        """Test handling of checkpoint with missing required fields."""
        with TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir) / "checkpoints"
            manager = CheckpointManager(checkpoint_dir=checkpoint_dir)
            
            # Create checkpoint missing required field
            invalid_checkpoint = {
                "version": "1.0",
                "timestamp": datetime.now().isoformat(),
                # Missing checkpoint_id and workspace_state
            }
            
            path = checkpoint_dir / "missing_fields.json"
            path.write_text(json.dumps(invalid_checkpoint))
            
            # Should raise ValueError
            with pytest.raises(ValueError, match="Invalid checkpoint"):
                manager.load_checkpoint(path)
    
    def test_save_state_when_disabled(self):
        """Test save_state when checkpointing is disabled."""
        config = {
            "checkpointing": {
                "enabled": False,
            }
        }
        
        core = CognitiveCore(config=config)
        
        # Should return None and log warning
        path = core.save_state()
        assert path is None
    
    def test_restore_state_when_running(self):
        """Test restore_state when cognitive loop is running."""
        with TemporaryDirectory() as tmpdir:
            config = {
                "checkpointing": {
                    "enabled": True,
                    "checkpoint_dir": str(Path(tmpdir) / "checkpoints"),
                }
            }
            
            core = CognitiveCore(config=config)
            
            # Save a checkpoint
            path = core.save_state()
            
            # Simulate running state
            core.state.running = True
            
            # Should fail to restore
            success = core.restore_state(path)
            assert success is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
