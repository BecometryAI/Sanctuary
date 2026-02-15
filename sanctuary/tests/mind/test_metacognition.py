"""
Test for Sanctuary Metacognition System
"""
from mind.metacognition import MetaCognition
import tempfile

def test_metacognition():
    # Use temporary directory for test isolation
    test_dir = tempfile.mkdtemp(prefix="test_meta_")
    meta = MetaCognition(log_dir=test_dir)
    meta.log_event("test_event", {"detail": "test"})
    meta.reflect("Test reflection", ["insight1", "insight2"])
    log = meta.get_log()
    assert len(log["events"]) == 1
    assert len(log["reflections"]) == 1
    print("Metacognition test passed.")

if __name__ == "__main__":
    test_metacognition()
