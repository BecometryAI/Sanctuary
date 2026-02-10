"""
Test for Sanctuary Meta-cognition Module
"""
from .metacognition import MetaCognition

def test_metacognition():
    meta = MetaCognition(log_dir="test_meta_logs")
    meta.log_event("decision", {"action": "respond", "input": "Hello Sanctuary!"})
    meta.reflect("Test reflection", ["Learned to log events.", "Reflection works."])
    log = meta.get_log()
    assert len(log["events"]) == 1
    assert len(log["reflections"]) == 1
    print("Meta-cognition test passed.")

if __name__ == "__main__":
    test_metacognition()
