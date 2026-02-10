"""
Sanctuary Meta-cognition Module: Self-monitoring and Reflection
"""
import datetime
import json
from pathlib import Path

class MetaCognition:
    def __init__(self, log_dir: str = "data/meta_logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / f"metacog_{datetime.date.today()}.json"
        if not self.log_file.exists():
            self._init_log()

    def _init_log(self):
        with open(self.log_file, "w", encoding="utf-8") as f:
            json.dump({"events": [], "reflections": []}, f)

    def log_event(self, event_type: str, details: dict):
        with open(self.log_file, "r+", encoding="utf-8") as f:
            data = json.load(f)
            data["events"].append({
                "timestamp": datetime.datetime.now().isoformat(),
                "type": event_type,
                "details": details
            })
            f.seek(0)
            json.dump(data, f, indent=2)
            f.truncate()

    def reflect(self, summary: str, insights: list):
        with open(self.log_file, "r+", encoding="utf-8") as f:
            data = json.load(f)
            data["reflections"].append({
                "timestamp": datetime.datetime.now().isoformat(),
                "summary": summary,
                "insights": insights
            })
            f.seek(0)
            json.dump(data, f, indent=2)
            f.truncate()

    def get_log(self):
        with open(self.log_file, "r", encoding="utf-8") as f:
            return json.load(f)
