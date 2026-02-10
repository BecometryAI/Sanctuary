#!/usr/bin/env python3
"""Convert sanctuary_continuity_archive.json from array to dict format."""

import json
from pathlib import Path

def convert_archive():
    """Convert the continuity archive from array to dict."""
    archive_path = Path(__file__).parent.parent / "data" / "Core_Archives" / "sanctuary_continuity_archive.json"
    
    # Read current array format
    with open(archive_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if not isinstance(data, list):
        print(f"File is already a dict, skipping conversion")
        return
    
    print(f"Converting {len(data)} array items to dict format...")
    
    # Merge all items into a single dict
    result = {}
    for item in data:
        result.update(item)
    
    # Write back as dict
    with open(archive_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Converted successfully!")
    print(f"  Top-level keys: {list(result.keys())}")

if __name__ == "__main__":
    convert_archive()
