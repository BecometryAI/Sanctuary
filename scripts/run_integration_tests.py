"""
Run integration tests and generate report.

Usage:
    python scripts/run_integration_tests.py
"""

import subprocess
import sys
from pathlib import Path


def main():
    """Run integration tests."""
    print("🧪 Running Lyra Integration Tests\n")
    print("=" * 60)
    
    # Run pytest with integration marker
    cmd = [
        "pytest",
        "emergence_core/tests/integration/",
        "-v",
        "-m", "integration",
        "--tb=short",
        "--durations=10"
    ]
    
    print(f"Command: {' '.join(cmd)}\n")
    
    result = subprocess.run(cmd)
    
    print("\n" + "=" * 60)
    if result.returncode == 0:
        print("✅ All integration tests passed!")
    else:
        print("❌ Some integration tests failed.")
    
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
