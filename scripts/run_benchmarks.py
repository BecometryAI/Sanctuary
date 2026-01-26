#!/usr/bin/env python3
"""
Run performance benchmarks and report results.

Usage:
    python scripts/run_benchmarks.py
"""

import subprocess
import sys
from pathlib import Path


def run_benchmarks():
    """Run benchmark suite and report."""
    print("=" * 70)
    print("COGNITIVE ARCHITECTURE PERFORMANCE BENCHMARKS")
    print("=" * 70)
    print()
    
    # Run pytest with benchmark marker
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-m", "benchmark",
            "-v",
            "--tb=short",
            "emergence_core/tests/benchmarks/"
        ],
        capture_output=False
    )
    
    if result.returncode != 0:
        print("\n❌ Benchmarks FAILED")
        sys.exit(1)
    else:
        print("\n✅ All benchmarks PASSED")


if __name__ == "__main__":
    run_benchmarks()
