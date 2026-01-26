#!/usr/bin/env python3
"""
Check for performance regressions by comparing current run to baseline.

Usage:
    python scripts/check_performance_regression.py --baseline data/benchmarks/baseline.json --current data/benchmarks/current.json
"""

import json
import sys
from pathlib import Path
import argparse


# Regression thresholds: subsystem -> allowed multiplier
REGRESSION_THRESHOLDS = {
    "attention": 1.3,           # 30% regression allowed
    "memory_retrieval": 1.5,    # 50% regression allowed
    "memory_consolidation": 1.5,
    "affect": 1.3,
    "meta_cognition": 1.5,
    "perception": 1.3,
    "action": 1.3,
    "workspace_update": 1.3,
    "broadcast": 1.3,
    "autonomous_initiation": 1.5,
    "cycle_avg": 1.2            # 20% regression allowed for overall
}


def check_regression(baseline_path: Path, current_path: Path) -> bool:
    """
    Check if current performance has regressed vs baseline.
    
    Args:
        baseline_path: Path to baseline benchmark JSON
        current_path: Path to current benchmark JSON
        
    Returns:
        True if no regressions detected, False otherwise
    """
    # Load JSON files
    try:
        with open(baseline_path) as f:
            baseline = json.load(f)
    except FileNotFoundError:
        print(f"⚠️  Baseline file not found: {baseline_path}")
        print("   Creating new baseline...")
        return True
    except json.JSONDecodeError as e:
        print(f"❌ Invalid baseline JSON: {e}")
        return False
    
    try:
        with open(current_path) as f:
            current = json.load(f)
    except FileNotFoundError:
        print(f"❌ Current results file not found: {current_path}")
        return False
    except json.JSONDecodeError as e:
        print(f"❌ Invalid current results JSON: {e}")
        return False
    
    regressions = []
    
    # Check each subsystem
    for subsystem, threshold in REGRESSION_THRESHOLDS.items():
        if subsystem not in baseline or subsystem not in current:
            continue
        
        baseline_time = baseline[subsystem].get('avg_ms', 0)
        current_time = current[subsystem].get('avg_ms', 0)
        
        if baseline_time == 0:
            continue
        
        ratio = current_time / baseline_time
        
        if ratio > threshold:
            regressions.append({
                'subsystem': subsystem,
                'baseline': baseline_time,
                'current': current_time,
                'ratio': ratio,
                'threshold': threshold,
                'exceeded_by': (ratio - threshold) * 100
            })
    
    # Report results
    if regressions:
        print("\n⚠️  PERFORMANCE REGRESSIONS DETECTED:\n")
        for reg in regressions:
            print(f"  {reg['subsystem']}:")
            print(f"    Baseline: {reg['baseline']:.2f}ms")
            print(f"    Current: {reg['current']:.2f}ms")
            print(f"    Ratio: {reg['ratio']:.2f}x (threshold: {reg['threshold']:.2f}x)")
            print(f"    Exceeded by: {reg['exceeded_by']:.1f}%")
            print()
        return False
    else:
        print("\n✅ No performance regressions detected")
        
        # Show summary
        print("\nPerformance Summary:")
        for subsystem in REGRESSION_THRESHOLDS.keys():
            if subsystem in baseline and subsystem in current:
                baseline_time = baseline[subsystem].get('avg_ms', 0)
                current_time = current[subsystem].get('avg_ms', 0)
                if baseline_time > 0:
                    ratio = current_time / baseline_time
                    change = (ratio - 1.0) * 100
                    symbol = "📈" if change > 0 else "📉"
                    print(f"  {subsystem}: {baseline_time:.1f}ms → {current_time:.1f}ms ({symbol} {change:+.1f}%)")
        
        return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Check for performance regressions")
    parser.add_argument("--baseline", type=Path, required=True, 
                       help="Path to baseline benchmark JSON")
    parser.add_argument("--current", type=Path, required=True,
                       help="Path to current benchmark JSON")
    
    args = parser.parse_args()
    
    if not check_regression(args.baseline, args.current):
        sys.exit(1)


if __name__ == "__main__":
    main()
