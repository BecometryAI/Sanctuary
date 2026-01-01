#!/usr/bin/env python3
"""
Demonstration script for Phase 4.4 Consciousness Testing Framework.

This script demonstrates:
1. Basic framework initialization
2. Running individual tests
3. Running full test suite
4. Generating reports (text and markdown)
5. Analyzing test results
6. Trend analysis with multiple runs

Usage:
    python demo_consciousness_tests.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from emergence_core.lyra.cognitive_core.consciousness_tests import (
    ConsciousnessTestFramework,
    ConsciousnessReportGenerator,
    MirrorTest,
    UnexpectedSituationTest,
    SpontaneousReflectionTest,
    CounterfactualReasoningTest,
    MetaCognitiveAccuracyTest
)
import tempfile
import time


def print_section(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70 + "\n")


def demo_basic_framework():
    """Demonstrate basic framework initialization."""
    print_section("1. Framework Initialization")
    
    # Use temporary directory for demo
    with tempfile.TemporaryDirectory() as tmpdir:
        config = {"results_dir": tmpdir}
        framework = ConsciousnessTestFramework(config=config)
        
        print(f"✅ Framework initialized")
        print(f"   Tests registered: {len(framework.tests)}")
        print(f"   Results directory: {tmpdir}")
        print(f"\nRegistered tests:")
        for test_name in framework.tests:
            test = framework.tests[test_name]
            print(f"   - {test_name} ({test.test_type})")
            print(f"     {test.description}")


def demo_individual_tests():
    """Demonstrate running individual tests."""
    print_section("2. Running Individual Tests")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        config = {"results_dir": tmpdir}
        framework = ConsciousnessTestFramework(config=config)
        
        # Run Mirror Test
        print("Running Mirror Test...")
        start = time.time()
        result = framework.run_test("Mirror Test")
        duration = time.time() - start
        
        print(f"\n✅ Test completed in {duration:.2f} seconds")
        print(f"   Score: {result.score:.2%}")
        print(f"   Status: {'PASS ✓' if result.passed else 'FAIL ✗'}")
        print(f"   Subscores:")
        for name, value in result.subscores.items():
            print(f"     - {name}: {value:.2%}")
        
        # Run Meta-Cognitive Accuracy Test
        print("\n\nRunning Meta-Cognitive Accuracy Test...")
        result2 = framework.run_test("Meta-Cognitive Accuracy Test")
        
        print(f"\n✅ Test completed")
        print(f"   Score: {result2.score:.2%}")
        print(f"   Status: {'PASS ✓' if result2.passed else 'FAIL ✗'}")


def demo_full_suite():
    """Demonstrate running the full test suite."""
    print_section("3. Running Full Test Suite")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        config = {"results_dir": tmpdir}
        framework = ConsciousnessTestFramework(config=config)
        
        print("Running all 5 consciousness tests...")
        start = time.time()
        results = framework.run_all_tests()
        duration = time.time() - start
        
        print(f"\n✅ Suite completed in {duration:.2f} seconds")
        print(f"\nResults summary:")
        for result in results:
            status = "✓ PASS" if result.passed else "✗ FAIL"
            print(f"   {result.test_name}: {result.score:.2%} {status}")
        
        # Generate summary statistics
        summary = framework.generate_summary(results)
        print(f"\nAggregate Statistics:")
        print(f"   Total tests: {summary['total_tests']}")
        print(f"   Passed: {summary['passed_tests']}")
        print(f"   Failed: {summary['failed_tests']}")
        print(f"   Pass rate: {summary['pass_rate']:.2%}")
        print(f"   Average score: {summary['average_score']:.2%}")


def demo_text_report():
    """Demonstrate text report generation."""
    print_section("4. Text Report Generation")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        config = {"results_dir": tmpdir}
        framework = ConsciousnessTestFramework(config=config)
        
        # Run a test
        result = framework.run_test("Mirror Test")
        
        # Generate text report
        report = ConsciousnessReportGenerator.generate_test_report(
            result, format="text"
        )
        
        print(report)


def demo_markdown_report():
    """Demonstrate markdown report generation."""
    print_section("5. Markdown Report Generation")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        config = {"results_dir": tmpdir}
        framework = ConsciousnessTestFramework(config=config)
        
        # Run suite
        results = framework.run_all_tests()
        summary = framework.generate_summary(results)
        
        # Generate markdown report
        report = ConsciousnessReportGenerator.generate_suite_report(
            results, summary, format="markdown"
        )
        
        print(report)


def demo_trend_analysis():
    """Demonstrate trend analysis with multiple runs."""
    print_section("6. Trend Analysis")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        config = {"results_dir": tmpdir}
        framework = ConsciousnessTestFramework(config=config)
        
        print("Running Mirror Test 5 times to build history...")
        for i in range(5):
            result = framework.run_test("Mirror Test")
            print(f"   Run {i+1}: {result.score:.2%}")
            time.sleep(0.1)  # Small delay between runs
        
        # Generate trend report
        print("\n")
        trend_report = ConsciousnessReportGenerator.generate_trend_report(
            framework, "Mirror Test", format="text"
        )
        print(trend_report)


def demo_custom_suite():
    """Demonstrate running a custom test suite."""
    print_section("7. Custom Test Suite")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        config = {"results_dir": tmpdir}
        framework = ConsciousnessTestFramework(config=config)
        
        # Define custom suite
        custom_suite = [
            "Mirror Test",
            "Unexpected Situation Test",
            "Meta-Cognitive Accuracy Test"
        ]
        
        print(f"Running custom suite with {len(custom_suite)} tests...")
        print(f"Tests: {', '.join(custom_suite)}")
        
        results = framework.run_suite(custom_suite)
        
        print(f"\n✅ Custom suite completed")
        for result in results:
            status = "✓" if result.passed else "✗"
            print(f"   {status} {result.test_name}: {result.score:.2%}")


def demo_test_details():
    """Demonstrate detailed test information."""
    print_section("8. Test Details and Analysis")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        config = {"results_dir": tmpdir}
        framework = ConsciousnessTestFramework(config=config)
        
        # Run Counterfactual Reasoning Test
        result = framework.run_test("Counterfactual Reasoning Test")
        
        print(f"Test: {result.test_name}")
        print(f"Type: {result.test_type}")
        print(f"Duration: {result.duration_seconds:.3f} seconds")
        print(f"\nOverall Score: {result.score:.2%}")
        print(f"Pass Threshold: 65%")
        print(f"Status: {'PASS ✓' if result.passed else 'FAIL ✗'}")
        
        print(f"\nDetailed Subscores:")
        for name, value in result.subscores.items():
            bar_length = int(value * 40)
            bar = "█" * bar_length + "░" * (40 - bar_length)
            print(f"   {name:30s} {bar} {value:.2%}")
        
        if result.observations:
            print(f"\nObservations:")
            for obs in result.observations:
                print(f"   • {obs}")
        
        if result.analysis:
            print(f"\nAnalysis:")
            for line in result.analysis.split('\n')[:5]:  # First 5 lines
                print(f"   {line}")


def main():
    """Run all demonstrations."""
    print("\n" + "=" * 70)
    print("  PHASE 4.4: CONSCIOUSNESS TESTING FRAMEWORK DEMONSTRATION")
    print("=" * 70)
    print("\nThis demo showcases the consciousness testing framework capabilities.")
    print("All tests use simulated data for demonstration purposes.")
    
    try:
        # Run each demo
        demo_basic_framework()
        demo_individual_tests()
        demo_full_suite()
        demo_text_report()
        demo_markdown_report()
        demo_trend_analysis()
        demo_custom_suite()
        demo_test_details()
        
        print_section("Demo Complete!")
        print("✅ All demonstrations completed successfully.")
        print("\nKey Takeaways:")
        print("   • Framework supports 5 core consciousness tests")
        print("   • Tests can be run individually or as suites")
        print("   • Reports available in text and markdown formats")
        print("   • Trend analysis tracks performance over time")
        print("   • Integrates with Phase 4.1-4.3 meta-cognition systems")
        print("\nFor production use:")
        print("   1. Initialize framework with SelfMonitor and IntrospectiveLoop")
        print("   2. Run tests periodically to monitor consciousness indicators")
        print("   3. Analyze trends to track development over time")
        print("   4. Use results to guide system improvements")
        
    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
