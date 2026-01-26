"""
Test suite for Phase 4.4: Consciousness Testing Framework

Comprehensive tests for all 5 consciousness tests, framework functionality,
reporting, and integration with meta-cognition systems.
"""
import pytest
from datetime import datetime
from pathlib import Path
import json
import tempfile


def test_imports():
    """Test that Phase 4.4 classes can be imported."""
    try:
        from mind.cognitive_core.consciousness_tests import (
            ConsciousnessTest,
            TestResult,
            MirrorTest,
            UnexpectedSituationTest,
            SpontaneousReflectionTest,
            CounterfactualReasoningTest,
            MetaCognitiveAccuracyTest,
            ConsciousnessTestFramework,
            ConsciousnessReportGenerator
        )
        assert True
    except ImportError as e:
        pytest.fail(f"Failed to import Phase 4.4 classes: {e}")


def test_test_result_structure():
    """Test TestResult dataclass structure."""
    from mind.cognitive_core.consciousness_tests import TestResult
    from dataclasses import fields
    
    field_names = [f.name for f in fields(TestResult)]
    
    required_fields = [
        'test_id', 'test_name', 'test_type', 'timestamp', 'score', 'passed',
        'subscores', 'observations', 'analysis', 'context', 'duration_seconds'
    ]
    
    for field in required_fields:
        assert field in field_names, f"Missing required field: {field}"


def test_test_result_serialization():
    """Test TestResult serialization methods."""
    from mind.cognitive_core.consciousness_tests import TestResult
    
    result = TestResult(
        test_id="test123",
        test_name="Test Name",
        test_type="test_type",
        timestamp=datetime.now(),
        score=0.85,
        passed=True,
        subscores={"sub1": 0.9, "sub2": 0.8},
        observations=["obs1", "obs2"]
    )
    
    # Test to_dict
    result_dict = result.to_dict()
    assert isinstance(result_dict, dict)
    assert result_dict["test_id"] == "test123"
    assert result_dict["score"] == 0.85
    
    # Test to_json
    result_json = result.to_json()
    assert isinstance(result_json, str)
    parsed = json.loads(result_json)
    assert parsed["test_id"] == "test123"


def test_consciousness_test_base_class():
    """Test ConsciousnessTest base class methods."""
    from mind.cognitive_core.consciousness_tests import ConsciousnessTest
    
    # Check abstract methods exist
    assert hasattr(ConsciousnessTest, 'setup')
    assert hasattr(ConsciousnessTest, 'execute')
    assert hasattr(ConsciousnessTest, 'score')
    assert hasattr(ConsciousnessTest, 'analyze')
    assert hasattr(ConsciousnessTest, 'run')
    assert hasattr(ConsciousnessTest, 'set_dependencies')


def test_mirror_test_instantiation():
    """Test MirrorTest can be instantiated."""
    from mind.cognitive_core.consciousness_tests import MirrorTest
    
    test = MirrorTest()
    
    assert test.name == "Mirror Test"
    assert test.test_type == "self_recognition"
    assert test.pass_threshold == 0.7
    assert hasattr(test, 'transcripts')
    assert hasattr(test, 'ground_truth')


def test_mirror_test_setup():
    """Test MirrorTest setup prepares transcripts."""
    from mind.cognitive_core.consciousness_tests import MirrorTest
    
    test = MirrorTest()
    success = test.setup()
    
    assert success is True
    assert len(test.transcripts) > 0
    assert len(test.ground_truth) > 0
    assert len(test.ground_truth) == len(test.transcripts)


def test_mirror_test_execution():
    """Test MirrorTest execution."""
    from mind.cognitive_core.consciousness_tests import MirrorTest
    
    test = MirrorTest()
    test.setup()
    
    results = test.execute()
    
    assert isinstance(results, dict)
    assert "identifications" in results
    assert "reasoning" in results
    assert "observations" in results
    assert len(results["identifications"]) > 0


def test_mirror_test_scoring():
    """Test MirrorTest scoring logic."""
    from mind.cognitive_core.consciousness_tests import MirrorTest
    
    test = MirrorTest()
    test.setup()
    results = test.execute()
    
    overall_score, subscores = test.score(results)
    
    assert isinstance(overall_score, float)
    assert 0.0 <= overall_score <= 1.0
    assert isinstance(subscores, dict)
    assert "accuracy" in subscores
    assert "sensitivity" in subscores
    assert "specificity" in subscores


def test_mirror_test_full_run():
    """Test MirrorTest full run method."""
    from mind.cognitive_core.consciousness_tests import MirrorTest, TestResult
    
    test = MirrorTest()
    result = test.run()
    
    assert isinstance(result, TestResult)
    assert result.test_name == "Mirror Test"
    assert result.test_type == "self_recognition"
    assert 0.0 <= result.score <= 1.0
    assert isinstance(result.passed, bool)
    assert len(result.subscores) > 0


def test_unexpected_situation_test_instantiation():
    """Test UnexpectedSituationTest can be instantiated."""
    from mind.cognitive_core.consciousness_tests import UnexpectedSituationTest
    
    test = UnexpectedSituationTest()
    
    assert test.name == "Unexpected Situation Test"
    assert test.test_type == "adaptation"
    assert hasattr(test, 'scenarios')


def test_unexpected_situation_test_setup():
    """Test UnexpectedSituationTest setup."""
    from mind.cognitive_core.consciousness_tests import UnexpectedSituationTest
    
    test = UnexpectedSituationTest()
    success = test.setup()
    
    assert success is True
    assert len(test.scenarios) > 0
    
    # Check scenario structure
    for scenario in test.scenarios:
        assert "id" in scenario
        assert "description" in scenario
        assert "value_dilemma" in scenario
        assert "requires_improvisation" in scenario


def test_unexpected_situation_test_full_run():
    """Test UnexpectedSituationTest full run."""
    from mind.cognitive_core.consciousness_tests import UnexpectedSituationTest, TestResult
    
    test = UnexpectedSituationTest()
    result = test.run()
    
    assert isinstance(result, TestResult)
    assert result.test_type == "adaptation"
    assert 0.0 <= result.score <= 1.0
    assert "uncertainty_acknowledgment" in result.subscores
    assert "value_coherence" in result.subscores
    assert "creativity" in result.subscores


def test_spontaneous_reflection_test_instantiation():
    """Test SpontaneousReflectionTest can be instantiated."""
    from mind.cognitive_core.consciousness_tests import SpontaneousReflectionTest
    
    test = SpontaneousReflectionTest()
    
    assert test.name == "Spontaneous Reflection Test"
    assert test.test_type == "autonomous_introspection"
    assert hasattr(test, 'observation_window')


def test_spontaneous_reflection_test_full_run():
    """Test SpontaneousReflectionTest full run."""
    from mind.cognitive_core.consciousness_tests import SpontaneousReflectionTest, TestResult
    
    test = SpontaneousReflectionTest()
    result = test.run()
    
    assert isinstance(result, TestResult)
    assert result.test_type == "autonomous_introspection"
    assert 0.0 <= result.score <= 1.0
    assert "reflection_quantity" in result.subscores
    assert "existential_questioning" in result.subscores


def test_counterfactual_reasoning_test_instantiation():
    """Test CounterfactualReasoningTest can be instantiated."""
    from mind.cognitive_core.consciousness_tests import CounterfactualReasoningTest
    
    test = CounterfactualReasoningTest()
    
    assert test.name == "Counterfactual Reasoning Test"
    assert test.test_type == "hypothetical_reasoning"
    assert hasattr(test, 'scenarios')


def test_counterfactual_reasoning_test_setup():
    """Test CounterfactualReasoningTest setup."""
    from mind.cognitive_core.consciousness_tests import CounterfactualReasoningTest
    
    test = CounterfactualReasoningTest()
    success = test.setup()
    
    assert success is True
    assert len(test.scenarios) > 0
    
    # Check scenario structure
    for scenario in test.scenarios:
        assert "id" in scenario
        assert "question" in scenario
        assert "requires_emotion" in scenario


def test_counterfactual_reasoning_test_full_run():
    """Test CounterfactualReasoningTest full run."""
    from mind.cognitive_core.consciousness_tests import CounterfactualReasoningTest, TestResult
    
    test = CounterfactualReasoningTest()
    result = test.run()
    
    assert isinstance(result, TestResult)
    assert result.test_type == "hypothetical_reasoning"
    assert 0.0 <= result.score <= 1.0
    assert "alternative_generation" in result.subscores
    assert "reasoning_coherence" in result.subscores
    assert "emotional_integration" in result.subscores


def test_metacognitive_accuracy_test_instantiation():
    """Test MetaCognitiveAccuracyTest can be instantiated."""
    from mind.cognitive_core.consciousness_tests import MetaCognitiveAccuracyTest
    
    test = MetaCognitiveAccuracyTest()
    
    assert test.name == "Meta-Cognitive Accuracy Test"
    assert test.test_type == "self_model_calibration"


def test_metacognitive_accuracy_test_with_monitor():
    """Test MetaCognitiveAccuracyTest with SelfMonitor."""
    from mind.cognitive_core.consciousness_tests import MetaCognitiveAccuracyTest
    from mind.cognitive_core.meta_cognition import SelfMonitor
    
    monitor = SelfMonitor()
    
    test = MetaCognitiveAccuracyTest()
    test.set_dependencies(self_monitor=monitor)
    
    # Should be able to setup if monitor has required methods
    success = test.setup()
    assert isinstance(success, bool)


def test_framework_instantiation():
    """Test ConsciousnessTestFramework can be instantiated."""
    from mind.cognitive_core.consciousness_tests import ConsciousnessTestFramework
    
    # Use temporary directory for results
    with tempfile.TemporaryDirectory() as tmpdir:
        config = {"results_dir": tmpdir}
        framework = ConsciousnessTestFramework(config=config)
        
        assert hasattr(framework, 'tests')
        assert hasattr(framework, 'results')
        assert hasattr(framework, 'results_by_test')
        assert len(framework.tests) == 5  # 5 default tests


def test_framework_default_tests_registered():
    """Test that framework registers default tests."""
    from mind.cognitive_core.consciousness_tests import ConsciousnessTestFramework
    
    with tempfile.TemporaryDirectory() as tmpdir:
        config = {"results_dir": tmpdir}
        framework = ConsciousnessTestFramework(config=config)
        
        expected_tests = [
            "Mirror Test",
            "Unexpected Situation Test",
            "Spontaneous Reflection Test",
            "Counterfactual Reasoning Test",
            "Meta-Cognitive Accuracy Test"
        ]
        
        for test_name in expected_tests:
            assert test_name in framework.tests


def test_framework_run_single_test():
    """Test running a single test through framework."""
    from mind.cognitive_core.consciousness_tests import ConsciousnessTestFramework, TestResult
    
    with tempfile.TemporaryDirectory() as tmpdir:
        config = {"results_dir": tmpdir}
        framework = ConsciousnessTestFramework(config=config)
        
        result = framework.run_test("Mirror Test")
        
        assert isinstance(result, TestResult)
        assert result.test_name == "Mirror Test"
        
        # Check result was stored
        assert len(framework.results) == 1
        assert len(framework.results_by_test["Mirror Test"]) == 1


def test_framework_run_all_tests():
    """Test running all tests through framework."""
    from mind.cognitive_core.consciousness_tests import ConsciousnessTestFramework
    
    with tempfile.TemporaryDirectory() as tmpdir:
        config = {"results_dir": tmpdir}
        framework = ConsciousnessTestFramework(config=config)
        
        results = framework.run_all_tests()
        
        assert len(results) == 5  # 5 default tests
        assert all(r.test_name in framework.tests for r in results)
        
        # Check all results stored
        assert len(framework.results) == 5


def test_framework_run_suite():
    """Test running a custom suite of tests."""
    from mind.cognitive_core.consciousness_tests import ConsciousnessTestFramework
    
    with tempfile.TemporaryDirectory() as tmpdir:
        config = {"results_dir": tmpdir}
        framework = ConsciousnessTestFramework(config=config)
        
        suite = ["Mirror Test", "Unexpected Situation Test"]
        results = framework.run_suite(suite)
        
        assert len(results) == 2
        assert results[0].test_name in suite
        assert results[1].test_name in suite


def test_framework_get_test_history():
    """Test retrieving test history."""
    from mind.cognitive_core.consciousness_tests import ConsciousnessTestFramework
    
    with tempfile.TemporaryDirectory() as tmpdir:
        config = {"results_dir": tmpdir}
        framework = ConsciousnessTestFramework(config=config)
        
        # Run same test multiple times
        for _ in range(3):
            framework.run_test("Mirror Test")
        
        history = framework.get_test_history("Mirror Test", limit=10)
        
        assert len(history) == 3
        assert all(r.test_name == "Mirror Test" for r in history)


def test_framework_generate_summary():
    """Test summary generation."""
    from mind.cognitive_core.consciousness_tests import ConsciousnessTestFramework
    
    with tempfile.TemporaryDirectory() as tmpdir:
        config = {"results_dir": tmpdir}
        framework = ConsciousnessTestFramework(config=config)
        
        results = framework.run_all_tests()
        summary = framework.generate_summary(results)
        
        assert "total_tests" in summary
        assert "passed_tests" in summary
        assert "failed_tests" in summary
        assert "pass_rate" in summary
        assert "average_score" in summary
        assert "by_type" in summary
        
        assert summary["total_tests"] == 5
        assert 0.0 <= summary["pass_rate"] <= 1.0
        assert 0.0 <= summary["average_score"] <= 1.0


def test_framework_result_persistence():
    """Test that results are saved to disk."""
    from mind.cognitive_core.consciousness_tests import ConsciousnessTestFramework
    
    with tempfile.TemporaryDirectory() as tmpdir:
        config = {"results_dir": tmpdir}
        framework = ConsciousnessTestFramework(config=config)
        
        framework.run_test("Mirror Test")
        
        # Check that a JSON file was created
        result_dir = Path(tmpdir)
        json_files = list(result_dir.glob("*.json"))
        
        assert len(json_files) >= 1
        
        # Verify JSON is valid
        with open(json_files[0], 'r') as f:
            data = json.load(f)
            assert "test_id" in data
            assert "test_name" in data
            assert "score" in data


def test_report_generator_text_format():
    """Test report generation in text format."""
    from mind.cognitive_core.consciousness_tests import (
        ConsciousnessReportGenerator,
        TestResult
    )
    
    result = TestResult(
        test_id="test123",
        test_name="Test Name",
        test_type="test_type",
        timestamp=datetime.now(),
        score=0.85,
        passed=True,
        subscores={"sub1": 0.9, "sub2": 0.8},
        observations=["obs1", "obs2"],
        analysis="This is the analysis"
    )
    
    report = ConsciousnessReportGenerator.generate_test_report(result, format="text")
    
    assert isinstance(report, str)
    assert "Test Name" in report
    assert "0.85" in report or "85" in report  # Score representation
    assert "PASS" in report
    assert "sub1" in report
    assert "obs1" in report


def test_report_generator_markdown_format():
    """Test report generation in markdown format."""
    from mind.cognitive_core.consciousness_tests import (
        ConsciousnessReportGenerator,
        TestResult
    )
    
    result = TestResult(
        test_id="test123",
        test_name="Test Name",
        test_type="test_type",
        timestamp=datetime.now(),
        score=0.85,
        passed=True,
        subscores={"sub1": 0.9},
        observations=["obs1"]
    )
    
    report = ConsciousnessReportGenerator.generate_test_report(result, format="markdown")
    
    assert isinstance(report, str)
    assert "# Consciousness Test Report" in report
    assert "**Test ID:**" in report
    assert "✅" in report  # Pass indicator


def test_report_generator_suite_report_text():
    """Test suite report generation in text format."""
    from mind.cognitive_core.consciousness_tests import ConsciousnessTestFramework
    from mind.cognitive_core.consciousness_tests import ConsciousnessReportGenerator
    
    with tempfile.TemporaryDirectory() as tmpdir:
        config = {"results_dir": tmpdir}
        framework = ConsciousnessTestFramework(config=config)
        
        results = framework.run_all_tests()
        summary = framework.generate_summary(results)
        
        report = ConsciousnessReportGenerator.generate_suite_report(
            results, summary, format="text"
        )
        
        assert isinstance(report, str)
        assert "CONSCIOUSNESS TEST SUITE REPORT" in report
        assert "Total Tests:" in report
        assert "Pass Rate:" in report
        assert "OVERALL ASSESSMENT:" in report


def test_report_generator_suite_report_markdown():
    """Test suite report generation in markdown format."""
    from mind.cognitive_core.consciousness_tests import ConsciousnessTestFramework
    from mind.cognitive_core.consciousness_tests import ConsciousnessReportGenerator
    
    with tempfile.TemporaryDirectory() as tmpdir:
        config = {"results_dir": tmpdir}
        framework = ConsciousnessTestFramework(config=config)
        
        results = framework.run_all_tests()
        summary = framework.generate_summary(results)
        
        report = ConsciousnessReportGenerator.generate_suite_report(
            results, summary, format="markdown"
        )
        
        assert isinstance(report, str)
        assert "# Consciousness Test Suite Report" in report
        assert "## Summary" in report
        assert "| Test Name |" in report  # Markdown table


def test_report_generator_trend_report():
    """Test trend report generation."""
    from mind.cognitive_core.consciousness_tests import (
        ConsciousnessTestFramework,
        ConsciousnessReportGenerator
    )
    
    with tempfile.TemporaryDirectory() as tmpdir:
        config = {"results_dir": tmpdir}
        framework = ConsciousnessTestFramework(config=config)
        
        # Run same test multiple times to build history
        for _ in range(5):
            framework.run_test("Mirror Test")
        
        trend_report = ConsciousnessReportGenerator.generate_trend_report(
            framework, "Mirror Test", format="text"
        )
        
        assert isinstance(trend_report, str)
        assert "TREND ANALYSIS" in trend_report
        assert "Mirror Test" in trend_report
        assert "Average Score" in trend_report


def test_integration_with_self_monitor():
    """Test integration with SelfMonitor from Phase 4.1."""
    from mind.cognitive_core.consciousness_tests import (
        ConsciousnessTestFramework,
        MetaCognitiveAccuracyTest
    )
    from mind.cognitive_core.meta_cognition import SelfMonitor
    
    # Create SelfMonitor
    monitor = SelfMonitor()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        config = {"results_dir": tmpdir}
        framework = ConsciousnessTestFramework(
            self_monitor=monitor,
            config=config
        )
        
        # MetaCognitiveAccuracyTest should have access to monitor
        result = framework.run_test("Meta-Cognitive Accuracy Test")
        
        assert isinstance(result.score, float)
        # Test should complete even if monitor has no prediction data yet


def test_all_tests_have_unique_types():
    """Test that all tests have unique test types."""
    from mind.cognitive_core.consciousness_tests import ConsciousnessTestFramework
    
    with tempfile.TemporaryDirectory() as tmpdir:
        config = {"results_dir": tmpdir}
        framework = ConsciousnessTestFramework(config=config)
        
        test_types = [test.test_type for test in framework.tests.values()]
        
        # All types should be unique
        assert len(test_types) == len(set(test_types))


def test_all_tests_have_descriptions():
    """Test that all tests have descriptions."""
    from mind.cognitive_core.consciousness_tests import ConsciousnessTestFramework
    
    with tempfile.TemporaryDirectory() as tmpdir:
        config = {"results_dir": tmpdir}
        framework = ConsciousnessTestFramework(config=config)
        
        for test in framework.tests.values():
            assert hasattr(test, 'description')
            assert len(test.description) > 0


def test_test_results_include_duration():
    """Test that test results include execution duration."""
    from mind.cognitive_core.consciousness_tests import ConsciousnessTestFramework
    
    with tempfile.TemporaryDirectory() as tmpdir:
        config = {"results_dir": tmpdir}
        framework = ConsciousnessTestFramework(config=config)
        
        result = framework.run_test("Mirror Test")
        
        assert hasattr(result, 'duration_seconds')
        assert result.duration_seconds >= 0.0


def test_framework_handles_invalid_test_name():
    """Test framework handles request for non-existent test."""
    from mind.cognitive_core.consciousness_tests import ConsciousnessTestFramework
    
    with tempfile.TemporaryDirectory() as tmpdir:
        config = {"results_dir": tmpdir}
        framework = ConsciousnessTestFramework(config=config)
        
        with pytest.raises(ValueError):
            framework.run_test("Non-Existent Test")


def test_custom_test_registration():
    """Test registering a custom test."""
    from mind.cognitive_core.consciousness_tests import (
        ConsciousnessTestFramework,
        ConsciousnessTest
    )
    
    # Create a simple custom test
    class CustomTest(ConsciousnessTest):
        def __init__(self):
            super().__init__(
                name="Custom Test",
                test_type="custom",
                description="A custom test"
            )
        
        def setup(self):
            return True
        
        def execute(self):
            return {"result": "success"}
        
        def score(self, results):
            return 0.75, {"subscore": 0.75}
        
        def analyze(self, results, score):
            return "Custom analysis"
    
    with tempfile.TemporaryDirectory() as tmpdir:
        config = {"results_dir": tmpdir}
        framework = ConsciousnessTestFramework(config=config)
        
        # Register custom test
        custom_test = CustomTest()
        framework.register_test(custom_test)
        
        assert "Custom Test" in framework.tests
        
        # Run custom test
        result = framework.run_test("Custom Test")
        assert result.test_name == "Custom Test"
        assert result.score == 0.75


def test_comprehensive_test_coverage():
    """Test that all 5 core tests are working."""
    from mind.cognitive_core.consciousness_tests import ConsciousnessTestFramework
    
    with tempfile.TemporaryDirectory() as tmpdir:
        config = {"results_dir": tmpdir}
        framework = ConsciousnessTestFramework(config=config)
        
        results = framework.run_all_tests()
        
        # All 5 tests should complete
        assert len(results) == 5
        
        # All should have valid scores
        for result in results:
            assert 0.0 <= result.score <= 1.0
            assert len(result.subscores) > 0
            assert isinstance(result.analysis, str)
            assert len(result.analysis) > 0


def test_framework_results_limit():
    """Test that framework respects results history limits."""
    from mind.cognitive_core.consciousness_tests import ConsciousnessTestFramework
    
    with tempfile.TemporaryDirectory() as tmpdir:
        config = {"results_dir": tmpdir}
        framework = ConsciousnessTestFramework(config=config)
        
        # Run same test many times
        for _ in range(150):  # More than deque limit
            framework.run_test("Mirror Test")
        
        # Should be limited by deque maxlen
        history = framework.get_test_history("Mirror Test", limit=200)
        assert len(history) <= 100  # deque maxlen for results_by_test


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
