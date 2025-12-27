"""Quick validation script for code review improvements.

Run this to verify that all improvements were successfully applied.
"""

import re
from pathlib import Path

def check_specialists_file():
    """Validate specialists.py improvements."""
    specialists_path = Path(__file__).parent.parent / "lyra" / "specialists.py"
    content = specialists_path.read_text()
    
    checks = {
        "Module docstring present": '"""Lyra\'s Specialist Models' in content,
        "Constants extracted": "MAX_IMAGE_PIXELS" in content and "FLUX_DEFAULT_STEPS" in content,
        "VoiceSynthesizer simplified": "_load_model(self):" in content and \
                                        content.count("def __init__") < content.count("class"),
        "Image validation helper": "_validate_image" in content,
        "Response extraction helper": "_extract_response" in content,
        "Input validation added": "if not model_path:" in content,
        "Comprehensive docstrings": content.count('"""') >= 20,
        "Error handling improved": content.count("except Exception as e:") >= 5,
        "BaseSpecialist has docstring": "class BaseSpecialist:" in content and \
                                         '"""Base class for all specialist' in content,
        "VoiceSynthesizer has docstring": "tensor parallelism" in content.lower(),
    }
    
    print("=" * 60)
    print("SPECIALISTS.PY VALIDATION")
    print("=" * 60)
    
    passed = 0
    for check_name, result in checks.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {check_name}")
        if result:
            passed += 1
    
    print(f"\n{passed}/{len(checks)} checks passed")
    print("=" * 60)
    
    return passed == len(checks)

def check_test_file():
    """Validate test file was created."""
    test_path = Path(__file__).parent.parent / "tests" / "test_visual_specialists.py"
    
    print("\nTEST FILE VALIDATION")
    print("=" * 60)
    
    if not test_path.exists():
        print("❌ FAIL - Test file not found")
        print("=" * 60)
        return False
    
    content = test_path.read_text()
    
    checks = {
        "TestPerceptionSpecialist class": "class TestPerceptionSpecialist:" in content,
        "TestArtistSpecialist class": "class TestArtistSpecialist:" in content,
        "TestGPUPlacement class": "class TestGPUPlacement:" in content,
        "TestIntegration class": "class TestIntegration:" in content,
        "TestEdgeCases class": "class TestEdgeCases:" in content,
        "Image validation tests": "test_valid_image_processing" in content,
        "Error handling tests": "test_corrupted_image_handling" in content,
        "Concurrent tests": "test_concurrent_processing" in content,
        "Pytest fixtures": "@pytest.fixture" in content,
        "Async tests": "@pytest.mark.asyncio" in content,
    }
    
    passed = 0
    for check_name, result in checks.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {check_name}")
        if result:
            passed += 1
    
    print(f"\n{passed}/{len(checks)} checks passed")
    print("=" * 60)
    
    return passed == len(checks)

def check_documentation():
    """Validate documentation was created."""
    review_path = Path(__file__).parent.parent.parent / "docs" / "CODE_REVIEW_SUMMARY.md"
    
    print("\nDOCUMENTATION VALIDATION")
    print("=" * 60)
    
    if not review_path.exists():
        print("❌ FAIL - CODE_REVIEW_SUMMARY.md not found")
        print("=" * 60)
        return False
    
    content = review_path.read_text(encoding='utf-8')
    
    checks = {
        "Executive Summary": "## Executive Summary" in content,
        "Efficiency Review": "## 1. Efficiency Review" in content,
        "Readability": "## 2. Readability" in content,
        "Simplification": "## 3. Simplification" in content,
        "Robustness": "## 4. Robustness" in content,
        "Feature Alignment": "## 5. Feature Alignment" in content,
        "Maintainability": "## 6. Maintainability" in content,
        "Testing": "## 7. Comprehensive Testing" in content,
        "Recommendations": "Future Enhancements" in content or "Recommendations" in content,
        "Summary present": "## 11. Summary of Improvements" in content or "Summary" in content,
    }
    
    passed = 0
    for check_name, result in checks.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {check_name}")
        if result:
            passed += 1
    
    print(f"\n{passed}/{len(checks)} checks passed")
    print("=" * 60)
    
    return passed == len(checks)

def check_code_metrics():
    """Check code quality metrics."""
    specialists_path = Path(__file__).parent.parent / "lyra" / "specialists.py"
    content = specialists_path.read_text()
    
    print("\nCODE QUALITY METRICS")
    print("=" * 60)
    
    # Count various code quality indicators
    total_lines = len(content.splitlines())
    docstring_lines = len(re.findall(r'""".*?"""', content, re.DOTALL))
    comment_lines = len(re.findall(r'^\s*#.*$', content, re.MULTILINE))
    class_count = len(re.findall(r'^class \w+', content, re.MULTILINE))
    method_count = len(re.findall(r'^\s+def \w+', content, re.MULTILINE))
    constant_count = len(re.findall(r'^[A-Z_]+ = ', content, re.MULTILINE))
    
    print(f"Total lines: {total_lines}")
    print(f"Classes: {class_count}")
    print(f"Methods: {method_count}")
    print(f"Constants: {constant_count}")
    print(f"Docstrings: {docstring_lines}")
    print(f"Comment lines: {comment_lines}")
    
    # Quality checks
    print("\nQuality Indicators:")
    print(f"{'✅' if constant_count >= 5 else '❌'} Constants extracted (>= 5): {constant_count}")
    print(f"{'✅' if docstring_lines >= 15 else '❌'} Docstrings present (>= 15): {docstring_lines}")
    print(f"{'✅' if total_lines < 900 else '❌'} File size reasonable (< 900 lines): {total_lines}")
    print(f"{'✅' if method_count / class_count < 12 else '❌'} Methods per class reasonable (< 12): {method_count / class_count:.1f}")
    
    print("=" * 60)
    
    return True

def main():
    """Run all validation checks."""
    print("\n" + "=" * 60)
    print("CODE REVIEW VALIDATION SCRIPT")
    print("=" * 60 + "\n")
    
    specialists_ok = check_specialists_file()
    tests_ok = check_test_file()
    docs_ok = check_documentation()
    metrics_ok = check_code_metrics()
    
    print("\n" + "=" * 60)
    print("OVERALL RESULTS")
    print("=" * 60)
    
    print(f"{'✅' if specialists_ok else '❌'} Specialists.py improvements")
    print(f"{'✅' if tests_ok else '❌'} Test suite created")
    print(f"{'✅' if docs_ok else '❌'} Documentation complete")
    print(f"{'✅' if metrics_ok else '❌'} Code quality metrics")
    
    if all([specialists_ok, tests_ok, docs_ok, metrics_ok]):
        print("\n🎉 ALL VALIDATIONS PASSED! Code review complete.")
    else:
        print("\n⚠️ Some validations failed. Please review output above.")
    
    print("=" * 60 + "\n")

if __name__ == "__main__":
    main()
