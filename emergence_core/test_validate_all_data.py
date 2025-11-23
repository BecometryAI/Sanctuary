#!/usr/bin/env python3
"""
Comprehensive Test Suite for Data Validation System

Tests cover:
1. Edge cases (empty files, malformed JSON, missing keys)
2. File formatting validation (trailing newlines, line endings)
3. Pydantic model validation (valid/invalid journal entries)
4. Directory traversal (recursive, exclusions, missing directories)
5. Error handling and reporting
6. Statistics tracking

Usage:
    pytest test_validate_all_data.py -v
    pytest test_validate_all_data.py -v --cov=validate_all_data
"""

import json
import pytest
import tempfile
from pathlib import Path
from datetime import datetime

from validate_all_data import DataValidator, ValidationStats


class TestValidationStats:
    """Test the ValidationStats class."""
    
    def test_initialization(self):
        """Test all counters initialize to zero."""
        stats = ValidationStats()
        assert stats.valid_journal_entries == 0
        assert stats.invalid_journal_entries == 0
        assert stats.valid_json_files == 0
        assert stats.invalid_json_files == 0
        assert stats.missing_trailing_newline == 0
        assert stats.total_files_checked == 0
    
    def test_increment_valid_counter(self):
        """Test incrementing valid counters."""
        stats = ValidationStats()
        stats.increment('valid_journal_entries')
        assert stats.valid_journal_entries == 1
        stats.increment('valid_journal_entries')
        assert stats.valid_journal_entries == 2
    
    def test_increment_invalid_counter(self):
        """Test incrementing invalid counter raises error."""
        stats = ValidationStats()
        with pytest.raises(ValueError, match="Invalid counter name"):
            stats.increment('nonexistent_counter')
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        stats = ValidationStats()
        stats.valid_journal_entries = 5
        stats.invalid_json_files = 2
        
        result = stats.to_dict()
        assert result['valid_journal_entries'] == 5
        assert result['invalid_json_files'] == 2
        assert isinstance(result, dict)


class TestDataValidatorInitialization:
    """Test DataValidator initialization and setup."""
    
    def test_init_with_valid_directory(self, tmp_path):
        """Test initialization with valid directory."""
        validator = DataValidator(tmp_path)
        assert validator.data_dir == tmp_path
        assert validator.verbose == False
        assert isinstance(validator.stats, ValidationStats)
    
    def test_init_with_nonexistent_directory(self):
        """Test initialization fails with nonexistent directory."""
        with pytest.raises(ValueError, match="Data directory not found"):
            DataValidator(Path("/nonexistent/path"))
    
    def test_init_with_file_not_directory(self, tmp_path):
        """Test initialization fails when path is a file."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("test")
        
        with pytest.raises(ValueError, match="Path is not a directory"):
            DataValidator(test_file)
    
    def test_init_verbose_mode(self, tmp_path):
        """Test verbose mode is set correctly."""
        validator = DataValidator(tmp_path, verbose=True)
        assert validator.verbose == True


class TestFileFormattingValidation:
    """Test file formatting checks."""
    
    def test_file_with_trailing_newline(self, tmp_path):
        """Test file with proper trailing newline passes."""
        test_file = tmp_path / "test.json"
        test_file.write_text('{"test": true}\n')
        
        validator = DataValidator(tmp_path)
        warnings = validator.check_file_formatting(test_file)
        
        assert len(warnings) == 0
        assert validator.stats.missing_trailing_newline == 0
    
    def test_file_without_trailing_newline(self, tmp_path):
        """Test file without trailing newline generates warning."""
        test_file = tmp_path / "test.json"
        test_file.write_text('{"test": true}')  # No newline
        
        validator = DataValidator(tmp_path)
        warnings = validator.check_file_formatting(test_file)
        
        assert len(warnings) == 1
        assert "Missing trailing newline" in warnings[0]
        assert validator.stats.missing_trailing_newline == 1
    
    def test_empty_file(self, tmp_path):
        """Test empty file generates warning."""
        test_file = tmp_path / "empty.json"
        test_file.write_text('')
        
        validator = DataValidator(tmp_path)
        warnings = validator.check_file_formatting(test_file)
        
        assert len(warnings) == 1
        assert validator.stats.missing_trailing_newline == 1
    
    def test_file_with_multiple_newlines(self, tmp_path):
        """Test file with multiple trailing newlines still passes."""
        test_file = tmp_path / "test.json"
        test_file.write_text('{"test": true}\n\n\n')
        
        validator = DataValidator(tmp_path)
        warnings = validator.check_file_formatting(test_file)
        
        assert len(warnings) == 0


class TestJSONValidation:
    """Test JSON file validation."""
    
    def test_valid_json_file(self, tmp_path):
        """Test valid JSON file passes."""
        test_file = tmp_path / "test.json"
        test_file.write_text('{"key": "value"}\n')
        
        validator = DataValidator(tmp_path)
        success, errors = validator.validate_json_file(test_file)
        
        assert success == True
        assert len(errors) == 0
        assert validator.stats.valid_json_files == 1
    
    def test_invalid_json_syntax(self, tmp_path):
        """Test invalid JSON syntax fails."""
        test_file = tmp_path / "invalid.json"
        test_file.write_text('{"key": invalid}\n')
        
        validator = DataValidator(tmp_path)
        success, errors = validator.validate_json_file(test_file)
        
        assert success == False
        assert len(errors) > 0
        assert any("JSON decode error" in e for e in errors)
        assert validator.stats.invalid_json_files == 1
    
    def test_archive_file_must_be_dict(self, tmp_path):
        """Test archive files must be dicts, not arrays."""
        test_file = tmp_path / "test_archive.json"
        test_file.write_text('[{"data": "value"}]\n')
        
        validator = DataValidator(tmp_path)
        success, errors = validator.validate_json_file(test_file)
        
        assert success == False
        assert any("Archive files should be dicts" in e for e in errors)
    
    def test_index_file_can_be_dict_or_array(self, tmp_path):
        """Test index files can be either dict or array."""
        # Test dict
        test_file1 = tmp_path / "test_index.json"
        test_file1.write_text('{"entries": []}\n')
        
        validator = DataValidator(tmp_path)
        success, errors = validator.validate_json_file(test_file1)
        assert success == True
        
        # Test array
        test_file2 = tmp_path / "another_index.json"
        test_file2.write_text('[1, 2, 3]\n')
        
        success, errors = validator.validate_json_file(test_file2)
        assert success == True


class TestJournalValidation:
    """Test journal entry validation."""
    
    def create_valid_journal_entry(self):
        """Helper to create a valid journal entry."""
        return {
            "journal_entry": {
                "timestamp": "2025-11-23T10:00:00-08:00",
                "label": "Test Entry",
                "entry_type": "journal",
                "emotional_tone": ["calm", "focused"],
                "description": "Test description",
                "key_insights": ["Insight 1"],
                "lyra_reflection": "Test reflection",
                "tags": ["test"],
                "stewardship_trace": {
                    "committed_by": "Test",
                    "witnessed_by": "Test",
                    "commitment_type": "test",
                    "reason": "Testing"
                }
            }
        }
    
    def test_valid_journal_file(self, tmp_path):
        """Test valid journal file passes validation."""
        test_file = tmp_path / "2025-11-23.json"
        journal_data = [self.create_valid_journal_entry()]
        test_file.write_text(json.dumps(journal_data) + '\n')
        
        validator = DataValidator(tmp_path)
        success, errors = validator.validate_journal_file(test_file)
        
        assert success == True
        assert validator.stats.valid_journal_entries == 1
        assert validator.stats.invalid_journal_entries == 0
    
    def test_journal_file_must_be_array(self, tmp_path):
        """Test journal file must be an array."""
        test_file = tmp_path / "2025-11-23.json"
        test_file.write_text('{"journal_entry": {}}\n')
        
        validator = DataValidator(tmp_path)
        success, errors = validator.validate_journal_file(test_file)
        
        assert success == False
        assert any("Expected array" in e for e in errors)
    
    def test_journal_entry_missing_key(self, tmp_path):
        """Test journal entry missing required key."""
        test_file = tmp_path / "2025-11-23.json"
        incomplete_entry = {
            "journal_entry": {
                "timestamp": "2025-11-23T10:00:00-08:00",
                # Missing required fields
            }
        }
        test_file.write_text(json.dumps([incomplete_entry]) + '\n')
        
        validator = DataValidator(tmp_path)
        success, errors = validator.validate_journal_file(test_file)
        
        assert success == False
        assert validator.stats.invalid_journal_entries == 1
    
    def test_journal_entry_wrong_structure(self, tmp_path):
        """Test journal entry with wrong structure."""
        test_file = tmp_path / "2025-11-23.json"
        wrong_structure = [
            {"wrong_key": {"data": "value"}}  # Should be "journal_entry"
        ]
        test_file.write_text(json.dumps(wrong_structure) + '\n')
        
        validator = DataValidator(tmp_path)
        success, errors = validator.validate_journal_file(test_file)
        
        assert success == False
        assert any("Missing 'journal_entry' key" in e for e in errors)
    
    def test_multiple_journal_entries(self, tmp_path):
        """Test file with multiple journal entries."""
        test_file = tmp_path / "2025-11-23.json"
        entries = [
            self.create_valid_journal_entry(),
            self.create_valid_journal_entry()
        ]
        test_file.write_text(json.dumps(entries) + '\n')
        
        validator = DataValidator(tmp_path)
        success, errors = validator.validate_journal_file(test_file)
        
        assert success == True
        assert validator.stats.valid_journal_entries == 2


class TestDirectoryValidation:
    """Test directory validation functionality."""
    
    def test_validate_empty_directory(self, tmp_path):
        """Test validating empty directory."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        
        validator = DataValidator(tmp_path)
        results = validator.validate_directory("empty")
        
        assert results['total_files'] == 0
        assert results['valid_files'] == 0
        assert results['invalid_files'] == 0
    
    def test_validate_nonexistent_directory(self, tmp_path):
        """Test validating nonexistent directory."""
        validator = DataValidator(tmp_path)
        results = validator.validate_directory("nonexistent")
        
        assert results.get('skipped') == True
        assert results.get('reason') == 'not_found'
    
    def test_recursive_validation(self, tmp_path):
        """Test recursive directory validation."""
        # Create nested structure
        subdir = tmp_path / "test" / "nested"
        subdir.mkdir(parents=True)
        
        (tmp_path / "test" / "file1.json").write_text('{"a": 1}\n')
        (subdir / "file2.json").write_text('{"b": 2}\n')
        
        validator = DataValidator(tmp_path)
        results = validator.validate_directory("test", recursive=True)
        
        assert results['total_files'] == 2
        assert results['valid_files'] == 2
    
    def test_non_recursive_validation(self, tmp_path):
        """Test non-recursive directory validation."""
        # Create nested structure
        subdir = tmp_path / "test" / "nested"
        subdir.mkdir(parents=True)
        
        (tmp_path / "test" / "file1.json").write_text('{"a": 1}\n')
        (subdir / "file2.json").write_text('{"b": 2}\n')
        
        validator = DataValidator(tmp_path)
        results = validator.validate_directory("test", recursive=False)
        
        assert results['total_files'] == 1  # Only file1, not file2
        assert results['valid_files'] == 1
    
    def test_exclude_files(self, tmp_path):
        """Test excluding files from validation."""
        test_dir = tmp_path / "test"
        test_dir.mkdir()
        
        (test_dir / "include.json").write_text('{"a": 1}\n')
        (test_dir / "exclude.json").write_text('{"b": 2}\n')
        
        validator = DataValidator(tmp_path)
        results = validator.validate_directory(
            "test", 
            exclude_files=['exclude.json']
        )
        
        assert results['total_files'] == 1
        assert results['valid_files'] == 1


class TestIsDateJournalFile:
    """Test the _is_date_journal_file helper method."""
    
    def test_date_journal_file(self, tmp_path):
        """Test file that is a date journal."""
        journal_dir = tmp_path / "journal"
        journal_dir.mkdir()
        test_file = journal_dir / "2025-11-23.json"
        test_file.touch()
        
        validator = DataValidator(tmp_path)
        assert validator._is_date_journal_file(test_file) == True
    
    def test_non_date_journal_file(self, tmp_path):
        """Test file in journal/ but not date-named."""
        journal_dir = tmp_path / "journal"
        journal_dir.mkdir()
        test_file = journal_dir / "manifest.json"
        test_file.touch()
        
        validator = DataValidator(tmp_path)
        assert validator._is_date_journal_file(test_file) == False
    
    def test_file_outside_journal(self, tmp_path):
        """Test date-named file outside journal/ directory."""
        other_dir = tmp_path / "other"
        other_dir.mkdir()
        test_file = other_dir / "2025-11-23.json"
        test_file.touch()
        
        validator = DataValidator(tmp_path)
        assert validator._is_date_journal_file(test_file) == False


class TestReportGeneration:
    """Test validation report generation and saving."""
    
    def test_save_report(self, tmp_path):
        """Test saving validation report."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        
        validator = DataValidator(data_dir)
        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': {'total_files': 0}
        }
        
        output_file = tmp_path / "report.json"
        validator.save_report(report, output_file)
        
        assert output_file.exists()
        
        # Verify it's valid JSON
        with open(output_file, 'r') as f:
            loaded_report = json.load(f)
        
        assert loaded_report['summary']['total_files'] == 0
        
        # Verify trailing newline was added
        with open(output_file, 'rb') as f:
            content = f.read()
        assert content.endswith(b'\n')


class TestEdgeCases:
    """Test edge cases and unusual inputs."""
    
    def test_unicode_in_json(self, tmp_path):
        """Test JSON with Unicode characters."""
        test_file = tmp_path / "unicode.json"
        test_file.write_text('{"emoji": "🚀", "chinese": "你好"}\n', encoding='utf-8')
        
        validator = DataValidator(tmp_path)
        success, errors = validator.validate_json_file(test_file)
        
        assert success == True
    
    def test_very_large_json(self, tmp_path):
        """Test handling of large JSON file."""
        test_file = tmp_path / "large.json"
        large_data = {f"key_{i}": f"value_{i}" for i in range(10000)}
        test_file.write_text(json.dumps(large_data) + '\n')
        
        validator = DataValidator(tmp_path)
        success, errors = validator.validate_json_file(test_file)
        
        assert success == True
    
    def test_deeply_nested_json(self, tmp_path):
        """Test deeply nested JSON structure."""
        test_file = tmp_path / "nested.json"
        nested = {"a": {"b": {"c": {"d": {"e": "value"}}}}}
        test_file.write_text(json.dumps(nested) + '\n')
        
        validator = DataValidator(tmp_path)
        success, errors = validator.validate_json_file(test_file)
        
        assert success == True
    
    def test_json_with_null_values(self, tmp_path):
        """Test JSON with null values."""
        test_file = tmp_path / "nulls.json"
        test_file.write_text('{"key": null, "arr": [null, 1, null]}\n')
        
        validator = DataValidator(tmp_path)
        success, errors = validator.validate_json_file(test_file)
        
        assert success == True
    
    def test_binary_file_as_json(self, tmp_path):
        """Test attempting to validate binary file fails gracefully."""
        test_file = tmp_path / "binary.json"
        test_file.write_bytes(b'\x00\x01\x02\x03')
        
        validator = DataValidator(tmp_path)
        success, errors = validator.validate_json_file(test_file)
        
        assert success == False
        assert len(errors) > 0


# Pytest fixtures
@pytest.fixture
def sample_data_structure(tmp_path):
    """Create a sample data directory structure for integration tests."""
    # Create directory structure
    (tmp_path / "journal").mkdir()
    (tmp_path / "Core_Archives").mkdir()
    (tmp_path / "Protocols").mkdir()
    
    # Create sample files
    (tmp_path / "journal" / "2025-11-23.json").write_text('[{"journal_entry": {}}]\n')
    (tmp_path / "Core_Archives" / "archive.json").write_text('{"data": "value"}\n')
    (tmp_path / "Protocols" / "protocol.json").write_text('{"name": "test"}\n')
    
    return tmp_path


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=validate_all_data", "--cov-report=html"])
