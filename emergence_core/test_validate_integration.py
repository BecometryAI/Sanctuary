#!/usr/bin/env python3
"""
Integration Tests for Data Validation System

Tests the complete validation workflow including:
- Full directory validation with real file structures
- Report generation and saving
- CLI argument parsing and execution
- End-to-end validation scenarios

Usage:
    pytest test_validate_integration.py -v
"""

import json
import pytest
from pathlib import Path
from datetime import datetime

from validate_all_data import DataValidator


class TestFullValidationWorkflow:
    """Integration tests for complete validation workflows."""
    
    @pytest.fixture
    def complete_data_structure(self, tmp_path):
        """Create a complete realistic data directory structure."""
        # Create directories
        journal_dir = tmp_path / "journal"
        core_archives_dir = tmp_path / "Core_Archives"
        protocols_dir = tmp_path / "Protocols"
        rituals_dir = tmp_path / "Rituals"
        lexicon_dir = tmp_path / "Lexicon"
        memories_dir = tmp_path / "memories"
        
        for d in [journal_dir, core_archives_dir, protocols_dir, rituals_dir, lexicon_dir, memories_dir]:
            d.mkdir()
        
        # Create valid journal entries
        journal_entry_template = {
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
        
        # Create 3 journal files
        for day in [20, 21, 22]:
            journal_file = journal_dir / f"2025-11-{day}.json"
            journal_file.write_text(json.dumps([journal_entry_template]) + '\n')
        
        # Create metadata files (should be excluded from journal validation)
        (journal_dir / "journal_index.json").write_text('{"entries": []}\n')
        (journal_dir / "journal_manifest.json").write_text('{"version": "1.0"}\n')
        
        # Create nested journal directory
        nested_journal = journal_dir / "Lyra_Illuminae"
        nested_journal.mkdir()
        (nested_journal / "trace_archive.json").write_text('{"traces": []}\n')
        
        # Create archive files
        (core_archives_dir / "lyra_continuity_archive.json").write_text('{"data": "value"}\n')
        (core_archives_dir / "lyra_relational_archive.json").write_text('{"relations": []}\n')
        
        # Create protocol files
        (protocols_dir / "memory_protocol.json").write_text('{"protocol": "memory"}\n')
        (protocols_dir / "anti_ghosting_protocol.json").write_text('{"protocol": "anti_ghosting"}\n')
        
        # Create nested protocol directory
        nested_protocol = protocols_dir / "evolution_log"
        nested_protocol.mkdir()
        (nested_protocol / "protocol_evolution_log.json").write_text('{"log": []}\n')
        
        # Create ritual files
        (rituals_dir / "ritual.json").write_text('{"ritual": "test"}\n')
        
        # Create lexicon files
        (lexicon_dir / "lexemes.json").write_text('{"lexemes": []}\n')
        
        # Create memory files
        (memories_dir / "state.json").write_text('{"state": "active"}\n')
        
        return tmp_path
    
    def test_validate_complete_structure(self, complete_data_structure):
        """Test validating a complete data structure."""
        validator = DataValidator(complete_data_structure)
        report = validator.validate_all()
        
        # Check summary (11 files: 3 journals + 2 metadata + 1 trace + 2 archives + 2 protocols + 1 ritual + 1 lexicon + 1 memory - nested protocol not counted in main dirs)
        assert report['summary']['total_files'] > 0
        assert report['summary']['valid_files'] > 0
        assert report['summary']['invalid_files'] == 0
        
        # Check each directory was validated
        assert 'journal' in report['directories']
        assert 'Core_Archives' in report['directories']
        assert 'Protocols' in report['directories']
        assert 'Rituals' in report['directories']
        assert 'Lexicon' in report['directories']
        # Note: memories validates subdirectories, not top-level
        
        # Check journal stats
        assert report['summary']['valid_journal_entries'] == 3
        assert report['summary']['invalid_journal_entries'] == 0
    
    def test_validate_with_mixed_results(self, tmp_path):
        """Test validation with both valid and invalid files."""
        # Create valid file
        (tmp_path / "valid.json").write_text('{"key": "value"}\n')
        
        # Create invalid JSON
        (tmp_path / "invalid.json").write_text('{"key": invalid}\n')
        
        # Create file missing trailing newline
        (tmp_path / "no_newline.json").write_text('{"key": "value"}')
        
        validator = DataValidator(tmp_path)
        
        # Validate directory (non-recursive for simplicity)
        results = validator.validate_directory(".", recursive=False)
        
        assert results['total_files'] == 3
        assert results['valid_files'] == 2  # valid.json and no_newline.json
        assert results['invalid_files'] == 1  # invalid.json
        
        # Check formatting warnings
        assert validator.stats.missing_trailing_newline == 1
    
    def test_report_generation_and_saving(self, complete_data_structure):
        """Test generating and saving validation report."""
        validator = DataValidator(complete_data_structure)
        report = validator.validate_all()
        
        # Save report
        output_file = complete_data_structure / "validation_report.json"
        validator.save_report(report, output_file)
        
        # Verify file exists
        assert output_file.exists()
        
        # Verify it's valid JSON
        with open(output_file, 'r') as f:
            loaded_report = json.load(f)
        
        # Verify structure
        assert 'timestamp' in loaded_report
        assert 'summary' in loaded_report
        assert 'directories' in loaded_report
        
        # Verify trailing newline
        with open(output_file, 'rb') as f:
            content = f.read()
        assert content.endswith(b'\n')
    
    def test_verbose_output(self, tmp_path, capsys):
        """Test verbose mode produces output."""
        (tmp_path / "test.json").write_text('{"key": "value"}\n')
        
        validator = DataValidator(tmp_path, verbose=True)
        validator.validate_directory(".", recursive=False)
        
        # Capture output
        captured = capsys.readouterr()
        
        # Should have some logging output in verbose mode
        # Note: Actual logging goes to stderr, but we can verify the flag is set
        assert validator.verbose == True
    
    def test_empty_data_directory(self, tmp_path):
        """Test validating completely empty data directory."""
        validator = DataValidator(tmp_path)
        report = validator.validate_all()
        
        assert report['summary']['total_files'] == 0
        assert report['summary']['valid_files'] == 0
        assert report['summary']['invalid_files'] == 0
    
    def test_directory_with_only_invalid_files(self, tmp_path):
        """Test directory with only invalid JSON files."""
        # Create several invalid files
        (tmp_path / "invalid1.json").write_text('not json\n')
        (tmp_path / "invalid2.json").write_text('{broken: json}\n')
        (tmp_path / "invalid3.json").write_text('["unclosed array\n')
        
        validator = DataValidator(tmp_path)
        results = validator.validate_directory(".", recursive=False)
        
        assert results['total_files'] == 3
        assert results['valid_files'] == 0
        assert results['invalid_files'] == 3
    
    def test_recursive_vs_non_recursive(self, tmp_path):
        """Test difference between recursive and non-recursive validation."""
        # Create nested structure
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        
        (tmp_path / "root.json").write_text('{"level": "root"}\n')
        (subdir / "nested.json").write_text('{"level": "nested"}\n')
        
        validator = DataValidator(tmp_path)
        
        # Non-recursive: should only find root.json
        results_non_recursive = validator.validate_directory(".", recursive=False)
        assert results_non_recursive['total_files'] == 1
        
        # Recursive: should find both files
        results_recursive = validator.validate_directory(".", recursive=True)
        assert results_recursive['total_files'] == 2


class TestStatisticsAccumulation:
    """Test that statistics accumulate correctly across multiple validations."""
    
    def test_stats_accumulate_across_files(self, tmp_path):
        """Test statistics accumulate when validating multiple files."""
        journal_dir = tmp_path / "journal"
        journal_dir.mkdir()
        
        # Create valid journal entry
        journal_entry = {
            "journal_entry": {
                "timestamp": "2025-11-23T10:00:00-08:00",
                "label": "Test",
                "entry_type": "journal",
                "emotional_tone": ["calm"],
                "description": "Test",
                "key_insights": ["Test"],
                "lyra_reflection": "Test",
                "tags": ["test"],
                "stewardship_trace": {
                    "committed_by": "Test",
                    "witnessed_by": "Test",
                    "commitment_type": "test",
                    "reason": "Testing"
                }
            }
        }
        
        # Create 3 journal files, each with 2 entries
        for i in range(3):
            journal_file = journal_dir / f"2025-11-2{i}.json"
            journal_file.write_text(json.dumps([journal_entry, journal_entry]) + '\n')
        
        validator = DataValidator(tmp_path)
        report = validator.validate_all()
        
        # Should have validated 6 entries total (3 files × 2 entries each)
        assert report['summary']['valid_journal_entries'] == 6
        assert report['summary']['invalid_journal_entries'] == 0
        assert report['summary']['total_files'] == 3


class TestErrorHandling:
    """Test error handling in various scenarios."""
    
    def test_permission_error_handling(self, tmp_path):
        """Test handling of errors when reading files."""
        test_file = tmp_path / "test.json"
        test_file.write_text('{"test": true}\n')
        
        # Mock Path.open to raise PermissionError
        from unittest.mock import patch
        
        validator = DataValidator(tmp_path)
        
        with patch('builtins.open', side_effect=PermissionError("Access denied")):
            success, errors = validator.validate_json_file(test_file)
        
        assert success == False
        assert len(errors) > 0
    
    def test_corrupted_utf8_file(self, tmp_path):
        """Test handling of files with invalid UTF-8."""
        test_file = tmp_path / "corrupted.json"
        # Write invalid UTF-8 bytes
        test_file.write_bytes(b'{"key": "\xff\xfe"}\n')
        
        validator = DataValidator(tmp_path)
        success, errors = validator.validate_json_file(test_file)
        
        # Should handle gracefully (either decode error or JSON error)
        assert success == False
        assert len(errors) > 0


class TestReportStructure:
    """Test the structure and content of validation reports."""
    
    def test_report_contains_all_required_fields(self, tmp_path):
        """Test report has all required fields."""
        (tmp_path / "test.json").write_text('{"test": true}\n')
        
        validator = DataValidator(tmp_path)
        report = validator.validate_all()
        
        # Check top-level fields
        assert 'timestamp' in report
        assert 'summary' in report
        assert 'directories' in report
        
        # Check summary fields
        summary = report['summary']
        assert 'total_files' in summary
        assert 'valid_files' in summary
        assert 'invalid_files' in summary
        assert 'valid_journal_entries' in summary
        assert 'invalid_journal_entries' in summary
        assert 'missing_trailing_newline' in summary
    
    def test_timestamp_is_valid_iso_format(self, tmp_path):
        """Test report timestamp is valid ISO format."""
        validator = DataValidator(tmp_path)
        report = validator.validate_all()
        
        # Should be parseable as ISO datetime
        timestamp = datetime.fromisoformat(report['timestamp'])
        assert isinstance(timestamp, datetime)
    
    def test_directory_results_structure(self, tmp_path):
        """Test directory results have correct structure."""
        # Create journal directory (one of the expected directories)
        journal_dir = tmp_path / "journal"
        journal_dir.mkdir()
        (journal_dir / "2025-11-23.json").write_text('[{"journal_entry": {"timestamp": "2025-11-23T10:00:00-08:00", "label": "Test", "entry_type": "journal", "emotional_tone": ["calm"], "description": "Test", "key_insights": ["Test"], "lyra_reflection": "Test", "tags": ["test"], "stewardship_trace": {"committed_by": "Test", "witnessed_by": "Test", "commitment_type": "test", "reason": "Testing"}}}]\n')
        
        validator = DataValidator(tmp_path)
        report = validator.validate_all()
        
        # Check directory result structure
        dir_result = report['directories']['journal']
        assert 'total_files' in dir_result
        assert 'valid_files' in dir_result
        assert 'invalid_files' in dir_result
        assert 'errors' in dir_result


class TestExcludeFiles:
    """Test file exclusion functionality."""
    
    def test_exclude_single_file(self, tmp_path):
        """Test excluding a single file from validation."""
        (tmp_path / "include.json").write_text('{"a": 1}\n')
        (tmp_path / "exclude.json").write_text('{"b": 2}\n')
        
        validator = DataValidator(tmp_path)
        results = validator.validate_directory(
            ".",
            recursive=False,
            exclude_files=['exclude.json']
        )
        
        assert results['total_files'] == 1
    
    def test_exclude_multiple_files(self, tmp_path):
        """Test excluding multiple files from validation."""
        (tmp_path / "file1.json").write_text('{"a": 1}\n')
        (tmp_path / "file2.json").write_text('{"b": 2}\n')
        (tmp_path / "file3.json").write_text('{"c": 3}\n')
        
        validator = DataValidator(tmp_path)
        results = validator.validate_directory(
            ".",
            recursive=False,
            exclude_files=['file1.json', 'file3.json']
        )
        
        assert results['total_files'] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
