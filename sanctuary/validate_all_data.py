#!/usr/bin/env python3
"""
Comprehensive Data Validation and Formatting Tool

This script validates all JSON files in the Sanctuary data directory, ensuring:
1. Valid JSON syntax
2. Correct Pydantic model compliance (for journal entries)
3. Proper file formatting (POSIX compliance, trailing newlines)
4. Structural integrity (archives are dicts, indices are dicts, etc.)

Directory Structure Validated:
- journal/: Daily journal entries with LegacyJournalEntry validation
- Core_Archives/: Archive files (continuity, relational, sovereign charter)
- Protocols/: Protocol definitions
- Rituals/: Ritual definitions
- Lexicon/: Lexicon entries
- memories/: Memory storage subdirectories

Usage Examples:
    # Basic validation
    python validate_all_data.py
    
    # Verbose output with detailed error messages
    python validate_all_data.py --verbose
    
    # Save detailed report to file
    python validate_all_data.py --report validation_report.json
    
    # Custom data directory
    python validate_all_data.py --data-dir /path/to/data

Exit Codes:
    0: All files valid
    1: Validation errors found or execution error

Author: Sanctuary Emergence Team
Date: November 23, 2025
Version: 2.0
"""

import json
import logging
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime

from pydantic import ValidationError

from mind.legacy_parser import LegacyJournalEntry

# Configure logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class ValidationStats:
    """
    Tracks validation statistics.
    
    Separated into its own class for better organization and easier testing.
    """
    
    def __init__(self):
        """Initialize all counters to zero."""
        self.valid_journal_entries: int = 0
        self.invalid_journal_entries: int = 0
        self.valid_json_files: int = 0
        self.invalid_json_files: int = 0
        self.missing_trailing_newline: int = 0
        self.total_files_checked: int = 0
    
    def to_dict(self) -> Dict[str, int]:
        """Convert stats to dictionary for JSON serialization."""
        return {
            'valid_journal_entries': self.valid_journal_entries,
            'invalid_journal_entries': self.invalid_journal_entries,
            'valid_json_files': self.valid_json_files,
            'invalid_json_files': self.invalid_json_files,
            'missing_trailing_newline': self.missing_trailing_newline,
            'total_files_checked': self.total_files_checked
        }
    
    def increment(self, counter: str) -> None:
        """
        Safely increment a counter.
        
        Args:
            counter: Name of counter to increment
            
        Raises:
            ValueError: If counter name is invalid
        """
        if not hasattr(self, counter):
            raise ValueError(f"Invalid counter name: {counter}")
        setattr(self, counter, getattr(self, counter) + 1)


class DataValidator:
    """
    Validates all JSON files in the data directory.
    
    This class handles:
    - JSON syntax validation
    - Pydantic model validation for journal entries
    - File formatting checks (POSIX compliance)
    - Structural validation (archives, indices, etc.)
    """
    
    # Constants for file type detection
    DATE_PREFIX = '20'  # Files starting with '20' are date-named journals
    METADATA_FILES = {'journal_index.json', 'journal_manifest.json'}
    
    def __init__(self, data_dir: Path, verbose: bool = False):
        """
        Initialize validator.
        
        Args:
            data_dir: Path to data directory
            verbose: Whether to show detailed output
            
        Raises:
            ValueError: If data directory doesn't exist
        """
        self.data_dir = Path(data_dir)
        self.verbose = verbose
        self.stats = ValidationStats()
        
        if not self.data_dir.exists():
            raise ValueError(f"Data directory not found: {data_dir}")
        
        if not self.data_dir.is_dir():
            raise ValueError(f"Path is not a directory: {data_dir}")
    
    def check_file_formatting(self, filepath: Path) -> List[str]:
        """
        Check file formatting standards (POSIX compliance).
        
        POSIX standard requires text files to end with a newline character.
        This ensures compatibility across different tools and systems.
        
        Args:
            filepath: Path to file to check
            
        Returns:
            List of formatting warnings (empty if all checks pass)
        """
        warnings = []
        
        try:
            # Read in binary mode to handle all line ending types correctly
            with open(filepath, 'rb') as f:
                content = f.read()
            
            # Check for trailing newline (POSIX standard)
            if not content.endswith(b'\n'):
                warnings.append("Missing trailing newline (POSIX standard)")
                self.stats.increment('missing_trailing_newline')
            
            # Future checks can be added here:
            # - Mixed line endings (CR, LF, CRLF)
            # - Trailing whitespace on lines
            # - Tab vs space consistency
            # - File size limits
            
        except IOError as e:
            warnings.append(f"Could not read file for formatting check: {e}")
        except Exception as e:
            warnings.append(f"Unexpected error during formatting check: {e}")
        
        return warnings
    
    def validate_journal_file(self, filepath: Path) -> Tuple[bool, List[str]]:
        """
        Validate a journal JSON file using LegacyJournalEntry model.
        
        Journal files must be arrays of objects where each object contains
        a 'journal_entry' key with the entry data.
        
        Args:
            filepath: Path to journal file
            
        Returns:
            Tuple of (success: bool, errors: List[str])
            success is True only if all validations pass
        """
        errors = []
        
        try:
            # Load JSON
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Check file formatting (non-blocking warnings)
            format_warnings = self.check_file_formatting(filepath)
            if format_warnings and self.verbose:
                for warning in format_warnings:
                    errors.append(f"Format warning: {warning}")
            
            # Validate structure - must be an array
            if not isinstance(data, list):
                errors.append(f"Expected array, got {type(data).__name__}")
                return False, errors
            
            # Validate each entry
            for idx, item in enumerate(data):
                if not isinstance(item, dict):
                    errors.append(f"Entry {idx}: Expected dict, got {type(item).__name__}")
                    continue
                
                if "journal_entry" not in item:
                    errors.append(f"Entry {idx}: Missing 'journal_entry' key")
                    continue
                
                # Validate using LegacyJournalEntry Pydantic model
                try:
                    entry_data = item["journal_entry"]
                    LegacyJournalEntry(**entry_data)
                    self.stats.increment('valid_journal_entries')
                    
                except ValidationError as e:
                    self.stats.increment('invalid_journal_entries')
                    errors.append(f"Entry {idx} validation error: {e.error_count()} issues")
                    if self.verbose:
                        for error in e.errors():
                            loc = ' -> '.join(str(l) for l in error['loc'])
                            errors.append(f"  - {loc}: {error['msg']}")
            
            # Success only if no errors (format warnings don't count)
            has_errors = any(not e.startswith("Format warning:") for e in errors)
            return not has_errors, errors
            
        except json.JSONDecodeError as e:
            errors.append(f"JSON decode error: {e}")
            return False, errors
        except IOError as e:
            errors.append(f"File I/O error: {e}")
            return False, errors
        except Exception as e:
            errors.append(f"Unexpected error: {type(e).__name__}: {e}")
            return False, errors
    
    def validate_json_file(self, filepath: Path, context: str = None) -> Tuple[bool, List[str]]:
        """
        Validate a generic JSON file.
        
        Performs basic validation:
        - Valid JSON syntax
        - File formatting checks
        - Structural validation based on filename patterns
        
        Args:
            filepath: Path to JSON file
            context: Optional context string (directory name) for better error messages
            
        Returns:
            Tuple of (success: bool, errors: List[str])
        """
        errors = []
        
        try:
            # Load JSON
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Check file formatting (non-blocking warnings)
            format_warnings = self.check_file_formatting(filepath)
            if format_warnings and self.verbose:
                for warning in format_warnings:
                    errors.append(f"Format warning: {warning}")
            
            # Increment valid count
            self.stats.increment('valid_json_files')
            
            # Structural validation based on filename patterns
            filename_lower = filepath.stem.lower()
            
            if 'archive' in filename_lower:
                if not isinstance(data, dict):
                    errors.append(f"Archive files should be dicts, got {type(data).__name__}")
            
            if 'index' in filename_lower:
                if not isinstance(data, (dict, list)):
                    errors.append(f"Index files should be dicts or arrays, got {type(data).__name__}")
            
            # Success only if no structural errors
            has_errors = any(not e.startswith("Format warning:") for e in errors)
            return not has_errors, errors
            
        except json.JSONDecodeError as e:
            self.stats.increment('invalid_json_files')
            errors.append(f"JSON decode error at line {e.lineno} column {e.colno}: {e.msg}")
            return False, errors
        except IOError as e:
            errors.append(f"File I/O error: {e}")
            return False, errors
        except Exception as e:
            errors.append(f"Unexpected error: {type(e).__name__}: {e}")
            return False, errors
    
    def _is_date_journal_file(self, filepath: Path) -> bool:
        """
        Check if file is a date-named journal file.
        
        Args:
            filepath: Path to check
            
        Returns:
            True if file is in journal/ directory and name starts with '20'
        """
        return (filepath.parent.name == 'journal' and 
                filepath.stem.startswith(self.DATE_PREFIX))
    
    def validate_directory(
        self, 
        subdir: str, 
        validator_func: Optional[callable] = None,
        exclude_files: Optional[List[str]] = None,
        recursive: bool = False
    ) -> Dict[str, Any]:
        """
        Validate all JSON files in a subdirectory.
        
        Args:
            subdir: Subdirectory name (e.g., 'journal', 'Protocols')
            validator_func: Optional custom validation function
            exclude_files: List of filenames to exclude from validation
            recursive: Whether to recursively search subdirectories
            
        Returns:
            Dictionary with validation results including:
            - total_files: Number of files checked
            - valid_files: Number of valid files
            - invalid_files: Number of invalid files
            - errors: List of error details
        """
        dir_path = self.data_dir / subdir
        
        # Check if directory exists
        if not dir_path.exists():
            logger.warning(f"Directory not found: {dir_path}")
            return {'skipped': True, 'reason': 'not_found'}
        
        results = {
            'total_files': 0,
            'valid_files': 0,
            'invalid_files': 0,
            'errors': []
        }
        
        # Get all JSON files, excluding specified files
        exclude_files = exclude_files or []
        if recursive:
            json_files = [
                f for f in dir_path.rglob("*.json") 
                if f.name not in exclude_files
            ]
        else:
            json_files = [
                f for f in dir_path.glob("*.json") 
                if f.name not in exclude_files
            ]
        
        results['total_files'] = len(json_files)
        self.stats.increment('total_files_checked')
        
        logger.info(f"\nValidating {subdir}/ ({len(json_files)} files)")
        
        # Validate each file
        for filepath in sorted(json_files):  # Sort for consistent output
            # Choose appropriate validator
            if validator_func and subdir == 'journal':
                # Special handling for journal directory
                if self._is_date_journal_file(filepath):
                    success, file_errors = validator_func(filepath)
                else:
                    # Non-date files in journal/ use generic validation
                    success, file_errors = self.validate_json_file(filepath, subdir)
            elif validator_func:
                success, file_errors = validator_func(filepath)
            else:
                success, file_errors = self.validate_json_file(filepath, subdir)
            
            # Record results
            if success:
                results['valid_files'] += 1
                if self.verbose:
                    logger.info(f"  ✓ {filepath.name}")
            else:
                results['invalid_files'] += 1
                logger.error(f"  ✗ {filepath.name}")
                for error in file_errors:
                    logger.error(f"    {error}")
                results['errors'].append({
                    'file': str(filepath.relative_to(self.data_dir)),
                    'errors': file_errors
                })
        
        return results
    
    def validate_all(self) -> Dict[str, Any]:
        """
        Validate all data directories.
        
        Returns:
            Complete validation report with:
            - timestamp: ISO format timestamp
            - data_dir: Path to data directory
            - directories: Results per directory
            - summary: Overall statistics
        """
        logger.info("=" * 70)
        logger.info("DATA VALIDATION REPORT")
        logger.info("=" * 70)
        logger.info(f"Data directory: {self.data_dir}")
        logger.info(f"Timestamp: {datetime.now().isoformat()}")
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'data_dir': str(self.data_dir),
            'directories': {}
        }
        
        # Validate journal entries (special handling with exclusions)
        report['directories']['journal'] = self.validate_directory(
            'journal',
            validator_func=self.validate_journal_file,
            exclude_files=list(self.METADATA_FILES),
            recursive=True
        )
        
        # Validate other directories (all recursive to catch subdirectories)
        for directory in ['Core_Archives', 'Protocols', 'Rituals', 'Lexicon']:
            report['directories'][directory] = self.validate_directory(
                directory, 
                recursive=True
            )
        
        # Validate memories subdirectories if they exist
        memories_path = self.data_dir / 'memories'
        if memories_path.exists() and memories_path.is_dir():
            for subdir in sorted(memories_path.iterdir()):
                if subdir.is_dir():
                    dir_name = f"memories/{subdir.name}"
                    report['directories'][dir_name] = self.validate_directory(dir_name)
        
        # Calculate summary statistics
        total_files = 0
        total_valid = 0
        total_invalid = 0
        
        for dir_name, results in report['directories'].items():
            if results.get('skipped'):
                continue
            
            total_files += results['total_files']
            total_valid += results['valid_files']
            total_invalid += results['invalid_files']
        
        # Print summary
        logger.info("\n" + "=" * 70)
        logger.info("VALIDATION SUMMARY")
        logger.info("=" * 70)
        
        for dir_name, results in report['directories'].items():
            if results.get('skipped'):
                continue
            
            status = "✓" if results['invalid_files'] == 0 else "✗"
            logger.info(
                f"{status} {dir_name:30s} "
                f"{results['valid_files']:3d}/{results['total_files']:3d} valid"
            )
        
        logger.info("=" * 70)
        logger.info(f"Total files:        {total_files}")
        if total_files > 0:
            logger.info(f"Valid files:        {total_valid} ({total_valid/total_files*100:.1f}%)")
        else:
            logger.info(f"Valid files:        {total_valid} (0.0%)")
        logger.info(f"Invalid files:      {total_invalid}")
        logger.info(f"Journal entries:    {self.stats.valid_journal_entries} valid, "
                   f"{self.stats.invalid_journal_entries} invalid")
        
        # Warn about formatting issues if any
        if self.stats.missing_trailing_newline > 0:
            logger.warning(
                f"Formatting issues:  {self.stats.missing_trailing_newline} files "
                f"missing trailing newline"
            )
        
        logger.info("=" * 70)
        
        # Add summary to report
        report['summary'] = {
            'total_files': total_files,
            'valid_files': total_valid,
            'invalid_files': total_invalid,
            'valid_journal_entries': self.stats.valid_journal_entries,
            'invalid_journal_entries': self.stats.invalid_journal_entries,
            'missing_trailing_newline': self.stats.missing_trailing_newline
        }
        
        return report
    
    def save_report(self, report: Dict[str, Any], output_file: Path) -> None:
        """
        Save validation report to JSON file.
        
        Args:
            output_file: Path to output JSON file
            report: Validation report dictionary
            
        Raises:
            IOError: If file cannot be written
        """
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
                f.write('\n')  # Add trailing newline (eating our own dogfood!)
            
            logger.info(f"\nDetailed report saved to: {output_file}")
        except IOError as e:
            logger.error(f"Failed to save report: {e}")
            raise


def main() -> int:
    """
    Main validation function with argument parsing.
    
    Returns:
        Exit code: 0 for success, 1 for validation errors or execution error
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Validate all JSON files in data directory',
        epilog='Exit codes: 0 = success, 1 = validation errors or execution error'
    )
    parser.add_argument(
        '--data-dir',
        type=Path,
        default=Path('../data'),
        help='Path to data directory (default: ../data relative to script)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Show detailed output for each file including format warnings'
    )
    parser.add_argument(
        '--report',
        type=Path,
        help='Save detailed validation report to specified JSON file'
    )
    
    args = parser.parse_args()
    
    try:
        # Resolve data directory path
        data_dir = args.data_dir
        if not data_dir.is_absolute():
            # Make path relative to script location
            script_dir = Path(__file__).parent
            data_dir = (script_dir / data_dir).resolve()
        
        # Run validation
        validator = DataValidator(data_dir, verbose=args.verbose)
        report = validator.validate_all()
        
        # Save report if requested
        if args.report:
            validator.save_report(report, args.report)
        
        # Determine exit code
        if report['summary']['invalid_files'] > 0:
            logger.error("\n❌ Validation failed - some files have errors")
            return 1
        else:
            logger.info("\n✅ All files validated successfully!")
            return 0
            
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        return 1
    except KeyboardInterrupt:
        logger.warning("\nValidation interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {type(e).__name__}: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
