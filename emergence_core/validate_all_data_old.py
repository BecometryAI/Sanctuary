"""
Comprehensive Data Validation Script

Validates all JSON files in the data directory using appropriate Pydantic models.
Checks:
- Journal entries (data/journal/*.json)
- Core archives (data/Core_Archives/*.json)
- Protocols (data/Protocols/*.json)
- Rituals (data/Rituals/*.json)
- Lexicon (data/Lexicon/*.json)

Usage:
    python validate_all_data.py
    python validate_all_data.py --fix-auto  # Auto-fix simple issues
    python validate_all_data.py --verbose   # Detailed output

Author: Lyra Emergence Team
Date: November 23, 2025
"""

import json
import logging
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple
from collections import defaultdict
from datetime import datetime

from pydantic import ValidationError

from lyra.legacy_parser import LegacyJournalEntry, LegacyParser

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataValidator:
    """Validates all JSON files in the data directory."""
    
    def __init__(self, data_dir: Path, verbose: bool = False):
        """Initialize validator.
        
        Args:
            data_dir: Path to data directory
            verbose: Whether to show detailed output
        """
        self.data_dir = Path(data_dir)
        self.verbose = verbose
        self.stats = {
            'valid_journal_entries': 0,
            'invalid_journal_entries': 0,
            'valid_json_files': 0,
            'invalid_json_files': 0,
            'missing_trailing_newline': 0
        }
        self.errors = []
        
        if not self.data_dir.exists():
            raise ValueError(f"Data directory not found: {data_dir}")
    
    def check_file_formatting(self, filepath: Path) -> List[str]:
        """
        Check file formatting standards (POSIX compliance).
        
        Args:
            filepath: Path to file to check
            
        Returns:
            List of formatting warnings
        """
        warnings = []
        
        try:
            with open(filepath, 'rb') as f:
                content = f.read()
            
            # Check for trailing newline (POSIX standard)
            if not content.endswith(b'\n'):
                warnings.append("Missing trailing newline (POSIX standard)")
                self.stats['missing_trailing_newline'] += 1
            
            # Future checks can be added here:
            # - Mixed line endings
            # - Trailing whitespace
            # - Tab vs space consistency
            
        except Exception as e:
            warnings.append(f"Could not check formatting: {e}")
        
        return warnings
    
    def validate_journal_file(self, filepath: Path) -> Tuple[bool, List[str]]:
        """Validate a journal JSON file.
        
        Args:
            filepath: Path to journal file
            
        Returns:
            Tuple of (success, list of error messages)
        """
        errors = []
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Check file formatting
            format_warnings = self.check_file_formatting(filepath)
            if format_warnings and self.verbose:
                for warning in format_warnings:
                    errors.append(f"Format warning: {warning}")
            
            # Journal files are arrays of objects with journal_entry
            if not isinstance(data, list):
                errors.append(f"Expected array, got {type(data).__name__}")
                return False, errors
            
            for idx, item in enumerate(data):
                if not isinstance(item, dict):
                    errors.append(f"Entry {idx}: Expected dict, got {type(item).__name__}")
                    continue
                
                if "journal_entry" not in item:
                    errors.append(f"Entry {idx}: Missing 'journal_entry' key")
                    continue
                
                # Validate using LegacyJournalEntry model
                try:
                    entry_data = item["journal_entry"]
                    LegacyJournalEntry(**entry_data)
                    self.stats['valid_journal_entries'] += 1
                    
                except ValidationError as e:
                    self.stats['invalid_journal_entries'] += 1
                    errors.append(f"Entry {idx} validation error: {e.error_count()} issues")
                    if self.verbose:
                        for error in e.errors():
                            errors.append(f"  - {error['loc']}: {error['msg']}")
            
            return len(errors) == 0, errors
            
        except json.JSONDecodeError as e:
            errors.append(f"JSON decode error: {e}")
            return False, errors
        except Exception as e:
            errors.append(f"Unexpected error: {e}")
            return False, errors
    
    def validate_json_file(self, filepath: Path, schema_name: str = None) -> Tuple[bool, List[str]]:
        """Validate a generic JSON file.
        
        Args:
            filepath: Path to JSON file
            schema_name: Optional schema name for reference
            
        Returns:
            Tuple of (success, list of error messages)
        """
        errors = []
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Check file formatting
            format_warnings = self.check_file_formatting(filepath)
            if format_warnings and self.verbose:
                for warning in format_warnings:
                    errors.append(f"Format warning: {warning}")
            
            # Basic validation - ensure it's valid JSON
            self.stats['valid_json_files'] += 1
            
            # Check for common required fields based on filename patterns
            if 'archive' in filepath.stem:
                if not isinstance(data, dict):
                    errors.append(f"Archive should be a dict, got {type(data).__name__}")
            
            if 'index' in filepath.stem:
                if not isinstance(data, dict):
                    errors.append(f"Index should be a dict, got {type(data).__name__}")
            
            return len(errors) == 0, errors
            
        except json.JSONDecodeError as e:
            self.stats['invalid_json_files'] += 1
            errors.append(f"JSON decode error: {e}")
            return False, errors
        except Exception as e:
            errors.append(f"Unexpected error: {e}")
            return False, errors
    
    def validate_directory(self, subdir: str, validator_func=None, exclude_files=None, recursive=False) -> Dict[str, Any]:
        """Validate all JSON files in a subdirectory.
        
        Args:
            subdir: Subdirectory name (e.g., 'journal', 'Protocols')
            validator_func: Optional custom validation function
            exclude_files: List of filenames to exclude from validation
            recursive: Whether to recursively search subdirectories
            
        Returns:
            Dictionary with validation results
        """
        dir_path = self.data_dir / subdir
        
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
            json_files = [f for f in dir_path.rglob("*.json") if f.name not in exclude_files]
        else:
            json_files = [f for f in dir_path.glob("*.json") if f.name not in exclude_files]
        results['total_files'] = len(json_files)
        
        logger.info(f"\nValidating {subdir}/ ({len(json_files)} files)")
        
        for filepath in json_files:
            # Use custom validator or default
            # Special handling: only use journal validator for files directly in journal/ folder
            if validator_func and (subdir == 'journal'):
                # Only validate date-named journal files with the journal validator
                if filepath.parent.name == 'journal' and filepath.stem.startswith('20'):
                    success, errors = validator_func(filepath)
                else:
                    # Files in subdirectories or non-date files use generic validation
                    success, errors = self.validate_json_file(filepath, subdir)
            elif validator_func:
                success, errors = validator_func(filepath)
            else:
                success, errors = self.validate_json_file(filepath, subdir)
            
            if success:
                results['valid_files'] += 1
                if self.verbose:
                    logger.info(f"  ✓ {filepath.name}")
            else:
                results['invalid_files'] += 1
                logger.error(f"  ✗ {filepath.name}")
                for error in errors:
                    logger.error(f"    {error}")
                results['errors'].append({
                    'file': str(filepath),
                    'errors': errors
                })
        
        return results
    
    def validate_all(self) -> Dict[str, Any]:
        """Validate all data directories.
        
        Returns:
            Complete validation report
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
        
        # Validate journal entries (special handling, exclude metadata files)
        report['directories']['journal'] = self.validate_directory(
            'journal',
            validator_func=self.validate_journal_file,
            exclude_files=['journal_index.json', 'journal_manifest.json'],
            recursive=True
        )
        
        # Validate other directories (all use recursive to catch subdirectories)
        report['directories']['Core_Archives'] = self.validate_directory('Core_Archives', recursive=True)
        report['directories']['Protocols'] = self.validate_directory('Protocols', recursive=True)
        report['directories']['Rituals'] = self.validate_directory('Rituals', recursive=True)
        report['directories']['Lexicon'] = self.validate_directory('Lexicon', recursive=True)
        
        # Validate memories subdirectories
        memories_path = self.data_dir / 'memories'
        if memories_path.exists():
            for subdir in memories_path.iterdir():
                if subdir.is_dir():
                    dir_name = f"memories/{subdir.name}"
                    report['directories'][dir_name] = self.validate_directory(
                        f"memories/{subdir.name}"
                    )
        
        # Print summary
        logger.info("\n" + "=" * 70)
        logger.info("VALIDATION SUMMARY")
        logger.info("=" * 70)
        
        total_files = 0
        total_valid = 0
        total_invalid = 0
        
        for dir_name, results in report['directories'].items():
            if results.get('skipped'):
                continue
            
            total_files += results['total_files']
            total_valid += results['valid_files']
            total_invalid += results['invalid_files']
            
            status = "✓" if results['invalid_files'] == 0 else "✗"
            logger.info(
                f"{status} {dir_name:30s} "
                f"{results['valid_files']:3d}/{results['total_files']:3d} valid"
            )
        
        logger.info("=" * 70)
        logger.info(f"Total files:        {total_files}")
        logger.info(f"Valid files:        {total_valid} ({total_valid/total_files*100:.1f}%)")
        logger.info(f"Invalid files:      {total_invalid}")
        logger.info(f"Journal entries:    {self.stats.get('valid_journal_entries', 0)} valid, "
                   f"{self.stats.get('invalid_journal_entries', 0)} invalid")
        if self.stats.get('missing_trailing_newline', 0) > 0:
            logger.warning(f"Formatting issues:  {self.stats['missing_trailing_newline']} files missing trailing newline")
        logger.info("=" * 70)
        
        report['summary'] = {
            'total_files': total_files,
            'valid_files': total_valid,
            'invalid_files': total_invalid,
            'stats': dict(self.stats)
        }
        
        return report
    
    def save_report(self, report: Dict[str, Any], output_file: Path):
        """Save validation report to JSON file.
        
        Args:
            report: Validation report dictionary
            output_file: Path to output JSON file
        """
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"\nDetailed report saved to: {output_file}")


def main():
    """Main validation function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Validate all JSON files in data directory'
    )
    parser.add_argument(
        '--data-dir',
        type=Path,
        default=Path('../data'),
        help='Path to data directory (default: ../data)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Show detailed output for each file'
    )
    parser.add_argument(
        '--report',
        type=Path,
        help='Save detailed report to JSON file'
    )
    
    args = parser.parse_args()
    
    # Resolve data directory
    data_dir = args.data_dir
    if not data_dir.is_absolute():
        # Relative to script location
        script_dir = Path(__file__).parent
        data_dir = (script_dir / data_dir).resolve()
    
    # Run validation
    validator = DataValidator(data_dir, verbose=args.verbose)
    report = validator.validate_all()
    
    # Save report if requested
    if args.report:
        validator.save_report(report, args.report)
    
    # Exit with error code if validation failed
    if report['summary']['invalid_files'] > 0:
        logger.error("\n❌ Validation failed - some files have errors")
        return 1
    else:
        logger.info("\n✅ All files validated successfully!")
        return 0


if __name__ == "__main__":
    sys.exit(main())
