#!/usr/bin/env python3
"""
Fix JSON file formatting issues across the data directory.

This script:
1. Adds trailing newlines to all JSON files (POSIX compliance)
2. Ensures consistent formatting
3. Validates JSON structure before and after changes
"""

import json
import logging
from pathlib import Path
from typing import Tuple, List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class JSONFormatter:
    """Handles JSON file formatting corrections."""
    
    def __init__(self, data_dir: Path):
        """
        Initialize the formatter.
        
        Args:
            data_dir: Path to the data directory
        """
        self.data_dir = Path(data_dir)
        self.stats = {
            'total_files': 0,
            'fixed_files': 0,
            'skipped_files': 0,
            'error_files': 0
        }
    
    def check_and_fix_file(self, filepath: Path) -> Tuple[bool, str]:
        """
        Check if file needs trailing newline and fix if needed.
        
        Args:
            filepath: Path to JSON file
            
        Returns:
            Tuple of (was_fixed, status_message)
        """
        try:
            # Read file content
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Validate JSON before modification
            try:
                json.loads(content)
            except json.JSONDecodeError as e:
                return False, f"Invalid JSON (skipped): {e}"
            
            # Check if file already has trailing newline
            if content.endswith('\n'):
                return False, "Already has trailing newline"
            
            # Add trailing newline
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content + '\n')
            
            # Validate JSON after modification
            with open(filepath, 'r', encoding='utf-8') as f:
                json.loads(f.read())
            
            return True, "Added trailing newline"
            
        except Exception as e:
            return False, f"Error: {e}"
    
    def process_all_files(self, dry_run: bool = False) -> None:
        """
        Process all JSON files in the data directory.
        
        Args:
            dry_run: If True, only report what would be changed
        """
        logger.info("=" * 70)
        logger.info("JSON FORMATTING FIX")
        logger.info("=" * 70)
        logger.info(f"Data directory: {self.data_dir}")
        logger.info(f"Mode: {'DRY RUN' if dry_run else 'LIVE'}")
        logger.info("")
        
        # Find all JSON files recursively
        json_files = sorted(self.data_dir.rglob("*.json"))
        self.stats['total_files'] = len(json_files)
        
        logger.info(f"Found {len(json_files)} JSON files")
        logger.info("")
        
        for filepath in json_files:
            relative_path = filepath.relative_to(self.data_dir)
            
            if dry_run:
                # In dry run, just check without modifying
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                if not content.endswith('\n'):
                    logger.info(f"  WOULD FIX: {relative_path}")
                    self.stats['fixed_files'] += 1
                else:
                    self.stats['skipped_files'] += 1
            else:
                # Actually fix the file
                was_fixed, message = self.check_and_fix_file(filepath)
                
                if was_fixed:
                    logger.info(f"  ✓ FIXED: {relative_path}")
                    self.stats['fixed_files'] += 1
                elif "Error" in message or "Invalid" in message:
                    logger.error(f"  ✗ ERROR: {relative_path} - {message}")
                    self.stats['error_files'] += 1
                else:
                    self.stats['skipped_files'] += 1
        
        # Print summary
        logger.info("")
        logger.info("=" * 70)
        logger.info("SUMMARY")
        logger.info("=" * 70)
        logger.info(f"Total files:     {self.stats['total_files']}")
        logger.info(f"Fixed:           {self.stats['fixed_files']}")
        logger.info(f"Already correct: {self.stats['skipped_files']}")
        logger.info(f"Errors:          {self.stats['error_files']}")
        logger.info("=" * 70)
        
        if dry_run:
            logger.info("")
            logger.info("This was a DRY RUN - no files were modified")
            logger.info("Run without --dry-run to apply changes")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Fix JSON formatting issues in data directory"
    )
    parser.add_argument(
        '--data-dir',
        type=Path,
        default=Path(__file__).parent.parent / 'data',
        help='Path to data directory (default: ../data)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be changed without modifying files'
    )
    
    args = parser.parse_args()
    
    formatter = JSONFormatter(args.data_dir)
    formatter.process_all_files(dry_run=args.dry_run)
    
    # Exit with error code if there were errors
    if formatter.stats['error_files'] > 0:
        exit(1)


if __name__ == "__main__":
    main()
