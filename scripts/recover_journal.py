#!/usr/bin/env python3
"""
Journal Recovery Tool: Validate, extract, and merge journal files.

This utility provides tools for working with JSONL journal files:
- Validate journal file integrity
- Repair corrupted JSONL files (skip malformed lines)
- Merge multiple journal files
- Extract entries by date range or type
- Convert JSONL to pretty-printed JSON for inspection

Usage:
    python scripts/recover_journal.py validate data/introspection/journal_2026-01-03.jsonl
    python scripts/recover_journal.py extract --type realization --days 7
    python scripts/recover_journal.py merge --output merged.jsonl
"""

import argparse
import gzip
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional


def load_journal_entries(journal_path: Path) -> List[Dict]:
    """
    Load entries from a journal file (handles both .jsonl and .jsonl.gz).
    
    Args:
        journal_path: Path to journal file
        
    Returns:
        List of valid journal entries
    """
    entries = []
    errors = []
    
    # Determine if compressed
    if journal_path.suffix == '.gz':
        def open_file(p):
            return gzip.open(p, 'rt', encoding='utf-8')
    else:
        def open_file(p):
            return open(p, 'r', encoding='utf-8')
    
    try:
        with open_file(journal_path) as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    entry = json.loads(line)
                    entries.append(entry)
                except json.JSONDecodeError as e:
                    errors.append({
                        "line": line_num,
                        "error": str(e),
                        "content": line[:50] + "..." if len(line) > 50 else line
                    })
    
    except Exception as e:
        print(f"❌ Error reading file {journal_path}: {e}", file=sys.stderr)
        return []
    
    if errors:
        print(f"⚠️  Found {len(errors)} corrupted lines in {journal_path.name}", file=sys.stderr)
        for err in errors[:5]:  # Show first 5 errors
            print(f"   Line {err['line']}: {err['error']}", file=sys.stderr)
    
    return entries


def validate_journal(journal_path: Path, verbose: bool = False) -> bool:
    """
    Validate journal file integrity.
    
    Args:
        journal_path: Path to journal file
        verbose: Print detailed information
        
    Returns:
        True if valid, False otherwise
    """
    if not journal_path.exists():
        print(f"❌ File not found: {journal_path}")
        return False
    
    print(f"🔍 Validating: {journal_path.name}")
    
    entries = load_journal_entries(journal_path)
    
    if not entries:
        print(f"❌ No valid entries found")
        return False
    
    # Check entry structure
    required_fields = {"timestamp", "type"}
    issues = []
    
    for i, entry in enumerate(entries):
        missing = required_fields - set(entry.keys())
        if missing:
            issues.append(f"Entry {i+1} missing fields: {missing}")
        
        # Validate timestamp format
        if "timestamp" in entry:
            try:
                datetime.fromisoformat(entry["timestamp"])
            except (ValueError, TypeError):
                issues.append(f"Entry {i+1} has invalid timestamp: {entry['timestamp']}")
    
    if issues:
        print(f"⚠️  Found {len(issues)} structural issues:")
        for issue in issues[:10]:
            print(f"   {issue}")
    
    print(f"✅ Valid entries: {len(entries)}")
    
    if verbose:
        # Show entry type distribution
        types = {}
        for entry in entries:
            entry_type = entry.get("type", "unknown")
            types[entry_type] = types.get(entry_type, 0) + 1
        
        print(f"\n📊 Entry type distribution:")
        for entry_type, count in sorted(types.items(), key=lambda x: -x[1]):
            print(f"   {entry_type}: {count}")
    
    return len(issues) == 0


def extract_entries(
    journal_dir: Path,
    entry_type: Optional[str] = None,
    days: Optional[int] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    output: Optional[Path] = None
) -> List[Dict]:
    """
    Extract entries matching criteria.
    
    Args:
        journal_dir: Directory containing journal files
        entry_type: Filter by entry type (observation, realization, question)
        days: Filter to last N days
        start_date: Filter by start date
        end_date: Filter by end date
        output: Optional output file path
        
    Returns:
        List of matching entries
    """
    if not journal_dir.exists():
        print(f"❌ Directory not found: {journal_dir}")
        return []
    
    # Find all journal files
    journal_files = sorted(journal_dir.glob("journal_*.jsonl"))
    journal_files.extend(sorted(journal_dir.glob("journal_*.jsonl.gz")))
    journal_files.sort()
    
    if not journal_files:
        print(f"❌ No journal files found in {journal_dir}")
        return []
    
    print(f"🔍 Scanning {len(journal_files)} journal files...")
    
    # Determine date range
    if days:
        start_date = datetime.now() - timedelta(days=days)
        end_date = datetime.now()
    
    # Extract entries
    matching_entries = []
    
    for journal_file in journal_files:
        entries = load_journal_entries(journal_file)
        
        for entry in entries:
            # Filter by type
            if entry_type and entry.get("type") != entry_type:
                continue
            
            # Filter by date
            if start_date or end_date:
                try:
                    entry_time = datetime.fromisoformat(entry["timestamp"])
                    
                    if start_date and entry_time < start_date:
                        continue
                    if end_date and entry_time > end_date:
                        continue
                except (ValueError, KeyError):
                    continue
            
            matching_entries.append(entry)
    
    print(f"✅ Found {len(matching_entries)} matching entries")
    
    # Write output if specified
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, 'w') as f:
            for entry in matching_entries:
                f.write(json.dumps(entry) + '\n')
        print(f"💾 Saved to: {output}")
    else:
        # Print to stdout
        for entry in matching_entries[:10]:  # Show first 10
            print(json.dumps(entry, indent=2))
        
        if len(matching_entries) > 10:
            print(f"\n... and {len(matching_entries) - 10} more entries")
    
    return matching_entries


def merge_journals(
    journal_dir: Path,
    output: Path,
    compress: bool = False
) -> int:
    """
    Merge multiple journal files into one.
    
    Args:
        journal_dir: Directory containing journal files
        output: Output file path
        compress: Whether to compress output
        
    Returns:
        Number of entries merged
    """
    if not journal_dir.exists():
        print(f"❌ Directory not found: {journal_dir}")
        return 0
    
    # Find all journal files
    journal_files = sorted(journal_dir.glob("journal_*.jsonl"))
    journal_files.extend(sorted(journal_dir.glob("journal_*.jsonl.gz")))
    journal_files.sort()
    
    if not journal_files:
        print(f"❌ No journal files found in {journal_dir}")
        return 0
    
    print(f"🔄 Merging {len(journal_files)} journal files...")
    
    # Collect all entries
    all_entries = []
    for journal_file in journal_files:
        entries = load_journal_entries(journal_file)
        all_entries.extend(entries)
        print(f"   {journal_file.name}: {len(entries)} entries")
    
    # Sort by timestamp
    all_entries.sort(key=lambda e: e.get("timestamp", ""))
    
    # Write merged file
    output.parent.mkdir(parents=True, exist_ok=True)
    
    if compress:
        with gzip.open(output, 'wt', encoding='utf-8') as f:
            for entry in all_entries:
                f.write(json.dumps(entry) + '\n')
    else:
        with open(output, 'w', encoding='utf-8') as f:
            for entry in all_entries:
                f.write(json.dumps(entry) + '\n')
    
    print(f"✅ Merged {len(all_entries)} entries into {output}")
    
    return len(all_entries)


def repair_journal(journal_path: Path, output: Path) -> int:
    """
    Repair corrupted journal file by skipping malformed lines.
    
    Args:
        journal_path: Path to corrupted journal file
        output: Path to write repaired file
        
    Returns:
        Number of valid entries recovered
    """
    if not journal_path.exists():
        print(f"❌ File not found: {journal_path}")
        return 0
    
    print(f"🔧 Repairing: {journal_path.name}")
    
    entries = load_journal_entries(journal_path)
    
    if not entries:
        print(f"❌ No valid entries found")
        return 0
    
    # Write repaired file
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, 'w', encoding='utf-8') as f:
        for entry in entries:
            f.write(json.dumps(entry) + '\n')
    
    print(f"✅ Recovered {len(entries)} valid entries to {output}")
    
    return len(entries)


def pretty_print_journal(journal_path: Path, limit: Optional[int] = None):
    """
    Convert JSONL to pretty-printed JSON for inspection.
    
    Args:
        journal_path: Path to journal file
        limit: Max number of entries to print (None = all)
    """
    if not journal_path.exists():
        print(f"❌ File not found: {journal_path}")
        return
    
    entries = load_journal_entries(journal_path)
    
    if not entries:
        print(f"❌ No valid entries found")
        return
    
    print(f"📄 Journal: {journal_path.name}")
    print(f"📊 Total entries: {len(entries)}\n")
    
    # Print entries
    for i, entry in enumerate(entries):
        if limit and i >= limit:
            print(f"\n... and {len(entries) - limit} more entries")
            break
        
        print(f"--- Entry {i+1} ---")
        print(json.dumps(entry, indent=2))
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Journal Recovery Tool: Validate, extract, and merge journal files"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Validate command
    validate_parser = subparsers.add_parser("validate", help="Validate journal file integrity")
    validate_parser.add_argument("journal", type=Path, help="Path to journal file")
    validate_parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    
    # Extract command
    extract_parser = subparsers.add_parser("extract", help="Extract entries by criteria")
    extract_parser.add_argument("--dir", type=Path, default=Path("data/introspection"), help="Journal directory")
    extract_parser.add_argument("--type", choices=["observation", "realization", "question"], help="Entry type")
    extract_parser.add_argument("--days", type=int, help="Last N days")
    extract_parser.add_argument("--start", type=str, help="Start date (ISO format)")
    extract_parser.add_argument("--end", type=str, help="End date (ISO format)")
    extract_parser.add_argument("-o", "--output", type=Path, help="Output file")
    
    # Merge command
    merge_parser = subparsers.add_parser("merge", help="Merge multiple journal files")
    merge_parser.add_argument("--dir", type=Path, default=Path("data/introspection"), help="Journal directory")
    merge_parser.add_argument("-o", "--output", type=Path, required=True, help="Output file")
    merge_parser.add_argument("--compress", action="store_true", help="Compress output")
    
    # Repair command
    repair_parser = subparsers.add_parser("repair", help="Repair corrupted journal file")
    repair_parser.add_argument("journal", type=Path, help="Path to corrupted journal file")
    repair_parser.add_argument("-o", "--output", type=Path, required=True, help="Output file")
    
    # Pretty print command
    pretty_parser = subparsers.add_parser("pretty", help="Pretty-print journal for inspection")
    pretty_parser.add_argument("journal", type=Path, help="Path to journal file")
    pretty_parser.add_argument("--limit", type=int, default=10, help="Max entries to print")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Execute command
    if args.command == "validate":
        success = validate_journal(args.journal, args.verbose)
        sys.exit(0 if success else 1)
    
    elif args.command == "extract":
        start_date = datetime.fromisoformat(args.start) if args.start else None
        end_date = datetime.fromisoformat(args.end) if args.end else None
        
        extract_entries(
            args.dir,
            entry_type=args.type,
            days=args.days,
            start_date=start_date,
            end_date=end_date,
            output=args.output
        )
    
    elif args.command == "merge":
        merge_journals(args.dir, args.output, args.compress)
    
    elif args.command == "repair":
        repair_journal(args.journal, args.output)
    
    elif args.command == "pretty":
        pretty_print_journal(args.journal, args.limit)


if __name__ == "__main__":
    main()
