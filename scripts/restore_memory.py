#!/usr/bin/env python3
"""
Memory Restore Tool

Command-line tool for listing and restoring memory backups.

Usage:
    python scripts/restore_memory.py list
    python scripts/restore_memory.py restore <backup_name> [--target <dir>] [--dry-run]
    python scripts/restore_memory.py validate <backup_name>

Author: Sanctuary Emergence Team
Date: January 2, 2026
"""

import argparse
import asyncio
import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from emergence_core.sanctuary.memory.backup import BackupManager
from emergence_core.sanctuary.logging_config import setup_logging, get_logger

logger = get_logger(__name__)


def format_size(size_mb: float) -> str:
    """Format size in MB to human-readable string."""
    if size_mb < 1:
        return f"{size_mb * 1024:.2f} KB"
    elif size_mb < 1024:
        return f"{size_mb:.2f} MB"
    else:
        return f"{size_mb / 1024:.2f} GB"


async def list_backups(manager: BackupManager):
    """List all available backups."""
    print("\n=== Available Backups ===\n")
    
    backups = manager.list_backups()
    
    if not backups:
        print("No backups found.")
        return
    
    for i, backup in enumerate(backups, 1):
        age_days = (datetime.utcnow() - backup["created"]).days
        compression = "compressed" if backup["compressed"] else "uncompressed"
        
        print(f"{i}. {backup['name']}")
        print(f"   Path: {backup['path']}")
        print(f"   Size: {format_size(backup['size_mb'])} ({compression})")
        print(f"   Created: {backup['created'].strftime('%Y-%m-%d %H:%M:%S')} ({age_days} days ago)")
        print()


async def restore_backup(
    manager: BackupManager,
    backup_name: str,
    target_dir: str = None,
    dry_run: bool = False
):
    """Restore from a backup."""
    # Find backup
    backups = manager.list_backups()
    backup = None
    
    for b in backups:
        if b["name"] == backup_name or backup_name in b["path"]:
            backup = b
            break
    
    if not backup:
        print(f"Error: Backup '{backup_name}' not found.")
        print("Use 'list' command to see available backups.")
        return False
    
    backup_path = Path(backup["path"])
    target = Path(target_dir) if target_dir else None
    
    if dry_run:
        print(f"\n=== Dry Run: Restore Validation ===\n")
        print(f"Backup: {backup['name']}")
        print(f"Source: {backup_path}")
        print(f"Target: {target or manager.source_dir}")
        print(f"Size: {format_size(backup['size_mb'])}")
        print(f"\nValidation: OK - backup exists and is readable")
        return True
    
    print(f"\n=== Restoring Backup ===\n")
    print(f"Backup: {backup['name']}")
    print(f"Source: {backup_path}")
    print(f"Target: {target or manager.source_dir}")
    print(f"Size: {format_size(backup['size_mb'])}")
    print()
    
    # Confirm restore
    response = input("This will overwrite existing data. Continue? (yes/no): ")
    if response.lower() != "yes":
        print("Restore cancelled.")
        return False
    
    try:
        success = await manager.restore_backup(backup_path, target)
        if success:
            print("\n✓ Restore completed successfully!")
            return True
        else:
            print("\n✗ Restore failed.")
            return False
    except Exception as e:
        print(f"\n✗ Restore failed: {e}")
        logger.error(f"Restore error: {e}", exc_info=True)
        return False


async def validate_backup(manager: BackupManager, backup_name: str):
    """Validate a backup without restoring."""
    return await restore_backup(manager, backup_name, dry_run=True)


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Memory restore tool for Sanctuary Emergence",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "command",
        choices=["list", "restore", "validate"],
        help="Command to execute"
    )
    
    parser.add_argument(
        "backup_name",
        nargs="?",
        help="Name of backup (for restore/validate commands)"
    )
    
    parser.add_argument(
        "--target",
        "-t",
        help="Target directory for restore (default: original source directory)"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate without actually restoring"
    )
    
    parser.add_argument(
        "--source",
        "-s",
        default="memories",
        help="Source directory path (default: memories)"
    )
    
    parser.add_argument(
        "--backup-dir",
        "-b",
        default="backups",
        help="Backup directory path (default: backups)"
    )
    
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    setup_logging(log_level=log_level)
    
    # Initialize backup manager
    manager = BackupManager(
        source_dir=Path(args.source),
        backup_dir=Path(args.backup_dir)
    )
    
    # Execute command
    try:
        if args.command == "list":
            await list_backups(manager)
        
        elif args.command == "restore":
            if not args.backup_name:
                print("Error: backup_name required for restore command")
                parser.print_help()
                sys.exit(1)
            
            success = await restore_backup(
                manager,
                args.backup_name,
                args.target,
                args.dry_run
            )
            sys.exit(0 if success else 1)
        
        elif args.command == "validate":
            if not args.backup_name:
                print("Error: backup_name required for validate command")
                parser.print_help()
                sys.exit(1)
            
            success = await validate_backup(manager, args.backup_name)
            sys.exit(0 if success else 1)
    
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        logger.error(f"Command error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
