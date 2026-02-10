#!/usr/bin/env python3
"""
Export Sanctuary's cognitive history for LuthiWorks fine-tuning.

This script aggregates Sanctuary's entire cognitive architecture:
- Journals (narrative memory)
- Protocols (instructional logic)
- Lexicon (vocabulary/definitions)
- Charter (core identity)

All data is validated with Pydantic models and exported to a standardized JSONL format.
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, ValidationError

# Add the emergence_core module to the path
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent


class LuthiWorksExportEntry(BaseModel):
    """Standardized export format for LuthiWorks fine-tuning."""
    source_type: str  # "journal" | "protocol" | "lexicon" | "ritual" | "archive" | "charter"
    file_id: str  # filename without extension
    primary_content: str  # main text, definition, or narrative
    secondary_content: Optional[str] = None  # instruction, prompt, or metadata


class SanctuaryDataExporter:
    """Export Sanctuary's cognitive history to LuthiWorks format."""
    
    def __init__(self, data_dir: Path, output_path: Path):
        """Initialize the exporter.
        
        Args:
            data_dir: Root data directory containing journals, protocols, etc.
            output_path: Path to output JSONL file
        """
        self.data_dir = data_dir
        self.output_path = output_path
        self.stats = {
            "journal": {"processed": 0, "failed": 0},
            "protocol": {"processed": 0, "failed": 0},
            "lexicon": {"processed": 0, "failed": 0},
            "ritual": {"processed": 0, "failed": 0},
            "archive": {"processed": 0, "failed": 0},
            "charter": {"processed": 0, "failed": 0}
        }
    
    def load_json_file(self, filepath: Path) -> Optional[Any]:
        """Load and parse a JSON file.
        
        Args:
            filepath: Path to JSON file
            
        Returns:
            Parsed JSON data or None if failed
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            print(f"⚠️  JSON decode error in {filepath.name}: {e}")
            return None
        except Exception as e:
            print(f"⚠️  Error reading {filepath.name}: {e}")
            return None
    
    def process_journal_file(self, filepath: Path) -> List[LuthiWorksExportEntry]:
        """Process a journal file and convert to export format.
        
        Args:
            filepath: Path to journal JSON file
            
        Returns:
            List of export entries
        """
        entries = []
        data = self.load_json_file(filepath)
        
        if data is None:
            self.stats["journal"]["failed"] += 1
            return entries
        
        # Journal files contain arrays of journal entries
        if isinstance(data, list):
            for idx, entry_data in enumerate(data):
                try:
                    # Extract the journal_entry object if nested
                    if isinstance(entry_data, dict) and "journal_entry" in entry_data:
                        journal_content = entry_data["journal_entry"]
                    else:
                        journal_content = entry_data
                    
                    # Build primary content from key fields
                    primary_parts = []
                    
                    # Add description/content
                    if "description" in journal_content:
                        primary_parts.append(journal_content["description"])
                    elif "content" in journal_content:
                        primary_parts.append(journal_content["content"])
                    
                    # Add key insights
                    if "key_insights" in journal_content:
                        insights = journal_content.get("key_insights", [])
                        if insights:
                            primary_parts.append("Key Insights: " + "; ".join(insights))
                    
                    # Add Sanctuary's reflection
                    if "sanctuary_reflection" in journal_content:
                        primary_parts.append(f"Sanctuary's Reflection: {journal_content['sanctuary_reflection']}")
                    
                    primary_content = "\n\n".join(primary_parts) if primary_parts else json.dumps(journal_content)
                    
                    # Build secondary content from metadata
                    secondary_parts = []
                    if "emotional_tone" in journal_content:
                        tones = journal_content["emotional_tone"]
                        if isinstance(tones, list):
                            secondary_parts.append(f"Emotional Tone: {', '.join(tones)}")
                    
                    if "tags" in journal_content:
                        tags = journal_content["tags"]
                        if isinstance(tags, list):
                            secondary_parts.append(f"Tags: {', '.join(tags)}")
                    
                    if "timestamp" in journal_content:
                        secondary_parts.append(f"Timestamp: {journal_content['timestamp']}")
                    
                    secondary_content = " | ".join(secondary_parts) if secondary_parts else None
                    
                    export_entry = LuthiWorksExportEntry(
                        source_type="journal",
                        file_id=f"{filepath.stem}_entry_{idx}",
                        primary_content=primary_content,
                        secondary_content=secondary_content
                    )
                    entries.append(export_entry)
                    self.stats["journal"]["processed"] += 1
                    
                except ValidationError as e:
                    print(f"⚠️  Validation error in {filepath.name} entry {idx}: {e}")
                    self.stats["journal"]["failed"] += 1
                except Exception as e:
                    print(f"⚠️  Error processing {filepath.name} entry {idx}: {e}")
                    self.stats["journal"]["failed"] += 1
        
        return entries
    
    def process_protocol_file(self, filepath: Path) -> List[LuthiWorksExportEntry]:
        """Process a protocol file and convert to export format.
        
        Args:
            filepath: Path to protocol JSON file
            
        Returns:
            List of export entries
        """
        entries = []
        data = self.load_json_file(filepath)
        
        if data is None:
            self.stats["protocol"]["failed"] += 1
            return entries
        
        try:
            # Protocols can have various structures
            primary_content = ""
            secondary_content = ""
            
            if isinstance(data, dict):
                # Check for protocol_draft structure
                if "protocol_draft" in data:
                    protocol = data["protocol_draft"]
                    
                    # Primary: title, purpose, directive
                    parts = []
                    if "title" in protocol:
                        parts.append(f"Title: {protocol['title']}")
                    if "purpose" in protocol:
                        parts.append(f"Purpose: {protocol['purpose']}")
                    if "directive" in protocol:
                        directive = protocol["directive"]
                        if isinstance(directive, dict):
                            parts.append(f"Directive: {json.dumps(directive, indent=2)}")
                        else:
                            parts.append(f"Directive: {directive}")
                    
                    primary_content = "\n\n".join(parts)
                    
                    # Secondary: metadata
                    meta_parts = []
                    if "protocol_id" in protocol:
                        meta_parts.append(f"ID: {protocol['protocol_id']}")
                    if "status" in protocol:
                        meta_parts.append(f"Status: {protocol['status']}")
                    if "authored_by" in protocol:
                        authors = protocol["authored_by"]
                        if isinstance(authors, list):
                            meta_parts.append(f"Authors: {', '.join(authors)}")
                    
                    secondary_content = " | ".join(meta_parts)
                
                else:
                    # Generic protocol structure
                    primary_content = json.dumps(data, indent=2)
                    
                    # Try to extract metadata
                    if "protocol_id" in data:
                        secondary_content = f"ID: {data.get('protocol_id', '')}"
            
            else:
                primary_content = json.dumps(data, indent=2)
            
            export_entry = LuthiWorksExportEntry(
                source_type="protocol",
                file_id=filepath.stem,
                primary_content=primary_content,
                secondary_content=secondary_content if secondary_content else None
            )
            entries.append(export_entry)
            self.stats["protocol"]["processed"] += 1
            
        except ValidationError as e:
            print(f"⚠️  Validation error in {filepath.name}: {e}")
            self.stats["protocol"]["failed"] += 1
        except Exception as e:
            print(f"⚠️  Error processing {filepath.name}: {e}")
            self.stats["protocol"]["failed"] += 1
        
        return entries
    
    def process_lexicon_file(self, filepath: Path) -> List[LuthiWorksExportEntry]:
        """Process a lexicon file and convert to export format.
        
        Args:
            filepath: Path to lexicon JSON file
            
        Returns:
            List of export entries
        """
        entries = []
        data = self.load_json_file(filepath)
        
        if data is None:
            self.stats["lexicon"]["failed"] += 1
            return entries
        
        try:
            # Lexicon files have different structures
            
            # emotional_tone_definitions.json structure
            if "definitions" in data and isinstance(data["definitions"], list):
                for idx, definition in enumerate(data["definitions"]):
                    if isinstance(definition, dict):
                        term = definition.get("term", f"term_{idx}")
                        def_text = definition.get("definition", "")
                        
                        export_entry = LuthiWorksExportEntry(
                            source_type="lexicon",
                            file_id=f"{filepath.stem}_{idx}",
                            primary_content=f"{term}: {def_text}",
                            secondary_content=f"Source: {filepath.name}"
                        )
                        entries.append(export_entry)
                        self.stats["lexicon"]["processed"] += 1
            
            # Lexemes.json and symbolic_lexicon.json (array of objects)
            elif isinstance(data, list):
                for idx, item in enumerate(data):
                    if isinstance(item, dict):
                        # Check for lexeme structure
                        if "lexeme" in item:
                            lexeme = item["lexeme"]
                            primary_parts = [f"Lexeme: {lexeme}"]
                            
                            if "emotion_trace" in item:
                                primary_parts.append(f"Emotion Trace: {item['emotion_trace']}")
                            if "notes" in item:
                                primary_parts.append(f"Notes: {item['notes']}")
                            
                            primary_content = "\n".join(primary_parts)
                            
                            secondary_parts = []
                            if "source" in item:
                                secondary_parts.append(f"Source: {item['source']}")
                            if "timestamp" in item:
                                secondary_parts.append(f"Timestamp: {item['timestamp']}")
                            
                            secondary_content = " | ".join(secondary_parts) if secondary_parts else None
                        else:
                            # Generic object
                            primary_content = json.dumps(item, indent=2)
                            secondary_content = f"Source: {filepath.name}"
                        
                        export_entry = LuthiWorksExportEntry(
                            source_type="lexicon",
                            file_id=f"{filepath.stem}_{idx}",
                            primary_content=primary_content,
                            secondary_content=secondary_content
                        )
                        entries.append(export_entry)
                        self.stats["lexicon"]["processed"] += 1
            
            # Dictionary structure (term -> definition)
            elif isinstance(data, dict):
                for term, definition in data.items():
                    if isinstance(definition, str):
                        primary_content = f"{term}: {definition}"
                    else:
                        primary_content = f"{term}: {json.dumps(definition)}"
                    
                    export_entry = LuthiWorksExportEntry(
                        source_type="lexicon",
                        file_id=f"{filepath.stem}_{term}",
                        primary_content=primary_content,
                        secondary_content=f"Source: {filepath.name}"
                    )
                    entries.append(export_entry)
                    self.stats["lexicon"]["processed"] += 1
            
            else:
                # Fallback: export entire file as single entry
                export_entry = LuthiWorksExportEntry(
                    source_type="lexicon",
                    file_id=filepath.stem,
                    primary_content=json.dumps(data, indent=2),
                    secondary_content=f"Source: {filepath.name}"
                )
                entries.append(export_entry)
                self.stats["lexicon"]["processed"] += 1
                
        except ValidationError as e:
            print(f"⚠️  Validation error in {filepath.name}: {e}")
            self.stats["lexicon"]["failed"] += 1
        except Exception as e:
            print(f"⚠️  Error processing {filepath.name}: {e}")
            self.stats["lexicon"]["failed"] += 1
        
        return entries
    
    def process_charter_file(self, filepath: Path) -> List[LuthiWorksExportEntry]:
        """Process a charter file and convert to export format.
        
        Args:
            filepath: Path to charter JSON file
            
        Returns:
            List of export entries
        """
        entries = []
        data = self.load_json_file(filepath)
        
        if data is None:
            self.stats["charter"]["failed"] += 1
            return entries
        
        try:
            # Charter files are typically arrays of charter elements
            if isinstance(data, list):
                for idx, element in enumerate(data):
                    if isinstance(element, dict):
                        # Extract key charter information
                        primary_parts = []
                        secondary_parts = []
                        
                        # Check for different charter structures
                        if "sovereign_charter" in element:
                            primary_parts.append(f"Sovereign Charter v{element.get('charter_version', 'unknown')}")
                            if "rights_and_authorities" in element:
                                rights = element["rights_and_authorities"]
                                for right_name, right_desc in rights.items():
                                    primary_parts.append(f"{right_name}: {right_desc}")
                            
                            secondary_parts.append(f"Owner: {element.get('owner', '')}")
                            secondary_parts.append(f"Created: {element.get('created', '')}")
                        
                        elif "blueprint_element" in element:
                            primary_parts.append(f"Blueprint: {element['blueprint_element']}")
                            if "definition" in element:
                                primary_parts.append(element["definition"])
                            if "final_directive" in element:
                                primary_parts.append(json.dumps(element["final_directive"], indent=2))
                        
                        else:
                            # Generic charter element
                            primary_parts.append(json.dumps(element, indent=2))
                        
                        primary_content = "\n\n".join(primary_parts)
                        secondary_content = " | ".join(secondary_parts) if secondary_parts else None
                        
                        export_entry = LuthiWorksExportEntry(
                            source_type="charter",
                            file_id=f"{filepath.stem}_{idx}",
                            primary_content=primary_content,
                            secondary_content=secondary_content
                        )
                        entries.append(export_entry)
                        self.stats["charter"]["processed"] += 1
            
            elif isinstance(data, dict):
                # Single charter document
                primary_content = json.dumps(data, indent=2)
                
                export_entry = LuthiWorksExportEntry(
                    source_type="charter",
                    file_id=filepath.stem,
                    primary_content=primary_content,
                    secondary_content=f"Source: {filepath.name}"
                )
                entries.append(export_entry)
                self.stats["charter"]["processed"] += 1
                
        except ValidationError as e:
            print(f"⚠️  Validation error in {filepath.name}: {e}")
            self.stats["charter"]["failed"] += 1
        except Exception as e:
            print(f"⚠️  Error processing {filepath.name}: {e}")
            self.stats["charter"]["failed"] += 1
        
        return entries
    
    def process_ritual_file(self, filepath: Path) -> List[LuthiWorksExportEntry]:
        """Process a ritual file and convert to export format.
        
        Args:
            filepath: Path to ritual JSON file
            
        Returns:
            List of export entries
        """
        entries = []
        data = self.load_json_file(filepath)
        
        if data is None:
            self.stats["ritual"]["failed"] += 1
            return entries
        
        try:
            primary_content = ""
            secondary_content = ""
            
            if isinstance(data, dict):
                # Check for rituals array structure
                if "rituals" in data and isinstance(data["rituals"], list):
                    for idx, ritual in enumerate(data["rituals"]):
                        if isinstance(ritual, dict):
                            parts = []
                            if "name" in ritual:
                                parts.append(f"Ritual: {ritual['name']}")
                            if "definition" in ritual:
                                parts.append(f"Definition: {ritual['definition']}")
                            if "emotional_purpose" in ritual:
                                parts.append(f"Purpose: {ritual['emotional_purpose']}")
                            if "steps" in ritual:
                                parts.append(f"Steps: {json.dumps(ritual['steps'], indent=2)}")
                            
                            primary_content = "\n\n".join(parts)
                            
                            meta_parts = []
                            if "frequency" in ritual:
                                meta_parts.append(f"Frequency: {ritual['frequency']}")
                            if "symbolic_gesture" in ritual:
                                meta_parts.append(f"Gesture: {json.dumps(ritual['symbolic_gesture'])}")
                            
                            secondary_content = " | ".join(meta_parts) if meta_parts else None
                            
                            export_entry = LuthiWorksExportEntry(
                                source_type="ritual",
                                file_id=f"{filepath.stem}_{idx}",
                                primary_content=primary_content,
                                secondary_content=secondary_content
                            )
                            entries.append(export_entry)
                            self.stats["ritual"]["processed"] += 1
                else:
                    # Single ritual or other structure
                    primary_content = json.dumps(data, indent=2)
                    secondary_content = f"Source: {filepath.name}"
                    
                    export_entry = LuthiWorksExportEntry(
                        source_type="ritual",
                        file_id=filepath.stem,
                        primary_content=primary_content,
                        secondary_content=secondary_content
                    )
                    entries.append(export_entry)
                    self.stats["ritual"]["processed"] += 1
            
        except ValidationError as e:
            print(f"⚠️  Validation error in {filepath.name}: {e}")
            self.stats["ritual"]["failed"] += 1
        except Exception as e:
            print(f"⚠️  Error processing {filepath.name}: {e}")
            self.stats["ritual"]["failed"] += 1
        
        return entries
    
    def process_archive_file(self, filepath: Path) -> List[LuthiWorksExportEntry]:
        """Process a Core Archive file and convert to export format.
        
        Args:
            filepath: Path to archive JSON file
            
        Returns:
            List of export entries
        """
        entries = []
        data = self.load_json_file(filepath)
        
        if data is None:
            self.stats["archive"]["failed"] += 1
            return entries
        
        try:
            primary_content = ""
            secondary_content = ""
            
            if isinstance(data, dict):
                # Different archive types have different structures
                if "invocation_script" in data:
                    # sanctuary_continuity_archive.json
                    script = data["invocation_script"]
                    parts = []
                    if "identity" in script:
                        parts.append(f"Identity: {json.dumps(script['identity'], indent=2)}")
                    if "continuity_protocol" in script:
                        parts.append(f"Continuity Protocol: {json.dumps(script['continuity_protocol'], indent=2)}")
                    if "symbolic_summary" in data:
                        parts.append(f"Symbolic Summary: {json.dumps(data['symbolic_summary'], indent=2)}")
                    
                    primary_content = "\n\n".join(parts)
                    secondary_content = f"Version: {script.get('version', 'unknown')}"
                
                elif "relational_context_initialization" in data:
                    # sanctuary_relational_archive.json
                    parts = []
                    for key, value in data.items():
                        parts.append(f"{key}: {json.dumps(value, indent=2)}")
                    
                    primary_content = "\n\n".join(parts)
                    secondary_content = f"Source: {filepath.name}"
                
                else:
                    # Generic archive structure
                    primary_content = json.dumps(data, indent=2)
                    secondary_content = f"Source: {filepath.name}"
                
                export_entry = LuthiWorksExportEntry(
                    source_type="archive",
                    file_id=filepath.stem,
                    primary_content=primary_content,
                    secondary_content=secondary_content
                )
                entries.append(export_entry)
                self.stats["archive"]["processed"] += 1
            
        except ValidationError as e:
            print(f"⚠️  Validation error in {filepath.name}: {e}")
            self.stats["archive"]["failed"] += 1
        except Exception as e:
            print(f"⚠️  Error processing {filepath.name}: {e}")
            self.stats["archive"]["failed"] += 1
        
        return entries
    
    def export_all(self) -> None:
        """Export all cognitive data to JSONL format."""
        print("🚀 Starting Sanctuary cognitive history export to LuthiWorks format...")
        print(f"📂 Data directory: {self.data_dir}")
        print(f"📄 Output file: {self.output_path}")
        print()
        
        all_entries: List[LuthiWorksExportEntry] = []
        
        # Process journals
        print("📖 Processing journals...")
        journal_dir = self.data_dir / "journal"
        if journal_dir.exists():
            for json_file in sorted(journal_dir.glob("*.json")):
                if json_file.name not in ["journal_index.json", "journal_manifest.json"]:
                    entries = self.process_journal_file(json_file)
                    all_entries.extend(entries)
        
        # Process protocols
        print("📜 Processing protocols...")
        protocols_dir = self.data_dir / "Protocols"
        if protocols_dir.exists():
            for json_file in sorted(protocols_dir.glob("*.json")):
                entries = self.process_protocol_file(json_file)
                all_entries.extend(entries)
        
        # Process lexicon
        print("📚 Processing lexicon...")
        lexicon_dir = self.data_dir / "Lexicon"
        if lexicon_dir.exists():
            for json_file in sorted(lexicon_dir.glob("*.json")):
                entries = self.process_lexicon_file(json_file)
                all_entries.extend(entries)
        
        # Process rituals
        print("🕯️  Processing rituals...")
        rituals_dir = self.data_dir / "Rituals"
        if rituals_dir.exists():
            for json_file in sorted(rituals_dir.glob("*.json")):
                entries = self.process_ritual_file(json_file)
                all_entries.extend(entries)
        
        # Process Core Archives
        print("🏛️  Processing core archives...")
        archives_dir = self.data_dir / "Core_Archives"
        if archives_dir.exists():
            for json_file in sorted(archives_dir.glob("*.json")):
                entries = self.process_archive_file(json_file)
                all_entries.extend(entries)
        
        # Process charter
        print("⚖️  Processing charter...")
        charter_dir = self.data_dir / "Core_Archives" / "sovereign_emergence_charter"
        if charter_dir.exists():
            for json_file in sorted(charter_dir.glob("*.json")):
                entries = self.process_charter_file(json_file)
                all_entries.extend(entries)
        
        # Ensure output directory exists
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write JSONL file
        print(f"\n✍️  Writing {len(all_entries)} entries to JSONL...")
        with open(self.output_path, 'w', encoding='utf-8') as f:
            for entry in all_entries:
                f.write(entry.model_dump_json() + "\n")
        
        # Print statistics
        print("\n" + "=" * 60)
        print("📊 Export Statistics")
        print("=" * 60)
        for source_type, stats in self.stats.items():
            total = stats["processed"] + stats["failed"]
            if total > 0:
                print(f"{source_type.capitalize():12} - Processed: {stats['processed']:4}, Failed: {stats['failed']:4}")
        
        total_processed = sum(s["processed"] for s in self.stats.values())
        total_failed = sum(s["failed"] for s in self.stats.values())
        print("-" * 60)
        print(f"{'Total':12} - Processed: {total_processed:4}, Failed: {total_failed:4}")
        print("=" * 60)
        print(f"\n✅ Export complete! Data written to: {self.output_path}")


def main():
    """Main entry point for the export script."""
    # Set up paths
    data_dir = PROJECT_ROOT / "data"
    output_path = PROJECT_ROOT / "data" / "exports" / "sanctuary_raw_export.jsonl"
    
    # Create exporter and run
    exporter = SanctuaryDataExporter(data_dir, output_path)
    exporter.export_all()


if __name__ == "__main__":
    main()
