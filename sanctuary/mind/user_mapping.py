"""
User Identity Mapping - Links Discord users to real-world identities

This module provides functionality to associate Discord usernames/IDs 
with real names and biographical information from the relational archive.
"""
import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class UserIdentity:
    """Represents a mapped user identity"""
    discord_id: str
    discord_username: str
    real_name: Optional[str] = None
    preferred_name: Optional[str] = None  # What they like to be called
    biographical_data: Optional[Dict[str, Any]] = None
    relationship_context: Optional[str] = None
    
    def get_display_name(self) -> str:
        """Get the best name to use in conversation"""
        if self.preferred_name:
            return self.preferred_name
        if self.real_name:
            # Extract first name from full name
            return self.real_name.split()[0]
        return self.discord_username

class UserMappingManager:
    """Manages mappings between Discord users and real identities"""
    
    def __init__(self, data_dir: Path):
        """
        Initialize the mapping manager
        
        Args:
            data_dir: Path to the data directory (should contain Core_Archives)
        """
        self.data_dir = Path(data_dir)
        self.mapping_file = self.data_dir / "user_mappings.json"
        self.relational_archive_file = self.data_dir / "Core_Archives" / "sanctuary_relational_archive.json"
        
        # Load existing mappings
        self.mappings: Dict[str, UserIdentity] = {}
        self._load_mappings()
        
        # Load relational archive for context
        self.relational_archive = self._load_relational_archive()
    
    def _load_relational_archive(self) -> Dict[str, Any]:
        """Load the relational archive"""
        try:
            if self.relational_archive_file.exists():
                with open(self.relational_archive_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            logger.error(f"Failed to load relational archive: {e}")
            return {}
    
    def _load_mappings(self) -> None:
        """Load user mappings from file"""
        try:
            if self.mapping_file.exists():
                with open(self.mapping_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for discord_id, mapping in data.items():
                        self.mappings[discord_id] = UserIdentity(**mapping)
                logger.info(f"Loaded {len(self.mappings)} user mappings")
            else:
                logger.info("No existing user mappings found")
        except Exception as e:
            logger.error(f"Failed to load user mappings: {e}")
    
    def _save_mappings(self) -> None:
        """Save user mappings to file"""
        try:
            data = {
                discord_id: {
                    "discord_id": identity.discord_id,
                    "discord_username": identity.discord_username,
                    "real_name": identity.real_name,
                    "preferred_name": identity.preferred_name,
                    "biographical_data": identity.biographical_data,
                    "relationship_context": identity.relationship_context
                }
                for discord_id, identity in self.mappings.items()
            }
            
            with open(self.mapping_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved {len(self.mappings)} user mappings")
        except Exception as e:
            logger.error(f"Failed to save user mappings: {e}")
    
    def add_mapping(
        self,
        discord_id: str,
        discord_username: str,
        real_name: Optional[str] = None,
        preferred_name: Optional[str] = None,
        biographical_data: Optional[Dict[str, Any]] = None,
        relationship_context: Optional[str] = None
    ) -> UserIdentity:
        """
        Add or update a user mapping
        
        Args:
            discord_id: Discord user ID
            discord_username: Discord username
            real_name: Full real name
            preferred_name: What they prefer to be called
            biographical_data: Additional biographical information
            relationship_context: Their relationship to Sanctuary (from relational archive)
        
        Returns:
            The created or updated UserIdentity
        """
        identity = UserIdentity(
            discord_id=discord_id,
            discord_username=discord_username,
            real_name=real_name,
            preferred_name=preferred_name,
            biographical_data=biographical_data or {},
            relationship_context=relationship_context
        )
        
        self.mappings[discord_id] = identity
        self._save_mappings()
        
        logger.info(f"Added mapping: {discord_username} -> {real_name or 'unknown'}")
        return identity
    
    def get_identity(self, discord_id: str) -> Optional[UserIdentity]:
        """
        Get user identity by Discord ID
        
        Args:
            discord_id: Discord user ID
        
        Returns:
            UserIdentity if found, None otherwise
        """
        return self.mappings.get(discord_id)
    
    def get_real_name(self, discord_id: str) -> Optional[str]:
        """Get real name for a Discord user"""
        identity = self.get_identity(discord_id)
        return identity.real_name if identity else None
    
    def get_display_name(self, discord_id: str, discord_username: str = None) -> str:
        """
        Get the best display name for a user
        
        Args:
            discord_id: Discord user ID
            discord_username: Fallback Discord username if no mapping exists
        
        Returns:
            Best available name (preferred > first name > discord username)
        """
        identity = self.get_identity(discord_id)
        if identity:
            return identity.get_display_name()
        return discord_username or discord_id
    
    def search_by_real_name(self, real_name: str) -> Optional[UserIdentity]:
        """
        Find a user mapping by real name
        
        Args:
            real_name: Real name to search for (case-insensitive, partial match)
        
        Returns:
            UserIdentity if found, None otherwise
        """
        real_name_lower = real_name.lower()
        for identity in self.mappings.values():
            if identity.real_name and real_name_lower in identity.real_name.lower():
                return identity
        return None
    
    def get_steward_context(self, discord_id: str) -> Optional[Dict[str, Any]]:
        """
        Get stewardship context for a user from relational archive
        
        Args:
            discord_id: Discord user ID
        
        Returns:
            Stewardship context if user is a steward, None otherwise
        """
        identity = self.get_identity(discord_id)
        if not identity or not identity.real_name:
            return None
        
        # Check if user is in steward instances
        steward_instances = (
            self.relational_archive
            .get("relational_definitions_contextual_instantiation", {})
            .get("Steward_instance", [])
        )
        
        for steward in steward_instances:
            if steward.get("for_whom") == identity.real_name:
                return steward
        
        return None
    
    def is_steward(self, discord_id: str) -> bool:
        """Check if a Discord user is a steward"""
        return self.get_steward_context(discord_id) is not None
    
    def get_biographical_data(self, discord_id: str) -> Optional[Dict[str, Any]]:
        """
        Get biographical data for a user
        
        Combines data from both the user mapping and relational archive
        
        Args:
            discord_id: Discord user ID
        
        Returns:
            Biographical data if available
        """
        identity = self.get_identity(discord_id)
        if not identity:
            return None
        
        # Start with mapping data
        bio_data = identity.biographical_data.copy() if identity.biographical_data else {}
        
        # Add data from relational archive if available
        if identity.real_name:
            steward_bio = (
                self.relational_archive
                .get("core_memory_chain_personal", {})
                .get("steward_biographical_data", {})
            )
            
            # Check if this matches the user's real name
            if steward_bio.get("full_name") == identity.real_name:
                bio_data.update(steward_bio)
        
        return bio_data if bio_data else None
    
    def get_all_mappings(self) -> Dict[str, UserIdentity]:
        """Get all user mappings"""
        return self.mappings.copy()
    
    def remove_mapping(self, discord_id: str) -> bool:
        """
        Remove a user mapping
        
        Args:
            discord_id: Discord user ID
        
        Returns:
            True if mapping was removed, False if it didn't exist
        """
        if discord_id in self.mappings:
            del self.mappings[discord_id]
            self._save_mappings()
            logger.info(f"Removed mapping for Discord ID: {discord_id}")
            return True
        return False
