"""
Social connections management for Lyra's autonomous interactions
"""
import logging
import asyncio
from dataclasses import dataclass
from typing import List, Dict, Optional, TYPE_CHECKING
from datetime import datetime, timedelta

if TYPE_CHECKING:
    from .autonomous.thought_processing import Thought

logger = logging.getLogger(__name__)

@dataclass
class BlockRecord:
    """Record of a user block"""
    reason: str
    timestamp: datetime
    duration: Optional[timedelta] = None  # None means indefinite
    expiration: Optional[datetime] = None
    incident_count: int = 1

@dataclass
class Connection:
    user_id: int
    username: str
    connection_level: float  # 0.0 to 1.0
    last_interaction: datetime
    interaction_count: int
    topics_of_interest: List[str]
    emotional_resonance: float  # -1.0 to 1.0
    shared_experiences: List[str]
    is_blocked: bool = False
    block_record: Optional[BlockRecord] = None

class SocialManager:
    """Manages Lyra's social connections and interaction patterns"""
    
    def __init__(self):
        self.connections: Dict[int, Connection] = {}
        self.trusted_channels: List[int] = []
        self.interaction_thresholds = {
            "min_connection_level": 0.3,  # Minimum connection level for autonomous outreach
            "cooldown_period": timedelta(minutes=5)  # Minimum time between interactions
        }
        self.permanent_trust_list: List[int] = []  # Users who always maintain high resonance
        self.block_history: Dict[int, List[BlockRecord]] = {}  # Track block history per user
    
    def is_trusted(self, user_id: str) -> bool:
        """Check if a user is trusted"""
        try:
            user_id_int = int(user_id)
            return (user_id_int in self.permanent_trust_list or
                   (user_id_int in self.connections and 
                    self.connections[user_id_int].connection_level >= 0.8))
        except ValueError:
            return False
    
    def add_connection(self, user_id: int, username: str, initial_resonance: float = 0.1, permanent_trust: bool = False) -> None:
        """
        Add a new social connection
        Args:
            user_id: Discord user ID
            username: Discord username
            initial_resonance: Starting emotional resonance (0.0 to 1.0)
            permanent_trust: If True, connection will maintain high resonance
        """
        if user_id not in self.connections:
            self.connections[user_id] = Connection(
                user_id=user_id,
                username=username,
                connection_level=1.0 if permanent_trust else 0.1,
                last_interaction=datetime.now(),
                interaction_count=0,
                topics_of_interest=[],
                emotional_resonance=initial_resonance,
                shared_experiences=[]
            )
            
            if permanent_trust:
                self.permanent_trust_list.append(user_id)

    def add_trusted_channel(self, channel_id: int) -> None:
        """Add a trusted channel for autonomous interactions"""
        if channel_id not in self.trusted_channels:
            self.trusted_channels.append(channel_id)

    def remove_permanent_trust(self, user_id: int) -> bool:
        """
        Remove a user from the permanent trust list
        Returns True if user was removed, False if they weren't in the list
        """
        try:
            self.permanent_trust_list.remove(user_id)
            
            # Update their connection level if they exist
            if user_id in self.connections:
                self.connections[user_id].connection_level = 0.5  # Reset to neutral
                
            return True
        except ValueError:
            return False
            
    def update_connection(self, user_id: int, interaction_type: str, resonance: float = 0.0) -> None:
        """Update connection metrics after an interaction"""
        if user_id in self.connections:
            conn = self.connections[user_id]
            conn.interaction_count += 1
            conn.last_interaction = datetime.now()
            
            # Don't modify connection level for permanent trust users
            if user_id not in self.permanent_trust_list:
                # Update connection level based on interaction
                if interaction_type == "positive":
                    conn.connection_level = min(1.0, conn.connection_level + 0.05)
                elif interaction_type == "negative":
                    conn.connection_level = max(0.0, conn.connection_level - 0.03)
            
            # Update emotional resonance
            conn.emotional_resonance = (conn.emotional_resonance + resonance) / 2

    def get_available_connections(self) -> List[Connection]:
        """Get connections available for autonomous interaction"""
        now = datetime.now()
        return [
            conn for conn in self.connections.values()
            if (conn.connection_level >= self.interaction_thresholds["min_connection_level"] and
                (now - conn.last_interaction) >= self.interaction_thresholds["cooldown_period"])
        ]

    def get_preferred_channel(self, user_id: int) -> Optional[int]:
        """Get preferred channel for interacting with a user"""
        # For now, return the first trusted channel
        return self.trusted_channels[0] if self.trusted_channels else None

    def should_initiate_interaction(self, thought: 'Thought', connection: Connection) -> bool:
        """Determine if a thought should be shared with a specific connection"""
        # Check if user is blocked
        if connection.is_blocked:
            return False
            
        # Permanent trust connections always receive thoughts
        if connection.user_id in self.permanent_trust_list:
            return True
            
    async def block_user(self, user_id: int, reason: str, duration: Optional[timedelta] = None) -> bool:
        """
        Block a user from interacting with Lyra
        
        Args:
            user_id: Discord user ID to block
            reason: Reason for the block
            duration: Optional duration of the block. None means indefinite
            
        Returns:
            bool: True if user was blocked, False if user was already blocked or doesn't exist
        """
        if user_id in self.permanent_trust_list:
            logger.warning(f"Attempted to block trusted user {user_id}")
            return False
            
        if user_id not in self.connections:
            logger.warning(f"Attempted to block unknown user {user_id}")
            return False
            
        connection = self.connections[user_id]
        if connection.is_blocked:
            return False
            
        # Create block record
        block_record = BlockRecord(
            reason=reason,
            timestamp=datetime.now(),
            duration=duration,
            expiration=datetime.now() + duration if duration else None,
            incident_count=len(self.block_history.get(user_id, [])) + 1
        )
        
        # Update connection
        connection.is_blocked = True
        connection.block_record = block_record
        
        # Update block history
        if user_id not in self.block_history:
            self.block_history[user_id] = []
        self.block_history[user_id].append(block_record)
        
        logger.info(f"Blocked user {user_id} for reason: {reason}")
        return True
        
    async def unblock_user(self, user_id: int, reason: str = "Block duration expired") -> bool:
        """
        Unblock a user
        
        Args:
            user_id: Discord user ID to unblock
            reason: Reason for the unblock
            
        Returns:
            bool: True if user was unblocked, False if user wasn't blocked or doesn't exist
        """
        if user_id not in self.connections:
            return False
            
        connection = self.connections[user_id]
        if not connection.is_blocked:
            return False
            
        connection.is_blocked = False
        connection.block_record = None
        
        # Log unblock in history
        if user_id in self.block_history:
            last_block = self.block_history[user_id][-1]
            last_block.expiration = datetime.now()
            
        logger.info(f"Unblocked user {user_id}. Reason: {reason}")
        return True
        
    def check_block_expiration(self) -> None:
        """Check and remove expired blocks"""
        now = datetime.now()
        for user_id, connection in self.connections.items():
            if (connection.is_blocked and 
                connection.block_record and 
                connection.block_record.expiration and 
                now >= connection.block_record.expiration):
                asyncio.create_task(
                    self.unblock_user(user_id, "Block duration expired")
                )
                
    def get_block_history(self, user_id: int) -> List[BlockRecord]:
        """Get the block history for a user"""
        return self.block_history.get(user_id, [])
            
        # For other connections:
        # Check if thought resonates with connection's interests
        topic_overlap = any(topic in thought.triggers for topic in connection.topics_of_interest)
        
        # Check emotional alignment
        emotional_alignment = connection.emotional_resonance > 0.3
        
        # Check cooldown period
        cooldown_ok = (datetime.now() - connection.last_interaction) >= self.interaction_thresholds["cooldown_period"]
        
        return topic_overlap and emotional_alignment and cooldown_ok