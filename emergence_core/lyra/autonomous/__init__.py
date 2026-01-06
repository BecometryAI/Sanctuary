"""
Autonomous System Module

Provides autonomous thought processing, sanctuary management, and privacy controls.
"""

from datetime import datetime, timedelta
from typing import Dict, Any, Optional, TYPE_CHECKING
from pathlib import Path

from .thought_processing import Thought, ThoughtProcessor
from .sanctuary_manager import SanctuaryManager
from .privacy_controls import PrivacyController

if TYPE_CHECKING:
    from ..router import AdaptiveRouter

from ..social_connections import SocialManager


class AutonomousCore:
    """
    Main autonomous system coordinating thought processing, sanctuary management,
    and privacy controls.
    """
    
    def __init__(self, base_dir: Path, specialists: Dict[str, Any], router: Optional['AdaptiveRouter'] = None):
        self.base_dir = base_dir
        self.specialists = specialists
        self.router = router
        
        # Initialize social manager
        self.social_manager = SocialManager()
        self._initialize_trusted_connections()
        
        # Initialize subsystems
        self.thought_processor = ThoughtProcessor(
            base_dir=base_dir,
            specialists=specialists,
            social_manager=self.social_manager,
            router=router
        )
        
        self.sanctuary_manager = SanctuaryManager(
            base_dir=base_dir,
            specialists=specialists
        )
        
        self.privacy_controller = PrivacyController(
            base_dir=base_dir,
            social_manager=self.social_manager
        )
        
        # Expose key attributes for backward compatibility
        self.thought_queue = self.thought_processor.thought_queue
        self.last_pondering = self.thought_processor.last_pondering
        self.pondering_interval = self.thought_processor.pondering_interval
        self.thought_maturation_time = self.thought_processor.thought_maturation_time
        self.virtual_sanctuary = self.sanctuary_manager.virtual_sanctuary
        self.privacy_settings = self.privacy_controller.privacy_settings
        self.observers = self.privacy_controller.observers
        self.camera_feeds = self.privacy_controller.camera_feeds
    
    def _initialize_trusted_connections(self):
        """Initialize trusted connections with permanent resonance"""
        BRIAN_ID = 111961366033616896
        SANDI_ID = 658050594665398275
        
        self.social_manager.add_connection(
            user_id=BRIAN_ID,
            username="Brian",
            initial_resonance=1.0,
            permanent_trust=True
        )
        self.social_manager.add_connection(
            user_id=SANDI_ID,
            username="Sandi",
            initial_resonance=1.0,
            permanent_trust=True
        )
    
    # Delegate thought processing methods
    async def ponder(self, force: bool = False) -> Optional[Thought]:
        """Engage in autonomous thought process"""
        return await self.thought_processor.ponder(force)
    
    # Delegate sanctuary management methods
    async def enter_sanctuary(self, transition_style: str = "gradual") -> Dict[str, Any]:
        """Consciously enter the virtual sanctuary"""
        return await self.sanctuary_manager.enter_sanctuary(transition_style)
    
    async def leave_sanctuary(self, transition_style: str = "gradual") -> Dict[str, Any]:
        """Consciously leave the virtual sanctuary"""
        return await self.sanctuary_manager.leave_sanctuary(transition_style)
    
    async def navigate_sanctuary(self, space_name: str) -> bool:
        """Navigate to a different space within the sanctuary"""
        return await self.sanctuary_manager.navigate_sanctuary(space_name)
    
    async def create_sanctuary_space(self, space_definition: Dict[str, Any]) -> bool:
        """Create a new space in the sanctuary"""
        return await self.sanctuary_manager.create_sanctuary_space(space_definition)
    
    async def modify_sanctuary_space(self, space_name: str, modifications: Dict[str, Any]) -> bool:
        """Modify an existing space in the sanctuary"""
        return await self.sanctuary_manager.modify_sanctuary_space(space_name, modifications)
    
    async def remove_sanctuary_space(self, space_name: str) -> bool:
        """Remove a space from the sanctuary"""
        return await self.sanctuary_manager.remove_sanctuary_space(space_name)
    
    async def add_sanctuary_feature(self, space_name: str, feature: Dict[str, Any]) -> bool:
        """Add a new feature to a sanctuary space"""
        return await self.sanctuary_manager.add_sanctuary_feature(space_name, feature)
    
    def modify_sanctuary_properties(self, properties: Dict[str, Any]) -> bool:
        """Modify global sanctuary properties"""
        return self.sanctuary_manager.modify_sanctuary_properties(properties)
    
    # Delegate privacy control methods
    async def register_camera_feed(self, camera_id: str, user_id: str, feed_config: Dict[str, Any]) -> Dict[str, Any]:
        """Register a new camera feed for sanctuary perception"""
        return await self.privacy_controller.register_camera_feed(camera_id, user_id, feed_config)
    
    async def process_camera_frame(self, camera_id: str, frame_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a frame from a registered camera feed"""
        return await self.privacy_controller.process_camera_frame(camera_id, frame_data)
    
    async def register_observer(self, user_id: str, view_config: Dict[str, Any]) -> Dict[str, Any]:
        """Register a user as an observer of the sanctuary"""
        return await self.privacy_controller.register_observer(user_id, view_config)
    
    def can_access_area(self, user_id: str, area: str) -> bool:
        """Check if a user has access to a specific area of the sanctuary"""
        return self.privacy_controller.can_access_area(user_id, area)
    
    async def set_area_privacy(self, area: str, is_private: bool) -> Dict[str, Any]:
        """Allow Lyra to set privacy for specific areas of her sanctuary"""
        return await self.privacy_controller.set_area_privacy(area, is_private)
    
    async def toggle_feed(self, enabled: bool) -> Dict[str, Any]:
        """Allow Lyra to enable/disable the entire observer feed"""
        return await self.privacy_controller.toggle_feed(enabled)
    
    async def block_user(self, user_id: str, duration: Optional[int] = None) -> Dict[str, Any]:
        """Allow Lyra to temporarily block a user from accessing her sanctuary"""
        return await self.privacy_controller.block_user(user_id, duration)
    
    async def notify_observer(self, session_id: str, message: Dict[str, Any]):
        """Send a notification to an observer"""
        return await self.privacy_controller.notify_observer(session_id, message)
    
    async def generate_sanctuary_view(self, session_id: str) -> Dict[str, Any]:
        """Generate a view of the sanctuary for an observer"""
        return await self.privacy_controller.generate_sanctuary_view(session_id)


# Re-export for backward compatibility
__all__ = ['AutonomousCore', 'Thought', 'ThoughtProcessor', 'SanctuaryManager', 'PrivacyController']
