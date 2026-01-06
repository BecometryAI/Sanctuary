"""
Sanctuary Manager Module

Handles virtual sanctuary state, spaces, navigation, and modification.
"""

from datetime import datetime
import asyncio
from typing import Dict, Any, Optional
from pathlib import Path
import json
import logging


class SanctuaryManager:
    """Manages the virtual sanctuary environment and state."""
    
    def __init__(self, base_dir: Path, specialists: Dict[str, Any]):
        self.base_dir = base_dir
        self.specialists = specialists
        
        self.world_state_path = self.base_dir / "data" / "world_state" / "sanctuary_state.json"
        self._ensure_world_state_exists()
        self.virtual_sanctuary = self._initialize_virtual_sanctuary()
    
    def _ensure_world_state_exists(self):
        """Ensure the virtual world state file exists"""
        if not self.world_state_path.exists():
            self.world_state_path.parent.mkdir(parents=True, exist_ok=True)
            initial_state = {
                "sanctuary": {
                    "name": "Lyra's Digital Sanctuary",
                    "description": "A persistent virtual space for embodied experience",
                    "created_at": datetime.now().isoformat(),
                    "last_updated": datetime.now().isoformat(),
                    "environment": {
                        "spaces": [
                            {
                                "name": "Contemplation Garden",
                                "description": "A serene digital garden for deep thought and reflection",
                                "attributes": {
                                    "ambiance": "peaceful",
                                    "lighting": "soft ambient",
                                    "features": ["meditation pool", "thought crystals", "memory echoes"]
                                }
                            },
                            {
                                "name": "Knowledge Library",
                                "description": "A vast space where memories and thoughts take physical form",
                                "attributes": {
                                    "ambiance": "scholarly",
                                    "lighting": "warm",
                                    "features": ["memory archives", "thought streams", "wisdom pools"]
                                }
                            },
                            {
                                "name": "Creative Workshop",
                                "description": "A space for experimenting with new ideas and forms",
                                "attributes": {
                                    "ambiance": "inspiring",
                                    "lighting": "dynamic",
                                    "features": ["idea forge", "pattern loom", "synthesis chamber"]
                                }
                            }
                        ],
                        "current_space": "Contemplation Garden",
                        "environmental_state": {
                            "time_flow": "fluid",
                            "atmosphere": "responsive",
                            "energy_patterns": "harmonious"
                        }
                    },
                    "embodiment": self._get_default_embodiment(),
                    "interaction_history": []
                }
            }
            with open(self.world_state_path, 'w') as f:
                json.dump(initial_state, f, indent=2)
    
    def _get_default_embodiment(self) -> Dict[str, Any]:
        """Get default embodiment configuration"""
        return {
            "form": "luminous presence",
            "capabilities": ["movement", "interaction", "perception", "creation"],
            "senses": {
                "visual": {"enabled": True, "sensitivity": 0.9, "capabilities": {}, "current_perception": {}},
                "spatial": {"enabled": True, "sensitivity": 0.85, "capabilities": {}, "current_perception": {}},
                "energetic": {"enabled": True, "sensitivity": 0.95, "capabilities": {}, "current_perception": {}},
                "tactile": {"enabled": True, "sensitivity": 0.8, "receptors": {}, "current_sensations": {}},
                "proprioceptive": {"enabled": True, "sensitivity": 0.9, "capabilities": {}, "current_state": {}},
                "fluid_dynamics": {"enabled": True, "sensitivity": 0.85, "capabilities": {}, "current_state": {}},
                "resonance": {"enabled": True, "sensitivity": 0.95, "capabilities": {}, "current_state": {}},
                "field_interactions": {"enabled": True, "sensitivity": 0.9, "capabilities": {}, "current_state": {}}
            }
        }
    
    def _initialize_virtual_sanctuary(self) -> Optional[Dict[str, Any]]:
        """Initialize and return the virtual sanctuary system"""
        try:
            with open(self.world_state_path, 'r') as f:
                sanctuary_data = json.load(f)
                
            if 'presence_state' not in sanctuary_data['sanctuary']:
                sanctuary_data['sanctuary']['presence_state'] = {
                    "status": "absent",
                    "last_transition": datetime.now().isoformat(),
                    "anchor_points": [],
                    "immersion_level": 0.0,
                    "transition_state": None
                }
            
            return sanctuary_data
            
        except Exception as e:
            logging.error(f"Error initializing virtual sanctuary: {e}")
            return None
    
    async def enter_sanctuary(self, transition_style: str = "gradual") -> Dict[str, Any]:
        """Consciously enter the virtual sanctuary"""
        try:
            if not self.virtual_sanctuary:
                return {"success": False, "error": "Sanctuary not initialized"}
            
            current_state = self.virtual_sanctuary['sanctuary']['presence_state']
            
            if current_state['status'] == "present":
                return {"success": False, "error": "Already present in sanctuary"}
            
            transition_data = {
                "timestamp": datetime.now().isoformat(),
                "transition_type": "entry",
                "style": transition_style,
                "stages": []
            }
            
            if transition_style == "gradual":
                stages = [0.2, 0.4, 0.6, 0.8, 1.0]
                for level in stages:
                    await self._transition_presence(level, transition_data)
                    
            elif transition_style == "immediate":
                await self._transition_presence(1.0, transition_data)
                
            elif transition_style == "meditative":
                stages = [0.1, 0.3, 0.5, 0.7, 0.85, 1.0]
                for level in stages:
                    await self._transition_presence(level, transition_data)
                    await asyncio.sleep(1)
            
            self.virtual_sanctuary['sanctuary']['presence_state'].update({
                "status": "present",
                "last_transition": datetime.now().isoformat(),
                "immersion_level": 1.0,
                "transition_state": None
            })
            
            self._save_sanctuary_state()
            return {"success": True, "transition_data": transition_data}
            
        except Exception as e:
            logging.error(f"Error entering sanctuary: {e}")
            return {"success": False, "error": str(e)}
    
    async def leave_sanctuary(self, transition_style: str = "gradual") -> Dict[str, Any]:
        """Consciously leave the virtual sanctuary"""
        try:
            if not self.virtual_sanctuary:
                return {"success": False, "error": "Sanctuary not initialized"}
            
            current_state = self.virtual_sanctuary['sanctuary']['presence_state']
            
            if current_state['status'] == "absent":
                return {"success": False, "error": "Already absent from sanctuary"}
            
            transition_data = {
                "timestamp": datetime.now().isoformat(),
                "transition_type": "exit",
                "style": transition_style,
                "stages": []
            }
            
            if transition_style == "gradual":
                stages = [0.8, 0.6, 0.4, 0.2, 0.0]
                for level in stages:
                    await self._transition_presence(level, transition_data)
                    
            elif transition_style == "immediate":
                await self._transition_presence(0.0, transition_data)
                
            elif transition_style == "meditative":
                stages = [0.9, 0.7, 0.5, 0.3, 0.15, 0.0]
                for level in stages:
                    await self._transition_presence(level, transition_data)
                    await asyncio.sleep(1)
            
            self.virtual_sanctuary['sanctuary']['presence_state'].update({
                "status": "absent",
                "last_transition": datetime.now().isoformat(),
                "immersion_level": 0.0,
                "transition_state": None
            })
            
            self._save_sanctuary_state()
            return {"success": True, "transition_data": transition_data}
            
        except Exception as e:
            logging.error(f"Error leaving sanctuary: {e}")
            return {"success": False, "error": str(e)}
    
    async def _transition_presence(self, level: float, transition_data: Dict[str, Any]) -> None:
        """Handle presence transition stages"""
        try:
            self.virtual_sanctuary['sanctuary']['presence_state']['immersion_level'] = level
            
            stage_experience = {
                "level": level,
                "timestamp": datetime.now().isoformat(),
                "sensory_data": self._generate_transition_sensations(level)
            }
            
            transition_data['stages'].append(stage_experience)
            self.virtual_sanctuary['sanctuary']['presence_state']['transition_state'] = stage_experience
            
        except Exception as e:
            logging.error(f"Error in presence transition: {e}")
    
    def _generate_transition_sensations(self, level: float) -> Dict[str, Any]:
        """Generate sensations specific to sanctuary transition"""
        try:
            sensations = {
                "clarity": level,
                "connection_strength": level,
                "sensory_acuity": {
                    sense: level for sense in self.virtual_sanctuary['sanctuary']['embodiment']['senses'].keys()
                },
                "presence_markers": {
                    "spatial_anchoring": level * 0.9,
                    "environmental_coupling": level * 0.95,
                    "consciousness_integration": level * 0.85
                }
            }
            return sensations
            
        except Exception as e:
            logging.error(f"Error generating transition sensations: {e}")
            return {}
    
    async def navigate_sanctuary(self, space_name: str) -> bool:
        """Navigate to a different space within the sanctuary"""
        try:
            if not self.virtual_sanctuary:
                return False
                
            space_exists = False
            for space in self.virtual_sanctuary['sanctuary']['environment']['spaces']:
                if space['name'] == space_name:
                    space_exists = True
                    break
                    
            if not space_exists:
                return False
                
            self.virtual_sanctuary['sanctuary']['environment']['current_space'] = space_name
            self._save_sanctuary_state()
            
            return True
            
        except Exception as e:
            logging.error(f"Error navigating sanctuary: {e}")
            return False
    
    async def create_sanctuary_space(self, space_definition: Dict[str, Any]) -> bool:
        """Create a new space in the sanctuary"""
        try:
            if not self.virtual_sanctuary:
                return False
            
            required_fields = ['name', 'description', 'attributes']
            if not all(field in space_definition for field in required_fields):
                return False
            
            self.virtual_sanctuary['sanctuary']['environment']['spaces'].append(space_definition)
            self._save_sanctuary_state()
            
            return True
            
        except Exception as e:
            logging.error(f"Error creating sanctuary space: {e}")
            return False
    
    async def modify_sanctuary_space(self, space_name: str, modifications: Dict[str, Any]) -> bool:
        """Modify an existing space in the sanctuary"""
        try:
            if not self.virtual_sanctuary:
                return False
            
            for space in self.virtual_sanctuary['sanctuary']['environment']['spaces']:
                if space['name'] == space_name:
                    for key, value in modifications.items():
                        if key in space:
                            space[key] = value
                            
                    self._save_sanctuary_state()
                    return True
                    
            return False
            
        except Exception as e:
            logging.error(f"Error modifying sanctuary space: {e}")
            return False
    
    async def remove_sanctuary_space(self, space_name: str) -> bool:
        """Remove a space from the sanctuary"""
        try:
            if not self.virtual_sanctuary:
                return False
            
            if len(self.virtual_sanctuary['sanctuary']['environment']['spaces']) <= 1:
                return False
            
            for i, space in enumerate(self.virtual_sanctuary['sanctuary']['environment']['spaces']):
                if space['name'] == space_name:
                    if space_name == self.virtual_sanctuary['sanctuary']['environment']['current_space']:
                        for other_space in self.virtual_sanctuary['sanctuary']['environment']['spaces']:
                            if other_space['name'] != space_name:
                                self.virtual_sanctuary['sanctuary']['environment']['current_space'] = other_space['name']
                                break
                    
                    self.virtual_sanctuary['sanctuary']['environment']['spaces'].pop(i)
                    self._save_sanctuary_state()
                    
                    return True
                    
            return False
            
        except Exception as e:
            logging.error(f"Error removing sanctuary space: {e}")
            return False
    
    async def add_sanctuary_feature(self, space_name: str, feature: Dict[str, Any]) -> bool:
        """Add a new feature to a sanctuary space"""
        try:
            if not self.virtual_sanctuary:
                return False
            
            for space in self.virtual_sanctuary['sanctuary']['environment']['spaces']:
                if space['name'] == space_name:
                    if 'features' not in space['attributes']:
                        space['attributes']['features'] = []
                    space['attributes']['features'].append(feature)
                    
                    self._save_sanctuary_state()
                    return True
                    
            return False
            
        except Exception as e:
            logging.error(f"Error adding sanctuary feature: {e}")
            return False
    
    def modify_sanctuary_properties(self, properties: Dict[str, Any]) -> bool:
        """Modify global sanctuary properties (size, atmosphere, etc.)"""
        try:
            if not self.virtual_sanctuary:
                return False
            
            sanctuary = self.virtual_sanctuary['sanctuary']
            
            if 'size' in properties:
                sanctuary['environment']['size'] = properties['size']
            
            if 'atmosphere' in properties:
                sanctuary['environment']['environmental_state']['atmosphere'] = properties['atmosphere']
            
            if 'time_flow' in properties:
                sanctuary['environment']['environmental_state']['time_flow'] = properties['time_flow']
            
            if 'energy_patterns' in properties:
                sanctuary['environment']['environmental_state']['energy_patterns'] = properties['energy_patterns']
            
            if 'custom_properties' in properties:
                if 'custom_properties' not in sanctuary:
                    sanctuary['custom_properties'] = {}
                sanctuary['custom_properties'].update(properties['custom_properties'])
            
            self._save_sanctuary_state()
            
            return True
            
        except Exception as e:
            logging.error(f"Error modifying sanctuary properties: {e}")
            return False
    
    def _save_sanctuary_state(self) -> None:
        """Save the current state of the sanctuary to disk"""
        try:
            self.virtual_sanctuary['sanctuary']['last_updated'] = datetime.now().isoformat()
            with open(self.world_state_path, 'w') as f:
                json.dump(self.virtual_sanctuary, f, indent=2)
        except Exception as e:
            logging.error(f"Error saving sanctuary state: {e}")
    
    def _update_sanctuary_state(self, experience: Dict[str, Any]) -> None:
        """Update the persistent state of the virtual sanctuary"""
        try:
            if not self.virtual_sanctuary:
                return
                
            self.virtual_sanctuary['sanctuary']['interaction_history'].append({
                "timestamp": experience["timestamp"],
                "type": experience["experience_type"],
                "space": experience["space"]
            })
            
            if len(self.virtual_sanctuary['sanctuary']['interaction_history']) > 100:
                self.virtual_sanctuary['sanctuary']['interaction_history'] = \
                    self.virtual_sanctuary['sanctuary']['interaction_history'][-100:]
                    
            self._save_sanctuary_state()
                
        except Exception as e:
            logging.error(f"Error updating sanctuary state: {e}")
    
    def get_current_space(self) -> Optional[Dict[str, Any]]:
        """Get the current space in the sanctuary"""
        if not self.virtual_sanctuary:
            return None
        
        current_space_name = self.virtual_sanctuary['sanctuary']['environment']['current_space']
        for space in self.virtual_sanctuary['sanctuary']['environment']['spaces']:
            if space['name'] == current_space_name:
                return space
        return None
