"""
EnvironmentModel: Model of external world and other agents.

This module implements the external world representation, tracking:
- Entities: Objects, agents, locations in the world
- Relationships: Connections between entities
- Environmental predictions: What will happen in the world
"""

from __future__ import annotations

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime

from .prediction import Prediction

logger = logging.getLogger(__name__)


@dataclass
class EntityModel:
    """
    Model of an entity in the environment.
    
    Attributes:
        entity_id: Unique identifier
        entity_type: Type (e.g., "agent", "object", "location")
        properties: Entity properties and attributes
        last_observed: When entity was last observed
    """
    entity_id: str
    entity_type: str
    properties: Dict[str, Any] = field(default_factory=dict)
    last_observed: datetime = field(default_factory=datetime.now)


@dataclass
class Relationship:
    """
    Relationship between entities.
    
    Attributes:
        source: Source entity ID
        relation_type: Type of relationship (e.g., "speaks_to", "located_at")
        target: Target entity ID
        properties: Additional relationship properties
    """
    source: str
    relation_type: str
    target: str
    properties: Dict[str, Any] = field(default_factory=dict)


class EnvironmentModel:
    """
    Model of external world and other agents.
    
    Maintains representations of:
    - Entities: Objects, people, locations
    - Relationships: How entities relate to each other
    - Environmental predictions: What will happen
    """
    
    def __init__(self):
        """Initialize empty environment model."""
        # Entities by ID
        self.entities: Dict[str, EntityModel] = {}
        
        # Relationships between entities
        self.relationships: List[Relationship] = []
        
        # Active predictions about the environment
        self.predictions_about_world: List[Prediction] = []
        
        logger.info("EnvironmentModel initialized")
    
    def add_entity(self, entity: EntityModel):
        """
        Add or update an entity in the model.
        
        Args:
            entity: EntityModel to add/update
        """
        self.entities[entity.entity_id] = entity
        logger.debug(f"Added entity: {entity.entity_id} ({entity.entity_type})")
    
    def get_entity(self, entity_id: str) -> Optional[EntityModel]:
        """
        Get entity by ID.
        
        Args:
            entity_id: Entity identifier
            
        Returns:
            EntityModel if found, None otherwise
        """
        return self.entities.get(entity_id)
    
    def add_relationship(self, relationship: Relationship):
        """
        Add a relationship between entities.
        
        Args:
            relationship: Relationship to add
        """
        self.relationships.append(relationship)
        logger.debug(f"Added relationship: {relationship.source} -{relationship.relation_type}-> {relationship.target}")
    
    def get_relationships(
        self,
        source: Optional[str] = None,
        relation_type: Optional[str] = None,
        target: Optional[str] = None
    ) -> List[Relationship]:
        """
        Query relationships by source, type, and/or target.
        
        Args:
            source: Filter by source entity (optional)
            relation_type: Filter by relationship type (optional)
            target: Filter by target entity (optional)
            
        Returns:
            List of matching relationships
        """
        results = self.relationships
        
        if source is not None:
            results = [r for r in results if r.source == source]
        if relation_type is not None:
            results = [r for r in results if r.relation_type == relation_type]
        if target is not None:
            results = [r for r in results if r.target == target]
        
        return results
    
    def predict_environment(self, time_horizon: float, context: Dict[str, Any]) -> List[Prediction]:
        """
        Predict what will happen in environment.
        
        Args:
            time_horizon: How far into the future to predict (seconds)
            context: Current context for making predictions
            
        Returns:
            List of predictions about the environment
        """
        predictions = []
        
        # Simple heuristic predictions based on entities and relationships
        
        # Predict based on known entities
        for entity_id, entity in self.entities.items():
            if entity.entity_type == "agent":
                # Predict agent behavior
                prediction = Prediction(
                    content=f"Agent {entity_id} will continue current behavior",
                    confidence=0.5,
                    time_horizon=time_horizon,
                    source="environment_model",
                    created_at=datetime.now()
                )
                predictions.append(prediction)
        
        # Predict based on relationships
        if self.relationships:
            # If there are active interactions, predict continuation
            prediction = Prediction(
                content="Current interactions will continue",
                confidence=0.6,
                time_horizon=time_horizon,
                source="environment_model",
                created_at=datetime.now()
            )
            predictions.append(prediction)
        
        # Store predictions
        self.predictions_about_world.extend(predictions)
        
        # Keep predictions list bounded
        if len(self.predictions_about_world) > 100:
            self.predictions_about_world = self.predictions_about_world[-100:]
        
        return predictions
    
    def update_from_observation(self, observation: Dict[str, Any]):
        """
        Update environment model based on new observation.
        
        Args:
            observation: Observed data about the environment
        """
        # Extract entities from observation
        if "entities" in observation:
            for entity_data in observation["entities"]:
                entity = EntityModel(
                    entity_id=entity_data.get("id", "unknown"),
                    entity_type=entity_data.get("type", "object"),
                    properties=entity_data.get("properties", {}),
                    last_observed=datetime.now()
                )
                self.add_entity(entity)
        
        # Extract relationships
        if "relationships" in observation:
            for rel_data in observation["relationships"]:
                relationship = Relationship(
                    source=rel_data.get("source", ""),
                    relation_type=rel_data.get("type", "unknown"),
                    target=rel_data.get("target", ""),
                    properties=rel_data.get("properties", {})
                )
                self.add_relationship(relationship)
        
        logger.debug(f"Updated environment model from observation")
    
    def to_dict(self) -> Dict[str, Any]:
        """Export environment model as dictionary."""
        return {
            "num_entities": len(self.entities),
            "entities": {eid: {
                "type": e.entity_type,
                "properties": e.properties,
                "last_observed": e.last_observed.isoformat()
            } for eid, e in self.entities.items()},
            "num_relationships": len(self.relationships),
            "num_predictions": len(self.predictions_about_world),
        }
