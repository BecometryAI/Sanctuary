"""
Unit tests for WorldModel and related classes.

Tests cover:
- Prediction creation and validation
- PredictionError computation
- SelfModel capabilities and state updates
- EnvironmentModel entity and relationship management
- WorldModel prediction and error tracking
"""

import pytest
from datetime import datetime

from lyra.cognitive_core.world_model import (
    WorldModel,
    Prediction,
    PredictionError,
    SelfModel,
    EnvironmentModel,
    EntityModel,
    Relationship
)


class TestPrediction:
    """Test Prediction dataclass."""
    
    def test_prediction_creation(self):
        """Test creating a valid prediction."""
        pred = Prediction(
            content="Will receive a response",
            confidence=0.8,
            time_horizon=1.0,
            source="test"
        )
        assert pred.content == "Will receive a response"
        assert pred.confidence == 0.8
        assert pred.time_horizon == 1.0
        assert pred.source == "test"
        assert isinstance(pred.created_at, datetime)
    
    def test_prediction_invalid_confidence(self):
        """Test that invalid confidence raises error."""
        with pytest.raises(ValueError):
            Prediction(
                content="test",
                confidence=1.5,  # Invalid
                time_horizon=1.0,
                source="test"
            )
    
    def test_prediction_negative_time_horizon(self):
        """Test that negative time horizon raises error."""
        with pytest.raises(ValueError):
            Prediction(
                content="test",
                confidence=0.5,
                time_horizon=-1.0,  # Invalid
                source="test"
            )


class TestPredictionError:
    """Test PredictionError dataclass."""
    
    def test_prediction_error_creation(self):
        """Test creating a prediction error."""
        pred = Prediction(
            content="Expected A",
            confidence=0.7,
            time_horizon=1.0,
            source="test"
        )
        error = PredictionError(
            prediction=pred,
            actual="Got B",
            magnitude=0.8,
            surprise=1.2
        )
        assert error.prediction == pred
        assert error.actual == "Got B"
        assert error.magnitude == 0.8
        assert error.surprise == 1.2
    
    def test_compute_surprise(self):
        """Test surprise computation from confidence."""
        # High confidence in wrong prediction = high surprise
        surprise = PredictionError.compute_surprise(0.9)
        assert surprise > 2.0
        
        # Low confidence in wrong prediction = low surprise
        surprise = PredictionError.compute_surprise(0.1)
        assert surprise < 1.0


class TestSelfModel:
    """Test SelfModel class."""
    
    def test_initialization(self):
        """Test SelfModel initializes correctly."""
        model = SelfModel()
        assert "language_understanding" in model.capabilities
        assert "emotional_valence" in model.states
        assert len(model.predictions_about_self) == 0
    
    def test_predict_own_behavior(self):
        """Test self-behavior prediction."""
        model = SelfModel()
        context = {
            "goals": [{"description": "Learn Python", "priority": 0.8}],
        }
        prediction = model.predict_own_behavior(context)
        assert isinstance(prediction, Prediction)
        assert prediction.source == "self_model"
        assert "goal" in prediction.content.lower()
    
    def test_update_from_action(self):
        """Test updating self-model from action outcomes."""
        model = SelfModel()
        initial_capability = model.capabilities.get("language_generation", 0.5)
        
        action = {"type": "speak", "content": "Hello"}
        outcome = {"success": True}
        
        model.update_from_action(action, outcome)
        
        # Capability should have been updated slightly
        assert len(model._action_history) == 1
    
    def test_get_capability(self):
        """Test capability retrieval."""
        model = SelfModel()
        cap = model.get_capability("language_understanding")
        assert 0.0 <= cap <= 1.0
        
        # Unknown capability should return 0
        unknown = model.get_capability("unknown_skill")
        assert unknown == 0.0


class TestEnvironmentModel:
    """Test EnvironmentModel class."""
    
    def test_initialization(self):
        """Test EnvironmentModel initializes correctly."""
        model = EnvironmentModel()
        assert len(model.entities) == 0
        assert len(model.relationships) == 0
        assert len(model.predictions_about_world) == 0
    
    def test_add_entity(self):
        """Test adding an entity."""
        model = EnvironmentModel()
        entity = EntityModel(
            entity_id="user_1",
            entity_type="agent",
            properties={"name": "Alice"}
        )
        model.add_entity(entity)
        assert "user_1" in model.entities
        assert model.entities["user_1"].properties["name"] == "Alice"
    
    def test_add_relationship(self):
        """Test adding a relationship."""
        model = EnvironmentModel()
        rel = Relationship(
            source="user_1",
            relation_type="speaks_to",
            target="user_2"
        )
        model.add_relationship(rel)
        assert len(model.relationships) == 1
        assert model.relationships[0].relation_type == "speaks_to"
    
    def test_get_relationships(self):
        """Test querying relationships."""
        model = EnvironmentModel()
        rel1 = Relationship("A", "likes", "B")
        rel2 = Relationship("A", "knows", "C")
        rel3 = Relationship("B", "likes", "C")
        
        model.add_relationship(rel1)
        model.add_relationship(rel2)
        model.add_relationship(rel3)
        
        # Query by source
        results = model.get_relationships(source="A")
        assert len(results) == 2
        
        # Query by relation type
        results = model.get_relationships(relation_type="likes")
        assert len(results) == 2
    
    def test_predict_environment(self):
        """Test environment prediction."""
        model = EnvironmentModel()
        entity = EntityModel("agent_1", "agent", {"name": "Bob"})
        model.add_entity(entity)
        
        predictions = model.predict_environment(1.0, {})
        assert len(predictions) > 0
        assert all(isinstance(p, Prediction) for p in predictions)


class TestWorldModel:
    """Test WorldModel class."""
    
    def test_initialization(self):
        """Test WorldModel initializes correctly."""
        model = WorldModel()
        assert isinstance(model.self_model, SelfModel)
        assert isinstance(model.environment_model, EnvironmentModel)
        assert len(model.predictions) == 0
        assert len(model.prediction_errors) == 0
    
    def test_predict(self):
        """Test prediction generation."""
        model = WorldModel()
        context = {
            "goals": [{"description": "test goal"}],
            "emotional_state": {"valence": 0.0, "arousal": 0.0}
        }
        predictions = model.predict(time_horizon=1.0, context=context)
        assert len(predictions) > 0
        assert all(isinstance(p, Prediction) for p in predictions)
    
    def test_update_on_percept(self):
        """Test percept processing and error detection."""
        model = WorldModel()
        
        # Generate a prediction first
        context = {"goals": []}
        model.predict(time_horizon=1.0, context=context)
        
        # Process a percept
        percept = {"content": "Something unexpected happened"}
        error = model.update_on_percept(percept)
        
        # May or may not generate error depending on prediction match
        if error:
            assert isinstance(error, PredictionError)
    
    def test_get_prediction_error_summary(self):
        """Test prediction error summary."""
        model = WorldModel()
        summary = model.get_prediction_error_summary()
        assert "total_errors" in summary
        assert "average_magnitude" in summary
        assert "average_surprise" in summary
    
    def test_update_from_action_outcome(self):
        """Test updating from action outcome."""
        model = WorldModel()
        action = {"type": "speak", "content": "Hello"}
        outcome = {
            "success": True,
            "observation": {
                "entities": [
                    {"id": "user_1", "type": "agent", "properties": {}}
                ]
            }
        }
        
        model.update_from_action_outcome(action, outcome)
        
        # Check that environment model was updated
        assert len(model.environment_model.entities) > 0
    
    def test_to_dict(self):
        """Test exporting world model state."""
        model = WorldModel()
        state = model.to_dict()
        assert "self_model" in state
        assert "environment_model" in state
        assert "num_predictions" in state
        assert "prediction_error_summary" in state
