"""
Test suite for cue-dependent memory retrieval.

Tests verify:
- Cue-based activation from workspace state
- Spreading activation to associated memories
- Emotional congruence biasing retrieval
- Recency weighting with decay
- Competitive retrieval with interference
- Retrieval strengthening
"""
import pytest
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime, timedelta
import json

from ..memory.retrieval import CueDependentRetrieval
from ..memory.emotional_weighting import EmotionalWeighting
from ..memory.storage import MemoryStorage


@pytest.fixture
def mock_storage():
    """Create a mock MemoryStorage instance."""
    storage = Mock(spec=MemoryStorage)
    
    # Mock episodic memory collection
    storage.episodic_memory = Mock()
    storage.episodic_memory.count.return_value = 3
    
    # Mock semantic memory collection
    storage.semantic_memory = Mock()
    storage.semantic_memory.count.return_value = 2
    
    return storage


@pytest.fixture
def emotional_weighting():
    """Create an EmotionalWeighting instance."""
    return EmotionalWeighting()


@pytest.fixture
def cue_dependent_retrieval(mock_storage, emotional_weighting):
    """Create a CueDependentRetrieval instance."""
    return CueDependentRetrieval(
        storage=mock_storage,
        emotional_weighting=emotional_weighting,
        retrieval_threshold=0.3,
        inhibition_strength=0.4,
        strengthening_factor=0.05,
        spread_factor=0.3
    )


class TestCueDependentRetrieval:
    """Test cue-dependent retrieval mechanisms."""
    
    def test_initialization(self, cue_dependent_retrieval):
        """Test that CueDependentRetrieval initializes correctly."""
        assert cue_dependent_retrieval.retrieval_threshold == 0.3
        assert cue_dependent_retrieval.inhibition_strength == 0.4
        assert cue_dependent_retrieval.strengthening_factor == 0.05
        assert cue_dependent_retrieval.spread_factor == 0.3
        assert "total_retrievals" in cue_dependent_retrieval.metrics
    
    def test_encode_cues_from_workspace(self, cue_dependent_retrieval):
        """Test cue extraction from workspace state."""
        workspace_state = {
            "goals": [
                {"description": "respond to user query"},
                {"description": "retrieve relevant memories"}
            ],
            "percepts": {
                "p1": {"raw": "user asked about emotions", "modality": "text"},
                "p2": {"raw": "system processing", "modality": "introspection"}
            },
            "emotions": {"valence": 0.5, "arousal": 0.6, "dominance": 0.7},
            "memories": [
                {"content": "previous conversation about feelings"}
            ]
        }
        
        cue_text = cue_dependent_retrieval._encode_cues(workspace_state)
        
        assert "respond to user query" in cue_text
        assert "retrieve relevant memories" in cue_text
        assert "user asked about emotions" in cue_text
        assert "previous conversation about feelings" in cue_text
    
    def test_encode_cues_empty_workspace(self, cue_dependent_retrieval):
        """Test cue extraction with empty workspace."""
        workspace_state = {}
        cue_text = cue_dependent_retrieval._encode_cues(workspace_state)
        
        assert cue_text == "current context"
    
    def test_recency_weight_recent_memory(self, cue_dependent_retrieval):
        """Test recency weight for a recent memory."""
        recent_time = datetime.now() - timedelta(hours=1)
        metadata = {"last_accessed": recent_time.isoformat()}
        
        weight = cue_dependent_retrieval._recency_weight(metadata)
        
        # Recent memory should have high weight (close to 1.0)
        assert weight > 0.95
    
    def test_recency_weight_old_memory(self, cue_dependent_retrieval):
        """Test recency weight for an old memory."""
        old_time = datetime.now() - timedelta(days=30)
        metadata = {"last_accessed": old_time.isoformat()}
        
        weight = cue_dependent_retrieval._recency_weight(metadata)
        
        # Old memory should have lower weight
        assert weight < 0.5
    
    def test_recency_weight_no_timestamp(self, cue_dependent_retrieval):
        """Test recency weight with no timestamp."""
        metadata = {}
        weight = cue_dependent_retrieval._recency_weight(metadata)
        
        # Should return default value
        assert weight == 0.3
    
    def test_get_candidates(self, cue_dependent_retrieval, mock_storage):
        """Test candidate memory retrieval."""
        # Setup mock episodic query results
        mock_storage.query_episodic.return_value = {
            "ids": [["mem1", "mem2"]],
            "documents": [[
                json.dumps({"content": "memory 1", "type": "test"}),
                json.dumps({"content": "memory 2", "type": "test"})
            ]],
            "metadatas": [[
                {"timestamp": datetime.now().isoformat()},
                {"timestamp": datetime.now().isoformat()}
            ]],
            "distances": [[0.2, 0.3]]
        }
        
        # Setup mock semantic query results
        mock_storage.query_semantic.return_value = {
            "ids": [["mem3"]],
            "documents": [[json.dumps({"content": "semantic memory", "type": "semantic"})]],
            "metadatas": [[{"timestamp": datetime.now().isoformat()}]],
            "distances": [[0.4]]
        }
        
        candidates = cue_dependent_retrieval._get_candidates("test query", max_candidates=50)
        
        assert len(candidates) == 3
        assert "mem1" in candidates
        assert "mem2" in candidates
        assert "mem3" in candidates
        
        # Check similarity conversion (similarity = 1 - distance)
        assert candidates["mem1"]["similarity"] == 0.8  # 1 - 0.2
        assert candidates["mem2"]["similarity"] == 0.7  # 1 - 0.3
        assert candidates["mem3"]["similarity"] == 0.6  # 1 - 0.4
    
    def test_spread_activation(self, cue_dependent_retrieval, mock_storage):
        """Test spreading activation to associated memories."""
        initial_activations = {
            "mem1": 0.8,
            "mem2": 0.5,
            "mem3": 0.2
        }
        
        # Setup associations: mem1 -> mem4 (strong), mem2 -> mem5 (weak)
        mock_storage.get_memory_associations.side_effect = lambda mem_id, collection_type: {
            "mem1": [("mem4", 0.9)],
            "mem2": [("mem5", 0.3)],
            "mem3": []
        }.get(mem_id, [])
        
        spread_activations = cue_dependent_retrieval._spread_activation(
            initial_activations,
            spread_factor=0.3,
            iterations=1
        )
        
        # mem4 should receive activation from mem1
        assert "mem4" in spread_activations
        assert spread_activations["mem4"] > 0
        
        # mem5 should receive activation from mem2
        assert "mem5" in spread_activations
        assert spread_activations["mem5"] > 0
        
        # mem4 should get more activation than mem5 (stronger association and higher source)
        assert spread_activations["mem4"] > spread_activations["mem5"]
    
    def test_competitive_retrieval(self, cue_dependent_retrieval):
        """Test competitive retrieval with interference."""
        activations = {
            "mem1": 0.8,
            "mem2": 0.75,  # Similar to mem1
            "mem3": 0.5,
            "mem4": 0.2    # Below threshold
        }
        
        candidates = {
            "mem1": {
                "data": {"content": "high activation memory"},
                "similarity": 0.85,
                "collection": "episodic"
            },
            "mem2": {
                "data": {"content": "similar high activation"},
                "similarity": 0.80,
                "collection": "episodic"
            },
            "mem3": {
                "data": {"content": "medium activation"},
                "similarity": 0.5,
                "collection": "episodic"
            },
            "mem4": {
                "data": {"content": "low activation"},
                "similarity": 0.3,
                "collection": "episodic"
            }
        }
        
        retrieved = cue_dependent_retrieval._competitive_retrieval(
            activations,
            candidates,
            limit=3
        )
        
        # Should retrieve top memories
        assert len(retrieved) <= 3
        
        # mem1 should be first (highest activation)
        assert retrieved[0]["memory_id"] == "mem1"
        
        # mem2 might be inhibited by mem1 (both have high similarity to cues)
        # mem3 should be retrieved as it's different
        memory_ids = [m["memory_id"] for m in retrieved]
        assert "mem3" in memory_ids
        
        # mem4 should not be retrieved (below threshold)
        assert "mem4" not in memory_ids
    
    def test_strengthen_retrieved(self, cue_dependent_retrieval, mock_storage):
        """Test retrieval strengthening."""
        memories = [
            {"memory_id": "mem1", "collection": "episodic", "content": "test"},
            {"memory_id": "mem2", "collection": "semantic", "content": "test"}
        ]
        
        cue_dependent_retrieval._strengthen_retrieved(memories)
        
        # Should call update_retrieval_metadata for each memory
        assert mock_storage.update_retrieval_metadata.call_count == 2
        mock_storage.update_retrieval_metadata.assert_any_call("mem1", "episodic")
        mock_storage.update_retrieval_metadata.assert_any_call("mem2", "semantic")
        
        # Should update metrics
        assert cue_dependent_retrieval.metrics["strengthening_events"] == 2
    
    def test_retrieve_with_workspace_state(self, cue_dependent_retrieval, mock_storage):
        """Test full retrieval with workspace state."""
        workspace_state = {
            "goals": [{"description": "test goal"}],
            "percepts": {"p1": {"raw": "test percept", "modality": "text"}},
            "emotions": {"valence": 0.5, "arousal": 0.6, "dominance": 0.7},
            "memories": []
        }
        
        # Setup mock query results
        mock_storage.query_episodic.return_value = {
            "ids": [["mem1"]],
            "documents": [[json.dumps({"content": "test memory"})]],
            "metadatas": [[{
                "timestamp": datetime.now().isoformat(),
                "emotional_state": {"valence": 0.5, "arousal": 0.6, "dominance": 0.7}
            }]],
            "distances": [[0.2]]
        }
        
        mock_storage.query_semantic.return_value = {
            "ids": [[]],
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]]
        }
        
        mock_storage.get_memory_associations.return_value = []
        
        retrieved = cue_dependent_retrieval.retrieve(workspace_state, limit=5)
        
        # Should retrieve memories
        assert len(retrieved) > 0
        assert retrieved[0]["content"] == "test memory"
        
        # Should have activation score
        assert "activation" in retrieved[0]
        assert retrieved[0]["activation"] > 0
        
        # Should update metrics
        assert cue_dependent_retrieval.metrics["total_retrievals"] == 1


class TestEmotionalCongruence:
    """Test emotional congruence calculations."""
    
    def test_emotional_congruence_identical_states(self, emotional_weighting):
        """Test congruence with identical emotional states."""
        current_state = {"valence": 0.5, "arousal": 0.6, "dominance": 0.7}
        memory_state = {"valence": 0.5, "arousal": 0.6, "dominance": 0.7}
        
        congruence = emotional_weighting.emotional_congruence_pad(
            current_state,
            memory_state
        )
        
        # Identical states should have perfect congruence
        assert congruence == 1.0
    
    def test_emotional_congruence_opposite_states(self, emotional_weighting):
        """Test congruence with opposite emotional states."""
        current_state = {"valence": 1.0, "arousal": 1.0, "dominance": 1.0}
        memory_state = {"valence": -1.0, "arousal": 0.0, "dominance": 0.0}
        
        congruence = emotional_weighting.emotional_congruence_pad(
            current_state,
            memory_state
        )
        
        # Opposite states should have low congruence
        assert congruence < 0.5
    
    def test_emotional_congruence_similar_states(self, emotional_weighting):
        """Test congruence with similar emotional states."""
        current_state = {"valence": 0.5, "arousal": 0.6, "dominance": 0.7}
        memory_state = {"valence": 0.6, "arousal": 0.65, "dominance": 0.75}
        
        congruence = emotional_weighting.emotional_congruence_pad(
            current_state,
            memory_state
        )
        
        # Similar states should have high congruence
        assert congruence > 0.8
    
    def test_emotional_congruence_no_memory_state(self, emotional_weighting):
        """Test congruence when memory has no emotional state."""
        current_state = {"valence": 0.5, "arousal": 0.6, "dominance": 0.7}
        memory_state = None
        
        congruence = emotional_weighting.emotional_congruence_pad(
            current_state,
            memory_state
        )
        
        # Should return neutral value
        assert congruence == 0.5
    
    def test_emotional_congruence_partial_state(self, emotional_weighting):
        """Test congruence with partial emotional state (missing dimensions)."""
        current_state = {"valence": 0.5, "arousal": 0.6, "dominance": 0.7}
        memory_state = {"valence": 0.5}  # Missing arousal and dominance
        
        congruence = emotional_weighting.emotional_congruence_pad(
            current_state,
            memory_state
        )
        
        # Should use defaults (0.0) for missing dimensions
        assert 0.0 <= congruence <= 1.0


class TestRetrievalMetrics:
    """Test retrieval metrics tracking."""
    
    def test_metrics_initialization(self, cue_dependent_retrieval):
        """Test that metrics are initialized correctly."""
        metrics = cue_dependent_retrieval.get_metrics()
        
        assert "total_retrievals" in metrics
        assert "avg_cue_similarity" in metrics
        assert "spreading_activations" in metrics
        assert "interference_events" in metrics
        assert "strengthening_events" in metrics
        
        assert metrics["total_retrievals"] == 0
    
    def test_metrics_after_retrieval(self, cue_dependent_retrieval, mock_storage):
        """Test that metrics are updated after retrieval."""
        workspace_state = {
            "goals": [{"description": "test"}],
            "emotions": {"valence": 0.5, "arousal": 0.6, "dominance": 0.7}
        }
        
        mock_storage.query_episodic.return_value = {
            "ids": [["mem1"]],
            "documents": [[json.dumps({"content": "test"})]],
            "metadatas": [[{"timestamp": datetime.now().isoformat()}]],
            "distances": [[0.2]]
        }
        
        mock_storage.query_semantic.return_value = {
            "ids": [[]],
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]]
        }
        
        mock_storage.get_memory_associations.return_value = []
        
        cue_dependent_retrieval.retrieve(workspace_state, limit=5)
        
        metrics = cue_dependent_retrieval.get_metrics()
        assert metrics["total_retrievals"] == 1
        assert metrics["avg_cue_similarity"] > 0


class TestMemoryStorageExtensions:
    """Test storage extensions for retrieval tracking."""
    
    def test_update_retrieval_metadata(self, mock_storage):
        """Test updating retrieval metadata."""
        # Setup mock to return existing memory
        mock_storage.episodic_memory.get.return_value = {
            "documents": [["test"]],
            "metadatas": [[{"retrieval_count": 0, "timestamp": datetime.now().isoformat()}]]
        }
        
        # Create real storage instance for this test
        # (We can't fully test without a real ChromaDB, but we can test the logic)
        # For now, just verify the mock is called correctly
        from ..memory.storage import MemoryStorage
        
        # Test would require actual ChromaDB initialization
        # Skip for now as it needs actual persistence setup
    
    def test_add_memory_association(self, mock_storage):
        """Test adding memory associations."""
        # Setup mock to return existing memory
        mock_storage.episodic_memory.get.return_value = {
            "documents": [["test"]],
            "metadatas": [[{"associations": []}]]
        }
        
        # Test would require actual ChromaDB initialization
        # Skip for now as it needs actual persistence setup


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_retrieve_with_none_workspace(self, cue_dependent_retrieval):
        """Test retrieval with None workspace state."""
        result = cue_dependent_retrieval.retrieve(None, limit=5)
        assert result == []
    
    def test_retrieve_with_invalid_workspace(self, cue_dependent_retrieval):
        """Test retrieval with invalid workspace state."""
        result = cue_dependent_retrieval.retrieve("invalid", limit=5)
        assert result == []
    
    def test_retrieve_with_zero_limit(self, cue_dependent_retrieval, mock_storage):
        """Test retrieval with zero or negative limit."""
        workspace_state = {"goals": [], "emotions": {}}
        
        mock_storage.query_episodic.return_value = {
            "ids": [[]],
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]]
        }
        
        # Should use default limit of 5
        result = cue_dependent_retrieval.retrieve(workspace_state, limit=0)
        assert isinstance(result, list)
    
    def test_retrieve_with_negative_limit(self, cue_dependent_retrieval, mock_storage):
        """Test retrieval with negative limit."""
        workspace_state = {"goals": [], "emotions": {}}
        
        mock_storage.query_episodic.return_value = {
            "ids": [[]],
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]]
        }
        
        result = cue_dependent_retrieval.retrieve(workspace_state, limit=-5)
        assert isinstance(result, list)
    
    def test_encode_cues_with_mixed_types(self, cue_dependent_retrieval):
        """Test cue extraction with mixed goal types."""
        workspace_state = {
            "goals": [
                {"description": "goal 1"},
                "goal 2",  # String instead of dict
                None,  # None value
                {"description": ""}  # Empty description
            ],
            "percepts": {},
            "emotions": {}
        }
        
        cue_text = cue_dependent_retrieval._encode_cues(workspace_state)
        assert "goal 1" in cue_text
        assert "goal 2" in cue_text
    
    def test_recency_weight_with_invalid_timestamp(self, cue_dependent_retrieval):
        """Test recency weight with invalid timestamp."""
        metadata = {"last_accessed": "invalid-timestamp"}
        weight = cue_dependent_retrieval._recency_weight(metadata)
        assert weight == 0.3  # Should return default
    
    def test_recency_weight_with_future_timestamp(self, cue_dependent_retrieval):
        """Test recency weight with future timestamp."""
        future = (datetime.now() + timedelta(hours=1)).isoformat()
        metadata = {"last_accessed": future}
        weight = cue_dependent_retrieval._recency_weight(metadata)
        # Should handle gracefully (future timestamp gives weight > 1.0)
        assert 0.0 <= weight <= 2.0  # Allow for slight time variance
    
    def test_competitive_retrieval_all_below_threshold(self, cue_dependent_retrieval):
        """Test competitive retrieval when all activations below threshold."""
        activations = {
            "mem1": 0.2,
            "mem2": 0.1,
            "mem3": 0.05
        }
        
        candidates = {
            "mem1": {"data": {"content": "test1"}, "similarity": 0.5, "collection": "episodic"},
            "mem2": {"data": {"content": "test2"}, "similarity": 0.4, "collection": "episodic"},
            "mem3": {"data": {"content": "test3"}, "similarity": 0.3, "collection": "episodic"}
        }
        
        retrieved = cue_dependent_retrieval._competitive_retrieval(
            activations,
            candidates,
            limit=5
        )
        
        # All below threshold (0.3), so nothing should be retrieved
        assert len(retrieved) == 0
    
    def test_strengthen_retrieved_with_invalid_memory(self, cue_dependent_retrieval, mock_storage):
        """Test strengthening with invalid memory data."""
        memories = [
            {"content": "test", "collection": "episodic"},  # Missing memory_id
            {"memory_id": "mem1"}  # Missing collection
        ]
        
        # Should handle gracefully without crashing
        cue_dependent_retrieval._strengthen_retrieved(memories)
        # No assertion needed - just verify it doesn't crash
    
    def test_spread_activation_with_no_associations(self, cue_dependent_retrieval, mock_storage):
        """Test spreading activation when no associations exist."""
        initial_activations = {
            "mem1": 0.8,
            "mem2": 0.6
        }
        
        mock_storage.get_memory_associations.return_value = []
        
        spread = cue_dependent_retrieval._spread_activation(
            initial_activations,
            spread_factor=0.3,
            iterations=2
        )
        
        # Without associations, activations should remain the same
        assert spread["mem1"] == 0.8
        assert spread["mem2"] == 0.6
    
    def test_get_candidates_with_empty_results(self, cue_dependent_retrieval, mock_storage):
        """Test candidate retrieval with empty results."""
        mock_storage.episodic_memory.count.return_value = 0
        mock_storage.semantic_memory.count.return_value = 0
        
        candidates = cue_dependent_retrieval._get_candidates("test query")
        
        assert len(candidates) == 0
    
    def test_get_candidates_with_malformed_json(self, cue_dependent_retrieval, mock_storage):
        """Test candidate retrieval with malformed JSON."""
        mock_storage.episodic_memory.count.return_value = 2
        mock_storage.query_episodic.return_value = {
            "ids": [["mem1", "mem2"]],
            "documents": [["invalid json {", json.dumps({"content": "valid"})]],
            "metadatas": [[{}, {}]],
            "distances": [[0.2, 0.3]]
        }
        
        mock_storage.semantic_memory.count.return_value = 0
        
        candidates = cue_dependent_retrieval._get_candidates("test query")
        
        # Should skip invalid JSON and return only valid entry
        assert len(candidates) == 1
        assert "mem2" in candidates
