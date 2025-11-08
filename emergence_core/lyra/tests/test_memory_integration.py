"""
Test suite for memory weaving and RAG integration.
"""
import pytest
from unittest.mock import Mock, patch
import json
from datetime import datetime
from pathlib import Path

from ..memory_weaver import MemoryWeaver
from ..memory import MemoryManager
from ..rag_engine import MindVectorDB, RAGQueryEngine

@pytest.fixture
def mock_memory_manager():
    return Mock(spec=MemoryManager)

@pytest.fixture
def mock_vector_db():
    return Mock(spec=MindVectorDB)

@pytest.fixture
def mock_rag_engine():
    return Mock(spec=RAGQueryEngine)

@pytest.fixture
def memory_weaver(mock_memory_manager, mock_vector_db, mock_rag_engine):
    return MemoryWeaver(mock_memory_manager, mock_vector_db, mock_rag_engine)

class TestMemoryWeaver:
    @pytest.mark.asyncio
    async def test_process_memory_with_json(self, memory_weaver):
        test_response = '''Some text here
        {"memory": "test memory", "type": "test"}
        more text'''
        
        result = await memory_weaver.process_memory(test_response)
        assert result == True
        memory_weaver.memory_manager.store_experience.assert_called_once()
        memory_weaver.vector_db.index.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_memory_with_text(self, memory_weaver):
        test_response = '''Some text here
        Memory: This is a test memory
        more text'''
        
        result = await memory_weaver.process_memory(test_response)
        assert result == True
        
        # Verify correct memory format
        call_args = memory_weaver.memory_manager.store_experience.call_args[0][0]
        assert "content" in call_args
        assert call_args["type"] == "extracted_memory"

    @pytest.mark.asyncio
    async def test_find_similar_memories(self, memory_weaver):
        test_memory = {
            "content": "test memory content",
            "type": "test"
        }
        
        similar_memories = [
            {"content": "similar1", "block_hash": "hash1"},
            {"content": "similar2", "block_hash": "hash2"}
        ]
        
        memory_weaver.memory_manager.retrieve_relevant_memories.return_value = similar_memories
        result = await memory_weaver.find_similar_memories(test_memory)
        
        assert len(result) == 2
        memory_weaver.memory_manager.retrieve_relevant_memories.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_interaction(self, memory_weaver):
        query = "test query"
        response = "test response"
        context = {"some": "context"}
        
        result = await memory_weaver.process_interaction(query, response, context)
        
        assert result["status"] == "success"
        assert "memory_processed" in result
        assert "interaction_stored" in result
        assert "timestamp" in result

class TestRAGQueryEngine:
    @pytest.fixture
    def rag_engine(self, mock_memory_weaver):
        retriever = Mock()
        llm = Mock()
        return RAGQueryEngine(retriever, llm, mock_memory_weaver)

    @pytest.mark.asyncio
    async def test_query_with_verification(self, rag_engine):
        query = "test query"
        response = "test response"
        
        # Mock chain run
        rag_engine.qa_chain.run.return_value = response
        
        # Mock document retrieval
        doc = Mock()
        doc.metadata = {"block_hash": "test_hash", "token_id": "test_token"}
        rag_engine.qa_chain.retriever.get_relevant_documents.return_value = [doc]
        
        # Mock blockchain verification
        rag_engine.retriever.db.chain.verify_block.return_value = {"verified": True}
        
        result = await rag_engine.query(query, verify_response=True)
        
        assert result["response"] == response
        assert "verification" in result
        assert len(result["verification"]["verified_sources"]) == 1
        assert "memory_processing" in result

    @pytest.mark.asyncio
    async def test_query_without_verification(self, rag_engine):
        query = "test query"
        response = "test response"
        
        rag_engine.qa_chain.run.return_value = response
        
        result = await rag_engine.query(query, verify_response=False)
        
        assert result["response"] == response
        assert "verification" not in result
        assert "memory_processing" in result

class TestMindVectorDB:
    @pytest.fixture
    def vector_db(self, tmp_path):
        db_path = tmp_path / "vector_store"
        chain_dir = tmp_path / "chain"
        return MindVectorDB(str(db_path), "test_mind.json", str(chain_dir))

    def test_initialization(self, vector_db):
        assert vector_db.embeddings is not None
        assert vector_db.text_splitter is not None
        assert vector_db.collection is not None

    def test_as_retriever(self, vector_db):
        retriever = vector_db.as_retriever({"k": 5})
        assert retriever is not None