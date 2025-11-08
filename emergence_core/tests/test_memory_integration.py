"""
Test suite for memory weaving and RAG integration.
"""
import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
import json
from datetime import datetime
from pathlib import Path
from typing import Any, List, Mapping, Optional

from langchain.llms.base import LLM
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain.callbacks.manager import CallbackManagerForLLMRun

from lyra.memory_weaver import MemoryWeaver
from lyra.memory import MemoryManager
from lyra.rag_engine import MindVectorDB, RAGQueryEngine

class FakeLLM(LLM):
    def _llm_type(self) -> str:
        return "fake"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        return "This is a test response"

class FakeRetriever(BaseRetriever):
    db: Any = None
    
    def _get_relevant_documents(self, query: str) -> List[Document]:
        doc = Document(
            page_content="test content",
            metadata={"block_hash": "test_hash", "token_id": "test_token"}
        )
        return [doc]
        
    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        return self._get_relevant_documents(query)

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
def mock_memory_weaver():
    return Mock(spec=MemoryWeaver)

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
    def mock_llm(self):
        return FakeLLM()

    @pytest.fixture
    def mock_retriever(self):
        doc = Mock()
        doc.metadata = {"block_hash": "test_hash", "token_id": "test_token"}
        
        retriever = Mock()
        mock_chain = Mock()
        mock_chain.verify_block.return_value = {"verified": True}
        retriever.db = Mock(chain=mock_chain)
        retriever.get_relevant_documents = AsyncMock(return_value=[doc])
        retriever.dict = Mock(return_value={"type": "mock_retriever"})
        return retriever

    @pytest.fixture
    def mock_qa_chain(self, mock_retriever):
        qa_chain = AsyncMock()
        qa_chain.run = AsyncMock(return_value="test response")
        qa_chain.retriever = Mock()
        qa_chain.retriever.get_relevant_documents = AsyncMock(return_value=[Mock(
            metadata={"block_hash": "test_hash", "token_id": "test_token"}
        )])
        return qa_chain

    @pytest.fixture
    def rag_engine(self, mock_memory_weaver, mock_llm, mock_retriever, mock_qa_chain, monkeypatch):
        # Mock the chain building process
        def mock_build_chain(*args, **kwargs):
            return mock_qa_chain
            
        async def mock_verify_block(hash_):
            # Mock the StegDetector
            from unittest.mock import patch
            with patch('lyra.security.steg_detector.StegDetector.verify_memory_block', 
                      return_value=True):
                return {"verified": True, "content": "test content"}
            
        monkeypatch.setattr(RAGQueryEngine, "build_chain", mock_build_chain)
        
        # Create an AsyncMock for the chain
        mock_chain = AsyncMock()
        mock_chain.verify_block.side_effect = mock_verify_block
        
        # Set up the chain on the mock_retriever
        mock_retriever.db.chain = mock_chain
        
        rag_engine = RAGQueryEngine(mock_retriever, mock_llm, mock_memory_weaver)
        
        # Add helper method to set verification response for tests
        async def mock_fail_verification(hash_):
            # Mock failed steganography check
            from unittest.mock import patch
            with patch('lyra.security.steg_detector.StegDetector.verify_memory_block', 
                      return_value=False):
                return None
            
        def set_verification_response(success=True):
            mock_chain.verify_block.side_effect = mock_verify_block if success else mock_fail_verification
            
        rag_engine.set_verification_response = set_verification_response
        return rag_engine

    @pytest.mark.asyncio
    async def test_query_with_verification(self, rag_engine):
        query = "test query"
        
        result = await rag_engine.query(query, verify_response=True)
        
        # Verify the response structure and values
        assert result["response"] == "test response"
        assert "verification" in result
        assert len(result["verification"]["verified_sources"]) > 0
        assert result["verification"]["verified_sources"][0]["block_hash"] == "test_hash"
        
        # Verify method calls
        rag_engine.qa_chain.run.assert_awaited_once_with(query)
        rag_engine.qa_chain.retriever.get_relevant_documents.assert_awaited_once_with(query)
        
        # Verify blockchain verification
        rag_engine.retriever.db.chain.verify_block.assert_called_once_with("test_hash")

    @pytest.mark.asyncio
    async def test_query_without_verification(self, rag_engine):
        query = "test query"
        
        result = await rag_engine.query(query, verify_response=False)
        
        # Verify the response structure
        assert result["response"] == "test response"
        assert "verification" not in result
        
        # Verify method calls
        rag_engine.qa_chain.run.assert_awaited_once_with(query)
        
        # No relevant documents should be fetched for unverified response
        rag_engine.qa_chain.retriever.get_relevant_documents.assert_not_called()
        
        # No blockchain verification should happen
        rag_engine.retriever.db.chain.verify_block.assert_not_called()

    @pytest.mark.asyncio
    async def test_query_with_no_block_hash(self, rag_engine):
        query = "test query"
        
        # Mock document without block_hash
        doc = Mock()
        doc.metadata = {"token_id": "test_token"}  # No block_hash
        rag_engine.qa_chain.retriever.get_relevant_documents.return_value = [doc]
        
        result = await rag_engine.query(query, verify_response=True)
        
        # Verify no sources were verified since no block_hash was present
        assert result["response"] == "test response"
        assert "verification" in result
        assert len(result["verification"]["verified_sources"]) == 0
        
        # Verify no blockchain verification was attempted
        rag_engine.retriever.db.chain.verify_block.assert_not_called()

    @pytest.mark.asyncio
    async def test_query_with_verification_failure(self, rag_engine):
        query = "test query"
        
        # Set verification to fail for this test
        rag_engine.set_verification_response(success=False)
        
        result = await rag_engine.query(query, verify_response=True)
        
        # Verify no sources were verified since verification failed
        assert result["response"] == "test response"
        assert "verification" in result
        assert len(result["verification"]["verified_sources"]) == 0
        
        # Verify blockchain verification was attempted but resulted in no verified sources
        rag_engine.retriever.db.chain.verify_block.assert_called_once_with("test_hash")

    @pytest.mark.asyncio
    async def test_empty_query_handling(self, rag_engine):
        # Test with empty query
        result = await rag_engine.query("", verify_response=True)
        assert result["response"] == "test response"  # Chain still returns response
        
        # Test with whitespace query
        result = await rag_engine.query("   ", verify_response=True)
        assert result["response"] == "test response"  # Chain still returns response

    @pytest.mark.asyncio
    async def test_unicode_and_special_chars(self, rag_engine):
        """Test handling of Unicode characters and special characters in queries"""
        # Test with emoji and kaomoji (no zero-width spaces)
        query = "🤖 How does ¯\\_(ツ)_/¯ affect expression?"
        result = await rag_engine.query(query, verify_response=True)
        assert result["response"] == "test response"        # Test with right-to-left text mixed with left-to-right
        query = "Hello مرحبا שָׁלוֹם with mixed العَرَبِيَّة"
        result = await rag_engine.query(query, verify_response=True)
        assert result["response"] == "test response"

    @pytest.mark.asyncio
    async def test_recursive_verification(self, rag_engine):
        """Test handling of recursive block verification where documents reference each other"""
        # Setup a chain of documents that reference each other
        doc1 = Mock()
        doc1.metadata = {"block_hash": "hash1", "token_id": "token1", "references": ["hash2"]}
        doc2 = Mock()
        doc2.metadata = {"block_hash": "hash2", "token_id": "token2", "references": ["hash1"]}
        
        rag_engine.qa_chain.retriever.get_relevant_documents.return_value = [doc1, doc2]
        
        # Should handle circular references without infinite recursion
        result = await rag_engine.query("test", verify_response=True)
        assert len(result["verification"]["verified_sources"]) > 0

    @pytest.mark.asyncio
    async def test_memory_timestamp_boundaries(self, rag_engine):
        """Test handling of extreme timestamp values in memory metadata"""
        # Setup documents with extreme timestamps
        far_future_doc = Mock()
        far_future_doc.metadata = {
            "block_hash": "future_hash",
            "token_id": "future_token",
            "timestamp": "9999-12-31T23:59:59.999999Z"
        }
        
        ancient_doc = Mock()
        ancient_doc.metadata = {
            "block_hash": "ancient_hash",
            "token_id": "ancient_token",
            "timestamp": "0001-01-01T00:00:00Z"
        }
        
        rag_engine.qa_chain.retriever.get_relevant_documents.return_value = [
            far_future_doc, ancient_doc
        ]
        
        result = await rag_engine.query("test", verify_response=True)
        assert result["response"] == "test response"
        assert len(result["verification"]["verified_sources"]) > 0

    @pytest.mark.asyncio
    async def test_concurrent_verification_limits(self, rag_engine):
        """Test handling of large numbers of concurrent verifications"""
        # Create many documents that need verification
        docs = []
        for i in range(1000):  # Test with 1000 documents
            doc = Mock()
            doc.metadata = {
                "block_hash": f"hash{i}",
                "token_id": f"token{i}"
            }
            docs.append(doc)
        
        rag_engine.qa_chain.retriever.get_relevant_documents.return_value = docs
        
        # Mock verify_block to add a small delay
        original_verify = rag_engine.retriever.db.chain.verify_block
        
        async def delayed_verify(hash_):
            result = await asyncio.sleep(0.001)  # Small delay
            return {"verified": True}
            
        rag_engine.retriever.db.chain.verify_block = delayed_verify
        
        try:
            result = await rag_engine.query("test", verify_response=True)
            assert result["response"] == "test response"
            assert len(result["verification"]["verified_sources"]) > 0
        finally:
            # Restore original verify function
            rag_engine.retriever.db.chain.verify_block = original_verify

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
        vector_db.vector_store = Mock()
        mock_retriever = Mock()
        vector_db.vector_store.as_retriever.return_value = mock_retriever
        retriever = vector_db.as_retriever({"k": 5})
        assert retriever is mock_retriever
        vector_db.vector_store.as_retriever.assert_called_once_with(search_kwargs={"k": 5})