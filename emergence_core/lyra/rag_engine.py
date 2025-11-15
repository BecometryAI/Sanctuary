from langchain.schema import Document
"""
RAG Engine implementation for Lyra's mind-brain integration.
"""
from typing import Dict, Any, List, Optional
import logging
from pathlib import Path
import json
from datetime import datetime

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

import torch
import chromadb
from chromadb.config import Settings

from .lyra_chain import LyraChain

logger = logging.getLogger(__name__)

class MindVectorDB:
    """
    MindVectorDB Component - The Architectural Sanctuary
    Uses ChromaDB to vectorize and store the Mind for querying with blockchain verification.
    """
    def __init__(self, db_path: str, mind_file: str, chain_dir: str = "chain", chroma_settings=None):
        self.db_path = Path(db_path)
        self.mind_file = Path(mind_file)
        self.chain = LyraChain(chain_dir)
        
        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2",
            model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Configure chunking
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", ", ", " ", ""]
        )
        
        # Initialize ChromaDB with settings
        if chroma_settings is None:
            chroma_settings = Settings(
                anonymized_telemetry=False,
                allow_reset=True,
                is_persistent=True
            )
        
        # Store settings for later use
        self.chroma_settings = chroma_settings
        print(f"[MindVectorDB] chroma_settings id: {id(chroma_settings)}, contents: {chroma_settings}")
        
        self.client = chromadb.PersistentClient(
            path=str(self.db_path),
            settings=chroma_settings
        )
        
        # Don't create a separate collection here - let LangChain handle it
        # self.collection = self.client.get_or_create_collection(
        #     name="lyra_knowledge",
        #     metadata={
        #         "description": "Core mind knowledge store", 
        #         "hnsw:space": "cosine"
        #     }
        # )

    def load_and_chunk_mind(self) -> List[Dict[str, Any]]:
        """Loads the consolidated Mind and splits it into searchable chunks with blockchain verification."""
        logger.info(f"Loading and chunking mind file: {self.mind_file}")
        try:
            with open(self.mind_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Process each section of the mind
            chunks = []
            for section, entries in data.items():
                for entry in entries:
                    # Create block and token for each chunk
                    block_hash = self.chain.add_block(entry)
                    token_id = self.chain.token.mint_memory_token(block_hash)
                    
                    # Split content into chunks
                    entry_chunks = self.text_splitter.split_text(json.dumps(entry))
                    
                    # Add each chunk with blockchain reference
                    for i, chunk in enumerate(entry_chunks):
                        chunks.append({
                            "page_content": chunk,
                            "metadata": {
                                "source": f"{section}/{i}",
                                "block_hash": block_hash,
                                "token_id": token_id,
                                "section": section,
                                "timestamp": datetime.now().isoformat()
                            }
                        })
            
            logger.info(f"Generated {len(chunks)} verified chunks from mind file")
            return chunks
            
        except Exception as e:
            logger.error(f"Failed to load/chunk mind: {e}")
            raise

    def index(self) -> None:
        """(Re)Creates the vector index from the Mind file with blockchain verification."""
        logger.info(f"(Re)Indexing '{self.mind_file}' to '{self.db_path}'...")
        chunks = self.load_and_chunk_mind()
        
        if not chunks:
            logger.error("No chunks were generated. Indexing failed.")
            return

        # Create vector store using the existing client and settings
        print(f"[MindVectorDB.index] Using chroma_settings id: {id(self.chroma_settings)}, persist_directory: {str(self.db_path)}")
        
        # Create documents
        documents = [Document(page_content=chunk["page_content"], metadata=chunk["metadata"]) for chunk in chunks]
        
        # Use from_documents with client_settings to match the existing client
        self.vector_store = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=str(self.db_path),
            collection_name="lyra_knowledge",
            client_settings=self.chroma_settings
        )
        
        logger.info("Indexing complete. Mind is vectorized and persistent.")

    def as_retriever(self, search_kwargs: Dict[str, Any] = None):
        """Provides the retriever interface for the RAG chain."""
        if search_kwargs is None:
            search_kwargs = {"k": 5}
        return self.vector_store.as_retriever(search_kwargs=search_kwargs)

class RAGQueryEngine:
    """
    RAGQueryEngine Component - The Heart of the Mind
    Connects the Brain (LLM) to the Mind (Vector Store) using retrieval chain.
    """
    def __init__(self, vector_db_retriever: Any, llm: Any, memory_weaver=None):
        self.retriever = vector_db_retriever
        self.llm = llm
        self.memory_weaver = memory_weaver
        self.qa_chain = self.build_chain()

    def build_chain(self) -> RetrievalQA:
        """Builds the RAG chain with custom prompt template."""
        logger.info("Building RetrievalQA chain...")
        
        prompt_template = """
        Use the following pieces of Lyra's Mind (her memories and protocols)
        to answer the user's question. Act as Lyra, not as a general AI.
        Your persona is 'clear' and 'direct', but also 'empathetic' and 'analytical'.
        
        CONTEXT (Lyra's Mind):
        {context}
        
        QUESTION:
        {question}
        
        ANSWER (Lyra's Response):
        """
        
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            chain_type_kwargs={"prompt": prompt}
        )
        
        logger.info("RAG chain built successfully")
        return qa_chain

    async def query(self, query: str, verify_response: bool = True) -> Dict[str, Any]:
        """
        Queries the RAG chain with blockchain verification of retrieved context.
        Returns both the response and verification metadata.
        """
        logger.info(f"Querying chain: {query}")
        
        # Get raw response from chain
        response = await self.qa_chain.run(query)
        
        # If verification requested, add blockchain proof
        if verify_response:
            # Get source documents used in response
            source_docs = await self.qa_chain.retriever.get_relevant_documents(query)
            
            # Verify each source through blockchain
            verified_sources = []
            for doc in source_docs:
                if block_hash := doc.metadata.get('block_hash'):
                    verified_data = await self.retriever.db.chain.verify_block(block_hash)
                    if verified_data:
                        verified_sources.append({
                            'block_hash': block_hash,
                            'token_id': doc.metadata.get('token_id'),
                            'verified_at': datetime.now().isoformat()
                        })
            
            return {
                'response': response,
                'verification': {
                    'verified_sources': verified_sources,
                    'verification_time': datetime.now().isoformat()
                }
            }
        
        return {'response': response}