from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import DirectoryLoader, JSONLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.schema import Document
from typing import List, Dict, Any
import chromadb
import torch
import json
import os

class SanctuaryLibrarian:
    def __init__(self, base_dir: str, persist_dir: str):
        """Initialize the Sanctuary Librarian with paths and configurations.
        
        Args:
            base_dir: Root directory containing all Sanctuary's files
            persist_dir: Directory where ChromaDB will persist the vector store
        """
        self.base_dir = base_dir
        self.persist_dir = persist_dir
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2",
            model_kwargs=({'device': 'cuda'} if torch.cuda.is_available() else {'device': 'cpu'}),
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Configure text splitter for different document types
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", ", ", " ", ""]
        )
        
    def json_metadata_func(self) -> Any:
        """Extract and enhance metadata from JSON documents."""
        def metadata_func(record: Dict[str, Any], metadata: Dict[str, Any]) -> Dict[str, Any]:
            # Add file path and type to metadata
            metadata["source_type"] = "json"
            
            # Extract key information based on file type
            if "journal_entry" in record:
                metadata["entry_type"] = "journal"
                metadata["timestamp"] = record.get("timestamp", "")
                metadata["labels"] = record.get("labels", [])
            elif "directives" in record:
                metadata["entry_type"] = "protocol"
                metadata["protocol_type"] = record.get("purpose", "")
            elif "ritual_type" in record:
                metadata["entry_type"] = "ritual"
                metadata["ritual_name"] = record.get("name", "")
            
            return metadata
        return metadata_func

    def load_documents(self) -> List[Document]:
        """Load all documents from the base directory."""
        documents = []
        
        # Configure JSON loader
        json_loader = DirectoryLoader(
            self.base_dir,
            glob="**/*.json",
            loader_cls=JSONLoader,
            loader_kwargs={
                "jq_schema": ".",
                "text_content": False,
                "metadata_func": self.json_metadata_func()
            }
        )
        
        try:
            json_docs = json_loader.load()
            documents.extend(json_docs)
            print(f"Loaded {len(json_docs)} JSON documents")
        except Exception as e:
            print(f"Error loading JSON documents: {e}")
        
        return documents

    def process_documents(self, documents: List[Document]) -> List[Document]:
        """Process and split documents into chunks."""
        try:
            split_docs = self.text_splitter.split_documents(documents)
            print(f"Split {len(documents)} documents into {len(split_docs)} chunks")
            return split_docs
        except Exception as e:
            print(f"Error splitting documents: {e}")
            return []

    def create_vector_store(self, documents: List[Document]) -> None:
        """Create and persist the vector store."""
        try:
            # Initialize Chroma client with persistence
            client = chromadb.PersistentClient(path=self.persist_dir)
            
            # Create collection
            collection_name = "sanctuary_knowledge"
            collection = client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            
            # Process documents into chunks
            ids = [str(i) for i in range(len(documents))]
            texts = [doc.page_content for doc in documents]
            metadatas = [doc.metadata for doc in documents]
            
            # Create embeddings
            embeddings = self.embeddings.embed_documents(texts)
            
            # Add documents to collection in batches
            batch_size = 500
            for i in range(0, len(documents), batch_size):
                end_idx = min(i + batch_size, len(documents))
                collection.add(
                    embeddings=embeddings[i:end_idx],
                    documents=texts[i:end_idx],
                    metadatas=metadatas[i:end_idx],
                    ids=ids[i:end_idx]
                )
            
            print(f"Successfully created and persisted vector store at {self.persist_dir}")
            print(f"Total vectors: {len(ids)}")
            
        except Exception as e:
            print(f"Error creating vector store: {e}")

    def build_index(self):
        """Main method to build the vector index."""
        print("Starting Sanctuary Librarian indexing process...")
        
        # Create persist directory if it doesn't exist
        os.makedirs(self.persist_dir, exist_ok=True)
        
        # Load documents
        documents = self.load_documents()
        if not documents:
            print("No documents found to process")
            return
        
        # Process documents
        processed_docs = self.process_documents(documents)
        if not processed_docs:
            print("No processed documents to index")
            return
        
        # Create vector store
        self.create_vector_store(processed_docs)
        print("Indexing process completed")