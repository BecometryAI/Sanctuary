"""
ChromaDB Transaction Wrapper

Provides atomic operations for ChromaDB with rollback support.
Ensures data consistency by treating multiple operations as a single unit.

Author: Sanctuary Team
Date: January 2, 2026
"""

import logging
from typing import Optional, Any, List, Dict
from contextlib import contextmanager
import copy
import chromadb

from ..exceptions import MemoryError
from ..logging_config import get_logger, OperationContext

logger = get_logger(__name__)


class ChromaDBTransaction:
    """
    Context manager for atomic ChromaDB operations.
    
    Implements a simple checkpoint/rollback mechanism for ChromaDB operations.
    Note: ChromaDB doesn't have native transactions, so this implements
    a best-effort approach with state tracking.
    
    Example:
        async with ChromaDBTransaction(collection) as txn:
            txn.add(ids=["1"], documents=["doc1"])
            txn.add(ids=["2"], documents=["doc2"])
            # Both adds commit together or rollback on error
    """
    
    def __init__(self, collection: chromadb.Collection):
        """
        Initialize transaction for a ChromaDB collection.
        
        Args:
            collection: ChromaDB collection instance
        """
        self.collection = collection
        self.operations: List[Dict[str, Any]] = []
        self.committed = False
        self._checkpoint_ids: Optional[List[str]] = None
    
    def __enter__(self):
        """Enter transaction context."""
        # Create checkpoint by recording current IDs
        # Note: For large collections, this could be expensive. Consider implementing
        # operation-only tracking or sampling for production use with millions of documents.
        try:
            with OperationContext(operation="transaction_start", collection=self.collection.name):
                result = self.collection.get()
                self._checkpoint_ids = set(result.get('ids', []))
                logger.debug(
                    f"Transaction started for collection '{self.collection.name}' "
                    f"({len(self._checkpoint_ids)} existing documents)"
                )
        except Exception as e:
            logger.error(f"Failed to create transaction checkpoint: {e}")
            raise MemoryError(
                "Failed to start transaction",
                operation="transaction_start",
                context={"collection": self.collection.name, "error": str(e)}
            )
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit transaction context with commit or rollback."""
        if exc_type is not None:
            # Exception occurred - attempt rollback
            logger.warning(
                f"Transaction failed for collection '{self.collection.name}': {exc_val}"
            )
            self._rollback()
            return False
        
        # No exception - commit is already done via direct operations
        logger.debug(f"Transaction committed for collection '{self.collection.name}'")
        self.committed = True
        return False
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self.__enter__()
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        return self.__exit__(exc_type, exc_val, exc_tb)
    
    def add(self, ids: List[str], documents: Optional[List[str]] = None, 
            embeddings: Optional[List] = None, metadatas: Optional[List[Dict]] = None):
        """
        Add documents to collection within transaction.
        
        Args:
            ids: Document IDs
            documents: Document texts
            embeddings: Document embeddings
            metadatas: Document metadata
        """
        try:
            self.collection.add(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas
            )
            self.operations.append({
                "type": "add",
                "ids": ids
            })
            logger.debug(f"Added {len(ids)} documents in transaction")
        except Exception as e:
            logger.error(f"Failed to add documents in transaction: {e}")
            raise MemoryError(
                "Failed to add documents",
                operation="transaction_add",
                context={"ids": ids, "error": str(e)}
            )
    
    def update(self, ids: List[str], documents: Optional[List[str]] = None,
               embeddings: Optional[List] = None, metadatas: Optional[List[Dict]] = None):
        """
        Update documents in collection within transaction.
        
        Args:
            ids: Document IDs to update
            documents: Updated document texts
            embeddings: Updated embeddings
            metadatas: Updated metadata
        """
        try:
            self.collection.update(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas
            )
            self.operations.append({
                "type": "update",
                "ids": ids
            })
            logger.debug(f"Updated {len(ids)} documents in transaction")
        except Exception as e:
            logger.error(f"Failed to update documents in transaction: {e}")
            raise MemoryError(
                "Failed to update documents",
                operation="transaction_update",
                context={"ids": ids, "error": str(e)}
            )
    
    def delete(self, ids: List[str]):
        """
        Delete documents from collection within transaction.
        
        Args:
            ids: Document IDs to delete
        """
        try:
            self.collection.delete(ids=ids)
            self.operations.append({
                "type": "delete",
                "ids": ids
            })
            logger.debug(f"Deleted {len(ids)} documents in transaction")
        except Exception as e:
            logger.error(f"Failed to delete documents in transaction: {e}")
            raise MemoryError(
                "Failed to delete documents",
                operation="transaction_delete",
                context={"ids": ids, "error": str(e)}
            )
    
    def _rollback(self):
        """
        Attempt to rollback transaction changes.
        
        Note: This is best-effort since ChromaDB doesn't have native transactions.
        We delete any IDs that were added after the checkpoint.
        """
        try:
            with OperationContext(operation="transaction_rollback", collection=self.collection.name):
                # Get current IDs
                result = self.collection.get()
                current_ids = set(result.get('ids', []))
                
                # Find IDs added during transaction
                new_ids = current_ids - self._checkpoint_ids
                
                if new_ids:
                    # Delete newly added IDs
                    self.collection.delete(ids=list(new_ids))
                    logger.info(
                        f"Rolled back transaction: deleted {len(new_ids)} documents "
                        f"from collection '{self.collection.name}'"
                    )
                else:
                    logger.debug("No rollback needed - no new documents added")
        
        except Exception as e:
            logger.error(
                f"Failed to rollback transaction for collection '{self.collection.name}': {e}"
            )
            # Rollback failure is serious but we can't do much about it
            raise MemoryError(
                "Failed to rollback transaction",
                operation="transaction_rollback",
                context={
                    "collection": self.collection.name,
                    "error": str(e)
                },
                recoverable=False
            )


@contextmanager
def chroma_transaction(collection: chromadb.Collection):
    """
    Convenience context manager for ChromaDB transactions.
    
    Args:
        collection: ChromaDB collection
    
    Example:
        with chroma_transaction(collection) as txn:
            txn.add(ids=["1"], documents=["doc"])
    """
    txn = ChromaDBTransaction(collection)
    with txn:
        yield txn
