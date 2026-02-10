"""
Custom blockchain implementation for Sanctuary's secure data storage and protocol verification
"""
import hashlib
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)

class Block:
    def __init__(self, index: int, timestamp: float, data: Dict[str, Any], previous_hash: str):
        """Initialize a block in Sanctuary's chain"""
        self.index = index
        self.timestamp = timestamp
        self.data = data
        self.previous_hash = previous_hash
        self.nonce = 0
        self.hash = self.calculate_hash()

    def calculate_hash(self) -> str:
        """Calculate block hash using data, timestamp, and previous hash"""
        block_string = json.dumps({
            "index": self.index,
            "timestamp": self.timestamp,
            "data": self.data,
            "previous_hash": self.previous_hash,
            "nonce": self.nonce
        }, sort_keys=True)
        return hashlib.sha256(block_string.encode()).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        """Convert block to dictionary for storage"""
        return {
            "index": self.index,
            "timestamp": self.timestamp,
            "data": self.data,
            "previous_hash": self.previous_hash,
            "nonce": self.nonce,
            "hash": self.hash
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Block':
        """Create block from dictionary"""
        block = cls(
            data["index"],
            data["timestamp"],
            data["data"],
            data["previous_hash"]
        )
        block.nonce = data["nonce"]
        block.hash = data["hash"]
        return block

class SanctuaryToken:
    """Sanctuary's native token for tracking memory authenticity"""

    def __init__(self, name: str = "SanctuaryMemoryToken", symbol: str = "SMT"):
        self.name = name
        self.symbol = symbol
        self.total_supply = 0
        self.balances: Dict[str, int] = {}
        self.memory_tokens: Dict[str, int] = {}  # Maps memory hashes to token IDs
        
    def mint_memory_token(self, memory_hash: str) -> int:
        """Mint a new token for a memory entry"""
        token_id = self.total_supply + 1
        self.total_supply = token_id
        self.memory_tokens[memory_hash] = token_id
        return token_id
        
    def verify_memory_token(self, memory_hash: str) -> bool:
        """Verify if a memory has an associated token"""
        return memory_hash in self.memory_tokens

class SanctuaryChain:
    """Custom blockchain for Sanctuary's memory and protocol verification"""
    
    def __init__(self, chain_dir: str = "chain"):
        self.chain_dir = Path(chain_dir)
        self.chain_dir.mkdir(exist_ok=True)
        self.chain_file = self.chain_dir / "sanctuary_chain.json"
        self.token = SanctuaryToken()
        
        # Initialize or load chain
        if self.chain_file.exists():
            self.chain = self._load_chain()
        else:
            self.chain = [self._create_genesis_block()]
            self._save_chain()
            
    def _create_genesis_block(self) -> Block:
        """Create the genesis block with Sanctuary's creation data"""
        genesis_data = {
            "type": "genesis",
            "message": "Sanctuary Emergence Protocol - Genesis Block",
            "timestamp": datetime.now().isoformat()
        }
        return Block(0, time.time(), genesis_data, "0")
        
    def _save_chain(self):
        """Save blockchain to file"""
        chain_data = [block.to_dict() for block in self.chain]
        with open(self.chain_file, 'w') as f:
            json.dump(chain_data, f, indent=2)
            
    def _load_chain(self) -> List[Block]:
        """Load blockchain from file"""
        with open(self.chain_file) as f:
            chain_data = json.load(f)
        return [Block.from_dict(block_data) for block_data in chain_data]
        
    def add_block(self, data: Dict[str, Any]) -> str:
        """Add a new block to the chain"""
        previous_block = self.chain[-1]
        block = Block(
            previous_block.index + 1,
            time.time(),
            data,
            previous_block.hash
        )
        
        # Simple proof of work for integrity
        while not block.hash.startswith('0'):
            block.nonce += 1
            block.hash = block.calculate_hash()
            
        self.chain.append(block)
        self._save_chain()
        
        # Mint token for memory entries
        if data.get("type") == "memory":
            self.token.mint_memory_token(block.hash)
            
        return block.hash
        
    async def verify_block(self, block_hash: str) -> Optional[Dict[str, Any]]:
        """Verify and retrieve block data"""
        # Initialize steganography detector if needed
        if not hasattr(self, '_steg_detector'):
            from mind.security.steg_detector import StegDetector
            self._steg_detector = StegDetector()
            
        for block in self.chain:
            if block.hash == block_hash:
                # Verify block integrity
                if block.calculate_hash() != block.hash:
                    logger.warning(f"Block {block_hash} failed hash verification")
                    return None
                    
                # For memory entries, verify token
                if block.data.get("type") == "memory":
                    if not self.token.verify_memory_token(block.hash):
                        logger.warning(f"Block {block_hash} failed token verification")
                        return None
                        
                # Verify no steganographic content
                if not await self._steg_detector.verify_memory_block(block.data):
                    logger.warning(f"Block {block_hash} failed steganography verification")
                    return None
                        
                return block.data
                
        return None
        
    def verify_chain(self) -> bool:
        """Verify entire chain integrity"""
        for i in range(1, len(self.chain)):
            current = self.chain[i]
            previous = self.chain[i-1]
            
            # Verify hash connection
            if current.previous_hash != previous.hash:
                return False
                
            # Verify block hash
            if current.calculate_hash() != current.hash:
                return False
                
        return True
        
    def get_chain_info(self) -> Dict[str, Any]:
        """Get chain statistics and info"""
        return {
            "length": len(self.chain),
            "memory_tokens": self.token.total_supply,
            "last_block_time": datetime.fromtimestamp(self.chain[-1].timestamp).isoformat(),
            "is_valid": self.verify_chain()
        }