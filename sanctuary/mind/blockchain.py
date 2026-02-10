"""
Blockchain integration module for Sanctuary's secure data storage and protocol enforcement
"""
import json
import logging
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
from eth_account import Account
from web3 import AsyncWeb3
from web3.providers import AsyncHTTPProvider
import aioipfs

logger = logging.getLogger(__name__)

class SanctuaryBlockchain:
    """Manages blockchain integration for Sanctuary's core data integrity"""
    
    async def __init__(self, config_path: str):
        """Initialize blockchain connection and IPFS client"""
        self.config = self._load_config(config_path)
        
        # Initialize AsyncWeb3
        self.w3 = AsyncWeb3(AsyncHTTPProvider(self.config['ethereum']['rpc_url']))
        
        # Initialize account
        self.account = Account.from_key(self.config['ethereum']['private_key'])
        
        # Initialize async IPFS client
        self.ipfs = aioipfs.AsyncIPFS(self.config['ipfs']['api_url'])
        
        # Load smart contract ABIs
        self.contract_abis = self._load_contract_abis()
        
        # Initialize contracts
        self.memory_contract = self.w3.eth.contract(
            address=self.config['contracts']['memory_contract'],
            abi=self.contract_abis['memory_contract']
        )
        
        self.protocol_contract = self.w3.eth.contract(
            address=self.config['contracts']['protocol_contract'],
            abi=self.contract_abis['protocol_contract']
        )
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load blockchain configuration"""
        with open(config_path) as f:
            return json.load(f)
            
    def _load_contract_abis(self) -> Dict[str, Any]:
        """Load smart contract ABIs"""
        contract_dir = Path(__file__).parent / 'contracts'
        abis = {}
        
        for abi_file in contract_dir.glob('*.abi.json'):
            with open(abi_file) as f:
                abis[abi_file.stem.replace('.abi', '')] = json.load(f)
                
        return abis
        
    async def store_memory(self, memory_data: Dict[str, Any]) -> str:
        """Store memory data with blockchain verification
        
        Args:
            memory_data: Memory data to store
            
        Returns:
            str: IPFS hash of stored data
        """
        try:
            # Store data in IPFS
            ipfs_hash = await self.ipfs.add(json.dumps(memory_data))
            
            # Create memory record on blockchain
            tx_hash = self.memory_contract.functions.storeMemory(
                ipfs_hash,
                memory_data['timestamp'],
                memory_data['type'],
                memory_data.get('tags', [])
            ).transact({'from': self.account.address})
            
            # Wait for transaction confirmation
            self.w3.eth.wait_for_transaction_receipt(tx_hash)
            
            logger.info(f"Memory stored with IPFS hash: {ipfs_hash}")
            return ipfs_hash
            
        except Exception as e:
            logger.error(f"Failed to store memory: {e}")
            raise
            
    async def verify_protocol(self, protocol_name: str, action_data: Dict[str, Any]) -> bool:
        """Verify protocol execution against smart contract rules
        
        Args:
            protocol_name: Name of the protocol to verify
            action_data: Data about the action being verified
            
        Returns:
            bool: True if action is valid according to protocol
        """
        try:
            # Check protocol rules on blockchain
            is_valid = await self.protocol_contract.functions.verifyProtocol(
                protocol_name,
                json.dumps(action_data)
            ).call()
            
            if not is_valid:
                logger.warning(f"Protocol verification failed for {protocol_name}")
                
            return is_valid
            
        except Exception as e:
            logger.error(f"Protocol verification error: {e}")
            return False
            
    async def retrieve_memory(self, ipfs_hash: str) -> Optional[Dict[str, Any]]:
        """Retrieve memory data from IPFS with blockchain verification
        
        Args:
            ipfs_hash: IPFS hash of the memory to retrieve
            
        Returns:
            Optional[Dict[str, Any]]: Retrieved memory data if verified
        """
        try:
            # Verify memory exists on blockchain
            exists = await self.memory_contract.functions.verifyMemory(ipfs_hash).call()
            
            if not exists:
                logger.warning(f"Memory {ipfs_hash} not found on blockchain")
                return None
                
            # Retrieve from IPFS
            data = await self.ipfs.cat(ipfs_hash)
            return json.loads(data)
            
        except Exception as e:
            logger.error(f"Failed to retrieve memory: {e}")
            return None
            
    async def log_action(self, action_type: str, action_data: Dict[str, Any]) -> str:
        """Log an action to the blockchain for audit purposes
        
        Args:
            action_type: Type of action being logged
            action_data: Data about the action
            
        Returns:
            str: Transaction hash
        """
        try:
            tx_hash = self.protocol_contract.functions.logAction(
                action_type,
                json.dumps(action_data),
                int(datetime.now().timestamp())
            ).transact({'from': self.account.address})
            
            # Wait for transaction confirmation
            self.w3.eth.wait_for_transaction_receipt(tx_hash)
            
            logger.info(f"Action logged with transaction: {tx_hash.hex()}")
            return tx_hash.hex()
            
        except Exception as e:
            logger.error(f"Failed to log action: {e}")
            raise