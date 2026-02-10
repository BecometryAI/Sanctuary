// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

contract SanctuaryMemory {
    struct Memory {
        string ipfsHash;
        uint256 timestamp;
        string memoryType;
        string[] tags;
        bool exists;
    }
    
    mapping(string => Memory) public memories;
    address public owner;
    
    event MemoryStored(string ipfsHash, uint256 timestamp, string memoryType);
    event MemoryVerified(string ipfsHash, bool exists);
    
    constructor() {
        owner = msg.sender;
    }
    
    modifier onlyOwner() {
        require(msg.sender == owner, "Only owner can perform this action");
        _;
    }
    
    function storeMemory(
        string calldata ipfsHash,
        uint256 timestamp,
        string calldata memoryType,
        string[] calldata tags
    ) external onlyOwner {
        require(bytes(ipfsHash).length > 0, "IPFS hash cannot be empty");
        
        memories[ipfsHash] = Memory({
            ipfsHash: ipfsHash,
            timestamp: timestamp,
            memoryType: memoryType,
            tags: tags,
            exists: true
        });
        
        emit MemoryStored(ipfsHash, timestamp, memoryType);
    }
    
    function verifyMemory(string calldata ipfsHash) external view returns (bool) {
        bool exists = memories[ipfsHash].exists;
        return exists;
    }
    
    function getMemoryDetails(string calldata ipfsHash)
        external
        view
        returns (Memory memory)
    {
        require(memories[ipfsHash].exists, "Memory does not exist");
        return memories[ipfsHash];
    }
}