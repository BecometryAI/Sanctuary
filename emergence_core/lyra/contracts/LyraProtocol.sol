// SPDX-License-Identifier: MIT
pragma solidity ^0.8.19;

contract LyraProtocol {
    struct Protocol {
        string name;
        string rules;
        bool active;
    }
    
    struct Action {
        string actionType;
        string data;
        uint256 timestamp;
        address executor;
    }
    
    mapping(string => Protocol) public protocols;
    Action[] public actions;
    address public owner;
    
    event ProtocolVerified(string protocol, bool valid);
    event ActionLogged(string actionType, uint256 timestamp);
    
    constructor() {
        owner = msg.sender;
    }
    
    modifier onlyOwner() {
        require(msg.sender == owner, "Only owner can perform this action");
        _;
    }
    
    function registerProtocol(
        string calldata name,
        string calldata rules
    ) external onlyOwner {
        protocols[name] = Protocol({
            name: name,
            rules: rules,
            active: true
        });
    }
    
    function verifyProtocol(
        string calldata protocolName,
        string calldata actionData
    ) external view returns (bool) {
        // Protocol must exist and be active
        require(protocols[protocolName].active, "Protocol not found or inactive");
        
        // In a production environment, this would contain actual verification logic
        // based on the protocol rules. For now, we just verify the protocol exists
        return true;
    }
    
    function logAction(
        string calldata actionType,
        string calldata actionData,
        uint256 timestamp
    ) external onlyOwner {
        actions.push(Action({
            actionType: actionType,
            data: actionData,
            timestamp: timestamp,
            executor: msg.sender
        }));
        
        emit ActionLogged(actionType, timestamp);
    }
    
    function getAction(uint256 index)
        external
        view
        returns (
            string memory actionType,
            string memory data,
            uint256 timestamp,
            address executor
        )
    {
        require(index < actions.length, "Action index out of bounds");
        Action memory action = actions[index];
        return (
            action.actionType,
            action.data,
            action.timestamp,
            action.executor
        );
    }
    
    function getActionCount() external view returns (uint256) {
        return actions.length;
    }
}