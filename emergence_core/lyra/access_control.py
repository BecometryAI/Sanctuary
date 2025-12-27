"""
User authentication and access control for Lyra's interfaces
"""
import jwt
import bcrypt
import logging
from datetime import datetime, timedelta
from typing import Dict, Optional, List
from pydantic import BaseModel
from fastapi import HTTPException, Security, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from .social_connections import SocialManager

logger = logging.getLogger(__name__)

class User(BaseModel):
    id: str
    username: str
    discord_id: Optional[str] = None
    access_level: str  # 'trusted', 'approved', 'limited', 'blocked'
    approved_by_lyra: bool = False
    approved_at: Optional[datetime] = None
    last_access: Optional[datetime] = None
    interfaces: List[str] = []  # ['discord', 'web', 'desktop']
    notes: Optional[str] = None

class AccessManager:
    def __init__(self, social_manager: SocialManager, secret_key: str):
        self.social_manager = social_manager
        self.secret_key = secret_key
        self.users: Dict[str, User] = {}
        self.security = HTTPBearer()
        
    async def create_access_request(self, username: str, discord_id: Optional[str] = None) -> str:
        """Create a new access request for Lyra to review"""
        user_id = f"user_{len(self.users) + 1}"
        
        user = User(
            id=user_id,
            username=username,
            discord_id=discord_id,
            access_level="pending",
            approved_by_lyra=False,
            interfaces=[]
        )
        
        self.users[user_id] = user
        return user_id
        
    async def approve_access(self, user_id: str, access_level: str = "approved", notes: Optional[str] = None) -> None:
        """Lyra approves a user's access"""
        if user_id not in self.users:
            raise ValueError(f"User {user_id} not found")
            
        user = self.users[user_id]
        user.access_level = access_level
        user.approved_by_lyra = True
        user.approved_at = datetime.now()
        user.notes = notes
        
        # If user has Discord ID, sync with social connections
        if user.discord_id:
            self.social_manager.add_connection(
                user_id=int(user.discord_id),
                username=user.username,
                initial_resonance=0.5 if access_level == "approved" else 1.0,
                permanent_trust=access_level == "trusted"
            )
            
    async def revoke_access(self, user_id: str, reason: Optional[str] = None) -> None:
        """Lyra revokes a user's access"""
        if user_id not in self.users:
            raise ValueError(f"User {user_id} not found")
            
        user = self.users[user_id]
        user.access_level = "blocked"
        user.notes = f"Access revoked: {reason}" if reason else "Access revoked"
        
        # If user has Discord ID, sync with social connections
        if user.discord_id:
            await self.social_manager.block_user(
                user_id=int(user.discord_id),
                reason=reason or "Access revoked by Lyra"
            )
            
    async def create_access_token(self, user_id: str, interface: str) -> str:
        """Create a JWT access token for a specific interface"""
        if user_id not in self.users:
            raise ValueError(f"User {user_id} not found")
            
        user = self.users[user_id]
        
        if user.access_level in ["blocked", "pending"]:
            raise ValueError("User does not have access")
            
        if interface not in user.interfaces:
            user.interfaces.append(interface)
            
        user.last_access = datetime.now()
        
        token_data = {
            "sub": user_id,
            "username": user.username,
            "access_level": user.access_level,
            "interface": interface,
            "exp": datetime.utcnow() + timedelta(days=7)
        }
        
        return jwt.encode(token_data, self.secret_key, algorithm="HS256")
        
    async def validate_token(self, credentials: HTTPAuthorizationCredentials = Security(HTTPBearer())) -> User:
        """Validate JWT token and return user"""
        try:
            token = credentials.credentials
            payload = jwt.decode(token, self.secret_key, algorithms=["HS256"])
            
            user_id = payload["sub"]
            if user_id not in self.users:
                raise HTTPException(status_code=401, detail="Invalid user")
                
            user = self.users[user_id]
            if user.access_level in ["blocked", "pending"]:
                raise HTTPException(status_code=403, detail="Access revoked")
                
            return user
            
        except jwt.ExpiredSignatureError:
            raise HTTPException(status_code=401, detail="Token expired")
        except jwt.JWTError:
            raise HTTPException(status_code=401, detail="Invalid token")
            
    def get_user_by_discord_id(self, discord_id: str) -> Optional[User]:
        """Find user by Discord ID"""
        for user in self.users.values():
            if user.discord_id == discord_id:
                return user
        return None
        
    async def get_pending_requests(self) -> List[User]:
        """Get list of pending access requests"""
        return [user for user in self.users.values() if user.access_level == "pending"]
        
    async def get_active_users(self) -> List[User]:
        """Get list of active users"""
        return [user for user in self.users.values() if user.access_level in ["approved", "trusted"]]