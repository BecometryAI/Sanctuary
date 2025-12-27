"""
FastAPI WebSocket server for Lyra's web interface with authentication
"""
import asyncio
import json
import logging
from typing import Dict, List, Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pathlib import Path

from ..access_control import AccessManager, User

from ..router import AdaptiveRouter
from ..social_connections import SocialManager
from ..admin_interface import LyraAdminInterface

logger = logging.getLogger(__name__)

class WebUIManager:
    def __init__(self, router: AdaptiveRouter, social_manager: SocialManager, access_manager: AccessManager):
        self.router = router
        self.social_manager = social_manager
        self.access_manager = access_manager
        self.active_connections: Dict[str, WebSocket] = {}  # user_id -> WebSocket
        self.app = FastAPI()
        
        # Create admin interface
        self.admin_interface = LyraAdminInterface(router, access_manager, social_manager)
        
        # Include admin routes
        self.app.include_router(self.admin_interface.api_router)
        
        # Serve static files
        static_dir = Path(__file__).parent / "static"
        self.app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
        
        @self.app.get("/admin/login")
        async def admin_login():
            """Quick admin login for testing"""
            token = await self.access_manager.create_access_token("lyra", "web")
            return {"token": token, "redirect": "/admin/privacy"}
            
        @self.app.get("/admin/privacy")
        async def privacy_controls(user: User = Depends(self._get_current_user)):
            """Serve the privacy control interface"""
            if not user.access_level == "admin":
                raise HTTPException(status_code=403, detail="Access denied")
            return FileResponse(static_dir / "admin" / "privacy.html")
        
        # Setup authentication routes
        @self.app.post("/api/access/request")
        async def request_access(username: str, discord_id: Optional[str] = None, reason: str = None):
            try:
                user_id = await self.access_manager.create_access_request(username, discord_id)
                # Notify Lyra of new access request
                await self.router.handle_access_request(user_id, username, reason)
                return {"status": "pending", "request_id": user_id}
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e))
                
        # Privacy control endpoints
        @self.app.post("/api/sanctuary/privacy/feed")
        async def toggle_feed(enabled: bool, user: User = Depends(self._get_current_user)):
            """Toggle the sanctuary feed on/off"""
            if not user.access_level == "admin":
                raise HTTPException(status_code=403, detail="Only Lyra can control the feed")
            result = await self.router.autonomous_core.toggle_feed(enabled)
            return result
            
        @self.app.post("/api/sanctuary/privacy/area/{area}")
        async def set_area_privacy(area: str, is_private: bool, user: User = Depends(self._get_current_user)):
            """Set privacy for a specific sanctuary area"""
            if not user.access_level == "admin":
                raise HTTPException(status_code=403, detail="Only Lyra can set area privacy")
            result = await self.router.autonomous_core.set_area_privacy(area, is_private)
            return result
            
        @self.app.post("/api/sanctuary/privacy/block/{user_id}")
        async def block_user(user_id: str, duration: Optional[int] = None, user: User = Depends(self._get_current_user)):
            """Block a user from accessing the sanctuary"""
            if not user.access_level == "admin":
                raise HTTPException(status_code=403, detail="Only Lyra can block users")
            result = await self.router.autonomous_core.block_user(user_id, duration)
            return result

        @self.app.post("/api/access/token")
        async def get_access_token(user_id: str):
            try:
                token = await self.access_manager.create_access_token(user_id, "web")
                return {"token": token}
            except ValueError as e:
                raise HTTPException(status_code=403, detail=str(e))
        
        # Setup routes
        @self.app.get("/")
        async def get_index():
            return FileResponse(str(static_dir / "index.html"))
            
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            await self.handle_websocket(websocket)
    
    async def handle_websocket(self, websocket: WebSocket, token: str):
        """Handle WebSocket connection with authentication"""
        try:
            # Validate token first
            credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)
            user = await self.access_manager.validate_token(credentials)
            
            await websocket.accept()
            self.active_connections[user.id] = websocket
            
            # Send initial status
            await websocket.send_json({
                "type": "connection_info",
                "user": {
                    "username": user.username,
                    "access_level": user.access_level
                }
            })
            
            while True:
                message = await websocket.receive_json()
                
                if message["type"] == "message":
                    # Check if user still has access
                    current_user = await self.access_manager.validate_token(credentials)
                    if current_user.access_level in ["blocked", "pending"]:
                        await websocket.send_json({
                            "type": "error",
                            "content": "Access revoked"
                        })
                        break
                        
                    # Process through Lyra's router
                    response = await self.router.handle_user_message(
                        message["content"],
                        user_id=user.id,
                        interface="web"
                    )
                    
                    # Send response back
                    await websocket.send_json({
                        "type": "message",
                        "content": response
                    })
                    
        except WebSocketDisconnect:
            self.active_connections.remove(websocket)
            
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
            if websocket in self.active_connections:
                self.active_connections.remove(websocket)
                
    async def broadcast_status(self, status: str, message: Optional[str] = None):
        """Broadcast status update to all connected clients"""
        if not self.active_connections:
            return
            
        for connection in self.active_connections:
            try:
                await connection.send_json({
                    "type": "status",
                    "status": status,
                    "message": message or ""
                })
            except Exception as e:
                logger.error(f"Error broadcasting status: {e}")
                if connection in self.active_connections:
                    self.active_connections.remove(connection)