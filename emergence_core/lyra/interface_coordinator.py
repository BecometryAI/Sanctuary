"""
Interface coordinator for managing multiple communication channels
"""
import asyncio
import logging
from typing import Dict, Optional
from pathlib import Path

from .webui.server import WebUIManager
from .terminal.interface import TerminalInterface
from .router import AdaptiveRouter
from .social_connections import SocialManager

logger = logging.getLogger(__name__)

class InterfaceCoordinator:
    def __init__(self, router: AdaptiveRouter, social_manager: SocialManager):
        self.router = router
        self.social_manager = social_manager
        
        # Initialize interfaces
        self.webui = WebUIManager(router, social_manager)
        self.terminal = TerminalInterface(router, social_manager)
        
        self.active_interfaces = set()
        
    async def start_interface(self, interface_type: str):
        """Start a specific interface"""
        if interface_type == "webui":
            if "webui" not in self.active_interfaces:
                # Start FastAPI server
                import uvicorn
                config = uvicorn.Config(
                    self.webui.app,
                    host="0.0.0.0",
                    port=8000,
                    log_level="info"
                )
                server = uvicorn.Server(config)
                asyncio.create_task(server.serve())
                self.active_interfaces.add("webui")
                logger.info("Web UI started on http://localhost:8000")
                
        elif interface_type == "terminal":
            if "terminal" not in self.active_interfaces:
                asyncio.create_task(self.terminal.start())
                self.active_interfaces.add("terminal")
                logger.info("Terminal interface started")
                
        else:
            raise ValueError(f"Unknown interface type: {interface_type}")
            
    async def stop_interface(self, interface_type: str):
        """Stop a specific interface"""
        if interface_type == "webui" and "webui" in self.active_interfaces:
            # Cleanup WebUI
            self.active_interfaces.remove("webui")
            
        elif interface_type == "terminal" and "terminal" in self.active_interfaces:
            self.terminal.stop()
            self.active_interfaces.remove("terminal")
            
    async def broadcast_status(self, status: str, message: Optional[str] = None):
        """Broadcast status to all active interfaces"""
        if "webui" in self.active_interfaces:
            await self.webui.broadcast_status(status, message)
            
    async def start_all(self):
        """Start all interfaces"""
        await self.start_interface("webui")
        await self.start_interface("terminal")
        
    async def stop_all(self):
        """Stop all interfaces"""
        for interface in list(self.active_interfaces):
            await self.stop_interface(interface)