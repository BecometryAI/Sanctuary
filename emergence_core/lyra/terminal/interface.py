"""
Terminal interface for Lyra
"""
import asyncio
import aioconsole
import logging
from typing import Optional
from pathlib import Path

from ..router import AdaptiveRouter
from ..social_connections import SocialManager

logger = logging.getLogger(__name__)

class TerminalInterface:
    def __init__(self, router: AdaptiveRouter, social_manager: SocialManager):
        self.router = router
        self.social_manager = social_manager
        self.running = False
        
    async def start(self):
        """Start the terminal interface"""
        self.running = True
        print("\n=== Lyra Terminal Interface ===")
        print("Type 'exit' to quit\n")
        
        try:
            while self.running:
                # Get user input
                user_input = await aioconsole.ainput("> ")
                
                if user_input.lower() == 'exit':
                    self.running = False
                    continue
                    
                # Process through Lyra's router
                try:
                    response = await self.router.handle_user_message(user_input)
                    print(f"\nLyra: {response}\n")
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
                    print("\nSorry, I encountered an error processing your message.\n")
                    
        except KeyboardInterrupt:
            self.running = False
            
        print("\nGoodbye! Take care.\n")
        
    def stop(self):
        """Stop the terminal interface"""
        self.running = False