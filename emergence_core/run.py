# Start the Lyra Emergence server
# DEPRECATED: This server uses the old API which depends on deleted
# router/specialist architecture. Use run_cognitive_core.py instead.
import uvicorn
import sys
import signal
import logging
import threading
import time
import os
from contextlib import contextmanager
from pathlib import Path

# Set up logging with more detail
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Ensure we're in the right directory and Python path is set up
current_dir = Path(__file__).resolve().parent
if current_dir not in sys.path:
    sys.path.insert(0, str(current_dir))
logger.debug(f"Added {current_dir} to Python path")

from lyra.api import create_app

@contextmanager
def get_app():
    """Create FastAPI app with proper lifecycle management"""
    app = create_app()
    try:
        yield app
    finally:
        # Add cleanup if needed
        pass

def run_server():
    """Run the Uvicorn server"""
    try:
        with get_app() as app:
            config = uvicorn.Config(
                app,
                host="127.0.0.1",
                port=8000,
                log_level="debug",  # Increased logging level
                reload=False,
                access_log=True,    # Enable access logging
                workers=1,
                ws_ping_interval=20.0,  # Send ping every 20 seconds
                ws_ping_timeout=30.0,   # Wait 30 seconds for pong response
                timeout_keep_alive=65,  # Keep-alive timeout
                loop="asyncio"
            )
            server = uvicorn.Server(config)
            server.run()
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        raise

class ServerThread(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self._stop_event = threading.Event()
        self.daemon = False  # Don't daemonize to allow clean shutdown
        self.server = None
        self.error = None
        self.app = None
    
    def run(self):
        try:
            logger.info("Configuring server...")
            self.app = create_app()  # Create app instance
            config = uvicorn.Config(
                self.app,
                host="0.0.0.0",
                port=8000,
                log_level="info",
                reload=False,
                access_log=False,
                workers=1
            )
            self.server = uvicorn.Server(config)
            logger.info("Starting uvicorn server...")
            self.server.run()
        except Exception as e:
            self.error = e
            logger.error(f"Server error: {e}")
            raise
        finally:
            self._stop_event.set()
            logger.info("Server thread stopping")
    
    def stop(self):
        """Stop the server and clean up"""
        logger.info("Stopping server...")
        if self.server:
            self.server.should_exit = True
            logger.info("Server exit flag set")
        self._stop_event.set()
        self.join(timeout=5)  # Wait up to 5 seconds for thread to finish
        logger.info("Server stopped")
        
    def check_error(self):
        """Check if the server thread encountered an error"""
        if self.error:
            raise RuntimeError(f"Server thread error: {self.error}")
        return True

def signal_handler(signum, frame):
    """Handle shutdown signals gracefully"""
    logger.info(f"Received signal {signum}. Shutting down...")
    sys.exit(0)

def start_server():
    """Start the server and return the server thread"""
    try:
        logger.info("Starting Lyra Emergence server...")
        
        # Initialize server components
        try:
            from lyra.api import app
            logger.info("API components initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize API components: {e}")
            raise
        
        # Start server thread
        server_thread = ServerThread()
        server_thread.start()
        logger.info("Server thread started")
        
        # Wait briefly to ensure startup
        import time
        time.sleep(2)
        
        # Verify server is running
        if not server_thread.is_alive():
            raise RuntimeError("Server thread failed to start")
            
        # Test server health directly
        try:
            import requests
            response = requests.get('http://localhost:8000/health', timeout=5)
            if response.status_code == 200:
                logger.info("Server health check passed")
            else:
                logger.warning(f"Server responded with status {response.status_code}")
        except Exception as e:
            logger.warning(f"Health check failed (this is expected during startup): {e}")
        
        logger.info("Server startup completed successfully")
        return server_thread
    except Exception as e:
        logger.error(f"Error starting server: {e}")
        raise

def stop_server(server_thread):
    """Stop the server gracefully"""
    if server_thread:
        logger.info("Stopping server...")
        server_thread.stop()
        logger.info("Server stopped")
        # Force set the stop event and join thread
        server_thread._stop_event.set()
        server_thread.join(timeout=2)

if __name__ == "__main__":
    server_thread = None
    try:
        def handle_shutdown(signum, frame):
            logger.info(f"Received signal {signum}")
            if server_thread:
                stop_server(server_thread)
            sys.exit(0)
            
        # Set up signal handlers
        signal.signal(signal.SIGINT, handle_shutdown)
        signal.signal(signal.SIGTERM, handle_shutdown)
        
        server_thread = start_server()
        
        # Keep the main thread alive until interrupted
        while server_thread and server_thread.is_alive():
            time.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    except Exception as e:
        logger.error(f"Error in main thread: {e}")
    finally:
        if server_thread:
            stop_server(server_thread)
        sys.exit(0)