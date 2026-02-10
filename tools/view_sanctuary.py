"""
Simple viewer for Sanctuary's environment
"""
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_app():
    """Create a minimal FastAPI application just for viewing the sanctuary"""
    logger.info("Creating minimal sanctuary viewer...")
    
    app = FastAPI(
        title="Sanctuary Viewer", 
        docs_url="/docs"
    )

    # Mount static files
    static_dir = os.path.join(os.path.dirname(__file__), "sanctuary", "webui", "static")
    app.mount("/static", StaticFiles(directory=static_dir), name="static")
    
    @app.get("/")
    async def root():
        """Serve the sanctuary interface"""
        return FileResponse(os.path.join(static_dir, "sanctuary.html"))

    return app

if __name__ == "__main__":
    import uvicorn
    
    # Create the app
    app = create_app()
    
    import threading
    
    # Run the server in a background thread
    def run_server():
        uvicorn.run(
            app,
            host="127.0.0.1",
            port=8000,
            log_level="info"
        )
    
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    
    print("\nSanctuary viewer is running at http://127.0.0.1:8000")
    print("Press Ctrl+C to stop the server\n")