"""
API server for interacting with the consciousness core

DEPRECATED: This API server was built for the old "Cognitive Committee" 
specialist architecture. It is no longer functional after the removal of
the router/specialist system. Use run_cognitive_core.py instead.
"""
from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.websockets import WebSocketDisconnect
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
from .consciousness import ConsciousnessCore
import logging
import os
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_app():
    """Create and configure the FastAPI application"""
    logger.info("Creating FastAPI application...")
    
    app = FastAPI(
        title="Lyra Emergence API", 
        docs_url="/api/docs", 
        redoc_url="/api/redoc"
    )
    logger.info("FastAPI application created successfully")

    logger.info("Importing required modules...")
    try:
        from fastapi.middleware.cors import CORSMiddleware
        from .webui.server import WebUIManager
        from .access_control import AccessManager
        from .social_connections import SocialManager
        from .router import AdaptiveRouter
        logger.info("All modules imported successfully")
    except ImportError as e:
        logger.error(f"Failed to import required modules: {str(e)}")
        raise
    from .router import AdaptiveRouter
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Initialize core components
    try:
        logger.info("Initializing core components...")
        consciousness = ConsciousnessCore()
        social_manager = SocialManager()
        access_manager = AccessManager(social_manager=social_manager, secret_key="lyra_development_key")
        # Set up paths for router
        base_dir = os.path.dirname(os.path.dirname(__file__))
        model_dir = os.path.join(base_dir, "model_cache")
        chroma_dir = os.path.join(base_dir, "data", "chroma")
        
        # Create directories if they don't exist
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(chroma_dir, exist_ok=True)
        
        router = AdaptiveRouter(
            base_dir=base_dir,
            chroma_dir=chroma_dir,
            model_dir=model_dir,
            development_mode=True  # Enable development mode by default for now
        )
        router.consciousness = consciousness  # Set consciousness after initialization
        
        # Initialize WebUI
        webui = WebUIManager(router, social_manager, access_manager)
        
        # Mount the WebUI app
        app.mount("/", webui.app)
        
        logger.info("All components initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize components: {e}")
        raise
        
    # Mount static files
    static_dir = os.path.join(os.path.dirname(__file__), "webui", "static")
    app.mount("/static", StaticFiles(directory=static_dir), name="static")
    
    @app.get("/")
    async def root():
        """Serve the main web interface"""
        return FileResponse(os.path.join(static_dir, "index.html"))

    @app.get("/health")
    async def health_check():
        """Check if the system is healthy"""
        logger.info("Health check endpoint called")
        return {
            "status": "ok", 
            "message": "System is healthy",
            "websocket_enabled": True,
            "consciousness_loaded": consciousness is not None
        }

    @app.get("/state")
    async def get_state():
        """Get the current state of the consciousness system"""
        return {"state": consciousness.get_state()}

    @app.post("/process")
    async def process_input(data: Dict[str, Any]):
        """Process input through the consciousness system"""
        try:
            response = consciousness.process_input(data)
            return response
        except Exception as e:
            logger.error(f"Error processing input: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    class ConnectionManager:
        def __init__(self):
            self.active_connections: List[WebSocket] = []

        async def connect(self, websocket: WebSocket):
            await websocket.accept()
            self.active_connections.append(websocket)

        def disconnect(self, websocket: WebSocket):
            self.active_connections.remove(websocket)

        async def broadcast(self, message: dict):
            for connection in self.active_connections:
                await connection.send_json(message)

    manager = ConnectionManager()

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        logger.info("New WebSocket connection attempt")
        await manager.connect(websocket)
        logger.info("WebSocket connection established")
        
        async def send_status(status: str, message: str = None):
            try:
                await websocket.send_json({
                    "type": "status",
                    "status": status,
                    "message": message or status
                })
            except Exception as e:
                logger.error(f"Error sending status: {e}")
        
        try:
            # Send initial status
            await send_status("online", "Connected to Lyra")
            
            while True:
                try:
                    data = await websocket.receive_text()
                    logger.info(f"Received WebSocket message: {data}")
                    
                    try:
                        message = json.loads(data)
                    except json.JSONDecodeError as e:
                        logger.error(f"Invalid JSON received: {e}")
                        await send_status("error", "Invalid message format")
                        continue
                    
                    if message.get("type") == "status" and message.get("content") == "ping":
                        await send_status("online", "Connected to Lyra")
                        continue
                    
                    if message.get("type") == "message":
                        try:
                            # Process the message through consciousness
                            await send_status("processing", "Processing your message...")
                            
                            response = consciousness.process_input({"message": message["content"]})
                            logger.info(f"Processed message, response: {response}")
                            
                            # Send response back to the client
                            await websocket.send_json({
                                "type": "message",
                                "content": response.get("response", "I understand.")
                            })
                            
                            await send_status("online", "Ready for next message")
                            
                        except Exception as e:
                            logger.error(f"Error processing message: {e}")
                            await send_status("error", "Error processing message")
                    
                except WebSocketDisconnect:
                    raise
                except Exception as e:
                    logger.error(f"Error handling message: {e}")
                    await send_status("error", "Internal error")
                
        except WebSocketDisconnect:
            logger.info("WebSocket disconnected normally")
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
        finally:
            manager.disconnect(websocket)
            try:
                await websocket.close()
            except:
                pass

    app.state.consciousness = consciousness
    app.state.manager = manager
    return app

app = create_app()

class Input(BaseModel):
    """Input data model"""
    content: Dict[str, Any]
    type: str = "general"

class Response(BaseModel):
    """Response data model"""
    response: Dict[str, Any]
    internal_state: Dict[str, Any]
    status: str = "success"
    message: Optional[str] = None

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        state = app.state.consciousness.get_internal_state()
        return {
            "status": "healthy", 
            "consciousness_state": "active",
            "version": "0.1.0"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/process", response_model=Response)
async def process_input(input_data: Input):
    """Process input through the consciousness core"""
    try:
        logger.info(f"Processing input of type: {input_data.type}")
        response = app.state.consciousness.process_input(input_data.content)
        internal_state = app.state.consciousness.get_internal_state()
        
        return Response(
            response=response,
            internal_state=internal_state,
            status="success",
            message="Input processed successfully"
        )
    except Exception as e:
        logger.error(f"Error processing input: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/state")
async def get_state():
    """Get current internal state"""
    try:
        state = app.state.consciousness.get_internal_state()
        return {
            "status": "success",
            "state": state
        }
    except Exception as e:
        logger.error(f"Error getting state: {e}")
        raise HTTPException(status_code=500, detail=str(e))