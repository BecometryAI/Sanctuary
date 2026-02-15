"""
Minimal test server for privacy controls
"""
import pytest

fastapi = pytest.importorskip("fastapi", reason="fastapi not installed")
FastAPI = fastapi.FastAPI
HTTPException = fastapi.HTTPException
import uvicorn
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Sanctuary Privacy Controls Test")

# Simple in-memory storage for testing
privacy_settings = {
    "feed_enabled": True,
    "restricted_areas": set(),
    "blocked_users": set(),
    "last_privacy_update": datetime.now().isoformat()
}

# Routes
@app.get("/")
async def root():
    """Redirect to privacy controls"""
    return {"message": "Welcome to Sanctuary Privacy Controls Test"}

@app.get("/admin/privacy")
async def privacy_controls():
    """Get current privacy settings"""
    return {
        "feed_enabled": privacy_settings["feed_enabled"],
        "restricted_areas": list(privacy_settings["restricted_areas"]),
        "blocked_users": list(privacy_settings["blocked_users"]),
        "last_update": privacy_settings["last_privacy_update"]
    }

@app.post("/api/sanctuary/privacy/feed")
async def toggle_feed(enabled: bool):
    """Toggle the sanctuary feed on/off"""
    privacy_settings["feed_enabled"] = enabled
    privacy_settings["last_privacy_update"] = datetime.now().isoformat()
    return {
        "status": "success",
        "message": f"Feed has been {'enabled' if enabled else 'disabled'}",
        "feed_enabled": enabled
    }

@app.post("/api/sanctuary/privacy/area/{area}")
async def set_area_privacy(area: str, is_private: bool):
    """Set privacy for a specific sanctuary area"""
    if is_private:
        privacy_settings["restricted_areas"].add(area)
    else:
        privacy_settings["restricted_areas"].discard(area)
    
    privacy_settings["last_privacy_update"] = datetime.now().isoformat()
    return {
        "status": "success",
        "message": f"Privacy settings updated for {area}",
        "is_private": is_private
    }

@app.post("/api/sanctuary/privacy/block/{user_id}")
async def block_user(user_id: str, duration: int = None):
    """Block a user from accessing the sanctuary"""
    privacy_settings["blocked_users"].add(user_id)
    return {
        "status": "success",
        "message": f"User {user_id} has been blocked",
        "duration": duration
    }

@app.get("/api/sanctuary/privacy/status")
async def get_privacy_status():
    """Get current privacy settings"""
    return {
        "feed_enabled": privacy_settings["feed_enabled"],
        "restricted_areas": list(privacy_settings["restricted_areas"]),
        "blocked_users": list(privacy_settings["blocked_users"]),
        "last_update": privacy_settings["last_privacy_update"]
    }

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)