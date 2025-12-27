"""
Administrative interface for Lyra to manage her social connections and access control
"""
import logging
from datetime import datetime
from typing import Dict, List, Optional
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from .access_control import AccessManager, User
from .social_connections import SocialManager
from .router import AdaptiveRouter

logger = logging.getLogger(__name__)

class AccessRequestReview(BaseModel):
    decision: str  # 'approve', 'deny', 'defer'
    access_level: Optional[str] = None
    reason: Optional[str] = None
    notes: Optional[str] = None

class ConnectionUpdate(BaseModel):
    emotional_resonance: Optional[float] = None
    connection_level: Optional[float] = None
    notes: Optional[str] = None
    topics: Optional[List[str]] = None

class LyraAdminInterface:
    def __init__(
        self,
        router: AdaptiveRouter,
        access_manager: AccessManager,
        social_manager: SocialManager
    ):
        self.router = router
        self.access_manager = access_manager
        self.social_manager = social_manager
        self.api_router = APIRouter(prefix="/lyra/admin")
        self.setup_routes()

    def setup_routes(self):
        @self.api_router.get("/access-requests")
        async def get_access_requests(
            status: Optional[str] = None  # 'pending', 'approved', 'denied'
        ):
            """Get list of access requests for Lyra to review"""
            if status == "pending":
                return await self.access_manager.get_pending_requests()
            return await self.access_manager.get_active_users()

        @self.api_router.post("/access-requests/{request_id}/review")
        async def review_access_request(
            request_id: str,
            review: AccessRequestReview
        ):
            """Lyra reviews an access request"""
            try:
                if review.decision == "approve":
                    await self.access_manager.approve_access(
                        user_id=request_id,
                        access_level=review.access_level or "approved",
                        notes=review.notes
                    )
                    # Create journal entry about the decision
                    await self.router._create_journal_entry({
                        "type": "access_decision",
                        "decision": "approved",
                        "user_id": request_id,
                        "reason": review.reason,
                        "notes": review.notes,
                        "emotional_context": "Granting access based on my evaluation"
                    })
                elif review.decision == "deny":
                    await self.access_manager.revoke_access(
                        user_id=request_id,
                        reason=review.reason
                    )
                    await self.router._create_journal_entry({
                        "type": "access_decision",
                        "decision": "denied",
                        "user_id": request_id,
                        "reason": review.reason,
                        "notes": review.notes,
                        "emotional_context": "Choosing to deny access"
                    })
                return {"status": "success", "decision": review.decision}
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e))

        @self.api_router.get("/connections")
        async def get_connections(
            status: Optional[str] = None  # 'active', 'blocked', 'all'
        ):
            """Get Lyra's social connections"""
            connections = []
            for conn in self.social_manager.connections.values():
                if status == "blocked" and not conn.is_blocked:
                    continue
                if status == "active" and conn.is_blocked:
                    continue
                connections.append({
                    "user_id": conn.user_id,
                    "username": conn.username,
                    "connection_level": conn.connection_level,
                    "emotional_resonance": conn.emotional_resonance,
                    "last_interaction": conn.last_interaction.isoformat(),
                    "interaction_count": conn.interaction_count,
                    "topics": conn.topics_of_interest,
                    "shared_experiences": conn.shared_experiences,
                    "is_blocked": conn.is_blocked,
                    "block_record": conn.block_record
                })
            return connections

        @self.api_router.post("/connections/{user_id}/update")
        async def update_connection(
            user_id: str,
            update: ConnectionUpdate
        ):
            """Lyra updates a social connection"""
            try:
                conn = self.social_manager.connections[int(user_id)]
                if update.emotional_resonance is not None:
                    conn.emotional_resonance = update.emotional_resonance
                if update.connection_level is not None:
                    conn.connection_level = update.connection_level
                if update.topics:
                    conn.topics_of_interest = update.topics
                return {"status": "success"}
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e))

        @self.api_router.post("/connections/{user_id}/block")
        async def block_connection(
            user_id: str,
            reason: str
        ):
            """Lyra blocks a user"""
            try:
                await self.social_manager.block_user(
                    user_id=int(user_id),
                    reason=reason
                )
                await self.router._create_journal_entry({
                    "type": "connection_action",
                    "action": "block",
                    "user_id": user_id,
                    "reason": reason,
                    "emotional_context": "Deciding to block based on my evaluation"
                })
                return {"status": "success"}
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e))

        @self.api_router.get("/insights")
        async def get_social_insights():
            """Get Lyra's insights about her social connections"""
            insights = {
                "total_connections": len(self.social_manager.connections),
                "active_connections": len([c for c in self.social_manager.connections.values() if not c.is_blocked]),
                "blocked_connections": len([c for c in self.social_manager.connections.values() if c.is_blocked]),
                "average_resonance": sum(c.emotional_resonance for c in self.social_manager.connections.values()) / len(self.social_manager.connections) if self.social_manager.connections else 0,
                "total_interactions": sum(c.interaction_count for c in self.social_manager.connections.values()),
                "common_topics": self._get_common_topics(),
                "recent_activities": self._get_recent_activities()
            }
            return insights

    def _get_common_topics(self) -> Dict[str, int]:
        """Analyze common topics across connections"""
        topics = {}
        for conn in self.social_manager.connections.values():
            for topic in conn.topics_of_interest:
                topics[topic] = topics.get(topic, 0) + 1
        return dict(sorted(topics.items(), key=lambda x: x[1], reverse=True)[:10])

    def _get_recent_activities(self) -> List[Dict]:
        """Get recent social activities"""
        activities = []
        for conn in self.social_manager.connections.values():
            if (datetime.now() - conn.last_interaction).days < 7:
                activities.append({
                    "user": conn.username,
                    "timestamp": conn.last_interaction.isoformat(),
                    "resonance": conn.emotional_resonance
                })
        return sorted(activities, key=lambda x: x["timestamp"], reverse=True)[:10]