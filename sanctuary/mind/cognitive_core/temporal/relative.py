"""
Relative Time: Describing time relative to now or other reference points.

This module provides utilities for describing timestamps in human-friendly,
relative terms like "just now", "3 hours ago", "in 2 days", etc.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Optional

logger = logging.getLogger(__name__)


class RelativeTime:
    """
    Utilities for describing time relative to a reference point.
    
    Provides natural language descriptions of timestamps relative to now
    or another reference time.
    """
    
    @staticmethod
    def describe(
        timestamp: datetime,
        reference: Optional[datetime] = None
    ) -> str:
        """
        Describe time relative to reference (default: now).
        
        Args:
            timestamp: Time to describe
            reference: Reference time (default: now)
            
        Returns:
            Human-readable relative time description
        """
        if reference is None:
            reference = datetime.now()
        
        diff = reference - timestamp
        
        if diff < timedelta(0):
            # Future time
            return RelativeTime._describe_future(-diff)
        else:
            # Past time
            return RelativeTime._describe_past(diff)
    
    @staticmethod
    def _describe_past(diff: timedelta) -> str:
        """
        Describe a time in the past.
        
        Args:
            diff: Time difference (positive)
            
        Returns:
            Description like "3 minutes ago"
        """
        seconds = diff.total_seconds()
        
        if seconds < 10:
            return "just now"
        elif seconds < 60:
            return f"{int(seconds)} seconds ago"
        elif seconds < 3600:
            minutes = int(seconds / 60)
            return f"{minutes} minute{'s' if minutes != 1 else ''} ago"
        elif seconds < 86400:
            hours = int(seconds / 3600)
            return f"{hours} hour{'s' if hours != 1 else ''} ago"
        elif seconds < 604800:  # Less than a week
            days = int(seconds / 86400)
            return f"{days} day{'s' if days != 1 else ''} ago"
        elif seconds < 2592000:  # Less than ~30 days
            weeks = int(seconds / 604800)
            return f"{weeks} week{'s' if weeks != 1 else ''} ago"
        elif seconds < 31536000:  # Less than a year
            months = int(seconds / 2592000)
            return f"{months} month{'s' if months != 1 else ''} ago"
        else:
            years = int(seconds / 31536000)
            return f"{years} year{'s' if years != 1 else ''} ago"
    
    @staticmethod
    def _describe_future(diff: timedelta) -> str:
        """
        Describe a time in the future.
        
        Args:
            diff: Time difference (positive, representing future)
            
        Returns:
            Description like "in 3 hours"
        """
        seconds = diff.total_seconds()
        
        if seconds < 60:
            return "in a moment"
        elif seconds < 3600:
            minutes = int(seconds / 60)
            return f"in {minutes} minute{'s' if minutes != 1 else ''}"
        elif seconds < 86400:
            hours = int(seconds / 3600)
            return f"in {hours} hour{'s' if hours != 1 else ''}"
        elif seconds < 604800:  # Less than a week
            days = int(seconds / 86400)
            return f"in {days} day{'s' if days != 1 else ''}"
        elif seconds < 2592000:  # Less than ~30 days
            weeks = int(seconds / 604800)
            return f"in {weeks} week{'s' if weeks != 1 else ''}"
        elif seconds < 31536000:  # Less than a year
            months = int(seconds / 2592000)
            return f"in {months} month{'s' if months != 1 else ''}"
        else:
            years = int(seconds / 31536000)
            return f"in {years} year{'s' if years != 1 else ''}"
    
    @staticmethod
    def describe_duration(duration: timedelta) -> str:
        """
        Describe a duration in human-readable form.
        
        Args:
            duration: Time duration
            
        Returns:
            Description like "5 minutes" or "2 hours and 30 minutes"
        """
        seconds = abs(duration.total_seconds())
        
        if seconds < 60:
            secs = int(seconds)
            return f"{secs} second{'s' if secs != 1 else ''}"
        elif seconds < 3600:
            minutes = int(seconds / 60)
            return f"{minutes} minute{'s' if minutes != 1 else ''}"
        elif seconds < 86400:
            hours = int(seconds / 3600)
            remaining_minutes = int((seconds % 3600) / 60)
            if remaining_minutes > 0:
                return f"{hours} hour{'s' if hours != 1 else ''} and {remaining_minutes} minute{'s' if remaining_minutes != 1 else ''}"
            else:
                return f"{hours} hour{'s' if hours != 1 else ''}"
        else:
            days = int(seconds / 86400)
            remaining_hours = int((seconds % 86400) / 3600)
            if remaining_hours > 0:
                return f"{days} day{'s' if days != 1 else ''} and {remaining_hours} hour{'s' if remaining_hours != 1 else ''}"
            else:
                return f"{days} day{'s' if days != 1 else ''}"
    
    @staticmethod
    def is_recent(timestamp: datetime, threshold: timedelta = timedelta(hours=1)) -> bool:
        """
        Check if a timestamp is recent (within threshold of now).
        
        Args:
            timestamp: Time to check
            threshold: Recency threshold (default: 1 hour)
            
        Returns:
            True if within threshold of now
        """
        now = datetime.now()
        diff = abs(now - timestamp)
        return diff < threshold
    
    @staticmethod
    def is_today(timestamp: datetime) -> bool:
        """
        Check if a timestamp is from today.
        
        Args:
            timestamp: Time to check
            
        Returns:
            True if same calendar day as now
        """
        now = datetime.now()
        return timestamp.date() == now.date()
    
    @staticmethod
    def is_this_week(timestamp: datetime) -> bool:
        """
        Check if a timestamp is from this week.
        
        Args:
            timestamp: Time to check
            
        Returns:
            True if within current week
        """
        now = datetime.now()
        # Get start of week (Monday)
        week_start = now - timedelta(days=now.weekday())
        week_start = week_start.replace(hour=0, minute=0, second=0, microsecond=0)
        
        return timestamp >= week_start
    
    @staticmethod
    def categorize_recency(timestamp: datetime) -> str:
        """
        Categorize how recent a timestamp is.
        
        Args:
            timestamp: Time to categorize
            
        Returns:
            Category: "now", "recent", "today", "this_week", "recent_past", "long_ago"
        """
        now = datetime.now()
        diff = now - timestamp
        
        if diff < timedelta(minutes=5):
            return "now"
        elif diff < timedelta(hours=1):
            return "recent"
        elif RelativeTime.is_today(timestamp):
            return "today"
        elif RelativeTime.is_this_week(timestamp):
            return "this_week"
        elif diff < timedelta(days=30):
            return "recent_past"
        else:
            return "long_ago"
