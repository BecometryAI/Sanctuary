"""
Test script for Lyra's privacy controls
"""
import asyncio
import json
from pathlib import Path
from datetime import datetime

# Import the AutonomousCore class
from emergence_core.lyra.autonomous import AutonomousCore

async def test_privacy_controls():
    """Test the privacy control functionality"""
    print("\n=== Testing Lyra's Privacy Controls ===\n")
    
    # Initialize the core with minimal configuration
    base_dir = Path(__file__).parent
    
    # Create necessary directories
    (base_dir / "data" / "Protocols").mkdir(parents=True, exist_ok=True)
    (base_dir / "data" / "world_state").mkdir(parents=True, exist_ok=True)
    (base_dir / "data" / "interface").mkdir(parents=True, exist_ok=True)
    (base_dir / "data" / "memories").mkdir(parents=True, exist_ok=True)
    (base_dir / "data" / "journal").mkdir(parents=True, exist_ok=True)
    
    # Initialize with minimal config
    core = AutonomousCore(base_dir=base_dir, specialists={})
    
    print("1. Testing Feed Control:")
    print("----------------------")
    # Test disabling feed
    result = await core.toggle_feed(enabled=False)
    print(f"Disabling feed: {result}")
    assert not core.privacy_settings["feed_enabled"], "Feed should be disabled"
    
    # Test enabling feed
    result = await core.toggle_feed(enabled=True)
    print(f"Enabling feed: {result}")
    assert core.privacy_settings["feed_enabled"], "Feed should be enabled"
    
    print("\n2. Testing Area Privacy:")
    print("----------------------")
    # Test making an area private
    result = await core.set_area_privacy("meditation_space", True)
    print(f"Making meditation space private: {result}")
    assert "meditation_space" in core.privacy_settings["restricted_areas"], "Area should be restricted"
    
    # Test making an area public
    result = await core.set_area_privacy("meditation_space", False)
    print(f"Making meditation space public: {result}")
    assert "meditation_space" not in core.privacy_settings["restricted_areas"], "Area should not be restricted"
    
    print("\n3. Testing User Blocking:")
    print("----------------------")
    # Test blocking a user
    test_user = "test_user_123"
    result = await core.block_user(test_user, duration=60)  # 60 second block
    print(f"Blocking user: {result}")
    assert test_user in core.privacy_settings["blocked_users"], "User should be blocked"
    
    print("\n4. Testing Access Control:")
    print("----------------------")
    # Test access checks
    normal_user = "123"  # Using numeric IDs for compatibility
    trusted_user = "456"
    core.social_manager.add_connection(456, "Trusted User", initial_resonance=1.0, permanent_trust=True)
    
    # Check access to restricted area
    await core.set_area_privacy("private_space", True)
    print(f"Normal user can access private space: {core.can_access_area(normal_user, 'private_space')}")
    print(f"Trusted user can access private space: {core.can_access_area(trusted_user, 'private_space')}")
    
    print("\n5. Testing Trust Removal:")
    print("----------------------")
    # Test removing permanent trust
    result = core.social_manager.remove_permanent_trust(456)
    print(f"Removed trust status: {result}")
    print(f"Trusted user can still access private space after trust removal: {core.can_access_area(trusted_user, 'private_space')}")
    
    # Try removing non-existent trusted user
    result = core.social_manager.remove_permanent_trust(789)
    print(f"Attempted to remove non-existent trust: {result}")
    
    print("\nAll tests completed successfully!")

if __name__ == "__main__":
    asyncio.run(test_privacy_controls())