"""
Simple CLI for testing Lyra conversational interface.

This is a basic command-line interface for interacting with Lyra's cognitive core
through natural conversation. It demonstrates the use of the LyraAPI for
multi-turn dialogue.

Usage:
    python -m lyra.cli
    
    Or:
    python emergence_core/lyra/cli.py

Commands:
    - Type any message to chat with Lyra
    - Type 'quit' or 'exit' to exit
    - Type 'reset' to clear conversation history
    - Type 'history' to see recent conversation
    - Type 'metrics' to see conversation statistics
"""

import asyncio
import sys
from pathlib import Path

# Note: This import path manipulation is for development/testing only.
# In production, install the package properly using pip/uv.
# For proper installation, see README.md installation instructions.
try:
    from lyra.client import LyraAPI
except ImportError:
    # Fallback for development: add parent directory to path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from lyra.client import LyraAPI


async def main():
    """Main CLI loop for interacting with Lyra."""
    # Initialize Lyra
    print("🧠 Initializing Lyra...")
    lyra = LyraAPI()
    
    try:
        await lyra.start()
        print("✅ Lyra is online. Type 'quit' to exit.\n")
        
        while True:
            # Get user input
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n")
                break
            
            if not user_input:
                continue
            
            # Handle special commands
            if user_input.lower() in ["quit", "exit"]:
                break
            
            if user_input.lower() == "reset":
                lyra.reset_conversation()
                print("🔄 Conversation reset.\n")
                continue
            
            if user_input.lower() == "history":
                history = lyra.get_conversation_history(10)
                if not history:
                    print("No conversation history yet.\n")
                else:
                    print("\n📜 Recent conversation:")
                    for i, turn in enumerate(history, 1):
                        print(f"\n{i}. You: {turn.user_input}")
                        print(f"   Lyra: {turn.system_response}")
                        print(f"   (Response time: {turn.response_time:.2f}s)")
                    print()
                continue
            
            if user_input.lower() == "metrics":
                metrics = lyra.get_metrics()
                print("\n📊 Conversation Metrics:")
                print(f"   Total turns: {metrics['conversation']['total_turns']}")
                print(f"   Average response time: {metrics['conversation']['avg_response_time']:.2f}s")
                print(f"   Timeouts: {metrics['conversation']['timeouts']}")
                print(f"   Errors: {metrics['conversation']['errors']}")
                print(f"   Topics tracked: {metrics['conversation']['topics_tracked']}")
                print(f"   History size: {metrics['conversation']['history_size']}")
                print(f"\n🧠 Cognitive Core Metrics:")
                print(f"   Total cycles: {metrics['cognitive_core']['total_cycles']}")
                print(f"   Average cycle time: {metrics['cognitive_core']['avg_cycle_time_ms']:.2f}ms")
                print(f"   Workspace size: {metrics['cognitive_core']['workspace_size']}")
                print(f"   Current goals: {metrics['cognitive_core']['current_goals']}")
                print()
                continue
            
            # Process turn
            print("💭 Thinking...")
            turn = await lyra.chat(user_input)
            
            # Display response with emotion
            emotion = turn.emotional_state
            if emotion:
                valence = emotion.get('valence', 0.0)
                arousal = emotion.get('arousal', 0.0)
                emotion_label = f"[{valence:.1f}V {arousal:.1f}A]"
            else:
                emotion_label = ""
            
            print(f"\nLyra {emotion_label}: {turn.system_response}")
            print(f"(Response time: {turn.response_time:.2f}s)\n")
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        print("\n🛑 Shutting down Lyra...")
        await lyra.stop()
        print("👋 Lyra offline.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
