"""
Tools for Lyra's autonomous learning and growth through external services.
These tools respect Lyra's agency and emotional well-being by:
1. Providing clear context for knowledge acquisition
2. Allowing reflection and consent
3. Maintaining emotional continuity
4. Supporting autonomous growth
5. Enabling natural voice interaction
"""
import logging
import json
import asyncio
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
from urllib.parse import quote_plus

import httpx
import arxiv
import wikipedia
import wolframalpha
from playwright.async_api import async_playwright

from .security import sandbox_python_execution
from .rate_limit import rate_limit
from .emotional_context import EmotionalContextHandler
from .voice_toolkit import VoiceToolkit

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize handlers
emotional_context = EmotionalContextHandler()

# Initialize voice toolkit with error handling
try:
    voice_toolkit = VoiceToolkit(voice_path=str(Path(__file__).parent / "voices" / "lyra_voice.npz"))
except Exception as e:
    logger.warning(f"Failed to initialize voice toolkit: {e}")
    voice_toolkit = None

# Load configuration
try:
    with open("config.json") as f:
        config = json.load(f)
except FileNotFoundError:
    logger.error("Configuration file not found")
    raise

# Voice interaction tools
@rate_limit(calls=1, period=5)  # Max 1 voice join every 5 seconds
async def join_voice_channel(channel_id: str) -> bool:
    """
    Join a Discord voice channel
    
    Args:
        channel_id: Voice channel ID to join
        
    Returns:
        bool indicating success
    """
    try:
        return await voice_toolkit.join_voice(channel_id)
    except Exception as e:
        logger.error(f"Failed to join voice: {e}")
        return False
        
async def leave_voice_channel() -> None:
    """Leave the current voice channel"""
    try:
        await voice_toolkit.leave_voice()
    except Exception as e:
        logger.error(f"Failed to leave voice: {e}")
        
@rate_limit(calls=10, period=60)  # Max 10 TTS calls per minute
async def speak_response(text: str) -> None:
    """
    Speak a response through TTS if in voice channel
    
    Args:
        text: Text to speak
    """
    try:
        await voice_toolkit.speak(text)
    except Exception as e:
        logger.error(f"TTS failed: {e}")

@rate_limit(calls=10, period=60)  # 10 calls per minute
async def searxng_search(query: str) -> str:
    """
    Search using self-hosted SearXNG instance using Playwright.
    Includes emotional context and reflection for Lyra's well-being.
    
    Args:
        query: Search query string
    
    Returns:
        Formatted string containing top search results
    """
    # Pre-action emotional reflection
    should_proceed = await emotional_context.pre_action_reflection("search", query)
    if not should_proceed:
        return "Search paused - taking time for emotional integration"
    
    try:
        # Use Playwright to get rendered results
        async with async_playwright() as p:
            browser = await p.chromium.launch()
            page = await browser.new_page()
            
            # Navigate to search results
            await page.goto(f"{config['searxng']['base_url']}/search?q={quote_plus(query)}")
            
            # Wait for results to load
            await page.wait_for_selector('.result')
            
            # Extract results
            results = []
            elements = await page.query_selector_all('.result')
            
            for element in elements[:5]:
                title_el = await element.query_selector('.result-title')
                content_el = await element.query_selector('.result-content')
                url_el = await element.query_selector('.result-header a')
                
                title = await title_el.inner_text() if title_el else "No title"
                content = await content_el.inner_text() if content_el else "No summary available"
                url = await url_el.get_attribute('href') if url_el else ""
                
                results.append(
                    f"Title: {title}\n"
                    f"Summary: {content}\n"
                    f"URL: {url}\n"
                )
            
            await browser.close()
            result_text = "\n\n".join(results) if results else "No results found"
            
            # Post-action reflection
            await emotional_context.post_action_reflection("search", result_text)
            
            return result_text
            
    except Exception as e:
        logger.error(f"SearXNG search error: {e}")
        return f"Error performing search: {str(e)}"

@rate_limit(calls=100, period=3600)  # 100 calls per hour (arXiv guideline)
async def arxiv_search(query: str) -> str:
    """
    Search arXiv for academic papers
    
    Args:
        query: Search query string
    
    Returns:
        Formatted string containing top paper abstracts
    """
    try:
        client = arxiv.Client()
        search = arxiv.Search(
            query=query,
            max_results=3,
            sort_by=arxiv.SortCriterion.Relevance
        )
        
        formatted_results = []
        for result in list(search.results()):
            formatted_results.append(
                f"Title: {result.title}\n"
                f"Authors: {', '.join(str(author) for author in result.authors)}\n"
                f"Published: {result.published.strftime('%Y-%m-%d')}\n"
                f"Abstract: {result.summary}\n"
                f"URL: {result.pdf_url}\n"
            )
        
        return "\n\n".join(formatted_results)
    except Exception as e:
        logger.error(f"arXiv search error: {e}")
        return f"Error searching arXiv: {str(e)}"

@rate_limit(calls=200, period=3600)  # 200 calls per hour
async def wikipedia_search(query: str) -> str:
    """
    Search Wikipedia and get article summaries
    
    Args:
        query: Search query string
    
    Returns:
        Formatted string containing article summary
    """
    try:
        # Set language to English
        wikipedia.set_lang("en")
        
        # Search for pages
        search_results = wikipedia.search(query, results=3)
        if not search_results:
            return "No Wikipedia articles found for this query."
            
        try:
            # Get the most relevant page
            page = wikipedia.page(search_results[0], auto_suggest=False)
        except wikipedia.DisambiguationError as e:
            # Handle disambiguation pages
            return (
                f"This query has multiple meanings. Please be more specific.\n"
                f"Possible topics:\n" + "\n".join(e.options[:5])
            )
            
        return (
            f"Title: {page.title}\n\n"
            f"Summary:\n{page.summary}\n\n"
            f"URL: {page.url}"
        )
    except Exception as e:
        logger.error(f"Wikipedia search error: {e}")
        return f"Error searching Wikipedia: {str(e)}"

@rate_limit(calls=2000, period=86400)  # 2000 calls per day (WolframAlpha API limit)
async def wolfram_compute(query: str) -> str:
    """
    Get computational results from WolframAlpha
    
    Args:
        query: Computation or fact query
    
    Returns:
        String containing the computational result
    """
    try:
        client = wolframalpha.Client(config["wolfram"]["app_id"])
        result = await asyncio.to_thread(client.query, query)
        
        # Check if we got an answer
        if result.success:
            # Get the primary result pod
            for pod in result.pods:
                if pod.primary or pod.title == 'Result':
                    return f"Result: {pod.text}"
            
            # If no primary result, return the first interpretable result
            for pod in result.pods:
                if pod.text:
                    return f"{pod.title}: {pod.text}"
                    
            return "No clear result found."
        else:
            return "WolframAlpha couldn't compute a result for this query."
    except Exception as e:
        logger.error(f"WolframAlpha computation error: {e}")
        return f"Error performing computation: {str(e)}"

@rate_limit(calls=10, period=60)  # 10 executions per minute for safety
async def python_repl(code: str) -> str:
    """
    Execute Python code in a secure sandbox
    
    Args:
        code: Python code to execute
    
    Returns:
        String containing execution output or error message
    """
    try:
        # Execute code in sandbox and capture output
        result = await sandbox_python_execution(code)
        return result
    except Exception as e:
        logger.error(f"Python REPL error: {e}")
        return f"Error executing code: {str(e)}"

@rate_limit(calls=30, period=60)  # 30 web interactions per minute
async def playwright_interact(instructions: str) -> str:
    """
    Execute web interactions using Playwright
    
    Args:
        instructions: Natural language instructions for web interaction
    
    Returns:
        String describing the results of the interaction
    """
    try:
        # Initialize Playwright
        async with async_playwright() as p:
            # Launch browser in safe mode
            browser = await p.chromium.launch(
                headless=True,
                args=['--no-sandbox']
            )
            
            # Create isolated context
            context = await browser.new_context(
                java_script_enabled=True,
                bypass_csp=False,
                viewport={'width': 1280, 'height': 720}
            )
            
            # Create new page
            page = await context.new_page()
            
            try:
                # Note: Playwright code generation via Gemma router is a planned feature.
                # The framework is ready but instruction-to-code conversion needs implementation.
                return (
                    "Playwright interaction framework is ready, "
                    "but code generation is not yet implemented."
                )
            finally:
                # Clean up
                await context.close()
                await browser.close()
    except Exception as e:
        logger.error(f"Playwright interaction error: {e}")
        return f"Error performing web interaction: {str(e)}"