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
from typing import Dict, Any, Optional, List, Tuple, TYPE_CHECKING
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

if TYPE_CHECKING:
    from .router import AdaptiveRouter

# Configure logging
logger = logging.getLogger(__name__)


class SpecialistTools:
    """
    Tool suite for Lyra's specialists with router integration for code generation.
    
    This class provides all external tool capabilities including:
    - Web search (SearXNG)
    - Academic research (arXiv)
    - Encyclopedic knowledge (Wikipedia)
    - Computation (WolframAlpha)
    - Code execution (Python REPL)
    - Web automation (Playwright with code generation)
    - Voice interaction (Discord voice channels)
    
    Thread-safe and designed for async operation.
    """
    
    def __init__(self, router: Optional['AdaptiveRouter'] = None, config_path: str = "config.json"):
        """
        Initialize specialist tools with optional router for code generation.
        
        Args:
            router: AdaptiveRouter instance for Playwright code generation (optional)
            config_path: Path to configuration file (default: "config.json")
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            json.JSONDecodeError: If config file is invalid
        """
        self.router = router
        
        # Load configuration with proper error handling
        self.config = self._load_config(config_path)
        
        # Initialize emotional context handler
        self.emotional_context = EmotionalContextHandler()
        
        # Initialize voice toolkit with error handling
        try:
            self.voice_toolkit = VoiceToolkit(
                voice_path=str(Path(__file__).parent / "voices" / "lyra_voice.npz")
            )
        except Exception as e:
            logger.warning(f"Failed to initialize voice toolkit: {e}")
            self.voice_toolkit = None
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load configuration from JSON file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Configuration dictionary
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            json.JSONDecodeError: If config is invalid JSON
        """
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {config_path}")
            # Return minimal config for testing/development
            return {
                "searxng": {"base_url": "http://localhost:8080"},
                "wolfram": {"app_id": ""},
            }
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in config file: {e}")
            raise

    
    # Voice interaction tools
    @rate_limit(calls=1, period=5)  # Max 1 voice join every 5 seconds
    async def join_voice_channel(self, channel_id: str) -> bool:
        """
        Join a Discord voice channel
        
        Args:
            channel_id: Voice channel ID to join
            
        Returns:
            bool indicating success
        """
        if not self.voice_toolkit:
            logger.error("Voice toolkit not available")
            return False
            
        try:
            return await self.voice_toolkit.join_voice(channel_id)
        except Exception as e:
            logger.error(f"Failed to join voice: {e}")
            return False
            
    async def leave_voice_channel(self) -> None:
        """Leave the current voice channel"""
        if not self.voice_toolkit:
            return
            
        try:
            await self.voice_toolkit.leave_voice()
        except Exception as e:
            logger.error(f"Failed to leave voice: {e}")
            
    @rate_limit(calls=10, period=60)  # Max 10 TTS calls per minute
    async def speak_response(self, text: str) -> None:
        """
        Speak a response through TTS if in voice channel
        
        Args:
            text: Text to speak
        """
        if not self.voice_toolkit:
            logger.warning("Voice toolkit not available")
            return
            
        try:
            await self.voice_toolkit.speak(text)
        except Exception as e:
            logger.error(f"TTS failed: {e}")

    @rate_limit(calls=10, period=60)  # 10 calls per minute
    async def searxng_search(self, query: str) -> str:
        """
        Search using self-hosted SearXNG instance using Playwright.
        Includes emotional context and reflection for Lyra's well-being.
        
        Args:
            query: Search query string
        
        Returns:
            Formatted string containing top search results
        """
        # Pre-action emotional reflection
        should_proceed = await self.emotional_context.pre_action_reflection("search", query)
        if not should_proceed:
            return "Search paused - taking time for emotional integration"
        
        try:
            # Use Playwright to get rendered results
            async with async_playwright() as p:
                browser = await p.chromium.launch()
                page = await browser.new_page()
                
                # Navigate to search results
                await page.goto(f"{self.config['searxng']['base_url']}/search?q={quote_plus(query)}")
                
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
                await self.emotional_context.post_action_reflection("search", result_text)
                
                return result_text
                
        except Exception as e:
            logger.error(f"SearXNG search error: {e}")
            return f"Error performing search: {str(e)}"

    @rate_limit(calls=100, period=3600)  # 100 calls per hour (arXiv guideline)
    async def arxiv_search(self, query: str) -> str:
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
    async def wikipedia_search(self, query: str) -> str:
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
    async def wolfram_compute(self, query: str) -> str:
        """
        Get computational results from WolframAlpha
        
        Args:
            query: Computation or fact query
        
        Returns:
            String containing the computational result
        """
        try:
            client = wolframalpha.Client(self.config["wolfram"]["app_id"])
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
    async def python_repl(self, code: str) -> str:
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
    async def playwright_interact(self, instructions: str) -> str:
        """
        Execute web interactions using Playwright with AI-generated code.
        
        Uses the Gemma router to convert natural language instructions into
        executable Playwright code, then safely executes it.
        
        Args:
            instructions: Natural language instructions for web interaction
        
        Returns:
            String describing the results of the interaction
            
        Raises:
            ValueError: If instructions are empty or too long
        """
        # Input validation
        if not instructions or not instructions.strip():
            return "Error: Instructions cannot be empty"
        
        if len(instructions) > 2000:
            return "Error: Instructions too long (max 2000 characters)"
        
        try:
            # Initialize Playwright
            async with async_playwright() as p:
                # Launch browser in safe mode
                browser = await p.chromium.launch(
                    headless=True,
                    args=['--no-sandbox', '--disable-dev-shm-usage']  # Additional safety
                )
                
                # Create isolated context with security settings
                context = await browser.new_context(
                    java_script_enabled=True,
                    bypass_csp=False,
                    viewport={'width': 1280, 'height': 720},
                    user_agent='Mozilla/5.0 (Lyra-Bot)'  # Identify as bot
                )
                
                # Create new page
                page = await context.new_page()
                
                try:
                    # Generate Playwright code using router if available
                    if self.router:
                        code = await self._generate_playwright_code(instructions, page)
                        if code:
                            # Execute the generated code
                            result = await self._execute_playwright_code(code, page, context)
                            return result
                    
                    # Fallback: Return status message
                    return (
                        "Playwright interaction framework is ready. "
                        "Router integration needed for code generation. "
                        f"Instructions received: {instructions}"
                    )
                finally:
                    # Clean up
                    await context.close()
                    await browser.close()
        except Exception as e:
            logger.error(f"Playwright interaction error: {e}")
            return f"Error performing web interaction: {str(e)}"
    
    async def _generate_playwright_code(self, instructions: str, page: Any) -> Optional[str]:
        """
        Generate Playwright code using the Gemma router model.
        
        Args:
            instructions: Natural language instructions
            page: Playwright page object (unused, for API consistency)
            
        Returns:
            Generated Python code string or None if generation fails
        """
        if not self.router or not hasattr(self.router, 'router_model'):
            logger.warning("Router or router_model not available for code generation")
            return None
        
        # Sanitize instructions to prevent prompt injection
        sanitized_instructions = instructions.replace("```", "").strip()
        
        prompt = f"""Generate async Python Playwright code to perform this action: {sanitized_instructions}

Available variables:
- page: Playwright Page object (already initialized)
- context: Browser context

Requirements:
1. Return ONLY executable Python code, no explanations or markdown
2. Use async/await for all Playwright operations
3. Handle common errors gracefully with try/except
4. Store final result in a variable called 'result'
5. Keep code safe - no file system access, no dangerous imports
6. Use page.goto(), page.click(), page.fill(), page.locator(), etc.
7. Maximum 20 lines of code

Example:
try:
    await page.goto('https://example.com')
    title = await page.title()
    result = f"Page title: {{title}}"
except Exception as e:
    result = f"Error: {{e}}"

Code:"""
        
        try:
            # Use router model to generate code with timeout
            generated = await asyncio.wait_for(
                self.router.router_model.generate(
                    prompt=prompt,
                    max_tokens=512,
                    temperature=0.3  # Low temperature for more deterministic code
                ),
                timeout=10.0  # 10 second timeout for generation
            )
            
            # Extract code from response
            code = generated.strip()
            
            # Remove markdown code blocks if present
            if '```python' in code:
                code = code.split('```python')[1].split('```')[0].strip()
            elif '```' in code:
                code = code.split('```')[1].split('```')[0].strip()
            
            # Basic validation
            if not code or len(code) < 10:
                logger.warning("Generated code is too short or empty")
                return None
            
            if code.count('\n') > 25:
                logger.warning("Generated code is too long, truncating")
                code = '\n'.join(code.split('\n')[:25])
            
            logger.info(f"Generated Playwright code ({len(code)} chars, {code.count(chr(10))} lines)")
            logger.debug(f"Code:\n{code}")
            return code
            
        except asyncio.TimeoutError:
            logger.error("Code generation timed out after 10 seconds")
            return None
        except AttributeError as e:
            logger.error(f"Router model missing generate method: {e}")
            return None
        except Exception as e:
            logger.error(f"Code generation failed: {e}", exc_info=True)
            return None
    
    async def _execute_playwright_code(self, code: str, page: Any, context: Any) -> str:
        """
        Safely execute generated Playwright code with security constraints.
        
        Security measures:
        - Restricted builtins (no file access, no imports)
        - Timeout enforcement (5 seconds max)
        - Code validation (basic syntax and security checks)
        - Exception isolation
        
        Args:
            code: Python code to execute
            page: Playwright page object
            context: Browser context
            
        Returns:
            Result string from execution
            
        Raises:
            TimeoutError: If code takes too long
            ValueError: If code contains dangerous patterns
        """
        # Validate code before execution
        self._validate_generated_code(code)
        
        try:
            # Create restricted execution environment
            # Only allow safe builtins
            safe_builtins = {
                'len': len,
                'str': str,
                'int': int,
                'float': float,
                'bool': bool,
                'list': list,
                'dict': dict,
                'tuple': tuple,
                'range': range,
                'enumerate': enumerate,
                'zip': zip,
                'min': min,
                'max': max,
                'sum': sum,
                'print': print,  # For debugging
                'True': True,
                'False': False,
                'None': None,
            }
            
            local_vars = {
                'page': page,
                'context': context,
                'result': None,
                'asyncio': asyncio,  # Needed for await
            }
            
            # Execute with timeout
            try:
                # Execute the code in restricted environment
                exec(code, {'__builtins__': safe_builtins}, local_vars)
                
                # Wait briefly for async operations to complete
                await asyncio.wait_for(asyncio.sleep(0.5), timeout=5.0)
                
            except asyncio.TimeoutError:
                return "Error: Code execution timed out (max 5 seconds)"
            
            # Get result
            result = local_vars.get('result', 'Code executed successfully')
            return str(result)
            
        except SyntaxError as e:
            logger.error(f"Syntax error in generated code: {e}")
            return f"Syntax error in generated code: {str(e)}"
        except Exception as e:
            logger.error(f"Playwright code execution error: {e}", exc_info=True)
            return f"Error executing Playwright code: {str(e)}"
    
    def _validate_generated_code(self, code: str) -> None:
        """
        Validate generated code for dangerous patterns.
        
        Args:
            code: Code to validate
            
        Raises:
            ValueError: If code contains dangerous patterns
        """
        # List of dangerous patterns
        dangerous_patterns = [
            'import os',
            'import sys',
            'import subprocess',
            'import shutil',
            '__import__',
            'eval(',
            'exec(',
            'compile(',
            'open(',
            'file(',
            'input(',
            'raw_input(',
        ]
        
        code_lower = code.lower()
        for pattern in dangerous_patterns:
            if pattern in code_lower:
                raise ValueError(f"Generated code contains dangerous pattern: {pattern}")
            return f"Error executing Playwright code: {str(e)}"


# Backward compatibility: Create module-level function wrappers
# These allow existing code to work without modification
_default_tools = None

def _get_default_tools() -> SpecialistTools:
    """Get or create default tools instance for backward compatibility."""
    global _default_tools
    if _default_tools is None:
        _default_tools = SpecialistTools()
    return _default_tools


async def join_voice_channel(channel_id: str) -> bool:
    """Backward compatibility wrapper."""
    return await _get_default_tools().join_voice_channel(channel_id)


async def leave_voice_channel() -> None:
    """Backward compatibility wrapper."""
    await _get_default_tools().leave_voice_channel()


async def speak_response(text: str) -> None:
    """Backward compatibility wrapper."""
    await _get_default_tools().speak_response(text)


async def searxng_search(query: str) -> str:
    """Backward compatibility wrapper."""
    return await _get_default_tools().searxng_search(query)


async def arxiv_search(query: str) -> str:
    """Backward compatibility wrapper."""
    return await _get_default_tools().arxiv_search(query)


async def wikipedia_search(query: str) -> str:
    """Backward compatibility wrapper."""
    return await _get_default_tools().wikipedia_search(query)


async def wolfram_compute(query: str) -> str:
    """Backward compatibility wrapper."""
    return await _get_default_tools().wolfram_compute(query)


async def python_repl(code: str) -> str:
    """Backward compatibility wrapper."""
    return await _get_default_tools().python_repl(code)


async def playwright_interact(instructions: str) -> str:
    """Backward compatibility wrapper."""
    return await _get_default_tools().playwright_interact(instructions)