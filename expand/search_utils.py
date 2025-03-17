"""
Simple DuckDuckGo search implementation.
"""

import aiohttp
import asyncio
import json
import logging
from typing import List, Dict, Any
from urllib.parse import quote_plus

from rate_limiter import RateLimiter

# Configure logger
logger = logging.getLogger(__name__)

class DDGSearch:
    """Minimal DuckDuckGo search client with rate limiting."""
    
    def __init__(self, rate_limit: int = 20):
        """Initialize with configurable rate limit."""
        self.rate_limiter = RateLimiter(max_requests=rate_limit, time_window=1)
        self.session = None
        
    async def __aenter__(self):
        """Context manager entry."""
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if self.session:
            await self.session.close()
            
    async def search(self, query: str, num_results: int = 3) -> List[Dict[str, Any]]:
        """Perform a search with rate limiting."""
        # Apply rate limiting
        self.rate_limiter.wait_if_needed()
        
        # Create session if needed
        if not self.session:
            self.session = aiohttp.ClientSession()
            
        # Prepare request
        encoded_query = quote_plus(query)
        url = f"https://api.duckduckgo.com/?q={encoded_query}&format=json&pretty=1"
        
        try:
            async with self.session.get(url) as response:
                if response.status != 200:
                    logger.error(f"DDG API error: {response.status}")
                    return []
                    
                data = await response.json(content_type=None)
                
                # Process results
                results = []
                
                # Add instant answer if available
                if data.get("AbstractText"):
                    results.append({
                        "title": data.get("Heading", "Instant Answer"),
                        "content": data.get("AbstractText"),
                        "url": data.get("AbstractURL", "")
                    })
                
                # Add related topics
                for topic in data.get("RelatedTopics", [])[:num_results]:
                    if "Text" in topic and "FirstURL" in topic:
                        results.append({
                            "title": topic.get("Text").split(" - ")[0] if " - " in topic.get("Text", "") else topic.get("Text", ""),
                            "content": topic.get("Text"),
                            "url": topic.get("FirstURL", "")
                        })
                        
                return results[:num_results]
                
        except Exception as e:
            logger.error(f"DDG search error: {e}")
            return []

async def ddg_search(query: str, num_results: int = 3) -> str:
    """Simple function to perform a DDG search and format results as text."""
    async with DDGSearch() as search:
        results = await search.search(query, num_results)
        
        if not results:
            return f"No search results found for: {query}"
            
        # Format results
        formatted = f"Search results for: {query}\n\n"
        
        for i, result in enumerate(results, 1):
            formatted += f"RESULT {i}:\n"
            formatted += f"Title: {result.get('title', 'No title')}\n"
            formatted += f"Content: {result.get('content', 'No content available')}\n"
            if result.get('url'):
                formatted += f"URL: {result.get('url')}\n"
            formatted += "\n"
            
        return formatted.strip()