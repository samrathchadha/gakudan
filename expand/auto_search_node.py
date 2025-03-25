"""
Auto-search prompt node that lets the LLM decide how to search.
"""

import asyncio
import logging
import uuid
from typing import Dict, Any, List, Optional, Callable, Coroutine

from google import genai
from google.genai.types import (
    FunctionDeclaration,
    Schema,
    Type,
    Tool,
    GenerateContentConfig,
    ToolConfig,
    FunctionCallingConfig,
    FunctionCallingConfigMode,
    Content
)

from rate_limiter import RateLimiter
from search_utils import ddg_search

# Configure logger
logger = logging.getLogger(__name__)

class AutoSearchNode:
    """
    A prompt node that allows the LLM to decide whether to search externally
    or refine the RAG query.
    """
    
    def __init__(self, client, rate_limiter: Optional[RateLimiter] = None):
        """Initialize with a Gemini client."""
        self.client = client
        self.rate_limiter = rate_limiter or RateLimiter(max_requests=30, time_window=60)
        
        # Define the search function declaration
        self.search_function = FunctionDeclaration(
            name="search_web",
            description="Search the web for current information",
            parameters=Schema(
                type=Type.OBJECT,
                properties={
                    "query": Schema(
                        type=Type.STRING,
                        description="The search query to find information about"
                    )
                },
                required=["query"]
            )
        )
        
        # Create the search tool
        self.search_tool = Tool(
            function_declarations=[self.search_function]
        )
    
    def _prepare_system_prompt(self) -> str:
        """Prepare a system prompt instructing the LLM how to handle the query."""
        return (
            "You are a decision maker for a hierarchical prompt exploration system. "
            "Your task is to decide how to handle the given query.\n\n"
            "You have two options:\n"
            "1. Call the search_web function if the query requires current information "
            "that might not be in the knowledge base, or would benefit from external facts.\n"
            "2. Instead of searching, you can refine the RAG query to better retrieve information "
            "from the existing knowledge base. Reply with your refined query without calling any function.\n\n"
            "DO NOT explain your decision process or add any introductory text. "
            "Either call the function or provide only the refined query text.\n\n"
            "Remember: Search when you need external facts, refine when you need better retrieval."
        )
    
    async def process_prompt(self, 
                             prompt: str, 
                             knowledge_base_query: Callable[[str, int], List[Dict[str, Any]]],
                             num_docs: int = 0) -> Dict[str, Any]:
        """
        Process a prompt, letting the LLM decide whether to search or refine the RAG query.
        
        Args:
            prompt: The user prompt
            knowledge_base_query: Function to query the knowledge base
            num_docs: Number of documents in the knowledge base
            
        Returns:
            Dictionary with processed results
        """
        # Wait for rate limit if needed
        self.rate_limiter.wait_if_needed()
        
        # Prepare config for AUTO mode (let model decide)
        config = GenerateContentConfig(
            temperature=0.2,
            tools=[self.search_tool],
            tool_config=ToolConfig(
                function_calling_config=FunctionCallingConfig(
                    mode=FunctionCallingConfigMode.AUTO  # Let model decide
                )
            )
        )
        
        system_prompt = self._prepare_system_prompt()
        
        try:
            # First request: Let model decide whether to search or refine
            response = self.client.models.generate_content(
                model="gemini-2.0-flash-lite",
                config=config,
                contents=[
                    Content(
                        role="system",
                        parts=[system_prompt]
                    ),
                    Content(
                        role="user", 
                        parts=[prompt]
                    )
                ]
            )
            
            # Check if the model decided to call the search function
            if response.function_calls and response.function_calls[0].name == "search_web":
                # Extract the search query
                search_query = response.function_calls[0].args.get("query", prompt)
                logger.info(f"Performing web search for: {search_query}")
                
                # Perform the search
                search_results = await ddg_search(search_query)
                
                # Determine how many docs to retrieve from knowledge base
                top_k = self._determine_top_k(num_docs)
                
                # Get knowledge base results as well
                kb_results = knowledge_base_query(prompt, top_k)
                
                # Format KB results
                kb_formatted = self._format_kb_results(kb_results, prompt)
                
                # Combine contexts
                context = f"Search results:\n{search_results}\n\nKnowledge base results:\n{kb_formatted}"
                
                return {
                    "id": str(uuid.uuid4()),
                    "prompt": prompt,
                    "refined_prompt": None,
                    "search_query": search_query,
                    "context": context,
                    "method": "search",
                    "search_results": search_results,
                    "kb_results": kb_results
                }
            else:
                # Model chose to refine the RAG query
                refined_query = response.text.strip()
                logger.info(f"Using refined query: {refined_query}")
                
                # Determine how many docs to retrieve
                top_k = self._determine_top_k(num_docs)
                
                # Get knowledge base results with the refined query
                kb_results = knowledge_base_query(refined_query, top_k)
                
                # Format results
                context = self._format_kb_results(kb_results, refined_query)
                
                return {
                    "id": str(uuid.uuid4()),
                    "prompt": prompt,
                    "refined_prompt": refined_query,
                    "search_query": None,
                    "context": context,
                    "method": "refine",
                    "kb_results": kb_results
                }
                
        except Exception as e:
            logger.error(f"Error in auto search node: {e}")
            # Fall back to direct knowledge base query
            top_k = self._determine_top_k(num_docs)
            kb_results = knowledge_base_query(prompt, top_k)
            context = self._format_kb_results(kb_results, prompt)
            
            return {
                "id": str(uuid.uuid4()),
                "prompt": prompt,
                "error": str(e),
                "context": context,
                "method": "fallback",
                "kb_results": kb_results
            }
    
    def _determine_top_k(self, num_docs: int) -> int:
        """Determine how many documents to retrieve based on KB size."""
        if num_docs > 12:
            return 5
        elif num_docs > 5:
            return 4
        else:
            return 3
    
    def _format_kb_results(self, results: List[Dict[str, Any]], query: str) -> str:
        """Format knowledge base results for inclusion in context."""
        if not results:
            return f"No knowledge base results found for: {query}"
        
        formatted = f"Knowledge base results for: {query}\n\n"
        
        for i, result in enumerate(results, 1):
            node_id = result.get("node_id", "Unknown")
            similarity = result.get("similarity", 0)
            text = result.get("text", "No content available")
            
            formatted += f"RESULT {i} (similarity: {similarity:.2f}):\n"
            formatted += f"Source ID: {node_id}\n"
            formatted += f"Content: {text[:300]}{'...' if len(text) > 300 else ''}\n\n"
        
        return formatted.strip()
    
    async def generate_answer(self, prompt: str, context: str) -> str:
        """Generate a final answer based on the enriched context."""
        # Wait for rate limit if needed
        self.rate_limiter.wait_if_needed()
        
        system_prompt = (
            "You are an AI assistant providing helpful, accurate, and concise responses. "
            "Use the provided context to answer the question. "
            "If the context doesn't contain relevant information, say so clearly. "
            "Do not make up information or claim to know things that aren't supported by the context."
        )
        
        try:
            response = self.client.models.generate_content(
                model="gemini-2.0-flash-lite",
                contents=[
                    Content(
                        role="system",
                        parts=[system_prompt]
                    ),
                    Content(
                        role="user",
                        parts=[f"Question: {prompt}\n\nContext:\n{context}\n\nAnswer:"]
                    )
                ]
            )
            
            return response.text.strip()
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return f"Error generating answer: {e}"
    
    async def generate_sub_prompts(self, prompt: str, answer: str, max_sub_prompts: int = 3) -> List[str]:
        """Generate sub-prompts for further exploration."""
        # Wait for rate limit if needed
        self.rate_limiter.wait_if_needed()
        
        system_prompt = (
            "You are generating sub-prompts to further explore a question. "
            "Based on the main question and its answer, suggest specific follow-up questions "
            "that would deepen understanding or explore important related aspects. "
            "Return exactly one sub-prompt per line, with no numbering or additional text. "
            "Each sub-prompt should be a complete, self-contained question."
        )
        
        try:
            response = self.client.models.generate_content(
                model="gemini-2.0-flash-lite",
                contents=[
                    Content(
                        role="system",
                        parts=[system_prompt]
                    ),
                    Content(
                        role="user",
                        parts=[f"Main question: {prompt}\n\nAnswer: {answer}\n\nGenerate {max_sub_prompts} sub-prompts:"]
                    )
                ]
            )
            
            # Split response by lines and filter out empty lines
            sub_prompts = [line.strip() for line in response.text.strip().split('\n') if line.strip()]
            
            # Limit to max_sub_prompts
            return sub_prompts[:max_sub_prompts]
            
        except Exception as e:
            logger.error(f"Error generating sub-prompts: {e}")
            return []