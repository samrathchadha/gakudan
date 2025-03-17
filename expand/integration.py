"""
Integration module for connecting auto-search with the Expand prompt processor.
"""

import asyncio
import logging
import uuid
from typing import Dict, Any, List

from google import genai

from prompt_processor import GeminiPromptProcessor
from auto_search_node import AutoSearchNode

# Configure logger
logger = logging.getLogger(__name__)

class PromptProcessorExtension:
    """
    Extension for the GeminiPromptProcessor that adds auto-search capabilities.
    """
    
    def __init__(self, api_key: str):
        """Initialize with API key."""
        # Create the base client
        self.client = genai.Client(api_key=api_key)
        
        # Create the original processor
        self.base_processor = GeminiPromptProcessor(api_key)
        
        # Create the auto-search node
        self.search_node = AutoSearchNode(self.client)
        
        # Store the original method
        self._original_generate_thought_approaches = self.base_processor.generate_thought_approaches
        
        # Patch the processor's methods
        self.base_processor.generate_thought_approaches = self._patched_generate_thought_approaches
        self._original_process_sub_prompt = self.base_processor.process_sub_prompt
        self.base_processor.process_sub_prompt = self._patched_process_sub_prompt
    
    def _patched_generate_thought_approaches(self, main_prompt: str) -> List[Dict[str, str]]:
        """
        Patched version of generate_thought_approaches that uses auto-search.
        """
        logger.info(f"Generating thought approaches for: {main_prompt}")
        
        # First, create a search-augmented version of the main prompt
        try:
            # Create event loop for async operations
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            # Count documents in knowledge base
            num_docs = self.base_processor.prompt_graph.knowledge_base.get_document_count()
            
            # Define KB query function
            def kb_query(query: str, top_k: int):
                return self.base_processor.prompt_graph.knowledge_base.query(
                    query,
                    top_k=top_k
                )
            
            # Process with auto-search
            search_result = loop.run_until_complete(
                self.search_node.process_prompt(
                    main_prompt,
                    kb_query,
                    num_docs=num_docs
                )
            )
            
            # Generate an answer
            answer = loop.run_until_complete(
                self.search_node.generate_answer(
                    main_prompt,
                    search_result["context"]
                )
            )
            
            # Generate sub-prompts based on the enriched context and answer
            sub_prompts = loop.run_until_complete(
                self.search_node.generate_sub_prompts(
                    main_prompt,
                    answer
                )
            )
            
            # Close the loop
            loop.close()
            
            # Add the answer to the graph
            answer_node_id = "auto_search_answer"
            self.base_processor.prompt_graph.add_node(
                answer_node_id,
                prompt=main_prompt,
                response=answer,
                depth=0,
                context=search_result["context"],
                method=search_result["method"]
            )
            
            # Connect it to the root
            self.base_processor.prompt_graph.add_edge("root", answer_node_id)
            
            # Convert sub-prompts to the expected format
            approach_list = []
            for sub_prompt in sub_prompts:
                node_id = str(uuid.uuid4())
                self.base_processor.prompt_graph.add_node(
                    node_id,
                    prompt=sub_prompt,
                    depth=1
                )
                self.base_processor.prompt_graph.add_edge("root", node_id)
                
                approach_list.append({
                    "id": node_id,
                    "prompt": sub_prompt
                })
            
            # Ensure we have at least some approaches
            if not approach_list:
                # Fall back to original method
                return self._original_generate_thought_approaches(main_prompt)
            
            return approach_list
            
        except Exception as e:
            logger.error(f"Error in patched generate_thought_approaches: {e}")
            # Fall back to original method
            return self._original_generate_thought_approaches(main_prompt)
    
    def _patched_process_sub_prompt(self, task_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Patched version of process_sub_prompt that adds auto-search for sub-prompts.
        """
        sub_prompt = task_info["sub_prompt"]
        depth = task_info.get("depth", 0)
        
        # Skip auto-search for very simple prompts or deeper levels
        if len(sub_prompt.split()) < 5 or depth > 1:
            return self._original_process_sub_prompt(task_info)
        
        try:
            # Create event loop for async operations
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            # Count documents in knowledge base
            num_docs = self.base_processor.prompt_graph.knowledge_base.get_document_count()
            
            # Define KB query function
            def kb_query(query: str, top_k: int):
                return self.base_processor.prompt_graph.knowledge_base.query(
                    query,
                    top_k=top_k,
                    exclude_ids=[task_info.get("id")] if "id" in task_info else None,
                    current_node_id=task_info.get("id")
                )
            
            # Process with auto-search
            search_result = loop.run_until_complete(
                self.search_node.process_prompt(
                    sub_prompt,
                    kb_query,
                    num_docs=num_docs
                )
            )
            
            # Generate an answer
            answer = loop.run_until_complete(
                self.search_node.generate_answer(
                    sub_prompt,
                    search_result["context"]
                )
            )
            
            # Close the loop
            loop.close()
            
            # Update the task info with the augmented context
            augmented_prompt = f"{sub_prompt}\n\nContext:\n{search_result['context']}"
            
            # Store original for knowledge base
            original_prompt = task_info["sub_prompt"]
            task_info["sub_prompt"] = augmented_prompt
            task_info["auto_search_applied"] = True
            task_info["search_method"] = search_result["method"]
            
            # Process with the original method
            result = self._original_process_sub_prompt(task_info)
            
            # Update the knowledge base with the original prompt + response
            if "response" in result:
                node_id = task_info.get("id", result.get("id"))
                if node_id:
                    self.base_processor.prompt_graph.knowledge_base.add_document(
                        f"{original_prompt}\n{result['response']}",
                        node_id,
                        parent_id=task_info.get("parent_id")
                    )
            
            return result
            
        except Exception as e:
            logger.error(f"Error in patched process_sub_prompt: {e}")
            # Fall back to original method
            return self._original_process_sub_prompt(task_info)
    
    def process_main_prompt(self, main_prompt: str):
        """Process the main prompt with auto-search enhancements."""
        return self.base_processor.process_main_prompt(main_prompt)