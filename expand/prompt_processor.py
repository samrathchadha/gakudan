"""
Main processor for handling prompt analysis using Google's Generative AI.
Features:
- Rate limiting for API calls
- Concurrent processing with thread pool
- Hierarchical prompt exploration
- Knowledge base integration for context retrieval
- Graph-based tracking of prompt relationships
- Database storage for all data (no file operations)
"""

import threading
import queue
import logging
import uuid
import time
import concurrent.futures
from typing import List, Dict, Any, Tuple
from google import genai
import os
import sys

# Add parent directory to path for database imports if needed
parent_dir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from prompt_graph import PromptGraph
from rate_limiter import RateLimiter
from system_prompts import (
    THOUGHT_GENERATOR_SYSTEM_PROMPT,
    SUB_THOUGHT_GENERATOR_SYSTEM_PROMPT,
    SUB_PROMPT_SYSTEM_INSTRUCTION,
    COMBINER_SYSTEM_PROMPT
)

# Optional database imports - will be used by subclasses
try:
    import db
    from db_manager import db_manager
    HAS_DB = True
except ImportError:
    HAS_DB = False
    logging.warning("Database modules not available. Using in-memory storage only.")

class GeminiPromptProcessor:
    """
    Main processor for handling prompt analysis using Google's Generative AI.
    Features:
    - Rate limiting for API calls
    - Concurrent processing with thread pool
    - Hierarchical prompt exploration
    - Knowledge base integration for context retrieval
    - Graph-based tracking of prompt relationships
    """
    def __init__(self, api_key: str):
        """
        Initialize the prompt processor.
        
        Args:
            api_key: Google API key for Generative AI
        """
        # Initialize Google Generative AI client
        self.client = genai.Client(api_key=api_key)
        
        # Results storage
        self.thread_results = {}
        self.combined_result = None
        self.all_results = []
        
        # Task management
        self.task_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.max_depth = 2  # Limit recursion depth
        self.active_threads = 0
        self.active_threads_lock = threading.Lock()
        self.all_tasks_completed = threading.Event()
        
        # Color management for console output
        self.colors = [
            '\033[31m',  # Red
            '\033[91m',  # Bright Red
            '\033[33m',  # Yellow
            '\033[93m',  # Bright Yellow
            '\033[34m',  # Blue
            '\033[94m',  # Bright Blue
            '\033[35m',  # Magenta
            '\033[95m',  # Bright Magenta
            '\033[36m',  # Cyan
            '\033[96m',  # Bright Cyan
            '\033[32m',  # Green
            '\033[92m',  # Bright Green
            '\033[37m',  # White
            '\033[97m',  # Bright White
        ]
        self.color_lock = threading.Lock()
        self.color_index = 0
        
        # Graph tracking system
        self.prompt_graph = PromptGraph()
        self.prompt_graph.add_node('root', prompt='Main Prompt', depth=-1)
        
        # Rate limiter - 30 requests per minute
        self.rate_limiter = RateLimiter(max_requests=30, time_window=60)
    
    def get_next_color(self) -> str:
        """Get the next color in the rotation for console output."""
        with self.color_lock:
            color = self.colors[self.color_index % len(self.colors)]
            self.color_index += 1
            return color
    
    def combine_responses(self, responses: List[Dict[str, Any]], main_prompt: str) -> Dict[str, Any]:
        """
        Combine all responses with a hierarchical approach.
        
        Args:
            responses: List of response dictionaries
            main_prompt: The original prompt
            
        Returns:
            Combined result dictionary
        """
        # Format responses using our graph structure
        formatted_responses = []
        
        # Helper function to recursively format responses using the graph
        def format_responses_from_graph(node_id='root', depth=0):
            children = self.prompt_graph.get_children(node_id)
            if not children:
                return []
            
            results = []
            for i, child_id in enumerate(children, 1):
                indent = "  " * depth
                node_data = self.prompt_graph.get_node_data(child_id)
                prompt = node_data.get('prompt', 'N/A')
                response_text = node_data.get('response', 'No response')
                
                results.append(f"{indent}Perspective {i} (Level {depth}): {prompt}\n{indent}Response: {response_text}")
                
                # Process children
                child_results = format_responses_from_graph(child_id, depth + 1)
                results.extend(child_results)
            
            return results
        
        # Start with root responses
        formatted_responses = format_responses_from_graph('root')
        
        # Create combination prompt
        combination_prompt = (
            f"Main Prompt: {main_prompt}\n\n"
            f"Hierarchical Responses:\n" + 
            "\n".join(formatted_responses) + 
            "\n\nSynthesize these responses into a comprehensive summary. "
            "Pay attention to the hierarchy of responses, where deeper levels "
            "are more specific explorations of their parent perspectives. "
            "Create a cohesive narrative that incorporates insights from all levels."
        )
        
        try:
            combined_response = self.rate_limited_generate_content(
                model="gemini-2.0-flash-lite",
                config=genai.types.GenerateContentConfig(
                    system_instruction=COMBINER_SYSTEM_PROMPT
                ),
                contents=[combination_prompt]
            )
            
            logging.info("\033[37m[COMBINER] Comprehensive synthesis complete\033[0m")
            
            graph_data = self.prompt_graph.to_dict()
            
            return {
                'main_prompt': main_prompt,
                'combined_response': combined_response.text,
                'graph': graph_data
            }
        except Exception as e:
            logging.error(f"Response combination error: {e}")
            return {'error': str(e), 'main_prompt': main_prompt}
    
    def generate_system_prompts(self, query: str) -> str:
        """
        Generate custom system prompts based on the query.
        
        Args:
            query: The main query
            
        Returns:
            String with modifications made to prompts
        """
        global THOUGHT_GENERATOR_SYSTEM_PROMPT, SUB_PROMPT_SYSTEM_INSTRUCTION, COMBINER_SYSTEM_PROMPT
        
        try:
            # Generate thought generator prompt
            thought_generator_response = self.rate_limited_generate_content(
                model="gemini-2.0-flash-lite",
                config=genai.types.GenerateContentConfig(
                    system_instruction="You are a prompt engineer. Create a system prompt for generating diverse thought approaches. BE VERY CONCISE. SHORT ANSWERS ONLY. LIKE 3 SENTENCES MAX. DO NOT USE HEADINGS AS THE LLM WILL BE DIRECTLY FED YOUR ANSWER"
                ),
                contents=[f"Based on this question: '{query}', create a DIRECTION for team members to be made in. Do not tell what team members should be chosen just make reccomendations and hints. You are writing system instructions for the recruiter of the team"]
            )
            new_thought_generator = thought_generator_response.text
            
            # Generate sub-prompt system instruction
            sub_prompt_response = self.rate_limited_generate_content(
                model="gemini-2.0-flash-lite",
                config=genai.types.GenerateContentConfig(
                    system_instruction="You are a prompt engineer. Create VERY concise system instruction. LIKE 4 SENTENCES MAX. DO NOT USE HEADINGS AS THE LLM WILL BE DIRECTLY FED YOUR ANSWER"
                ),
                contents=[f"Based on this question: '{query}', create a brief system instruction for responding to sub-prompts. Your prompt should be in the direction of HOW to answer, not WHAT to answer. Each system has very specific tasks so this is a guideline of HOW to answer, not what topics to mention. do NOT be generic"]
            )
            new_sub_prompt = sub_prompt_response.text
            
            # Generate combiner prompt
            combiner_response = self.rate_limited_generate_content(
                model="gemini-2.0-flash-lite",
                config=genai.types.GenerateContentConfig(
                    system_instruction="You are a prompt engineer. Create a system prompt for synthesizing multiple perspectives. LIKE 4 SENTENCES MAX. DO NOT USE HEADINGS AS THE LLM WILL BE DIRECTLY FED YOUR ANSWER"
                ),
                contents=[f"Based on this question: '{query}', create a system prompt for combining multiple perspectives into one coherent answer. Your system prompt should be more like on how to plan the answer NOT on what to write"]
            )
            new_combiner = combiner_response.text
            
            if new_thought_generator.strip():
                THOUGHT_GENERATOR_SYSTEM_PROMPT = f"""
                Generate an appropriate number of sub-prompts to solve the given problem and explore the problem from specific perspectives and personality traits that are unique
                {new_thought_generator}
                - the point of the prompts is when their answers are summarized, the main propmt is very well explained, design them with that in mind.
                - make sure each prompt is very unique, dont make the prompts tasks.
                - the goal is to aswer the question cohesively and focus on all essential perspectives.
                - the prompts you give after defining personalities and perspective should be as close to main prompt as possible. phrase the prompt as a question 
                - boldly give personalities
                - the personality should bring unique perspective but shouldn't be TOO creative so you know roles that contribute is the priority
                - personalities could be designed taking into perspective what other personalities may need to complement them
                - do not ask questions that can be answered by yes or no
                - do not include any introductory messages
                - format start:
                    1)personality:prompt
                    2)personality:prompt
                    3)personality:prompt
                    4)personality:prompt
                    ..
                - format end
                do not deviate from format at all 
                """
                
            if new_sub_prompt.strip():
                SUB_PROMPT_SYSTEM_INSTRUCTION += " " + new_sub_prompt
                
            if new_combiner.strip():
                COMBINER_SYSTEM_PROMPT += " " + new_combiner
                
        except Exception as e:
            logging.error(f"Error generating system prompts: {e}")

        return f"Modifications to prompts:\nTHOUGHT_GENERATOR_SYSTEM_PROMPT: {new_thought_generator}\nSUB_PROMPT_SYSTEM_INSTRUCTION: {new_sub_prompt}\nCOMBINER_SYSTEM_PROMPT: {new_combiner}"

    def process_main_prompt(self, main_prompt: str) -> List[Dict[str, Any]]:
        """
        Main orchestration method for prompt processing.
        
        Args:
            main_prompt: The main question/prompt
            
        Returns:
            List of all results
        """
        # Reset state for new run
        self.all_results = []
        self.task_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self.all_tasks_completed = threading.Event()
        self.active_threads = 0
        
        # Reset the graph
        self.prompt_graph = PromptGraph()
        self.prompt_graph.add_node('root', prompt=main_prompt, depth=-1)
        
        # Generate system prompts based on query
        system_prompts_info = self.generate_system_prompts(main_prompt)
        logging.info(system_prompts_info)
        
        # Generate thought approach sub-prompts
        sub_prompts = self.generate_thought_approaches(main_prompt)
        
        # Initialize the task queue with root tasks
        for i, sub_prompt_info in enumerate(sub_prompts):
            color = self.colors[i % len(self.colors)]
            self.task_queue.put({
                'id': sub_prompt_info['id'],
                'sub_prompt': sub_prompt_info['prompt'],
                'parent_id': 'root',
                'depth': 0,
                'color': color
            })
        
        # Create worker threads
        num_workers = min(10, len(sub_prompts) * 2)  # Adjust based on number of sub-prompts
        workers = []
        
        for _ in range(num_workers):
            worker = threading.Thread(target=self.worker)
            worker.daemon = True
            worker.start()
            workers.append(worker)
        
        # Wait for completion or timeout
        max_wait_time = 600  # 10 minutes timeout
        start_time = time.time()
        
        while time.time() - start_time < max_wait_time:
            # Check if all tasks are completed
            with self.active_threads_lock:
                if self.active_threads == 0 and self.task_queue.empty():
                    self.all_tasks_completed.set()
                    time.sleep(3)  # Wait to ensure no new tasks are added
                    if self.active_threads == 0 and self.task_queue.empty():
                        break
            
            # Get results
            while not self.result_queue.empty():
                self.all_results.append(self.result_queue.get())
            
            time.sleep(0.1)
        
        # Signal workers to exit
        self.all_tasks_completed.set()
        
        # Get any remaining results
        while not self.result_queue.empty():
            self.all_results.append(self.result_queue.get())
        
        # Wait for workers to complete
        for worker in workers:
            worker.join(timeout=1)
        
        # Save results to database (implemented by subclasses)
        self.save_results(main_prompt, self.all_results)
        
        return self.all_results
    
    def save_results(self, main_prompt: str, results: List[Dict[str, Any]]) -> bool:
        """
        Save results for later use.
        
        Base implementation does nothing. Override in database-aware subclasses.
        
        Args:
            main_prompt: The main prompt
            results: The results to save
            
        Returns:
            True if successful, False otherwise
        """
        # Base implementation does nothing
        # This will be overridden in database-aware subclasses
        return True
    
    def rate_limited_generate_content(self, model: str, config: Any, contents: List[str]) -> Any:
        """
        Generate content with rate limiting applied.
        
        Args:
            model: Name of the model to use
            config: Configuration for the request
            contents: List of content strings
            
        Returns:
            Response from the API
        """
        # Apply rate limiting
        wait_time = self.rate_limiter.wait_if_needed()
        if wait_time > 0:
            logging.debug(f"Rate limited request, waited {wait_time:.2f} seconds")
            
        # Make the API call
        try:
            return self.client.models.generate_content(
                model=model,
                config=config,
                contents=contents
            )
        except Exception as e:
            logging.error(f"API call failed: {e}")
            # Implement exponential backoff for transient errors
            if "429" in str(e) or "timeout" in str(e).lower():
                retry_wait = min(wait_time * 2 + 1, 30)  # Cap at 30 seconds
                logging.warning(f"Rate limit or timeout error, retrying in {retry_wait}s")
                time.sleep(retry_wait)
                return self.rate_limited_generate_content(model, config, contents)
            raise
        
    def process_sub_prompt(self, task_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single sub-prompt task with enhanced RAG capabilities.
        Creates explicit RAG connections in the graph and avoids parent/sibling nodes.
        """
        sub_prompt = task_info['sub_prompt']
        depth = task_info.get('depth', 0)
        parent_id = task_info.get('parent_id', 'root')
        color = task_info.get('color', '\033[37m')  # Default to white
        node_id = task_info.get('id', str(uuid.uuid4()))
        rag_sources = task_info.get('rag_sources', [])
        
        thread_logger = logging.getLogger(node_id)
        thread_logger.setLevel(logging.INFO)
        
        try:
            with self.active_threads_lock:
                self.active_threads += 1
            
            thread_logger.info(f"{color}Starting processing of task (depth: {depth}, parent: {parent_id})\033[0m")
            
            # Add node to the graph first to establish relationship
            self.prompt_graph.add_node(node_id, prompt=sub_prompt, depth=depth)
            
            # Add hierarchy edge from parent to this node
            if parent_id != node_id:  # Avoid self-loops
                self.prompt_graph.add_edge(parent_id, node_id, edge_type="hierarchy")
            
            # Add RAG connections from existing sources
            for source_id in rag_sources:
                if source_id != node_id:  # Avoid self-loops
                    self.prompt_graph.add_rag_connection(source_id, node_id)
            
            # Query knowledge base for additional context with enhanced relationship awareness
            # This will explicitly avoid retrieving parent and sibling nodes
            kb_results = self.prompt_graph.query_knowledge_base(
                sub_prompt, 
                top_k=2,
                exclude_related=True,  # Explicitly exclude parent and siblings
                current_node_id=node_id
            )
            
            additional_context = ""
            new_rag_sources = []
            
            if kb_results:
                # Format all results into the additional context
                additional_context = "Related information:\n"
                for i, result in enumerate(kb_results, 1):
                    relevant_node = result["node_id"]
                    relevant_text = result["text"]
                    similarity = result["similarity"]
                    additional_context += f"Source {i} (relevance: {similarity:.2f}): {relevant_text}\n\n"
                    
                    # Add to RAG sources and create explicit connection
                    new_rag_sources.append(relevant_node)
                    
                    # Create explicit RAG connection with similarity score
                    if relevant_node != node_id:  # Avoid self-loops
                        self.prompt_graph.add_rag_connection(relevant_node, node_id, similarity)
                    
                    thread_logger.info(f"{color}Found relevant information from node {relevant_node} (similarity: {similarity:.2f})\033[0m")
            
            # Generate response using Gemini with rate limiting
            response = self.rate_limited_generate_content(
                model="gemini-2.0-flash-lite",
                config=genai.types.GenerateContentConfig(
                    system_instruction=SUB_PROMPT_SYSTEM_INSTRUCTION
                ),
                contents=[additional_context + sub_prompt]
            )
            response_text = response.text

            # Update the node with the response and RAG results
            self.prompt_graph.add_node(node_id, response=response_text, rag_results=kb_results)
            
            # Get the current prompt text
            prompt_text = self.prompt_graph.get_node_data(node_id).get('prompt', '')
            
            # If prompt is missing (which shouldn't happen), set it
            if not prompt_text:
                prompt_text = sub_prompt
                self.prompt_graph.add_node(node_id, prompt=prompt_text)
                
            combined_text = f"{prompt_text}\n{response_text}"
            
            # Update knowledge base with parent relationship for better filtering
            # This is crucial for our enhanced relationship awareness
            self.prompt_graph.knowledge_base.add_document(
                combined_text, 
                node_id, 
                parent_id=parent_id
            )

            # Check if we should generate sub-prompts for deeper exploration
            if depth < self.max_depth:
                sub_prompt_response = self.rate_limited_generate_content(
                    model="gemini-2.0-flash-lite",
                    config=genai.types.GenerateContentConfig(
                        system_instruction=SUB_THOUGHT_GENERATOR_SYSTEM_PROMPT
                    ),
                    contents=[
                        f"Previous response: {response_text}",
                        "Generate a maximum of three sub prompts, reply with JUST NO if you dont need sub prompts at all"
                    ]
                )
                r2 = sub_prompt_response.text
                if len(r2.split("\n")) == 1 and "no" in r2.lower():
                    print(f"Thread {node_id} did not require further analysis")
                else:
                    for new_prompt in r2.split("\n"):
                        if new_prompt.strip():
                            # Include RAG information when adding to queue
                            enhanced_prompt = response_text + "\n" + new_prompt
                            
                            # Combine existing and new RAG sources
                            all_rag_sources = list(set(rag_sources + new_rag_sources))
                            
                            if kb_results:
                                similarity_info = "\n".join([f"Similarity source {r['node_id']}: {r['similarity']:.2f}" for r in kb_results])
                                enhanced_prompt = similarity_info + "\n" + enhanced_prompt
                                
                            self.add_to_queue(enhanced_prompt, node_id, depth)
            
            result = {
                'id': node_id,
                'parent_id': parent_id,
                'sub_prompt': sub_prompt,
                'response': response_text,
                'depth': depth,
                'color': color,
                'rag_sources': list(set(rag_sources + new_rag_sources)),
                'rag_results': kb_results
            }
            
            print(f"{color}[DEPTH: {depth}] {sub_prompt}"+ "\n\n" + response_text+"\033[0m")
            
            return result
        except Exception as e:
            thread_logger.error(f"{color}Error processing sub-prompt: {e}\033[0m")
            # Update node with error
            self.prompt_graph.add_node(node_id, error=str(e))
            
            return {
                'id': node_id,
                'parent_id': parent_id,
                'sub_prompt': sub_prompt,
                'error': str(e),
                'depth': depth,
                'color': color,
                'rag_sources': rag_sources
            }
        finally:
            with self.active_threads_lock:
                self.active_threads -= 1
                if self.active_threads == 0 and self.task_queue.empty():
                    self.all_tasks_completed.set()
                    
    def add_to_queue(self, sub_prompt: str, parent_id: str, depth: int = 0) -> bool:
        """
        Add a new task to the processing queue with enhanced RAG information.
        Ensures all RAG connections are explicitly created.
        """
        if depth >= self.max_depth:
            logging.warning(f"Max depth limit reached for task: {sub_prompt}")
            return False
        
        color = self.get_next_color()
        new_id = str(uuid.uuid4())
        logging.info(f"{color}Adding new task to queue from parent {parent_id}: {sub_prompt}\033[0m")
        
        # Get RAG results from the parent node
        parent_node_data = self.prompt_graph.get_node_data(parent_id)
        rag_results = parent_node_data.get('rag_results', [])
        
        # Format RAG context for the new prompt
        rag_context = ""
        rag_sources = []
        
        if rag_results:
            rag_context = "Previous relevant information:\n"
            for i, result in enumerate(rag_results, 1):
                relevant_node = result["node_id"]
                relevant_text = result["text"]
                similarity = result["similarity"]
                rag_context += f"Source {i} (relevance: {similarity:.2f}): {relevant_text}\n\n"
                rag_sources.append(relevant_node)
        
        # Extract the actual prompt from multi-line prompts
        if len(sub_prompt.split("\n")) > 1:
            prompt_text = sub_prompt.split("\n")[-1]
        else:
            prompt_text = sub_prompt
            
        # Add to prompt graph with explicit RAG context
        self.prompt_graph.add_node(new_id, prompt=prompt_text, depth=depth, color=color, 
                                rag_context=rag_context if rag_context else None)
        
        # Create hierarchical edge from parent
        self.prompt_graph.add_edge(parent_id, new_id, edge_type="hierarchy")
        
        # Create explicit RAG connections for each source
        for source_id in rag_sources:
            if source_id != new_id:  # Avoid self-loops
                similarity = next((r["similarity"] for r in rag_results if r["node_id"] == source_id), None)
                self.prompt_graph.add_rag_connection(source_id, new_id, similarity)
        
        # Add to processing queue with enhanced RAG context
        enhanced_sub_prompt = rag_context + sub_prompt if rag_context else sub_prompt
        
        self.task_queue.put({
            'id': new_id,
            'sub_prompt': enhanced_sub_prompt,
            'parent_id': parent_id,
            'depth': depth + 1,
            'color': color,
            'rag_sources': rag_sources
        })
        
        return True

    def worker(self):
        """Worker function that processes tasks from the queue."""
        while True:
            try:
                task = self.task_queue.get(block=False)
                result = self.process_sub_prompt(task)
                self.result_queue.put(result)
                self.task_queue.task_done()
            except queue.Empty:
                # Check if we should exit
                if self.all_tasks_completed.is_set():
                    time.sleep(1)
                    if self.all_tasks_completed.is_set():
                        break
                # No tasks available, wait a bit
                time.sleep(0.1)
            except Exception as e:
                logging.error(f"Worker error: {e}")
                self.task_queue.task_done()
    
    def generate_thought_approaches(self, main_prompt: str) -> List[Dict[str, str]]:
        """
        Generate multiple thought approaches for the main prompt.
        
        Args:
            main_prompt: The main question/prompt
            
        Returns:
            List of approaches with IDs and prompts
        """
        try:
            response = self.rate_limited_generate_content(
                model="gemini-2.0-flash-lite",
                config=genai.types.GenerateContentConfig(
                    system_instruction=THOUGHT_GENERATOR_SYSTEM_PROMPT
                ),
                contents=[f"main prompt: how to fix {main_prompt}\ngenerate unique sub-prompts. one per line "]
            )
            approaches_content = response.text
            
            # Print markdown formatted content
            print("\n\n")
            print(approaches_content)
            print("\n\n")
            
            approach_list = []
            for line in approaches_content.split('\n'):
                if line.strip():
                    node_id = str(uuid.uuid4())
                    # Add to our graph
                    self.prompt_graph.add_node(node_id, prompt=line.strip(), depth=0)
                    self.prompt_graph.add_edge('root', node_id)
                    
                    approach_list.append({
                        'id': node_id,
                        'prompt': line.strip()
                    })
            
            return approach_list
        except Exception as e:
            logging.error(f"Error generating thought approaches: {e}")
            # Return some fallback approaches with IDs for testing
            fallbacks = ["Approach A", "Approach B", "Approach C", "Approach D"]
            approach_list = []
            for fallback in fallbacks:
                node_id = str(uuid.uuid4())
                self.prompt_graph.add_node(node_id, prompt=fallback, depth=0)
                self.prompt_graph.add_edge('root', node_id)
                approach_list.append({
                    'id': node_id,
                    'prompt': fallback
                })
            return approach_list


class DatabasePromptProcessor(GeminiPromptProcessor):
    """
    Extended prompt processor that uses database for all storage.
    """
    def __init__(self, api_key: str, session_id: str = None):
        """
        Initialize with database support.
        
        Args:
            api_key: Google API key
            session_id: Optional session ID for database operations
        """
        super().__init__(api_key)
        self.session_id = session_id
        
        # Ensure we have database support
        if not HAS_DB:
            raise ImportError("Database modules not available. Cannot use DatabasePromptProcessor.")
            
        # Initialize database connection
        db.initialize_db()
        
        # Use a database-aware PromptGraph if available
        if hasattr(db_manager, 'get_graph_for_session') and self.session_id:
            self.prompt_graph = db_manager.get_graph_for_session(self.session_id)
        
        logging.info(f"DatabasePromptProcessor initialized with session {session_id}")
    
    def save_results(self, main_prompt: str, results: List[Dict[str, Any]]) -> bool:
        """
        Save results to database.
        
        Args:
            main_prompt: The main prompt
            results: The results to save
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create a session if we don't have one
            if not self.session_id:
                self.session_id, client_id = db_manager.create_session(main_prompt, "local_run")
                logging.info(f"Created new session {self.session_id} for database storage")
            
            # Format the data for database
            data = self.prompt_graph.format_for_database()
            
            # Save contract data
            db_manager.save_contract_data(self.session_id, data)
            
            # Update session status
            db_manager.update_session_status(self.session_id, "completed")
            
            logging.info(f"Results saved to database for session {self.session_id}")
            return True
        except Exception as e:
            logging.error(f"Error saving results to database: {e}")
            return False
    
    def process_sub_prompt(self, task_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Override to ensure database updates.
        """
        # Process normally
        result = super().process_sub_prompt(task_info)
        
        # Save intermediate results to database
        if self.session_id:
            try:
                node_id = result['id']
                prompt = result.get('sub_prompt')
                response = result.get('response')
                depth = result.get('depth')
                
                # Create a copy of attributes without key fields for JSONB storage
                attributes = dict(result)
                for key in ['id', 'sub_prompt', 'response', 'depth', 'parent_id']:
                    if key in attributes:
                        attributes.pop(key, None)
                
                # Add to database
                db_manager.add_node(
                    node_id=node_id,
                    session_id=self.session_id,
                    prompt=prompt,
                    response=response,
                    depth=depth,
                    attributes=attributes
                )
                
                # Add link to parent
                parent_id = result.get('parent_id')
                if parent_id and parent_id != node_id:
                    db_manager.add_link(
                        session_id=self.session_id,
                        source_id=parent_id,
                        target_id=node_id,
                        edge_type="hierarchy"
                    )
                
                # Add RAG connections
                rag_sources = result.get('rag_sources', [])
                for source_id in rag_sources:
                    if source_id != node_id:
                        # Try to get similarity score
                        similarity = None
                        rag_results = result.get('rag_results', [])
                        for rag_result in rag_results:
                            if rag_result.get('node_id') == source_id:
                                similarity = rag_result.get('similarity')
                                break
                        
                        db_manager.add_link(
                            session_id=self.session_id,
                            source_id=source_id,
                            target_id=node_id,
                            edge_type="rag",
                            similarity=similarity
                        )
                
            except Exception as e:
                logging.error(f"Error updating database during sub-prompt processing: {e}")
        
        return result
