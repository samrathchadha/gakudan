#!/usr/bin/env python
"""
Modified contract.py that uses PostgreSQL database instead of files.
"""

import json
import uuid
import logging
import threading
import time
import argparse
import os
import sys
import networkx as nx
from typing import Dict, List, Optional
import concurrent.futures
from google import genai

# Add parent directory to path to import database modules
parent_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, parent_dir)

# Import database modules
import db
from db_manager import db_manager

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PromptGraph:
    def __init__(self, session_id=None, api_key=None):
        self.graph = nx.MultiDiGraph()  # Changed to MultiDiGraph to support multiple edge types
        self.node_map = {}  # id -> node
        self.summarized_map = {}  # summarized_id -> original_parent_id
        self.lock = threading.Lock()  # Thread lock for safe concurrent operations
        self.request_timestamps = []  # Track timestamps for rate limiting
        self.rate_limit_lock = threading.Lock()  # Lock for rate limiting
        self.session_id = session_id
        
        # Initialize Google Generative AI client
        if api_key:
            self.client = genai.Client(api_key=api_key)
        else:
            logger.warning("No API key provided for Google Generative AI")
            self.client = None
        
        if session_id:
            self.load_from_database(session_id)
    
    def load_from_database(self, session_id):
        """Load graph from the database"""
        logger.info(f"Loading graph from database for session {session_id}")
        
        try:
            # Get all nodes for this session
            query = db.Node.select().where(db.Node.session_id == session_id)
            
            # Add nodes to graph
            for node in query:
                node_data = {
                    "id": node.node_id,
                    "prompt": node.prompt,
                    "depth": node.depth
                }
                
                if node.response:
                    node_data["response"] = node.response
                
                # Add any additional attributes from JSONB
                if node.attributes:
                    for key, value in node.attributes.items():
                        node_data[key] = value
                
                # Add to graph and node map
                self.graph.add_node(node.node_id, **node_data)
                self.node_map[node.node_id] = node_data
            
            # Get all links for this session
            link_query = db.Link.select().where(db.Link.session_id == session_id)
            
            # Add links to graph
            for link in link_query:
                # Create attributes dictionary
                attrs = {
                    "edge_type": link.edge_type or "hierarchy"
                }
                
                if link.link_type:
                    attrs["link_type"] = link.link_type
                
                if link.similarity is not None:
                    attrs["similarity"] = link.similarity
                
                # Add any additional attributes from JSONB
                if link.attributes:
                    for key, value in link.attributes.items():
                        attrs[key] = value
                
                # Add edge to graph
                self.graph.add_edge(link.source_id, link.target_id, **attrs)
            
            # Clean up the graph
            self.clean_graph()
            
            logger.info(f"Loaded graph from database: {len(self.graph.nodes)} nodes, {len(self.graph.edges)} edges")
            
        except Exception as e:
            logger.error(f"Error loading graph from database: {e}", exc_info=True)
            raise
    
    def _is_invalid_node(self, node):
        """Check if a node is invalid (missing prompt or response)"""
        if "id" not in node:
            return True
        if "depth" in node:
            if node["depth"] == -1:
                return False
        if "prompt" not in node or not node["prompt"]:
            return True
        if "response" not in node or not node["response"]:
            # Only check response for non-synthesis nodes
            if not node["id"].startswith("S_"):
                return True
        return False
        
    def clean_graph(self):
        """Ensure every node has at least one link or remove it and validate nodes"""
        # First, remove invalid nodes
        invalid_node_ids = []
        for node_id, node_data in self.graph.nodes(data=True):
            if self._is_invalid_node(node_data):
                invalid_node_ids.append(node_id)
                logger.info(f"Marked invalid node for removal: {node_id}")
        
        # Remove invalid nodes
        for node_id in invalid_node_ids:
            self.graph.remove_node(node_id)
            if node_id in self.node_map:
                del self.node_map[node_id]
        
        logger.info(f"Removed {len(invalid_node_ids)} invalid nodes")
        
        # Find orphaned nodes
        node_ids = set(self.graph.nodes())
        linked_nodes = set()
        
        # Find all nodes that are in links
        for u, v in self.graph.edges():
            linked_nodes.add(u)
            linked_nodes.add(v)
        
        # Find orphaned nodes
        orphaned_nodes = node_ids - linked_nodes
        
        # Remove orphaned nodes
        for node_id in orphaned_nodes:
            self.graph.remove_node(node_id)
            if node_id in self.node_map:
                del self.node_map[node_id]
        
        logger.info(f"Removed {len(orphaned_nodes)} orphaned nodes")
        
        # Rebuild node map
        self.node_map = {
            node_id: dict(data) 
            for node_id, data in self.graph.nodes(data=True)
        }
    
    def get_max_depth(self):
        """Find the maximum depth in the graph"""
        max_depth = -1
        for _, data in self.graph.nodes(data=True):
            if "depth" in data and data["depth"] > max_depth:
                max_depth = data["depth"]
        return max_depth
    
    def get_min_depth(self):
        """Find the minimum depth in the graph"""
        min_depth = float('inf')
        for _, data in self.graph.nodes(data=True):
            if "depth" in data and data["depth"] < min_depth:
                min_depth = data["depth"]
        return min_depth if min_depth != float('inf') else 0
        
    def get_nodes_at_depth(self, depth):
        """Get all nodes at a specific depth"""
        return [
            (node_id, data) 
            for node_id, data in self.graph.nodes(data=True) 
            if data.get("depth") == depth
        ]
    
    def get_children(self, node_id):
        """Get all children of a node"""
        return list(self.graph.successors(node_id))
    
    def get_parents(self, node_id):
        """Get all parents of a node"""
        return list(self.graph.predecessors(node_id))
    
    def add_node(self, id, prompt, depth, response=None):
        """Add a new node to the graph with specified ID"""
        node = {
            "id": id,
            "prompt": prompt,
            "depth": depth
        }
        
        if response:
            node["response"] = response
            
        with self.lock:
            self.graph.add_node(id, **node)
            self.node_map[id] = node
            
            # Save to database if session_id is provided
            if self.session_id:
                db_manager.add_node(
                    node_id=id,
                    session_id=self.session_id,
                    prompt=prompt,
                    response=response,
                    depth=depth
                )
            
        return id
    
    def add_link(self, source_id, target_id, edge_type="hierarchy"):
        """Add a link between nodes"""
        with self.lock:
            # Check if this link already exists
            if not self.graph.has_edge(source_id, target_id):
                self.graph.add_edge(
                    source_id, 
                    target_id, 
                    edge_type=edge_type,
                    link_type=edge_type
                )
                
                # Save to database if session_id is provided
                if self.session_id:
                    db_manager.add_link(
                        session_id=self.session_id,
                        source_id=source_id,
                        target_id=target_id,
                        edge_type=edge_type,
                        link_type=edge_type
                    )
    
    def edit_node(self, node_id, prompt=None, response=None, depth=None):
        """Edit an existing node"""
        if node_id not in self.node_map:
            logger.warning(f"Node {node_id} not found")
            return False

        with self.lock:
            node_data = self.graph.nodes[node_id]

            if prompt is not None:
                node_data["prompt"] = prompt

            if response is not None:
                node_data["response"] = response

            if depth is not None:
                node_data["depth"] = depth

            # Update the node map
            self.node_map[node_id] = dict(node_data)
            
            # Save to database if session_id is provided
            if self.session_id:
                db_manager.add_node(
                    node_id=node_id,
                    session_id=self.session_id,
                    prompt=prompt if prompt is not None else node_data.get("prompt"),
                    response=response if response is not None else node_data.get("response"),
                    depth=depth if depth is not None else node_data.get("depth")
                )

        return True
    
    def is_leaf_node(self, node_id):
        """Check if a node is a leaf node (has no children)"""
        return self.graph.out_degree(node_id) == 0
    
    def find_or_create_root_node(self):
        """Find or create a root node at depth -1"""
        # First try to find existing root node at depth -1
        root_nodes = self.get_nodes_at_depth(-1)
        
        if root_nodes:
            # Use the first one if multiple exist
            root_node_id, root_node = root_nodes[0]
            logger.info(f"Found existing root node: {root_node_id}")
            return root_node_id
        
        # Otherwise, create a new root node
        min_depth = self.get_min_depth()
        first_level_nodes = self.get_nodes_at_depth(min_depth)
        
        if not first_level_nodes:
            logger.warning("No nodes found to create a root from")
            return None
        
        # Extract a common theme for the root question
        if len(first_level_nodes) > 0:
            _, sample_node = first_level_nodes[0]
            sample_prompt = sample_node.get('prompt', '')
            # Create a more general prompt for the root
            root_prompt = f"Main Question: {sample_prompt.split('?')[0]}?"
        else:
            root_prompt = "Main Research Question"
            
        root_id = "ROOT"
        self.add_node(
            id=root_id,
            prompt=root_prompt,
            depth=-1,
            response="Root question node"
        )
        
        # Connect root to all nodes at min_depth
        for node_id, _ in first_level_nodes:
            self.add_link(root_id, node_id)
            
        logger.info(f"Created new root node {root_id} at depth -1")
        return root_id
    
    
    def create_synthesis_structure(self):
        """Create a proper hierarchical synthesis structure"""
        # Find the maximum depth in the graph
        max_depth = self.get_max_depth()
        logger.info(f"Maximum depth in graph: {max_depth}")
        
        # Find or create the root node
        root_id = self.find_or_create_root_node()
        
        # Process from deepest nodes to shallowest
        for depth in range(max_depth, -2, -1):
            nodes_at_depth = self.get_nodes_at_depth(depth)
            logger.info(f"Processing {len(nodes_at_depth)} nodes at depth {depth}")
            
            for node_id, node in nodes_at_depth:
                # Skip if this is already a synthesis node
                if node_id.startswith("S_"):
                    continue
                
                # Create a synthesis node for this node
                s_node_id = f"S_{node_id}"
                
                # Synthesis nodes are placed at same depth as their original node
                # This maintains proper hierarchy
                if depth == -1:
                    synthesis_depth = 10000
                synthesis_depth = -depth
                
                # Add synthesis node if it doesn't exist
                if s_node_id not in self.node_map:
                    self.add_node(
                        id=s_node_id,
                        prompt=f"Synthesize insights from responses to: {node['prompt']} into a fluent immersive answer that explores everything talked about",
                        depth=synthesis_depth
                    )
                    
                    # Skip leaf nodes, they don't need synthesis
                    if self.is_leaf_node(node_id):
                        continue
                    # Find all children of this node
                    children = self.get_children(node_id)
                    
                    # Connect children to this synthesis node
                    for child_id in children:
                        # If child has its own synthesis node, connect that instead
                        s_child_id = f"S_{child_id}"
                        if s_child_id in self.node_map and not self.is_leaf_node(child_id):
                            self.add_link(s_child_id, s_node_id)
                        else:
                            # Direct connection if no synthesis node exists
                            self.add_link(child_id, s_node_id)
                    
                    # Map this synthesis back to its original
                    self.summarized_map[s_node_id] = node_id
                    
    def _check_rate_limit(self):
        """Check and enforce rate limit of 30 requests per minute"""
        current_time = time.time()
        
        with self.rate_limit_lock:
            # Remove timestamps older than 60 seconds
            self.request_timestamps = [t for t in self.request_timestamps if current_time - t < 60]
            
            # If we're at the limit, calculate wait time
            if len(self.request_timestamps) >= 30:
                oldest_timestamp = min(self.request_timestamps)
                wait_time = 60 - (current_time - oldest_timestamp) + 0.1  # Add a small buffer
            else:
                wait_time = 0
        
        # Wait outside the lock if needed
        if wait_time > 0:
            logger.info(f"Rate limit reached. Waiting {wait_time:.2f} seconds...")
            time.sleep(wait_time)
            # Recursive call to check again after waiting
            return self._check_rate_limit()
        
        # Add current timestamp and proceed
        with self.rate_limit_lock:
            self.request_timestamps.append(time.time())
        return True
    
    def generate_synthesis(self, node_id, input_nodes, model="gemini-2.0-flash-lite", system_instruction=None):
        """Generate a synthesis from multiple input nodes using Google Generative AI"""
        if not self.client:
            logger.error("Error: No Google Generative AI client initialized. Please provide an API key.")
            return None
        
        # Get the node's prompt
        node = self.node_map[node_id]
        prompt = node["prompt"]
        
        # Collect insights from input nodes
        insights = []
        for input_id in input_nodes:
            if input_id in self.node_map and "response" in self.node_map[input_id]:
                insights.append({
                    "id": input_id,
                    "prompt": self.node_map[input_id]["prompt"],
                    "response": self.node_map[input_id]["response"]
                })
        
        # If no valid insights, skip synthesis
        if not insights:
            logger.warning(f"No valid insights for {node_id}, skipping synthesis")
            return None
        
        # Create the synthesis prompt
        thing = "".join([f"INSIGHT {i+1} from question: {insight['prompt']}\n{insight['response']}\n{'-' * 40}\n" for i, insight in enumerate(insights)])
        
        # Default system instruction if none provided
        if not system_instruction:
            system_instruction = """
            You are synthesizing multiple insights to generate a solution-oriented response.
            Focus on integration, not just summarization. Reference specific points from the insights
            and use direct quotes where helpful. Maintain specific details, provide concrete solutions,
            and keep your response focused on the main question with specific details.
            Do not generalize the information or gentrify the emotion - maintain original tone and specificity.
            Do not show the end user what insights you use to create your synthesis.
            """
        
        synthesis_prompt = f"""
        MAIN QUESTION: {prompt}

        Here are the insights to synthesize into a beautiful answer that includes all info present:

        {'-' * 40}
        {thing}

        Your task:
        1. Don't just summarize - create a new, integrated solution or perspective
        2. Reference specific points from the insights and use direct quotes where helpful
        3. Maintain the specific details and don't generalize the information
        4. Provide concrete, actionable solutions
        5. Keep your response focused on addressing the main question with specific details
        6. DO NOT gentrify the emotion - maintain the original tone and specificity
        7. DO NOT show the end user what insights you use to create your synthesis.
        8. You do not need to quote or show the users what insights/sources you are using btw. LIKE DO NOT USE ANY QUOTES, JUST USE THEM AS CONTEXT AND BEAUTIFULLY SYNTHESIZE THEM. if applicable, use the insight word for word but DO not quote it and use bullet points wherever you need where necessary

        Synthesized response:
        """
        
        # Check and enforce rate limit
        self._check_rate_limit()
        
        try:
            # Generate the synthesis
            response = self.client.models.generate_content(
                model=model,
                contents = system_instruction + synthesis_prompt
            )
            
            response_text = response.text
            
            # Update the node with the response
            self.edit_node(node_id, response=response_text)
            
            return response_text
        except Exception as e:
            logger.error(f"Error generating synthesis for {node_id}: {e}")
            return None
    
    def _process_synthesis_at_depth(self, depth, model="gemini-2.0-flash-lite", system_instruction=None):
        """Process all synthesis nodes at a specific depth"""
        synthesis_nodes = [
            (node_id, node) for node_id, node in self.graph.nodes(data=True)
            if (node_id.startswith("S_")) and node.get("depth") == depth
        ]
        
        logger.info(f"Processing {len(synthesis_nodes)} synthesis nodes at depth {depth}")
        results = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:  # Limit concurrent requests
            # Create a list to keep track of futures and their node IDs
            future_to_node = {}
            
            for node_id, node in synthesis_nodes:
                # Skip nodes that already have responses
                if "response" in node and node["response"]:
                    logger.info(f"Skipping node {node_id} as it already has a response")
                    continue
                
                # Find all nodes that feed into this synthesis node
                input_nodes = []
                for source, target in self.graph.in_edges(node_id):
                    input_nodes.append(source)
                
                if input_nodes:
                    logger.info(f"Queuing synthesis for {node_id} from {len(input_nodes)} inputs")
                    # Submit the job to the executor
                    future = executor.submit(
                        self.generate_synthesis, 
                        node_id, 
                        input_nodes, 
                        model, 
                        system_instruction
                    )
                    future_to_node[future] = node_id
            
            # Process results as they complete
            for future in concurrent.futures.as_completed(future_to_node):
                node_id = future_to_node[future]
                try:
                    result = future.result()
                    if result:
                        results.append((node_id, True))
                        logger.info(f"Successfully synthesized {node_id}")
                    else:
                        results.append((node_id, False))
                except Exception as e:
                    logger.error(f"Error processing {node_id}: {e}")
                    results.append((node_id, False))
        
        successful = sum(1 for _, success in results if success)
        logger.info(f"Completed {successful} out of {len(synthesis_nodes)} synthesis nodes at depth {depth}")
        return successful > 0
    
    def run_synthesis(self, model="gemini-2.0-flash-lite", system_instruction=None):
        """Run the synthesis process multithreaded based on node depth"""
        # First make sure we have the synthesis structure with root and final nodes
        if not any(node_id.startswith("S_") for node_id in self.graph.nodes()):
            logger.info("Creating synthesis structure...")
            self.create_synthesis_structure()
        
        # Get unique synthesis depths, including special nodes
        synthesis_depths = sorted([
            data["depth"] for node_id, data in self.graph.nodes(data=True)
            if (node_id.startswith("S_")) and "depth" in data
        ])
        
        if not synthesis_depths:
            logger.warning("No synthesis nodes found. Make sure to run create_synthesis_structure() first.")
            return
        
        logger.info(f"Found synthesis nodes at depths: {synthesis_depths}")
        
        # Process synthesis nodes from lowest to highest depth (deepest to shallowest)
        processed_depths = set()  # Track which depths we've already processed
        for depth in synthesis_depths:
            if depth in processed_depths:
                logger.info(f"Depth {depth} already processed, skipping")
                continue
                
            logger.info(f"\n--- Starting processing at depth {depth} ---")
            success = self._process_synthesis_at_depth(depth, model, system_instruction)
            if not success:
                logger.warning(f"Warning: No successful syntheses at depth {depth}")
            
            processed_depths.add(depth)  # Mark this depth as processed
    
    def save_to_database(self):
        """Save the final contract data to the database."""
        if not self.session_id:
            logger.warning("No session ID provided, cannot save to database")
            return False
        
        try:
            # Convert the graph to a format suitable for JSON
            result_data = {
                "directed": True,
                "multigraph": True,
                "graph": {},
                "nodes": [],
                "links": []
            }
            
            # Add nodes
            for node_id, data in self.graph.nodes(data=True):
                node_data = {"id": node_id}
                node_data.update(data)
                result_data["nodes"].append(node_data)
            
            # Add links
            for source, target, data in self.graph.edges(data=True):
                link_data = {"source": source, "target": target}
                link_data.update(data)
                result_data["links"].append(link_data)
            
            # Save to the database
            db_manager.save_contract_data(self.session_id, result_data)
            logger.info(f"Contract data saved to database for session {self.session_id}")
            
            # Update session status
            db_manager.update_contract_status(self.session_id, "completed")
            db_manager.update_session_status(self.session_id, "completed")
            
            return True
        except Exception as e:
            logger.error(f"Error saving contract data to database: {e}", exc_info=True)
            db_manager.update_contract_status(self.session_id, "error", str(e))
            return False


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Process expand data into contract result.')
    parser.add_argument('--session-id', required=True, help='Session ID for database operations')
    parser.add_argument('--api-key', required=True, help='API Key for Google Generative AI')
    
    return parser.parse_args()

def main():
    """Main entry point for the script."""
    args = parse_arguments()
    
    # Initialize the database
    db.initialize_db()
    
    # Set session ID and API key
    session_id = args.session_id
    api_key = args.api_key
    
    logger.info(f"Processing contract for session {session_id}")
    
    # Update session status
    db_manager.update_contract_status(session_id, "processing")
    
    try:
        # Initialize the PromptGraph with the session ID and API key
        graph = PromptGraph(session_id=session_id, api_key=api_key)
        
        # Create the synthesis structure
        graph.create_synthesis_structure()
        
        # Run the synthesis process
        graph.run_synthesis(model="gemini-2.0-flash-lite")
        
        # Save the results to the database
        graph.save_to_database()
        
        logger.info(f"Processing complete for session {session_id}")

    except Exception as e:
        logger.error(f"Error processing contract: {e}", exc_info=True)
        db_manager.update_contract_status(session_id, "error", str(e))
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
