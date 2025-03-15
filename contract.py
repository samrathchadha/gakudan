import json
import uuid
import matplotlib.pyplot as plt
import networkx as nx
from typing import Dict, List, Optional
import concurrent.futures
import threading
import time
from google import genai
import time

class PromptGraph:
    def __init__(self, json_file=None, api_key=None):
        self.graph = {
            "directed": True,
            "multigraph": False,
            "graph": {},
            "nodes": [],
            "links": []
        }
        
        self.node_map = {}  # id -> node
        self.summarized_map = {}  # summarized_id -> original_parent_id
        self.lock = threading.Lock()  # Thread lock for safe concurrent operations
        self.request_timestamps = []  # Track timestamps for rate limiting
        self.rate_limit_lock = threading.Lock()  # Lock for rate limiting
        
        # Initialize Google Generative AI client
        if api_key:
            self.client = genai.Client(api_key=api_key)
        else:
            print("Warning: No API key provided for Google Generative AI")
            self.client = None
        
        if json_file:
            self.load_from_json(json_file)
    
    def load_from_json(self, json_file):
        """Load graph from a JSON file"""
        with open(json_file, 'r') as f:
            self.graph = json.load(f)
            
        # Build node map for quick access
        for node in self.graph["nodes"]:
            # Skip invalid nodes
            if self._is_invalid_node(node):
                continue
            self.node_map[node["id"]] = node
        
        # Clean up the graph
        self.clean_graph()
        
        # Save the cleaned graph
        with open(json_file, 'w') as f:
            json.dump(self.graph, f, indent=4)
            
        print(f"Loaded and cleaned graph from {json_file}")
    
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
        valid_nodes = []
        for node in self.graph["nodes"]:
            if not self._is_invalid_node(node):
                valid_nodes.append(node)
            else:
                print(f"Removed invalid node {node.get('id', 'UNKNOWN')}")
        
        self.graph["nodes"] = valid_nodes
        
        # Rebuild node map
        self.node_map = {node["id"]: node for node in self.graph["nodes"]}
        
        # Find orphaned nodes
        node_ids = {node["id"] for node in self.graph["nodes"]}
        linked_nodes = set()
        
        # Find all nodes that are in links
        valid_links = []
        for link in self.graph["links"]:
            # Check if both source and target exist
            if link["source"] in node_ids and link["target"] in node_ids:
                linked_nodes.add(link["source"])
                linked_nodes.add(link["target"])
                valid_links.append(link)
            else:
                print(f"Removed invalid link: {link['source']} -> {link['target']}")
        
        self.graph["links"] = valid_links
        
        # Find orphaned nodes
        orphaned_nodes = node_ids - linked_nodes
        
        # Remove orphaned nodes
        if orphaned_nodes:
            self.graph["nodes"] = [node for node in self.graph["nodes"] if node["id"] not in orphaned_nodes]
            print(f"Removed {len(orphaned_nodes)} orphaned nodes")
            
            # Rebuild node map
            self.node_map = {node["id"]: node for node in self.graph["nodes"]}
    
    def get_max_depth(self):
        """Find the maximum depth in the graph"""
        max_depth = -1
        for node in self.graph["nodes"]:
            if "depth" in node and node["depth"] > max_depth:
                max_depth = node["depth"]
        return max_depth
    
    def get_min_depth(self):
        """Find the minimum depth in the graph"""
        min_depth = float('inf')
        for node in self.graph["nodes"]:
            if "depth" in node and node["depth"] < min_depth:
                min_depth = node["depth"]
        return min_depth if min_depth != float('inf') else 0
        
    def get_nodes_at_depth(self, depth):
        """Get all nodes at a specific depth"""
        return [node for node in self.graph["nodes"] if node.get("depth") == depth]
    
    def get_children(self, node_id):
        """Get all children of a node"""
        return [link["target"] for link in self.graph["links"] if link["source"] == node_id]
    
    def get_parents(self, node_id):
        """Get all parents of a node"""
        return [link["source"] for link in self.graph["links"] if link["target"] == node_id]
    
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
            self.graph["nodes"].append(node)
            self.node_map[id] = node
            
        return id
    
    def add_link(self, source_id, target_id):
        """Add a link between nodes"""
        with self.lock:
            # Check if this link already exists
            if not any(link["source"] == source_id and link["target"] == target_id 
                      for link in self.graph["links"]):
                self.graph["links"].append({
                    "source": source_id,
                    "target": target_id
                })
    
    def edit_node(self, node_id, prompt=None, response=None, depth=None):
        """Edit an existing node"""
        if node_id not in self.node_map:
            print(f"Node {node_id} not found")
            return False

        with self.lock:
            node = self.node_map[node_id]

            if prompt is not None:
                node["prompt"] = prompt

            if response is not None:
                node["response"] = response

            if depth is not None:
                node["depth"] = depth

            # Update the node in the graph["nodes"] list
            for graph_node in self.graph["nodes"]:
                if graph_node["id"] == node_id:
                    graph_node.update(node) # This line updates the node in the graph list!
                    break

        return True
    
    def is_leaf_node(self, node_id):
        """Check if a node is a leaf node (has no children)"""
        return not any(link["source"] == node_id for link in self.graph["links"])
    
    def find_or_create_root_node(self):
        """Find or create a root node at depth -1"""
        # First try to find existing root node at depth -1
        root_nodes = self.get_nodes_at_depth(-1)
        
        if root_nodes:
            # Use the first one if multiple exist
            root_node = root_nodes[0]
            print(f"Found existing root node: {root_node['id']}")
            return root_node['id']
        
        # Otherwise, create a new root node
        min_depth = self.get_min_depth()
        first_level_nodes = self.get_nodes_at_depth(min_depth)
        
        if not first_level_nodes:
            print("No nodes found to create a root from")
            return None
        
        # Extract a common theme for the root question
        if len(first_level_nodes) > 0:
            sample_prompt = first_level_nodes[0].get('prompt', '')
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
        for node in first_level_nodes:
            self.add_link(root_id, node['id'])
            
        print(f"Created new root node {root_id} at depth -1")
        return root_id
    
    
    def create_synthesis_structure(self):
        """Create a proper hierarchical synthesis structure"""
        # Find the maximum depth in the graph
        max_depth = self.get_max_depth()
        print(f"Maximum depth in graph: {max_depth}")
        
        # Find or create the root node
        root_id = self.find_or_create_root_node()
        
        # Process from deepest nodes to shallowest
        for depth in range(max_depth, -2, -1):
            nodes_at_depth = self.get_nodes_at_depth(depth)
            print(f"Processing {len(nodes_at_depth)} nodes at depth {depth}")
            
            for node in nodes_at_depth:
                node_id = node["id"]
                
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
            print(f"Rate limit reached. Waiting {wait_time:.2f} seconds...")
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
            print("Error: No Google Generative AI client initialized. Please provide an API key.")
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
            print(f"No valid insights for {node_id}, skipping synthesis")
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
            print(f"Error generating synthesis for {node_id}: {e}")
            return None
    
    def _process_synthesis_at_depth(self, depth, model="gemini-2.0-flash-lite", system_instruction=None):
        """Process all synthesis nodes at a specific depth"""
        synthesis_nodes = [
            node for node in self.graph["nodes"] 
            if (node["id"].startswith("S_") ) and node.get("depth") == depth
        ]
        
        print(f"Processing {len(synthesis_nodes)} synthesis nodes at depth {depth}")
        results = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:  # Limit concurrent requests
            # Create a list to keep track of futures and their node IDs
            future_to_node = {}
            
            for node in synthesis_nodes:
                s_id = node["id"]
                
                # Skip nodes that already have responses
                if "response" in node and node["response"]:
                    print(f"Skipping node {s_id} as it already has a response")
                    continue
                
                # Find all nodes that feed into this synthesis node
                input_nodes = []
                for link in self.graph["links"]:
                    if link["target"] == s_id:
                        input_nodes.append(link["source"])
                
                if input_nodes:
                    print(f"Queuing synthesis for {s_id} from {len(input_nodes)} inputs")
                    # Submit the job to the executor
                    future = executor.submit(
                        self.generate_synthesis, 
                        s_id, 
                        input_nodes, 
                        model, 
                        system_instruction
                    )
                    future_to_node[future] = s_id
            
            # Process results as they complete
            for future in concurrent.futures.as_completed(future_to_node):
                node_id = future_to_node[future]
                try:
                    result = future.result()
                    if result:
                        results.append((node_id, True))
                        print(f"Successfully synthesized {node_id}")
                    else:
                        results.append((node_id, False))
                except Exception as e:
                    print(f"Error processing {node_id}: {e}")
                    results.append((node_id, False))
        
        successful = sum(1 for _, success in results if success)
        print(f"Completed {successful} out of {len(synthesis_nodes)} synthesis nodes at depth {depth}")
        return successful > 0
    
    def run_synthesis(self, model="gemini-2.0-flash-lite", system_instruction=None):
        """Run the synthesis process multithreaded based on node depth"""
        # First make sure we have the synthesis structure with root and final nodes
        if not any(node["id"].startswith("S_") for node in self.graph["nodes"]):
            print("Creating synthesis structure...")
            self.create_synthesis_structure()
        
        # Get unique synthesis depths, including special nodes
        synthesis_depths = sorted([
            node["depth"] for node in self.graph["nodes"] 
            if (node["id"].startswith("S_")) and "depth" in node
        ])
        
        if not synthesis_depths:
            print("No synthesis nodes found. Make sure to run create_synthesis_structure() first.")
            return
        
        print(f"Found synthesis nodes at depths: {synthesis_depths}")
        
        # Process synthesis nodes from lowest to highest depth (deepest to shallowest)
        processed_depths = set()  # Track which depths we've already processed
        for depth in synthesis_depths:
            if depth in processed_depths:
                print(f"Depth {depth} already processed, skipping")
                continue
                
            print(f"\n--- Starting processing at depth {depth} ---")
            success = self._process_synthesis_at_depth(depth, model, system_instruction)
            if not success:
                print(f"Warning: No successful syntheses at depth {depth}")
            
            processed_depths.add(depth)  # Mark this depth as processed
        
                    
    def visualize(self):
        """Visualize the graph using NetworkX and Matplotlib"""
        G = nx.DiGraph()
        
        # Add nodes with colors based on type
        for node in self.graph["nodes"]:
            node_id = node["id"]
            
            # Determine node type
            if node_id == "ROOT":
                node_type = "root"
            elif node_id.lower() == "s_root":
                node_type = "final"
            elif node_id.startswith("S_"):
                node_type = "summary"
            else:
                node_type = "regular"
            
            # Check if the node has a response (completed)
            has_response = "response" in node and node["response"]
            
            # Set color based on type and completion
            if node_type == "root":
                color = "darkred" if has_response else "red"
            elif node_type == "final":
                color = "purple" if has_response else "plum"
            elif node_type == "summary":
                color = "darkgreen" if has_response else "lightgreen"
            else:
                color = "royalblue" if has_response else "lightblue"
            
            G.add_node(node_id, label=node_id, prompt=node.get("prompt", ""), color=color)
        
        # Add edges
        for link in self.graph["links"]:
            G.add_edge(link["source"], link["target"])
        
        # Get node colors
        node_colors = [G.nodes[n]["color"] for n in G.nodes]
        
        # Create positions using hierarchical layout
        depth_map = {node["id"]: node.get("depth", 0) for node in self.graph["nodes"]}
        pos = nx.multipartite_layout(G, subset_key=lambda x: depth_map.get(x, 0))
        
        # Draw the graph
        plt.figure(figsize=(1500, 1000))
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_size=500, node_color=node_colors)
        
        # Draw edges
        nx.draw_networkx_edges(G, pos, arrows=True)
        
        # Draw labels
        # nx.draw_networkx_labels(G, pos)
        
        plt.title("Prompt Graph Visualization")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig("prompt_graph.png", dpi=300)
        plt.show()
        
        print("Graph visualization saved as prompt_graph.png")
    
    def visualize_hierarchical(self):
        """Visualize the graph as a hierarchical tree"""
        G = nx.DiGraph()

        # Add nodes with colors based on type
        for node in self.graph["nodes"]:
            node_id = node["id"]

            # Determine node type
            if node_id == "ROOT":
                node_type = "root"
            elif node_id.lower() == "s_root":
                node_type = "final"
            elif node_id.startswith("S_"):
                node_type = "summary"
            else:
                node_type = "regular"

            # Check if the node has a response (completed)
            has_response = "response" in node and node["response"]

            # Set color based on type and completion
            if node_type == "root":
                color = "darkred" if has_response else "red"
            elif node_type == "final":
                color = "purple" if has_response else "plum"
            elif node_type == "summary":
                color = "darkgreen" if has_response else "lightgreen"
            else:
                color = "royalblue" if has_response else "lightblue"

            G.add_node(node_id, label=node_id, prompt=node.get("prompt", ""), color=color)

        # Add edges
        for link in self.graph["links"]:
            G.add_edge(link["source"], link["target"])

        # Get node colors
        node_colors = [G.nodes[n]["color"] for n in G.nodes]

        # Create a more tree-like layout
        pos = nx.nx_agraph.graphviz_layout(G, prog="dot")

        # Draw the graph
        plt.figure(figsize=(150, 150)) # Make plot 20 times bigger

        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_size=500, node_color=node_colors)

        # Draw edges
        nx.draw_networkx_edges(G, pos, arrows=True)

        # Draw labels
        nx.draw_networkx_labels(G, pos)

        plt.axis("off")
        plt.tight_layout()
        plt.savefig("prompt_graph_tree.png", dpi=150)

        print("Hierarchical visualization saved as prompt_graph_tree.png")
    
    def save_to_json(self, filename):
        """Save the graph to a JSON file"""
        with open(filename, 'w') as f:
            json.dump(self.graph, f, indent=4)
        print(f"Graph saved to {filename}")


# Example usage
if __name__ == "__main__":
    # Load an existing graph, create the synthesis structure, and run synthesis
    api_key = "AIzaSyBe8kjRD-siRLDQh30xVRka5TmrsAZVwYc"  # Replace with your actual API key
    graph = PromptGraph("./expand.json", api_key=api_key)
    graph.create_synthesis_structure()
    graph.run_synthesis(model="gemini-2.0-flash-lite")
    time.sleep(60)
    graph.visualize_hierarchical()
    graph.save_to_json("contract.json")
