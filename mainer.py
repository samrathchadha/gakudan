from rich.console import Console
from rich.markdown import Markdown
import threading
import queue
import logging
import colorlog
import uuid
from typing import List, Dict, Any, Optional, Union
import networkx as nx
import matplotlib.pyplot as plt
import json
from google import genai
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import time
import concurrent.futures
import os

# Configure Colorful Logging
handler = colorlog.StreamHandler()
handler.setFormatter(colorlog.ColoredFormatter(
    '%(log_color)s[%(levelname)s][%(threadName)s] %(message)s',
    log_colors={
        'DEBUG': 'cyan',
        'INFO': 'green',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'red,bg_white',
    }
))
logging.basicConfig(level=logging.INFO, handlers=[handler])
logger = logging.getLogger(__name__)

# Rich console for improved output formatting
console = Console()

def mprint(string):
    """Print markdown-formatted text with clear separation."""
    print("\n\n")
    console.print(Markdown(string))
    print("\n\n")

# System prompts - will be customized per query
THOUGHT_GENERATOR_SYSTEM_PROMPT = """Generate an appropriate number of sub-prompts to solve the given problem and explore the problem from specific perspectives and personality traits that are unique
- the point of the prompts is when their answers are summarized, the main propmt is very well explained, design them with that in mind.
- make sure each prompt is very unique, dont make the prompts tasks.
- the goal is to aswer the question cohesively and focus on all essential perspectives.
- the prompts you give after defining personalities and perspective should be as close to main prompt as possible. phrase the prompt as a question 
- boldly give personalities
- the personality should bring unique perspective but shouldn't be TOO creative so you know roles that contribute is the priority
- personalities could be designed taking into perspective what other personalities may need to complement them
- do not ask questions that can be answered by yes or no
- ensure there is a new line between points 
- do not include any introductory messages
- DO NOT GIVE THEM NAMES
- format start:
    1)personality:prompt
    2)personality:prompt
    3)personality:prompt
    4)personality:prompt
    ..
- format end
do not deviate from format at all 
"""

SUB_THOUGHT_GENERATOR_SYSTEM_PROMPT = """
IF you feel like this needs further deliberation, you can simply respond with what personalities should deliberate it and what prompt they should use. they will already be given your current answer, you can ask them further questoins on it
PLEASE NOTE: there is no need to generate them, if you think the depth of exploration here is sufficient you are not required to further elaborate
Generate a MAXIMUM of 3 sub-prompts to solve the given problem and explore the problem from specific perspectives and personality traits that are unique
- the point of the prompts is when their answers are summarized, the main propmt is very well explained, design them with that in mind.
- make sure each prompt is very unique, dont make the prompts tasks.
- the goal is to aswer the question cohesively and focus on all essential perspectives.
- the prompts you give after defining personalities and perspective should be as close to main prompt as possible. phrase the prompt as a question 
- boldly give personalities
- DO NOT GIVE THEM NAMES
- the personality should bring unique perspective but shouldn't be TOO creative so you know roles that contribute is the priority
- personalities could be designed taking into perspective what other personalities may need to complement them
- do not ask questions that can be answered by yes or no
- ensure there is a new line between points if given
- do not include any introductory messages
- format start:
    1)personality:prompt
    2)personality:prompt
    ..
- format end
do not deviate from format at all 
"""

SUB_PROMPT_SYSTEM_INSTRUCTION = "You are an AI assistant. Provide EXTREMELY CONCISE and PRECISE responses. Do not answer with yes or no, build unique content. Provide strong tangible solutions to the task at hand. do not include any introductory messages or conclusive messages"

COMBINER_SYSTEM_PROMPT = 'You have been given several perspectives from many different people.INCLUDE EVERY GOOD POINT ANYONE HAS MENTIONED. You will now create a formal plan with all the work everyone has done. Dont summarize, synthesize a coherent synchronous answer that is a final answer to the original prompt, not just a summary of the answers you have. DO NOT GIVE ANY TOPICS. JUST GIVE IT INSTRUCTIONS ON HOW TO DO NOT WHAT TO DO IT ON. do not include any introductory messages or conclusive messages. You do not need to be concise.'

class RateLimiter:
    """
    Implements rate limiting for API calls with a sliding window approach.
    Ensures no more than max_requests are made within the time_window.
    """
    def __init__(self, max_requests: int = 30, time_window: int = 60):
        """
        Initialize rate limiter with configurable parameters.
        
        Args:
            max_requests: Maximum number of requests allowed in the time window
            time_window: Time window in seconds
        """
        self.max_requests = max_requests
        self.time_window = time_window
        self.request_timestamps = []
        self.lock = threading.Lock()
        logger.info(f"Rate limiter initialized: {max_requests} requests per {time_window} seconds")
    
    def wait_if_needed(self) -> float:
        """
        Check if rate limit is reached and wait if necessary.
        
        Returns:
            float: Time waited in seconds (0 if no waiting was needed)
        """
        current_time = time.time()
        wait_time = 0
        
        with self.lock:
            # Remove timestamps older than time_window
            self.request_timestamps = [
                t for t in self.request_timestamps 
                if current_time - t < self.time_window
            ]
            
            # If we're at the limit, calculate wait time
            if len(self.request_timestamps) >= self.max_requests:
                oldest_timestamp = min(self.request_timestamps)
                wait_time = self.time_window - (current_time - oldest_timestamp) + 0.1  # Small buffer
        
        # Wait outside the lock if needed
        if wait_time > 0:
            logger.debug(f"Rate limit reached. Waiting {wait_time:.2f} seconds...")
            time.sleep(wait_time)
            # Recursive call to check again after waiting
            return wait_time + self.wait_if_needed()
        
        # Add current timestamp and proceed
        with self.lock:
            self.request_timestamps.append(time.time())
        
        return wait_time
    
    def __call__(self, func):
        """
        Decorator for rate-limiting functions.
        
        Args:
            func: The function to rate-limit
            
        Returns:
            Wrapped function that respects rate limits
        """
        def wrapper(*args, **kwargs):
            self.wait_if_needed()
            return func(*args, **kwargs)
        return wrapper

class KnowledgeBase:
    """
    Vector database for semantic search of previous responses.
    Uses TF-IDF and cosine similarity for retrieval with improved uniqueness.
    """
    def __init__(self):
        self.documents = []
        self.vectorizer = TfidfVectorizer(
            min_df=1, 
            max_df=0.9,             # Ignore terms that appear in >90% of documents
            stop_words='english',    # Remove common English stopwords
            ngram_range=(1, 2)       # Include unigrams and bigrams
        )
        self.vectors = None
        self.lock = threading.Lock()
        self.node_id_to_index = {}   # Maps node_ids to their indices in documents
        
    def add_document(self, text: str, node_id: str):
        """
        Add document to knowledge base and update vector index.
        If node already exists, update it instead of creating a duplicate.
        """
        with self.lock:
            # Check if node already exists
            if node_id in self.node_id_to_index:
                # Update existing document
                index = self.node_id_to_index[node_id]
                self.documents[index]["text"] = text
                logger.debug(f"Updated existing document for node {node_id}")
            else:
                # Add new document
                self.documents.append({"text": text, "node_id": node_id})
                self.node_id_to_index[node_id] = len(self.documents) - 1
                logger.debug(f"Added new document for node {node_id}")
            
            # Recompute vectors if we have documents
            if self.documents:
                texts = [doc["text"] for doc in self.documents]
                self.vectors = self.vectorizer.fit_transform(texts)
    
    def query(self, query_text: str, top_k: int = 1) -> List[Dict[str, Any]]:
        """
        Find most similar documents to the query, ensuring unique results.
        
        Args:
            query_text: The search query
            top_k: Number of results to return
            
        Returns:
            List of matches with text, node_id and similarity score
        """
        with self.lock:
            if not self.documents or len(self.documents) <= 5:
                return []
            
            query_vector = self.vectorizer.transform([query_text])
            similarities = cosine_similarity(query_vector, self.vectors).flatten()
            
            # Track used node_ids to avoid duplicates
            used_nodes = set()
            results = []
            
            # Get indices sorted by similarity (descending)
            sorted_indices = similarities.argsort()[::-1]
            
            # Add top results ensuring uniqueness
            for idx in sorted_indices:
                node_id = self.documents[idx]["node_id"]
                similarity = similarities[idx]
                
                # Skip if we've already used this node or if similarity is too low
                if node_id in used_nodes or similarity < 0.1:
                    continue
                    
                used_nodes.add(node_id)
                results.append({
                    "text": self.documents[idx]["text"],
                    "node_id": node_id,
                    "similarity": similarity
                })
                
                # Stop when we have enough unique results
                if len(results) >= top_k:
                    break
            
            # Log information about the query results
            if results:
                logger.debug(f"Query found {len(results)} unique results from {len(self.documents)} documents")
                for i, result in enumerate(results):
                    logger.debug(f"Result {i+1}: node={result['node_id']}, similarity={result['similarity']:.2f}")
            else:
                logger.debug(f"Query found no results from {len(self.documents)} documents")
                
            return results
            
    def get_document_count(self) -> int:
        """Get the number of documents in the knowledge base."""
        with self.lock:
            return len(self.documents)
            
    def clear(self):
        """Clear all documents from the knowledge base."""
        with self.lock:
            self.documents = []
            self.vectors = None
            self.node_id_to_index = {}

class PromptGraph:
    """
    Graph data structure for tracking relationships between prompts.
    Includes visualization capabilities and knowledge base integration.
    """
    def __init__(self):
        self.graph = nx.DiGraph()
        self.lock = threading.Lock()
        self.knowledge_base = KnowledgeBase()
        
    def add_node(self, node_id: str, **attributes):
        """
        Add or update a node in the graph.
        
        Args:
            node_id: Unique identifier for the node
            **attributes: Node attributes (prompt, response, depth, etc.)
        """
        with self.lock:
            if node_id in self.graph:
                # Update existing node attributes
                for key, value in attributes.items():
                    self.graph.nodes[node_id][key] = value
            else:
                # Add new node
                self.graph.add_node(node_id, **attributes)
                
            # If response is provided, add to knowledge base
            if 'response' in attributes and 'prompt' in self.graph.nodes[node_id]:
                prompt_text = self.graph.nodes[node_id]['prompt']
                response_text = attributes['response']
                combined_text = f"{prompt_text}\n{response_text}"
                self.knowledge_base.add_document(combined_text, node_id)
    
    def add_edge(self, parent_id: str, child_id: str, edge_attrs: Dict = None):
        """
        Add an edge between nodes, maintaining proper depth hierarchy.
        
        Args:
            parent_id: ID of the parent node
            child_id: ID of the child node
            edge_attrs: Optional attributes for the edge (e.g., origins)
        """
        with self.lock:
            # Add edge with attributes if provided
            if edge_attrs:
                self.graph.add_edge(parent_id, child_id, **edge_attrs)
            else:
                self.graph.add_edge(parent_id, child_id)
            
            # Fix depth: ensure child depth is greater than parent
            if parent_id in self.graph and child_id in self.graph:
                parent_depth = self.graph.nodes[parent_id].get('depth', 0)
                child_depth = self.graph.nodes[child_id].get('depth', 0)
                
                if child_depth <= parent_depth:
                    self.graph.nodes[child_id]['depth'] = parent_depth + 1

    def get_children(self, node_id: str) -> List[str]:
        """Get all child nodes of the given node."""
        with self.lock:
            return list(self.graph.successors(node_id))
    
    def get_parent(self, node_id: str) -> Optional[str]:
        """Get parent node of the given node (if any)."""
        with self.lock:
            parents = list(self.graph.predecessors(node_id))
            return parents[0] if parents else None
    
    def query_knowledge_base(self, query_text: str, top_k: int = 1) -> List[Dict[str, Any]]:
        """
        Query the knowledge base for relevant information.
        
        Args:
            query_text: The search query text
            top_k: Number of results to return (default: 1)
            
        Returns:
            List of matches with text, node_id and similarity score
        """
        return self.knowledge_base.query(query_text, top_k)
    
    def get_node_data(self, node_id: str) -> Dict[str, Any]:
        """Get all attributes for a specific node."""
        with self.lock:
            if node_id in self.graph:
                return dict(self.graph.nodes[node_id])
            return {}
    
    def get_nodes_at_depth(self, depth: int) -> List[str]:
        """Get all nodes at a specific depth level."""
        with self.lock:
            return [
                node for node, attrs in self.graph.nodes(data=True) 
                if attrs.get('depth') == depth
            ]
    
    def visualize_hierarchical(self, output_file: str = "prompt_graph_hierarchical.png"):
        """
        Visualize the graph as a strict hierarchical tree with proper depth relationships.
        Ensures that nodes are organized by their depth levels.
        """
        with self.lock:
            # Create a directed graph for visualization
            G = nx.DiGraph()
            
            # First pass: Add nodes with depth info and colors
            node_colors = []
            node_sizes = []
            labels = {}
            
            for node_id, attrs in self.graph.nodes(data=True):
                # Get node depth
                depth = attrs.get('depth', 0)
                
                # Determine node color based on depth
                if node_id == 'root' or depth == -1:
                    color = "darkred"
                    size = 800
                elif depth == 0:
                    color = "orange"
                    size = 600
                elif depth == 1:
                    color = "green"
                    size = 500
                elif depth == 2:
                    color = "blue"
                    size = 400
                else:
                    color = "purple"
                    size = 300
                
                # Create short label from prompt
                prompt = attrs.get('prompt', '')
                if prompt:
                    short_label = (prompt[:25] + '...') if len(prompt) > 25 else prompt
                else:
                    short_label = node_id
                    
                # Add node to graph
                G.add_node(node_id, depth=depth)
                node_colors.append(color)
                node_sizes.append(size)
                labels[node_id] = short_label
            
            # Second pass: Add edges with hierarchy enforcement
            for u, v, attrs in self.graph.edges(data=True):
                # Get depths
                u_depth = self.graph.nodes[u].get('depth', 0)
                v_depth = self.graph.nodes[v].get('depth', 0)
                
                # Only add edges that maintain hierarchy (lower depth to higher depth)
                if u_depth <= v_depth:
                    G.add_edge(u, v)
                else:
                    G.add_edge(v, u)  # Reverse the edge
            
            # Create hierarchical layout
            try:
                # Try to use graphviz for better hierarchical layout
                pos = nx.nx_agraph.graphviz_layout(G, prog="dot")
            except:
                # Fall back to networkx's built-in layout
                logger.warning("Graphviz not available, using alternative layout")
                
                # Custom positioning based on depth
                pos = {}
                depths = {}
                
                # Group nodes by depth
                for node, data in G.nodes(data=True):
                    depth = data.get('depth', 0)
                    if depth not in depths:
                        depths[depth] = []
                    depths[depth].append(node)
                
                # Position nodes by depth (y-axis) and evenly spread on x-axis
                sorted_depths = sorted(depths.keys())
                for i, depth in enumerate(sorted_depths):
                    nodes = depths[depth]
                    y = -i * 2  # Vertical position by depth
                    
                    for j, node in enumerate(nodes):
                        # Spread nodes horizontally
                        width = max(6, len(nodes) * 0.5)
                        x = (j - len(nodes)/2) * width / max(1, len(nodes)-1)
                        pos[node] = (x, y)
            
            # Create the figure with sufficient size
            plt.figure(figsize=(20, 16))
            
            # Draw the nodes
            nx.draw_networkx_nodes(G, pos, 
                                node_size=node_sizes, 
                                node_color=node_colors,
                                alpha=0.9)
            
            # Draw the edges with arrows
            nx.draw_networkx_edges(G, pos, 
                                arrows=True, 
                                arrowsize=15, 
                                width=1.5, 
                                alpha=0.7)
            
            # Draw the labels
            nx.draw_networkx_labels(G, pos, 
                                labels=labels,
                                font_size=9, 
                                font_weight='bold',
                                font_color='black')
            
            # Add legend
            import matplotlib.patches as mpatches
            
            legend_elements = [
                mpatches.Patch(color="darkred", label="Root Node (Depth -1)"),
                mpatches.Patch(color="orange", label="Depth 0"),
                mpatches.Patch(color="green", label="Depth 1"),
                mpatches.Patch(color="blue", label="Depth 2"),
                mpatches.Patch(color="purple", label="Depth 3+")
            ]
            
            plt.legend(handles=legend_elements, loc='upper right')
            
            # Remove axes and set title
            plt.axis('off')
            plt.title("Hierarchical Prompt Graph", fontsize=16)
            plt.tight_layout()
            
            # Save the figure
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Hierarchical visualization saved to {output_file}")
            
    def visualize(self, output_file: str = "prompt_graph.png"):
        """
        Create a visualization of the graph and save to file with RAG connections.
        """
        with self.lock:
            plt.figure(figsize=(15, 12))  # Larger figure for better visibility
            
            # Create a copy of the graph with explicit RAG connections
            temp_graph = self.graph.copy()
            
            # Add explicit edges for RAG connections
            for u, v, data in list(temp_graph.edges(data=True)):
                if 'origins' in data and data['origins']:
                    for origin in data['origins']:
                        # Add direct RAG connection edges if they don't already exist
                        if not temp_graph.has_edge(origin, v):
                            temp_graph.add_edge(origin, v, rag_connection=True)
            
            # Create node colors based on depth
            node_colors = []
            for node in temp_graph.nodes():
                if node == 'root':
                    node_colors.append('red')
                else:
                    depth = temp_graph.nodes[node].get('depth', 0)
                    if depth == 0:
                        node_colors.append('orange')
                    elif depth == 1:
                        node_colors.append('yellow')
                    else:
                        node_colors.append('green')
            
            # Create node labels
            node_labels = {}
            for node in temp_graph.nodes():
                if node == 'root':
                    node_labels[node] = 'Main Prompt'
                else:
                    label = temp_graph.nodes[node].get('prompt', '')
                    if label and len(label) > 20:
                        label = label[:17] + "..."
                    node_labels[node] = f"{node[:4]}...: {label}"
            
            # Get edge colors and styles
            edge_colors = []
            edge_styles = []
            for u, v, data in temp_graph.edges(data=True):
                if data.get('rag_connection', False):
                    edge_colors.append('blue')  # RAG connection
                    edge_styles.append('dashed')  # Dashed line for RAG
                elif data and 'origins' in data and data['origins']:
                    edge_colors.append('purple')  # Edge with RAG origins
                    edge_styles.append('solid')
                else:
                    edge_colors.append('black')  # Regular edge
                    edge_styles.append('solid')
            
            # Generate layout and draw the graph
            pos = nx.spring_layout(temp_graph, seed=42)
            
            # Draw edges with different styles
            for i, (u, v, data) in enumerate(temp_graph.edges(data=True)):
                if edge_styles[i] == 'dashed':
                    nx.draw_networkx_edges(
                        temp_graph, pos, edgelist=[(u, v)], 
                        edge_color=[edge_colors[i]], style='dashed',
                        width=1.5, alpha=0.7, arrowsize=15
                    )
                else:
                    nx.draw_networkx_edges(
                        temp_graph, pos, edgelist=[(u, v)], 
                        edge_color=[edge_colors[i]], style='solid',
                        width=2.0, arrowsize=15
                    )
            
            # Draw nodes
            nx.draw_networkx_nodes(temp_graph, pos, node_color=node_colors, node_size=500)
            nx.draw_networkx_labels(temp_graph, pos, labels=node_labels, font_size=8)
            
            # Add legend for edge types
            import matplotlib.patches as mpatches
            from matplotlib.lines import Line2D
            
            legend_elements = [
                Line2D([0], [0], color='black', lw=2, label='Regular Connection'),
                Line2D([0], [0], color='purple', lw=2, label='Edge with RAG Origins'),
                Line2D([0], [0], color='blue', ls='--', lw=1.5, label='RAG Connection')
            ]
            
            plt.legend(handles=legend_elements, loc='upper right')
            plt.title("Prompt Hierarchy Graph with Enhanced RAG Connections")
            plt.savefig(output_file)
            plt.close()
            logger.info(f"Enhanced graph visualization saved to {output_file}")
    
    def save_to_json(self, filename: str = "./prompt_graph.json"):
        """
        Save the graph to a JSON file with enhanced RAG connections.
        
        Args:
            filename: Path to save the JSON file
        """
        with self.lock:
            # Make a copy of the graph to avoid modifying the original
            temp_graph = self.graph.copy()
            
            # Add explicit edges for RAG connections
            rag_connections_added = 0
            for u, v, data in list(temp_graph.edges(data=True)):
                if 'origins' in data and data['origins']:
                    for origin in data['origins']:
                        # Add direct RAG connection edges if they don't already exist
                        if not temp_graph.has_edge(origin, v):
                            temp_graph.add_edge(origin, v, rag_connection=True)
                            rag_connections_added += 1
            
            # Format the data explicitly to match what visualizer.html expects
            data = {
                "directed": True,
                "multigraph": False,
                "graph": {},
                "nodes": [],
                "links": []
            }
            
            # Add nodes with all their attributes
            for node_id, attrs in temp_graph.nodes(data=True):
                node_data = {"id": node_id}
                node_data.update(attrs)  # Add all attributes
                data["nodes"].append(node_data)
            
            # Add links with all their attributes
            for u, v, attrs in temp_graph.edges(data=True):
                link_data = {"source": u, "target": v}
                link_data.update(attrs)  # Add all attributes
                data["links"].append(link_data)
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)
            
            # Save to file
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Graph saved to {filename} with {rag_connections_added} enhanced RAG connections")
    
    def save_to_json1(self, filename: str = "./prompt_graph1.json"):
        """Just dump everything - duplicates are allowed."""
        with self.lock:
            data = {
                "directed": True,
                "multigraph": True,
                "graph": {},
                "nodes": [],
                "links": []
            }
            
            # Add all nodes
            for node_id, attrs in self.graph.nodes(data=True):
                node_data = {"id": node_id}
                for key, value in attrs.items():
                    # Ensure serializable
                    try:
                        json.dumps({key: value})
                        node_data[key] = value
                    except:
                        node_data[key] = str(value)
                data["nodes"].append(node_data)
            
            # Add ALL edges - don't worry about duplicates
            for u, v, attrs in self.graph.edges(data=True):
                link_data = {"source": u, "target": v}
                for key, value in attrs.items():
                    try:
                        json.dumps({key: value})
                        link_data[key] = value
                    except:
                        link_data[key] = str(value)
                data["links"].append(link_data)
            
            # EXPLICITLY add ALL RAG connections too
            for u, v, attrs in list(self.graph.edges(data=True)):
                if 'origins' in attrs and attrs['origins']:
                    for origin in attrs['origins']:
                        # Add regardless of whether it exists
                        data["links"].append({
                            "source": origin,
                            "target": v, 
                            "rag_connection": True,
                            "explicit_rag": True
                        })
            
            # Save the messy but complete graph
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Graph saved with {len(data['nodes'])} nodes and {len(data['links'])} links")

    def save_to_json2(self, filename: str = "./prompt_graph2.json"):
        """Bypass NetworkX and create raw dictionaries."""
        # Create direct dictionaries that match visualizer.html format
        nodes = []
        links = []
        
        # First convert all graph nodes to dictionaries
        for node_id in self.graph.nodes():
            node_dict = {"id": node_id}
            # Copy all attributes
            for key, value in self.graph.nodes[node_id].items():
                try:
                    json.dumps({key: value})  # Test JSON serialization
                    node_dict[key] = value
                except:
                    node_dict[key] = str(value)
            nodes.append(node_dict)
        
        # Then convert all edges and explicitly add RAG connections
        added_edges = set()  # Track edges we've added
        
        # First add all direct edges
        for u, v, data in self.graph.edges(data=True):
            edge_key = (u, v)
            added_edges.add(edge_key)
            
            edge_dict = {"source": u, "target": v}
            for key, value in data.items():
                try:
                    json.dumps({key: value})
                    edge_dict[key] = value
                except:
                    edge_dict[key] = str(value)
            links.append(edge_dict)
        
        # Then explicitly add all RAG connections
        for u, v, data in self.graph.edges(data=True):
            if 'origins' in data and data['origins']:
                for origin in data['origins']:
                    # Always add these RAG connections
                    rag_edge = {
                        "source": origin,
                        "target": v,
                        "rag_connection": True,
                        "type": "rag"
                    }
                    links.append(rag_edge)
        
        # Create final output
        output = {
            "directed": True,
            "multigraph": True,
            "graph": {},
            "nodes": nodes,
            "links": links
        }
        
        # Write to file
        with open(filename, 'w') as f:
            json.dump(output, f, indent=2)
        
        logger.info(f"Saved raw dictionary graph: {len(nodes)} nodes, {len(links)} links")
        
    def save_to_json3(self, filename: str = "./prompt_graph3.json"):
        """Keep original structure but transform to JSON."""
        with self.lock:
            # Create simple data dicts with no special handling
            node_data = []
            for node_id, attrs in self.graph.nodes(data=True):
                node = {"id": node_id}
                # Add all attributes as-is
                for k, v in attrs.items():
                    node[k] = v  # Let JSON encoder handle serialization
                node_data.append(node)
            
            link_data = []
            # Add all original links
            for u, v, attrs in self.graph.edges(data=True):
                link = {"source": u, "target": v}
                for k, v in attrs.items():
                    link[k] = v
                link_data.append(link)
            
            # Add all RAG connections
            for u, v, attrs in self.graph.edges(data=True):
                if 'origins' in attrs and attrs['origins']:
                    for origin in attrs['origins']:
                        link_data.append({
                            "source": origin,
                            "target": v,
                            "rag_connection": True
                        })
            
            # Create the output structure
            output = {
                "directed": True,
                "multigraph": True,
                "graph": {},
                "nodes": node_data,
                "links": link_data
            }
            
            # Save to file with a custom encoder to handle non-serializable objects
            class CustomEncoder(json.JSONEncoder):
                def default(self, obj):
                    try:
                        return super().default(obj)
                    except:
                        return str(obj)
            
            with open(filename, 'w') as f:
                json.dump(output, f, indent=2, cls=CustomEncoder)
    
    def save_to_json4(self, filename: str = "./prompt_graph4.json"):
        """Create a modified structure with link types."""
        # Structure with explicit types
        data = {
            "directed": True,
            "multigraph": True,
            "graph": {},
            "nodes": [],
            "links": []
        }
        
        # Process all nodes
        for node_id, attrs in self.graph.nodes(data=True):
            node = {"id": node_id}
            for k, v in attrs.items():
                try:
                    json.dumps({k: v})
                    node[k] = v
                except:
                    node[k] = str(v)
            data["nodes"].append(node)
        
        # Process all links with types
        for u, v, attrs in self.graph.edges(data=True):
            # Determine link type
            if 'rag_connection' in attrs and attrs['rag_connection']:
                link_type = "rag"
            elif 'origins' in attrs and attrs['origins']:
                link_type = "hybrid"
            else:
                link_type = "standard"
            
            # Create base link
            link = {
                "source": u,
                "target": v,
                "link_type": link_type
            }
            
            # Add all attributes
            for k, v in attrs.items():
                try:
                    json.dumps({k: v})
                    link[k] = v
                except:
                    link[k] = str(v)
            
            data["links"].append(link)
        
        # Add all RAG connections explicitly
        for u, v, attrs in self.graph.edges(data=True):
            if 'origins' in attrs and attrs['origins']:
                for origin in attrs['origins']:
                    data["links"].append({
                        "source": origin,
                        "target": v,
                        "link_type": "explicit_rag",
                        "rag_connection": True
                    })
        
        # Save with custom encoder
        class CustomEncoder(json.JSONEncoder):
            def default(self, obj):
                try:
                    return super().default(obj)
                except:
                    return str(obj)
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2, cls=CustomEncoder)
            
    def save_to_json5(self, filename: str = "./prompt_graph5.json"):
        """
        Save the graph to a JSON file with strict hierarchical structure enforcement.
        Ensures that connections between nodes of different depths are properly maintained.
        """
        with self.lock:
            # Create basic structure
            data = {
                "directed": True,
                "multigraph": False,
                "graph": {},
                "nodes": [],
                "links": []
            }
            
            # First pass: Add all nodes and organize by depth
            nodes_by_depth = {}
            node_map = {}
            
            # Process all nodes and organize by depth
            for node_id, attrs in self.graph.nodes(data=True):
                # Create clean node data
                node_data = {"id": node_id}
                for k, v in attrs.items():
                    try:
                        # Test serialization
                        json.dumps({k: v})
                        node_data[k] = v
                    except:
                        node_data[k] = str(v)
                
                # Store in our output
                data["nodes"].append(node_data)
                node_map[node_id] = node_data
                
                # Organize by depth for hierarchical processing
                depth = attrs.get('depth', 0)
                if depth not in nodes_by_depth:
                    nodes_by_depth[depth] = []
                nodes_by_depth[depth].append(node_id)
            
            # Second pass: Add all existing edges from the graph, enforcing hierarchy
            edges_to_add = set()
            for u, v, attrs in self.graph.edges(data=True):
                # Get depths to determine correct direction
                u_depth = self.graph.nodes[u].get('depth', 0)
                v_depth = self.graph.nodes[v].get('depth', 0)
                
                # Make sure edge goes from lower depth to higher depth
                if u_depth <= v_depth:
                    source, target = u, v
                    edge_attrs = attrs
                else:
                    # Reverse the edge direction to maintain hierarchy
                    source, target = v, u
                    edge_attrs = attrs  # Keep original attributes
                
                # Add to our set of edges to create
                edges_to_add.add((source, target, json.dumps(edge_attrs)))
            
            # Third pass: Ensure every node has at least one parent from the previous depth
            depths = sorted(nodes_by_depth.keys())
            
            for i in range(1, len(depths)):
                current_depth = depths[i]
                parent_depth = depths[i-1]
                
                # Process all nodes at current depth
                for child_id in nodes_by_depth[current_depth]:
                    # Check if this node has a parent from previous depth
                    has_proper_parent = False
                    
                    for edge in edges_to_add:
                        source, target, _ = edge
                        if target == child_id and source in nodes_by_depth[parent_depth]:
                            has_proper_parent = True
                            break
                    
                    # If no proper parent exists, create a synthetic parent connection
                    if not has_proper_parent and nodes_by_depth[parent_depth]:
                        # Find first available parent
                        parent_id = nodes_by_depth[parent_depth][0]
                        
                        # Add synthetic connection
                        synthetic_attrs = {"synthetic": True}
                        edges_to_add.add((parent_id, child_id, json.dumps(synthetic_attrs)))
            
            # Fourth pass: Create all links from our processed edges
            for source, target, attrs_json in edges_to_add:
                # Create the link
                link_data = {
                    "source": source,
                    "target": target
                }
                
                # Add attributes
                attrs = json.loads(attrs_json)
                for k, v in attrs.items():
                    link_data[k] = v
                
                data["links"].append(link_data)
            
            # Fifth pass: Add RAG connections
            for u, v, attrs in self.graph.edges(data=True):
                if 'origins' in attrs and attrs['origins']:
                    for origin in attrs['origins']:
                        # Always add RAG connections
                        data["links"].append({
                            "source": origin,
                            "target": v,
                            "rag_connection": True
                        })
            
            # Save to file
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Hierarchical graph saved to {filename} with {len(data['nodes'])} nodes and {len(data['links'])} links")
            
    def load_from_json(self, filename: str) -> bool:
        """
        Load graph from a JSON file.
        
        Args:
            filename: Path to the JSON file
            
        Returns:
            Success status
        """
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
            
            # Create a new DiGraph
            with self.lock:
                self.graph = nx.DiGraph()
                
                # Add nodes with attributes
                if "nodes" in data:
                    for node_data in data["nodes"]:
                        node_id = node_data.pop("id")
                        self.graph.add_node(node_id, **node_data)
                
                # Add edges with attributes
                if "links" in data:
                    for link in data["links"]:
                        source = link.pop("source")
                        target = link.pop("target")
                        self.graph.add_edge(source, target, **link)
            
            # Rebuild knowledge base
            for node_id, attrs in self.graph.nodes(data=True):
                if 'prompt' in attrs and 'response' in attrs:
                    combined_text = f"{attrs['prompt']}\n{attrs['response']}"
                    self.knowledge_base.add_document(combined_text, node_id)
            
            logger.info(f"Graph loaded from {filename}")
            return True
        except Exception as e:
            logger.error(f"Error loading graph from {filename}: {e}")
            return False
    
    def to_dict(self) -> Dict[str, Dict]:
        """Convert graph to dictionary format for serialization."""
        with self.lock:
            nodes = {}
            for node in self.graph.nodes():
                nodes[node] = {
                    'attributes': dict(self.graph.nodes[node]),
                    'children': list(self.graph.successors(node)),
                    'parent': list(self.graph.predecessors(node))
                }
            return nodes

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
    
    def add_to_queue(self, sub_prompt: str, parent_id: str, depth: int = 0) -> bool:
        """
        Add a new task to the processing queue with enhanced RAG information.
        """
        if depth >= self.max_depth:
            logger.warning(f"Max depth limit reached for task: {sub_prompt}")
            return False
        
        color = self.get_next_color()
        new_id = str(uuid.uuid4())
        logger.info(f"{color}Adding new task to queue from parent {parent_id}: {sub_prompt}\033[0m")
        
        # Enhanced: Get RAG results from the parent node
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
        
        # Add to prompt graph
        if len(sub_prompt.split("\n")) > 1:
            # Include the RAG context in the prompt
            enhanced_prompt = sub_prompt.split("\n")[-1]
            self.prompt_graph.add_node(new_id, prompt=enhanced_prompt, depth=depth, color=color, 
                                    rag_context=rag_context if rag_context else None)
        else:
            return "not so skibidi...."
        
        # Enhanced: Add edge with origins information
        self.prompt_graph.add_edge(parent_id, new_id, {'origins': rag_sources})
        
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
            logger.debug(f"Rate limited request, waited {wait_time:.2f} seconds")
            
        # Make the API call
        try:
            return self.client.models.generate_content(
                model=model,
                config=config,
                contents=contents
            )
        except Exception as e:
            logger.error(f"API call failed: {e}")
            # Implement exponential backoff for transient errors
            if "429" in str(e) or "timeout" in str(e).lower():
                retry_wait = min(wait_time * 2 + 1, 30)  # Cap at 30 seconds
                logger.warning(f"Rate limit or timeout error, retrying in {retry_wait}s")
                time.sleep(retry_wait)
                return self.rate_limited_generate_content(model, config, contents)
            raise
    
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
            mprint(approaches_content)
            
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
            logger.error(f"Error generating thought approaches: {e}")
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


    def process_sub_prompt(self, task_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single sub-prompt task with enhanced RAG capabilities.
        """
        sub_prompt = task_info['sub_prompt']
        depth = task_info.get('depth', 0)
        parent_id = task_info.get('parent_id', 'root')
        color = task_info.get('color', '\033[37m')  # Default to white
        node_id = task_info.get('id', str(uuid.uuid4()))
        rag_sources = task_info.get('rag_sources', [])
        
        thread_logger = colorlog.getLogger(node_id)
        thread_logger.setLevel(logging.INFO)
        
        try:
            with self.active_threads_lock:
                self.active_threads += 1
            
            thread_logger.info(f"{color}Starting processing of task (depth: {depth}, parent: {parent_id})\033[0m")
            
            # Enhanced: Query knowledge base for top-k (k=2) relevant information
            kb_results = self.prompt_graph.query_knowledge_base(sub_prompt, top_k=2)
            additional_context = ""
            
            if kb_results:
                # Format all results into the additional context
                additional_context = "Related information:\n"
                for i, result in enumerate(kb_results, 1):
                    relevant_node = result["node_id"]
                    relevant_text = result["text"]
                    similarity = result["similarity"]
                    additional_context += f"Source {i} (relevance: {similarity:.2f}): {relevant_text}\n\n"
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

            # Enhanced: Immediately update the knowledge base with the new response
            self.prompt_graph.add_node(node_id, response=response_text, rag_results=kb_results)
            prompt_text = self.prompt_graph.get_node_data(node_id).get('prompt', '')
            combined_text = f"{prompt_text}\n{response_text}"
            self.prompt_graph.knowledge_base.add_document(combined_text, node_id)

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
                    mprint(f"Thread {node_id} did not require further analysis")
                else:
                    for new_prompt in r2.split("\n"):
                        if new_prompt.strip():
                            # Enhanced: Include RAG information when adding to queue
                            enhanced_prompt = response_text + "\n" + new_prompt
                            if kb_results:
                                similarity_info = "\n".join([f"Similarity source {r['node_id']}: {r['similarity']:.2f}" for r in kb_results])
                                enhanced_prompt = similarity_info + "\n" + enhanced_prompt
                            self.add_to_queue(enhanced_prompt, node_id, depth)
            else:
                # Ensure max depth responses are stored
                logger.info(f"{color}Max depth reached for node {node_id}, ensuring response is stored\033[0m")
                # We've already added it to the knowledge base above, so just confirm
                thread_logger.info(f"{color}Confirmed max depth response stored for node {node_id}\033[0m")
            
            result = {
                'id': node_id,
                'parent_id': parent_id,
                'sub_prompt': sub_prompt,
                'response': response_text,
                'depth': depth,
                'color': color,
                'rag_sources': rag_sources,
                'rag_results': kb_results
            }
            
            mprint(f"{color}[DEPTH: {depth}] {sub_prompt}"+ "\n\n" + response_text+"\033[0m")
            
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
                logger.error(f"Worker error: {e}")
                self.task_queue.task_done()

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
            
            logger.info("\033[37m[COMBINER] Comprehensive synthesis complete\033[0m")
            
            # Visualize and save the graph
            self.prompt_graph.visualize()
            
            try:
                self.prompt_graph.save_to_json()
            except:
                print("approach 0 didnt work")
            try:
                self.prompt_graph.save_to_json1()
            except:
                print("approach 1 didnt work")
            try:
                self.prompt_graph.save_to_json2()
            except:
                print("approach 2 didnt work")
            try:
                self.prompt_graph.save_to_json3()
            except:
                print("approach 3 didnt work")
            try:
                self.prompt_graph.save_to_json4()
            except:
                print("approach 4 didnt work")
            try:
                self.prompt_graph.save_to_json5()
            except:
                print("approach 5 didnt work")
            try:
                self.prompt_graph.visualize_hierarchical()
            except:
                print("visualize_hierarachal didnt work")

            
            
            return {
                'main_prompt': main_prompt,
                'combined_response': combined_response.text,
                'graph': self.prompt_graph.to_dict()
            }
        except Exception as e:
            logger.error(f"Response combination error: {e}")
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

    def process_main_prompt(self, main_prompt: str) -> tuple:
        """
        Main orchestration method for prompt processing.
        
        Args:
            main_prompt: The main question/prompt
            
        Returns:
            Tuple of (all_results, combined_output)
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
        logger.info(system_prompts_info)
        
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
        
        # Combine responses
        mprint("# Final response is getting evaluated")
        combined_result = self.combine_responses(self.all_results, main_prompt)
        
        return self.all_results, combined_result

def main():
    """Main function to run the prompt processor."""
    api_key = "AIzaSyBgjwc8ihSASWnCm1qiRUYdEu1jtWuzcW4"  # Replace with environment variable in production
    processor = GeminiPromptProcessor(api_key)
    
    try:
        while True:
            console.clear()
            main_prompt = input("Enter a problem that can be solved via planning: ")
            if not main_prompt.strip():
                continue
                
            mprint(f"# {main_prompt}?")
            results, combined_output = processor.process_main_prompt(main_prompt)
            
            # Final output processing
            logger.info("\033[37m[FINAL RESULT] Comprehensive Analysis Complete\033[0m")
            
            if 'error' in combined_output and not combined_output.get('combined_response'):
                mprint(f"Error occurred during analysis: {combined_output['error']}")
            else:
                mprint(combined_output['combined_response'])
            
            # Show graph info
            logger.info("\033[37m[GRAPH] Graph structure saved and visualized\033[0m")
            
            # Ask if user wants to continue
            while True:
                choice = input("Would you like to enter a new question? [y/n]: ")
                if choice.lower() in ['y', 'n']:
                    break
            
            if choice.lower() == 'n':
                break
    except KeyboardInterrupt:
        print("\nExiting program...")
    except Exception as e:
        logger.critical(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
