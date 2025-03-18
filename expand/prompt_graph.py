"""
Graph data structure for tracking relationships between prompts.
RAG connections are treated as first-class citizens, ensuring they're always visible.
"""

import threading
import json
import logging
import os
import networkx as nx
import matplotlib.pyplot as plt
from typing import List, Dict, Any
from knowledge_base import KnowledgeBase

class PromptGraph:
    """
    Enhanced graph data structure for tracking relationships between prompts.
    RAG connections are treated as first-class citizens, ensuring they're always visible.
    """
    def __init__(self):
        self.graph = nx.MultiDiGraph()  # Changed to MultiDiGraph to support multiple edge types
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
    
    def add_edge(self, parent_id: str, child_id: str, edge_attrs: Dict = None, edge_type: str = "hierarchy"):
        """
        Add an edge between nodes with explicit edge type.
        
        Args:
            parent_id: ID of the parent node
            child_id: ID of the child node
            edge_attrs: Optional attributes for the edge
            edge_type: Type of edge ("hierarchy" or "rag")
        """
        with self.lock:
            if edge_attrs is None:
                edge_attrs = {}
                
            # Always set the edge type explicitly
            edge_attrs["edge_type"] = edge_type
            
            # Add edge with attributes
            self.graph.add_edge(parent_id, child_id, **edge_attrs)
            
            logging.debug(f"Added {edge_type} edge: {parent_id} -> {child_id}")
            
            # Fix depth for hierarchy edges: ensure child depth is greater than parent
            if edge_type == "hierarchy" and parent_id in self.graph and child_id in self.graph:
                parent_depth = self.graph.nodes[parent_id].get('depth', 0)
                child_depth = self.graph.nodes[child_id].get('depth', 0)
                
                if child_depth <= parent_depth:
                    self.graph.nodes[child_id]['depth'] = parent_depth + 1
    
    def add_rag_connection(self, source_id: str, target_id: str, similarity: float = None):
        """
        Add an explicit RAG connection between nodes.
        
        Args:
            source_id: ID of the source node (providing context)
            target_id: ID of the target node (receiving context)
            similarity: Optional similarity score
        """
        with self.lock:
            # Create attributes for the RAG connection
            attrs = {
                "rag_connection": True,
                "edge_type": "rag"
            }
            
            if similarity is not None:
                attrs["similarity"] = similarity
                
            # Add the RAG edge
            self.graph.add_edge(source_id, target_id, **attrs)
            logging.debug(f"Added explicit RAG edge: {source_id} -> {target_id} (similarity: {similarity})")

    def get_children(self, node_id: str, edge_type: str = None) -> List[str]:
        """
        Get child nodes of the given node, optionally filtered by edge type.
        
        Args:
            node_id: ID of the parent node
            edge_type: Optional edge type to filter by ("hierarchy" or "rag")
            
        Returns:
            List of child node IDs
        """
        with self.lock:
            children = []
            
            for _, child, data in self.graph.out_edges(node_id, data=True):
                if edge_type is None or data.get("edge_type") == edge_type:
                    children.append(child)
                    
            return children
    
    def to_dict(self) -> Dict[str, Dict]:
        """Convert graph to dictionary format for serialization."""
        with self.lock:
            nodes = {}
            for node in self.graph.nodes():
                # Get hierarchy connections
                hierarchy_children = [target for target, data in self.graph.adj[node].items() 
                                    for _, edge_data in data.items() 
                                    if edge_data.get('edge_type') == 'hierarchy']
                
                hierarchy_parents = [source for source, edges in self.graph.pred[node].items() 
                                    for _, edge_data in edges.items() 
                                    if edge_data.get('edge_type') == 'hierarchy']
                
                # Get RAG connections
                rag_sources = [source for source, edges in self.graph.pred[node].items() 
                            for _, edge_data in edges.items() 
                            if edge_data.get('edge_type') == 'rag' or edge_data.get('rag_connection', False)]
                
                rag_targets = [target for target, data in self.graph.adj[node].items() 
                            for _, edge_data in data.items() 
                            if edge_data.get('edge_type') == 'rag' or edge_data.get('rag_connection', False)]
                
                # Store in dictionary format
                nodes[node] = {
                    'attributes': dict(self.graph.nodes[node]),
                    'children': hierarchy_children,
                    'parents': hierarchy_parents,
                    'rag_sources': rag_sources,
                    'rag_targets': rag_targets
                }
            return nodes

    def get_parents(self, node_id: str, edge_type: str = None) -> List[str]:
        """
        Get parent nodes of the given node, optionally filtered by edge type.
        
        Args:
            node_id: ID of the child node
            edge_type: Optional edge type to filter by ("hierarchy" or "rag")
            
        Returns:
            List of parent node IDs
        """
        with self.lock:
            parents = []
            
            for parent, _, data in self.graph.in_edges(node_id, data=True):
                if edge_type is None or data.get("edge_type") == edge_type:
                    parents.append(parent)
                    
            return parents
    
    def get_rag_sources(self, node_id: str) -> List[str]:
        """
        Get all RAG source nodes for the given node.
        
        Args:
            node_id: ID of the target node
            
        Returns:
            List of source node IDs providing RAG context
        """
        return self.get_parents(node_id, edge_type="rag")
    
    def query_knowledge_base(self, query_text: str, top_k: int = 1, 
                                exclude_ids: List[str] = None, 
                                exclude_related: bool = True,
                                current_node_id: str = None) -> List[Dict[str, Any]]:
        """
        Query the knowledge base for relevant information.
        With enhanced relationship awareness - explicitly excludes parent nodes and sibling nodes.
        
        Args:
            query_text: The search query text
            top_k: Number of results to return (default: 1)
            exclude_ids: Specific node IDs to exclude
            exclude_related: Whether to exclude parent and siblings
            current_node_id: Current node ID for relationship exclusion
            
        Returns:
            List of matches with text, node_id and similarity score
        """
        # Ensure current_node_id has been added to the knowledge base's relationship tracking
        if current_node_id and exclude_related:
            # Get parent of current node from our graph
            parents = self.get_parents(current_node_id, edge_type="hierarchy")
            
            # If we have a parent, ensure the knowledge base knows about this relationship
            if parents and current_node_id not in self.knowledge_base.node_parent_map:
                parent_id = parents[0]  # Use the first parent if there are multiple
                node_data = self.get_node_data(current_node_id)
                prompt = node_data.get('prompt', '')
                response = node_data.get('response', '')
                
                # Make sure knowledge base knows about this relationship
                if prompt:
                    combined_text = f"{prompt}\n{response}" if response else prompt
                    self.knowledge_base.add_document(combined_text, current_node_id, parent_id=parent_id)
                    logging.debug(f"Updated KB relationship for {current_node_id} with parent {parent_id}")
        
        # Forward the query to the knowledge base with relationship exclusion
        return self.knowledge_base.query(
            query_text, 
            top_k=top_k,
            exclude_ids=exclude_ids,
            exclude_related=exclude_related,
            current_node_id=current_node_id
        )
    
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
    
    def save_to_json(self, filename: str = "./expand.json"):
        """
        Save the graph to a JSON file with explicit RAG connections.
        
        Args:
            filename: Path to save the JSON file
        """
        with self.lock:
            # Format the data explicitly for visualization
            data = {
                "directed": True,
                "multigraph": True,  # Changed to True to support multiple edge types
                "graph": {},
                "nodes": [],
                "links": []
            }
            
            # Add nodes with all their attributes
            for node_id, attrs in self.graph.nodes(data=True):
                node_data = {"id": node_id}
                # Add all attributes, ensuring they're JSON serializable
                for key, value in attrs.items():
                    try:
                        json.dumps({key: value})  # Test JSON serialization
                        node_data[key] = value
                    except:
                        node_data[key] = str(value)
                data["nodes"].append(node_data)
            
            # Add all edges, explicitly marking edge types
            for u, v, attrs in self.graph.edges(data=True):
                link_data = {
                    "source": u, 
                    "target": v
                }
                
                # Determine edge type and set appropriate flags
                edge_type = attrs.get("edge_type", "hierarchy")
                
                if edge_type == "rag":
                    link_data["rag_connection"] = True
                    link_data["edge_type"] = "rag"
                    link_data["link_type"] = "rag"  # For visualizer compatibility
                else:
                    link_data["edge_type"] = "hierarchy"
                    link_data["link_type"] = "hierarchy"  # For visualizer compatibility
                
                # Add all other attributes
                for key, value in attrs.items():
                    if key not in link_data:  # Don't overwrite
                        try:
                            json.dumps({key: value})  # Test serialization
                            link_data[key] = value
                        except:
                            link_data[key] = str(value)
                
                data["links"].append(link_data)
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)
            
            # Save to file
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2)
            
            hierarchy_count = sum(1 for link in data["links"] if link.get("edge_type") == "hierarchy")
            rag_count = sum(1 for link in data["links"] if link.get("edge_type") == "rag")
            
            logging.info(f"Graph saved to {filename} with {len(data['nodes'])} nodes, "
                       f"{hierarchy_count} hierarchy connections, and {rag_count} RAG connections")
            
    def visualize(self, output_file: str = "./prompt_graph.png"):
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
            logging.info(f"Enhanced graph visualization saved to {output_file}")
