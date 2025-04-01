"""
Fixed DatabasePromptGraph that properly saves all nodes and links.
"""

import threading
import logging
import networkx as nx
import numpy as np
from typing import List, Dict, Any
from knowledge_base import KnowledgeBase

# Include NumPy conversion locally
def convert_numpy_types(obj):
    """Convert NumPy types to Python native types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif hasattr(obj, 'item'):  # Handle other numpy scalar types
        return obj.item()
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    elif isinstance(obj, set):
        return set(convert_numpy_types(item) for item in obj)
    else:
        return obj

class DatabasePromptGraph:
    """
    Database-first graph data structure for tracking relationships between prompts.
    Explicitly saves all nodes and links to the database.
    """
    def __init__(self, session_id: str):
        """
        Initialize with session ID for database storage.
        
        Args:
            session_id: ID of the session this graph belongs to
        """
        self.graph = nx.MultiDiGraph()  # Changed to MultiDiGraph to support multiple edge types
        self.lock = threading.Lock()
        self.knowledge_base = KnowledgeBase()
        self.session_id = session_id
        self.dirty = False  # Track if there are unsaved changes
        
        # Import db_manager here to avoid circular imports
        from db_manager import db_manager
        self.db_manager = db_manager
        
        logger.info(f"Initialized DatabasePromptGraph for session {session_id}")
    
    def add_node(self, node_id: str, **attributes):
        """
        Add or update a node in the graph.
        Does not save to database until explicitly requested.
        
        Args:
            node_id: Unique identifier for the node
            **attributes: Node attributes (prompt, response, depth, etc.)
        """
        with self.lock:
            # Convert any NumPy types in attributes
            clean_attributes = {k: convert_numpy_types(v) for k, v in attributes.items()}
            
            if node_id in self.graph:
                # Update existing node attributes
                for key, value in clean_attributes.items():
                    self.graph.nodes[node_id][key] = value
            else:
                # Add new node
                self.graph.add_node(node_id, **clean_attributes)
                
            # If response is provided, add to knowledge base
            if 'response' in clean_attributes and 'prompt' in self.graph.nodes[node_id]:
                prompt_text = self.graph.nodes[node_id]['prompt']
                response_text = clean_attributes['response']
                combined_text = f"{prompt_text}\n{response_text}"
                self.knowledge_base.add_document(combined_text, node_id)
                
            self.dirty = True
    
    def add_edge(self, parent_id: str, child_id: str, edge_attrs: Dict = None, edge_type: str = "hierarchy"):
        """
        Add an edge between nodes with explicit edge type.
        Does not save to database until explicitly requested.
        
        Args:
            parent_id: ID of the parent node
            child_id: ID of the child node
            edge_attrs: Optional attributes for the edge
            edge_type: Type of edge ("hierarchy" or "rag")
        """
        with self.lock:
            if edge_attrs is None:
                edge_attrs = {}
            else:
                # Convert any NumPy types in edge attributes
                edge_attrs = {k: convert_numpy_types(v) for k, v in edge_attrs.items()}
                
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
                    
            self.dirty = True
    
    def add_rag_connection(self, source_id: str, target_id: str, similarity: float = None):
        """
        Add an explicit RAG connection between nodes.
        Does not save to database until explicitly requested.
        
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
                # Convert NumPy float to Python float
                similarity = convert_numpy_types(similarity)
                attrs["similarity"] = similarity
                
            # Add the RAG edge
            self.graph.add_edge(source_id, target_id, **attrs)
            logging.debug(f"Added explicit RAG edge: {source_id} -> {target_id} (similarity: {similarity})")
            
            self.dirty = True
    
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
        results = self.knowledge_base.query(
            query_text, 
            top_k=top_k,
            exclude_ids=exclude_ids,
            exclude_related=exclude_related,
            current_node_id=current_node_id
        )
        
        # Convert NumPy types in results
        for result in results:
            if 'similarity' in result and hasattr(result['similarity'], 'item'):
                result['similarity'] = float(result['similarity'].item())
        
        return results
    
    def get_node_data(self, node_id: str) -> Dict[str, Any]:
        """Get all attributes for a specific node."""
        with self.lock:
            if node_id in self.graph:
                # Convert NumPy types to ensure JSON compatibility
                return {k: convert_numpy_types(v) for k, v in self.graph.nodes[node_id].items()}
            return {}
    
    def get_nodes_at_depth(self, depth: int) -> List[str]:
        """Get all nodes at a specific depth level."""
        with self.lock:
            return [
                node for node, attrs in self.graph.nodes(data=True) 
                if attrs.get('depth') == depth
            ]
    
    def format_for_database(self) -> Dict[str, Any]:
        """
        Format the graph data for database storage.
        
        Returns:
            Dictionary formatted for database storage
        """
        with self.lock:
            data = {
                "directed": True,
                "multigraph": True,
                "graph": {},
                "nodes": [],
                "links": []
            }
            
            # Add nodes with all their attributes
            for node_id, attrs in self.graph.nodes(data=True):
                # Convert NumPy types in attributes
                clean_attrs = {k: convert_numpy_types(v) for k, v in attrs.items()}
                
                node_data = {"id": node_id}
                node_data.update(clean_attrs)
                
                # Handle large string attributes
                for key, value in node_data.items():
                    if isinstance(value, str) and len(value) > 100000:
                        node_data[key] = value[:100000] + "... [truncated]"
                
                data["nodes"].append(node_data)
            
            # Add all edges, explicitly marking edge types
            for u, v, attrs in self.graph.edges(data=True):
                # Convert NumPy types in attributes
                clean_attrs = {k: convert_numpy_types(v) for k, v in attrs.items()}
                
                link_data = {
                    "source": u, 
                    "target": v
                }
                
                # Determine edge type and set appropriate flags
                edge_type = clean_attrs.get("edge_type", "hierarchy")
                
                if edge_type == "rag":
                    link_data["rag_connection"] = True
                    link_data["edge_type"] = "rag"
                    link_data["link_type"] = "rag"  # For visualizer compatibility
                else:
                    link_data["edge_type"] = "hierarchy"
                    link_data["link_type"] = "hierarchy"  # For visualizer compatibility
                
                # Add all other attributes
                for key, value in clean_attrs.items():
                    if key not in link_data:  # Don't overwrite
                        link_data[key] = value
                
                data["links"].append(link_data)
                
            return data
    
    def save_to_database(self):
        """
        Save all graph data to the database.
        Saves both to contract_results and to the individual nodes and links tables.
        
        Returns:
            True if successful, False otherwise
        """
        if not self.dirty:
            logger.info(f"No changes to save for session {self.session_id}")
            return True
            
        try:
            # Format data for database
            data = self.format_for_database()
            
            # Save contract data in one operation (will overwrite existing)
            self.db_manager.save_contract_data(self.session_id, data)
            
            # Save individual nodes and links
            self.save_nodes_and_links(data)
            
            # Reset dirty flag
            self.dirty = False
            
            logger.info(f"All graph data saved to database for session {self.session_id}")
            return True
        except Exception as e:
            logger.error(f"Error saving graph to database: {e}", exc_info=True)
            return False
    
    def save_nodes_and_links(self, formatted_data=None):
        """
        Save all nodes and links to their respective database tables.
        
        Args:
            formatted_data: Optional pre-formatted data, will generate if not provided
            
        Returns:
            True if successful, False otherwise
        """
        import db  # Local import to avoid circular dependencies
        
        try:
            # Get formatted data if not provided
            if formatted_data is None:
                formatted_data = self.format_for_database()
            
            nodes = formatted_data["nodes"]
            links = formatted_data["links"]
            
            # Log the count of nodes and links to be saved
            logger.info(f"Saving {len(nodes)} nodes and {len(links)} links to database for session {self.session_id}")
            
            with db.db_transaction() as transaction:
                # First, clear existing data for this session to avoid duplicates
                db.Node.delete().where(db.Node.session_id == self.session_id).execute()
                db.Link.delete().where(db.Link.session_id == self.session_id).execute()
                
                # Bulk insert nodes
                for node in nodes:
                    node_id = node["id"]
                    prompt = node.get("prompt")
                    response = node.get("response")
                    depth = node.get("depth")
                    
                    # Extract attributes (everything else)
                    attributes = dict(node)
                    for key in ["id", "prompt", "response", "depth"]:
                        if key in attributes:
                            attributes.pop(key, None)
                    
                    # Add or update node in database
                    db.Node.create(
                        node_id=node_id,
                        session_id=self.session_id,
                        prompt=prompt,
                        response=response,
                        depth=depth,
                        attributes=attributes if attributes else None
                    )
                
                # Bulk insert links
                for link in links:
                    source = link["source"]
                    target = link["target"]
                    edge_type = link.get("edge_type", "hierarchy")
                    link_type = link.get("link_type", edge_type)
                    similarity = link.get("similarity")
                    
                    # Extract attributes (everything else)
                    attributes = dict(link)
                    for key in ["source", "target", "edge_type", "link_type", "similarity"]:
                        if key in attributes:
                            attributes.pop(key, None)
                    
                    # Add link to database
                    db.Link.create(
                        session_id=self.session_id,
                        source_id=source,
                        target_id=target,
                        edge_type=edge_type,
                        link_type=link_type,
                        similarity=similarity,
                        attributes=attributes if attributes else None
                    )
            
            logger.info(f"Successfully saved {len(nodes)} nodes and {len(links)} links to database")
            return True
        except Exception as e:
            logger.error(f"Error saving nodes and links to database: {e}", exc_info=True)
            return False
