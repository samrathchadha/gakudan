"""
Knowledge Base implementation for semantic search with relationship awareness.
Avoids retrieving documents from the same parent or the parent itself.
"""

import threading
import logging
import numpy as np
from typing import List, Dict, Any, Set, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

class KnowledgeBase:
    """
    Enhanced vector database for semantic search with relationship awareness.
    Avoids retrieving documents from the same parent or the parent itself.
    """
    def __init__(self, use_transformer: bool = False, model_name: str = "all-MiniLM-L6-v2"):
        self.documents = []
        self.use_transformer = use_transformer
        
        # TF-IDF vectorizer (original method)
        self.vectorizer = TfidfVectorizer(
            min_df=1, 
            max_df=0.9,             # Ignore terms that appear in >90% of documents
            stop_words='english',    # Remove common English stopwords
            ngram_range=(1, 2)       # Include unigrams and bigrams
        )
        
        # Sentence transformer for better semantic search (if enabled)
        self.transformer = None
        if use_transformer:
            try:
                self.transformer = SentenceTransformer(model_name)
                logging.info(f"Using Sentence Transformer model: {model_name}")
            except Exception as e:
                logging.error(f"Failed to load transformer model: {e}")
                self.use_transformer = False
                
        self.vectors = None
        self.lock = threading.Lock()
        self.node_id_to_index = {}   # Maps node_ids to their indices in documents
        
        # Additional mappings for relationship tracking
        self.node_parent_map = {}    # Maps node_ids to their parent_ids
        self.node_siblings_map = {}  # Maps node_ids to their sibling_ids
        
    def add_document(self, text: str, node_id: str, parent_id: Optional[str] = None):
        """
        Add document to knowledge base and update vector index.
        Also records the parent relationship for filtering.
        
        Args:
            text: Document text
            node_id: Unique identifier for the node
            parent_id: Optional parent node ID for relationship tracking
        """
        with self.lock:
            # Check if node already exists
            if node_id in self.node_id_to_index:
                # Update existing document
                index = self.node_id_to_index[node_id]
                self.documents[index]["text"] = text
                logging.debug(f"Updated existing document for node {node_id}")
            else:
                # Add new document
                self.documents.append({"text": text, "node_id": node_id})
                self.node_id_to_index[node_id] = len(self.documents) - 1
                logging.debug(f"Added new document for node {node_id}")
            
            # Update relationship tracking
            if parent_id:
                # Record parent relationship
                self.node_parent_map[node_id] = parent_id
                
                # Update sibling relationships
                if parent_id not in self.node_siblings_map:
                    self.node_siblings_map[parent_id] = set()
                
                # Add this node to its parent's children
                self.node_siblings_map[parent_id].add(node_id)
                
            # Recompute vectors if we have documents
            if self.documents:
                texts = [doc["text"] for doc in self.documents]
                if self.use_transformer and self.transformer:
                    self.vectors = self.transformer.encode(texts)
                else:
                    self.vectors = self.vectorizer.fit_transform(texts)
    
    def get_related_nodes(self, node_id: str) -> Set[str]:
        """
        Get all nodes related to this node (parent and siblings).
        
        Args:
            node_id: The node ID to find related nodes for
            
        Returns:
            Set of node IDs that are related (parent, siblings)
        """
        related = set()
        
        # Add parent if it exists
        if node_id in self.node_parent_map:
            parent_id = self.node_parent_map[node_id]
            related.add(parent_id)
            
            # Add siblings (all children of the parent)
            if parent_id in self.node_siblings_map:
                related.update(self.node_siblings_map[parent_id])
        
        return related
    
    def query(self, query_text: str, top_k: int = 1, exclude_ids: List[str] = None, 
            exclude_related: bool = True, current_node_id: str = None) -> List[Dict[str, Any]]:
        """
        Find most similar documents to the query, ensuring diverse results.
        Excludes specified nodes and optionally related nodes.
        
        Args:
            query_text: The search query
            top_k: Number of results to return
            exclude_ids: Explicit list of node IDs to exclude
            exclude_related: Whether to exclude parent and siblings
            current_node_id: Current node ID for determining relationships
            
        Returns:
            List of matches with text, node_id and similarity score
        """
        with self.lock:
            if not self.documents or len(self.documents) <= 5:
                return []
            
            # Build exclusion set
            exclude_set = set(exclude_ids) if exclude_ids else set()
            
            # Add related nodes to exclusion set if requested
            if exclude_related and current_node_id:
                exclude_set.update(self.get_related_nodes(current_node_id))
                # Always exclude the current node itself
                exclude_set.add(current_node_id)
            
            # Get query vector
            if self.use_transformer and self.transformer:
                query_vector = self.transformer.encode([query_text])[0].reshape(1, -1)
                similarities = cosine_similarity(query_vector, self.vectors).flatten()
            else:
                query_vector = self.vectorizer.transform([query_text])
                similarities = cosine_similarity(query_vector, self.vectors).flatten()
            
            # Track used node_ids to avoid duplicates
            used_nodes = set()
            results = []
            
            # Get indices sorted by similarity (descending)
            sorted_indices = similarities.argsort()[::-1]
            
            # Add top results ensuring uniqueness and exclusion criteria
            for idx in sorted_indices:
                node_id = self.documents[idx]["node_id"]
                similarity = similarities[idx]
                
                # Skip if we've already used this node, similarity is too low, or it's in the exclude list
                if (node_id in used_nodes or 
                    similarity < 0.1 or 
                    node_id in exclude_set):
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
                logging.debug(f"Query found {len(results)} diverse results from {len(self.documents)} documents")
                logging.debug(f"Excluded {len(exclude_set)} related or explicitly excluded nodes")
                for i, result in enumerate(results):
                    logging.debug(f"Result {i+1}: node={result['node_id']}, similarity={result['similarity']:.2f}")
            else:
                logging.debug(f"Query found no results from {len(self.documents)} documents after filtering {len(exclude_set)} nodes")
                
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
            self.node_parent_map = {}
            self.node_siblings_map = {}