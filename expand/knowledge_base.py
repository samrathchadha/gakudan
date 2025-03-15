"""
Knowledge Base implementation for semantic search of previous responses.
Uses TF-IDF and cosine similarity for retrieval with improved uniqueness.
"""

import threading
import logging
import numpy as np
from typing import List, Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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
                logging.debug(f"Updated existing document for node {node_id}")
            else:
                # Add new document
                self.documents.append({"text": text, "node_id": node_id})
                self.node_id_to_index[node_id] = len(self.documents) - 1
                logging.debug(f"Added new document for node {node_id}")
            
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
                logging.debug(f"Query found {len(results)} unique results from {len(self.documents)} documents")
                for i, result in enumerate(results):
                    logging.debug(f"Result {i+1}: node={result['node_id']}, similarity={result['similarity']:.2f}")
            else:
                logging.debug(f"Query found no results from {len(self.documents)} documents")
                
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