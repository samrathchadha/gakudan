#!/usr/bin/env python
"""
Modified runner.py that uses PostgreSQL database instead of local files.
"""

import os
import sys
import time
import uuid
import logging
import argparse
import colorlog
import numpy as np
from typing import Dict, Any, Optional
from prompt_processor import GeminiPromptProcessor
from prompt_graph import PromptGraph

# Add parent directory to path to import database modules
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

# Import database modules
import db
from db_manager import db_manager

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

class DatabasePromptGraph(PromptGraph):
    """
    Enhanced PromptGraph that stores nodes and links in the database.
    """
    def __init__(self, session_id: str):
        """
        Initialize with session ID for database storage.
        
        Args:
            session_id: ID of the session this graph belongs to
        """
        super().__init__()
        self.session_id = session_id
        self.db_manager = db_manager
        logger.info(f"Initialized DatabasePromptGraph for session {session_id}")
    
    def add_node(self, node_id: str, **attributes):
        """
        Add or update a node in both the graph and database.
        
        Args:
            node_id: Unique identifier for the node
            **attributes: Node attributes (prompt, response, depth, etc.)
        """
        # Call parent method to update in-memory graph
        super().add_node(node_id, **attributes)
        
        # Extract key attributes
        prompt = attributes.get('prompt')
        response = attributes.get('response')
        depth = attributes.get('depth')
        
        # Create a copy of attributes without key fields for JSONB storage
        db_attributes = dict(attributes)
        for key in ['prompt', 'response', 'depth']:
            if key in db_attributes:
                del db_attributes[key]
        
        # Add to database
        db_success = self.db_manager.add_node(
            node_id=node_id,
            session_id=self.session_id,
            prompt=prompt,
            response=response,
            depth=depth,
            attributes=db_attributes if db_attributes else None
        )
        
        if not db_success:
            logger.warning(f"Failed to add node {node_id} to database for session {self.session_id}")
    
    def add_edge(self, parent_id: str, child_id: str, edge_attrs: Dict = None, edge_type: str = "hierarchy"):
        """
        Add an edge between nodes in both the graph and database.
        
        Args:
            parent_id: ID of the parent node
            child_id: ID of the child node
            edge_attrs: Optional attributes for the edge
            edge_type: Type of edge ("hierarchy" or "rag")
        """
        # Call parent method to update in-memory graph
        super().add_edge(parent_id, child_id, edge_attrs, edge_type)
        
        # Extract similarity if present
        similarity = None
        if edge_attrs and "similarity" in edge_attrs:
            similarity = edge_attrs["similarity"]
        
        # Create a copy of attributes without key fields for JSONB storage
        db_attributes = dict(edge_attrs) if edge_attrs else {}
        for key in ['edge_type', 'similarity']:
            if key in db_attributes:
                del db_attributes[key]
        
        # Add to database
        db_success = self.db_manager.add_link(
            session_id=self.session_id,
            source_id=parent_id,
            target_id=child_id,
            edge_type=edge_type,
            link_type=edge_type,  # Use edge_type as link_type for compatibility
            similarity=similarity,
            attributes=db_attributes if db_attributes else None
        )
        
        if not db_success:
            logger.warning(f"Failed to add edge from {parent_id} to {child_id} in database for session {self.session_id}")
    
    def add_rag_connection(self, source_id: str, target_id: str, similarity: float = None):
        """
        Add an explicit RAG connection between nodes in both the graph and database.
        
        Args:
            source_id: ID of the source node (providing context)
            target_id: ID of the target node (receiving context)
            similarity: Optional similarity score
        """
        # Call parent method to update in-memory graph
        super().add_rag_connection(source_id, target_id, similarity)
        
        # Add to database
        db_success = self.db_manager.add_link(
            session_id=self.session_id,
            source_id=source_id,
            target_id=target_id,
            edge_type="rag",
            link_type="rag",
            similarity=similarity
        )
        
        if not db_success:
            logger.warning(f"Failed to add RAG connection from {source_id} to {target_id} in database for session {self.session_id}")
    
    def save_to_database(self):
        """
        Ensure all graph data is saved to the database.
        This replaces the save_to_json method.
        """
        graph_data = self.to_dict()
        
        # Log stats
        node_count = len(self.graph.nodes())
        edge_count = len(self.graph.edges())
        logger.info(f"Saved graph to database: {node_count} nodes, {edge_count} edges")

# Create a subclass of GeminiPromptProcessor that uses our DatabasePromptGraph
class DatabasePromptProcessor(GeminiPromptProcessor):
    def __init__(self, api_key, session_id):
        super().__init__(api_key)
        self.session_id = session_id
        
        # Replace the PromptGraph with our database version
        self.prompt_graph = DatabasePromptGraph(session_id)
        self.prompt_graph.add_node('root', prompt='Main Prompt', depth=-1)
        
        logger.info(f"Initialized DatabasePromptProcessor for session {session_id}")
    
    def process_sub_prompt(self, task_info):
        # Call the original method
        result = super().process_sub_prompt(task_info)
        
        # Ensure data is saved to the database
        self.prompt_graph.save_to_database()
        
        return result

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run Expand prompt processor with database storage.')
    parser.add_argument('--prompt', required=True, help='Prompt to process')
    parser.add_argument('--api-key', required=True, help='API Key for Google Generative AI')
    parser.add_argument('--session-id', required=True, help='Session ID for database storage')
    
    return parser.parse_args()

def verify_database_connection(session_id):
    """Verify the database connection is working."""
    try:
        # Initialize database
        db.initialize_db()
        
        # Test connection by querying for the session
        session_data = db_manager.get_session_data(session_id)
        if not session_data:
            logger.critical(f"Session {session_id} not found in database")
            return False
            
        logger.info(f"Successfully verified database connection for session {session_id}")
        return True
    except Exception as e:
        logger.critical(f"Failed to verify database connection: {e}", exc_info=True)
        return False

def main():
    """Main function to run the prompt processor with database storage."""
    # Parse command line arguments
    args = parse_arguments()
    session_id = args.session_id
    
    # Verify database connection
    if not verify_database_connection(session_id):
        logger.critical("Database verification failed. Exiting.")
        sys.exit(1)
    
    # Initialize our processor with database support
    processor = DatabasePromptProcessor(args.api_key, session_id)
    
    try:
        # Process the prompt
        logger.info(f"Processing prompt for session {session_id}: {args.prompt}")
        results = processor.process_main_prompt(args.prompt)
        
        # Final save to database
        processor.prompt_graph.save_to_database()
        
        # Update session status
        db_manager.update_session_status(session_id, "expand_completed")
        
        logger.info("Processing completed")
        
    except Exception as e:
        logger.critical(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        
        # Update session status with error
        db_manager.update_session_status(session_id, "expand_error", str(e))
        
        sys.exit(1)

if __name__ == "__main__":
    main()
