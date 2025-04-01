#!/usr/bin/env python
"""
Modified runner.py that uses PostgreSQL database exclusively - no file operations.
Minimizes database writes by saving only at the end of processing.
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

# Add parent directory to path to import database modules
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

# Debug environment variables
logger.info(f"Environment variables for database:")
for key in ['DB_HOST', 'DB_PORT', 'DB_NAME', 'DB_USER', 'DB_PASSWORD']:
    logger.info(f"{key}={os.environ.get(key, 'NOT SET')}")

# NumPy type conversion utilities
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

try:
    # Import required libraries
    from google import genai
    from prompt_processor import GeminiPromptProcessor
    from prompt_graph import PromptGraph
    
    # Import database modules
    import db
    from db_manager import db_manager
    
    logger.info("All modules imported successfully")
except Exception as e:
    logger.critical(f"Failed to import required modules: {e}", exc_info=True)
    sys.exit(1)

class DatabasePromptGraph(PromptGraph):
    """
    Enhanced PromptGraph that stores nodes and links in the database.
    Only saves to database when explicitly requested.
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
        self.dirty = False  # Track if there are unsaved changes
        logger.info(f"Initialized DatabasePromptGraph for session {session_id}")
    
    def add_node(self, node_id: str, **attributes):
        """
        Add or update a node in the graph. Does not save to database immediately.
        
        Args:
            node_id: Unique identifier for the node
            **attributes: Node attributes (prompt, response, depth, etc.)
        """
        # Call parent method to update in-memory graph
        super().add_node(node_id, **attributes)
        self.dirty = True
    
    def add_edge(self, parent_id: str, child_id: str, edge_attrs: Dict = None, edge_type: str = "hierarchy"):
        """
        Add an edge between nodes. Does not save to database immediately.
        
        Args:
            parent_id: ID of the parent node
            child_id: ID of the child node
            edge_attrs: Optional attributes for the edge
            edge_type: Type of edge ("hierarchy" or "rag")
        """
        # Call parent method to update in-memory graph
        super().add_edge(parent_id, child_id, edge_attrs, edge_type)
        self.dirty = True
    
    def add_rag_connection(self, source_id: str, target_id: str, similarity: float = None):
        """
        Add an explicit RAG connection between nodes. Does not save to database immediately.
        
        Args:
            source_id: ID of the source node (providing context)
            target_id: ID of the target node (receiving context)
            similarity: Optional similarity score
        """
        # Call parent method to update in-memory graph
        super().add_rag_connection(source_id, target_id, similarity)
        self.dirty = True
    
    def save_to_database(self):
        """
        Save all graph data to the database at once.
        Overwrites existing data for this session.
        
        Returns:
            True if successful, False otherwise
        """
        if not self.dirty:
            logger.info(f"No changes to save for session {self.session_id}")
            return True
            
        try:
            # Format the data explicitly for database
            data = self.format_for_database()
            
            # Save contract data in one operation (will overwrite existing)
            db_manager.save_contract_data(self.session_id, data)
            logger.info(f"Graph data saved to database for session {self.session_id}")
            
            # Reset dirty flag
            self.dirty = False
            
            return True
        except Exception as e:
            logger.error(f"Error saving graph to database: {e}", exc_info=True)
            return False
            
    def save_nodes_and_links(self):
        """
        Save all nodes and links to the database in a single operation.
        """
        if not self.dirty:
            logger.info(f"No changes to save for session {self.session_id}")
            return True
            
        try:
            # Format data for database
            data = self.format_for_database()
            
            # Get nodes and links separately
            nodes = data["nodes"]
            links = data["links"]
            
            # Save in bulk operations
            with db.db_transaction() as transaction:
                # First, clear existing data for this session
                db.Node.delete().where(db.Node.session_id == self.session_id).execute()
                db.Link.delete().where(db.Link.session_id == self.session_id).execute()
                
                # Bulk insert nodes
                for node in nodes:
                    node_id = node["id"]
                    prompt = node.get("prompt")
                    response = node.get("response")
                    depth = node.get("depth")
                    
                    # Extract attributes
                    attributes = dict(node)
                    for key in ["id", "prompt", "response", "depth"]:
                        if key in attributes:
                            attributes.pop(key, None)
                    
                    # Add to database
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
                    similarity = link.get("similarity")
                    
                    # Extract attributes
                    attributes = dict(link)
                    for key in ["source", "target", "edge_type", "similarity"]:
                        if key in attributes:
                            attributes.pop(key, None)
                    
                    # Add to database
                    db.Link.create(
                        session_id=self.session_id,
                        source_id=source,
                        target_id=target,
                        edge_type=edge_type,
                        link_type=edge_type,  # Use edge_type as link_type for compatibility
                        similarity=similarity,
                        attributes=attributes if attributes else None
                    )
            
            # Reset dirty flag
            self.dirty = False
            
            logger.info(f"Saved {len(nodes)} nodes and {len(links)} links to database for session {self.session_id}")
            return True
        except Exception as e:
            logger.error(f"Error saving nodes and links to database: {e}", exc_info=True)
            return False

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
        # Call the original method - does not save to database
        result = super().process_sub_prompt(task_info)
        return result
    
    def add_to_queue(self, sub_prompt, parent_id, depth=0):
        # Call original method - does not save to database
        result = super().add_to_queue(sub_prompt, parent_id, depth)
        return result
    
    def process_main_prompt(self, main_prompt):
        # Set root prompt
        self.prompt_graph.add_node('root', prompt=main_prompt, depth=-1)
        
        # Process normally - does not save to database during processing
        results = super().process_main_prompt(main_prompt)
        
        # Save final state to database
        logger.info("Processing complete, saving final state to database")
        self.prompt_graph.save_to_database()
        
        # Update session status
        db_manager.update_session_status(self.session_id, "expand_completed")
        
        return results

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
    
    logger.info(f"Runner starting for session {session_id}")
    logger.info(f"Prompt: {args.prompt}")
    
    # Verify database connection
    if not verify_database_connection(session_id):
        logger.critical("Database verification failed. Exiting.")
        sys.exit(1)
    
    # Initialize our processor with database support
    processor = DatabasePromptProcessor(args.api_key, session_id)
    
    try:
        # Process the prompt - saves to database at the end
        logger.info(f"Processing prompt for session {session_id}: {args.prompt}")
        results = processor.process_main_prompt(args.prompt)
        
        logger.info("Processing completed successfully")
        return 0
        
    except Exception as e:
        logger.critical(f"Unexpected error: {e}", exc_info=True)
        import traceback
        traceback.print_exc()
        
        # Update session status with error
        db_manager.update_session_status(session_id, "expand_error", str(e))
        
        sys.exit(1)

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except Exception as e:
        logger.critical(f"Fatal error in main: {e}", exc_info=True)
        sys.exit(1)
