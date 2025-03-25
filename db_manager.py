#!/usr/bin/env python
"""
Database manager handling operations for the Expand application.
Handles graph construction, querying, and JSON conversion.
"""

import logging
import datetime
import json
import time
import uuid
from typing import Dict, List, Any, Optional, Tuple
from peewee import fn, SQL, DoesNotExist
import db

logger = logging.getLogger("db_manager")

class DatabaseManager:
    """Manager for database operations with graph handling."""
    
    def __init__(self):
        """Initialize the database manager."""
        self.db = db.database
    
    def create_session(self, prompt: str, api_key: str) -> Tuple[str, str]:
        """
        Create a new session in the database.
        
        Args:
            prompt: The main prompt for the session
            api_key: API key for the session
            
        Returns:
            Tuple of (session_id, client_id)
        """
        session_id = str(uuid.uuid4())
        client_id = str(uuid.uuid4())
        current_time = datetime.datetime.now()
        
        # Ensure DB is connected
        if self.db.is_closed():
            db.initialize_db()
        
        with db.db_transaction() as transaction:
            # Create session
            session = db.Session.create(
                session_id=session_id,
                client_id=client_id,
                prompt=prompt,
                status="processing",
                contract_status="pending",
                created_at=current_time,
                last_updated=current_time
            )
            
            # Create client tracking
            client_tracking = db.ClientTracking.create(
                client_id=client_id,
                session_id=session_id,
                last_check=current_time
            )
            
            # Create root node for the session
            db.Node.create(
                node_id='root',
                session_id=session_id,
                prompt=prompt,
                depth=-1,
                created_at=current_time,
                updated_at=current_time
            )
            
            logger.info(f"Created new session {session_id} for client {client_id}")
            
        return session_id, client_id
    
    def update_session_status(self, session_id: str, status: str, error: str = None) -> bool:
        """
        Update the status of a session.
        
        Args:
            session_id: ID of the session to update
            status: New status value
            error: Optional error message
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Ensure DB is connected
            if self.db.is_closed():
                db.initialize_db()
                
            with db.db_transaction() as transaction:
                session = db.Session.get(db.Session.session_id == session_id)
                session.status = status
                session.last_updated = datetime.datetime.now()
                
                if error:
                    session.error = error
                    
                session.save()
                logger.info(f"Updated session {session_id} status to {status}")
                return True
                
        except DoesNotExist:
            logger.error(f"Session {session_id} not found")
            return False
        except Exception as e:
            logger.error(f"Error updating session {session_id}: {e}", exc_info=True)
            return False

    def update_contract_status(self, session_id: str, status: str, error: str = None) -> bool:
        """
        Update the contract status of a session.
        
        Args:
            session_id: ID of the session to update
            status: New contract status value
            error: Optional error message
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Ensure DB is connected
            if self.db.is_closed():
                db.initialize_db()
                
            with db.db_transaction() as transaction:
                session = db.Session.get(db.Session.session_id == session_id)
                session.contract_status = status
                session.last_updated = datetime.datetime.now()
                
                if error:
                    session.contract_error = error
                    
                session.save()
                logger.info(f"Updated session {session_id} contract status to {status}")
                return True
                
        except DoesNotExist:
            logger.error(f"Session {session_id} not found")
            return False
        except Exception as e:
            logger.error(f"Error updating contract status for session {session_id}: {e}", exc_info=True)
            return False
    
    def add_node(self, node_id: str, session_id: str, prompt: str = None, 
                 response: str = None, depth: int = None, 
                 attributes: Dict = None) -> bool:
        """
        Add or update a node in the database using composite primary key.
        
        Args:
            node_id: Node identifier (now part of composite key with session_id)
            session_id: ID of the session this node belongs to
            prompt: Node prompt text
            response: Node response text
            depth: Node depth in the hierarchy
            attributes: Additional node attributes
            
        Returns:
            True if successful, False otherwise
        """
        try:
            current_time = datetime.datetime.now()
            
            # Ensure DB is connected
            if self.db.is_closed():
                db.initialize_db()
                
            with db.db_transaction() as transaction:
                # Check if node exists using composite key
                try:
                    node = db.Node.get(
                        (db.Node.session_id == session_id) & 
                        (db.Node.node_id == node_id)
                    )
                    # Update existing node
                    if prompt is not None:
                        node.prompt = prompt
                    if response is not None:
                        node.response = response
                    if depth is not None:
                        node.depth = depth
                    if attributes is not None:
                        node.attributes = attributes
                    
                    node.updated_at = current_time
                    node.save()
                    logger.debug(f"Updated node {node_id} in session {session_id}")
                    
                except DoesNotExist:
                    # Create new node
                    node = db.Node.create(
                        node_id=node_id,
                        session_id=session_id,
                        prompt=prompt,
                        response=response,
                        depth=depth,
                        attributes=attributes,
                        created_at=current_time,
                        updated_at=current_time
                    )
                    logger.debug(f"Created node {node_id} in session {session_id}")
                
                return True
                
        except Exception as e:
            logger.error(f"Error adding/updating node {node_id} in session {session_id}: {e}", exc_info=True)
            return False
    
    def add_link(self, session_id: str, source_id: str, target_id: str,
                edge_type: str = None, link_type: str = None,
                similarity: float = None, attributes: Dict = None) -> bool:
        """
        Add a link between nodes in the database.
        
        Args:
            session_id: ID of the session this link belongs to
            source_id: ID of the source node
            target_id: ID of the target node
            edge_type: Type of edge ('hierarchy' or 'rag')
            link_type: Link type for compatibility
            similarity: Similarity score for RAG connections
            attributes: Additional link attributes
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Ensure DB is connected
            if self.db.is_closed():
                db.initialize_db()
                
            with db.db_transaction() as transaction:
                # Check if the same link already exists
                existing_link = db.Link.select().where(
                    (db.Link.session_id == session_id) &
                    (db.Link.source_id == source_id) &
                    (db.Link.target_id == target_id) &
                    (db.Link.edge_type == edge_type)
                ).first()
                
                if existing_link:
                    # Link already exists, update attributes if provided
                    if attributes:
                        existing_link.attributes = attributes
                    if similarity is not None:
                        existing_link.similarity = similarity
                    if link_type is not None:
                        existing_link.link_type = link_type
                    
                    existing_link.save()
                    logger.debug(f"Updated link from {source_id} to {target_id} in session {session_id}")
                else:
                    # Create new link
                    link = db.Link.create(
                        session_id=session_id,
                        source_id=source_id,
                        target_id=target_id,
                        edge_type=edge_type,
                        link_type=link_type or edge_type,  # Use edge_type as link_type if not provided
                        similarity=similarity,
                        attributes=attributes,
                        created_at=datetime.datetime.now()
                    )
                    logger.debug(f"Created link from {source_id} to {target_id} in session {session_id}")
                
                return True
                
        except Exception as e:
            logger.error(f"Error adding link from {source_id} to {target_id} in session {session_id}: {e}", exc_info=True)
            return False
    
    def get_node_data(self, node_id: str, session_id: str) -> Dict:
        """
        Get data for a specific node using composite key.
        
        Args:
            node_id: ID of the node to retrieve
            session_id: ID of the session the node belongs to
            
        Returns:
            Dictionary with node data
        """
        try:
            # Ensure DB is connected
            if self.db.is_closed():
                db.initialize_db()
                
            node = db.Node.get(
                (db.Node.session_id == session_id) & 
                (db.Node.node_id == node_id)
            )
            
            node_data = {
                'id': node.node_id,  # Return just the node_id to the user
                'prompt': node.prompt,
                'response': node.response,
                'depth': node.depth
            }
            
            # Add attributes if they exist
            if node.attributes:
                for key, value in node.attributes.items():
                    node_data[key] = value
                    
            return node_data
        except DoesNotExist:
            logger.warning(f"Node {node_id} in session {session_id} not found")
            return {}
        except Exception as e:
            logger.error(f"Error getting node data for {node_id} in session {session_id}: {e}", exc_info=True)
            return {}
    
    def get_session_data(self, session_id: str) -> Dict:
        """
        Get detailed data for a session including status.
        
        Args:
            session_id: ID of the session to retrieve
            
        Returns:
            Dictionary with session data
        """
        try:
            # Ensure DB is connected
            if self.db.is_closed():
                db.initialize_db()
                
            session = db.Session.get(db.Session.session_id == session_id)
            return {
                'session_id': session.session_id,
                'client_id': session.client_id,
                'prompt': session.prompt,
                'status': session.status,
                'contract_status': session.contract_status,
                'created_at': session.created_at.timestamp(),
                'last_updated': session.last_updated.timestamp(),
                'error': session.error,
                'contract_error': session.contract_error
            }
        except DoesNotExist:
            logger.warning(f"Session {session_id} not found")
            return {}
        except Exception as e:
            logger.error(f"Error getting session data for {session_id}: {e}", exc_info=True)
            return {}
    
    def format_graph_json(self, session_id: str, 
                          after_time: Optional[datetime.datetime] = None) -> Dict:
        """
        Format nodes and links as a graph JSON structure.
        
        Args:
            session_id: The session ID to get data for
            after_time: Optional timestamp to get only data after this time
            
        Returns:
            Dictionary with nodes and links in the expected format
        """
        try:
            # Ensure DB is connected
            if self.db.is_closed():
                db.initialize_db()
                
            # Add detailed logging
            if after_time:
                logger.info(f"Getting graph data for session {session_id} after {after_time}")
            else:
                logger.info(f"Getting all graph data for session {session_id}")
                
            nodes_query = db.Node.select().where(db.Node.session_id == session_id)
            links_query = db.Link.select().where(db.Link.session_id == session_id)
            
            # Filter by time if provided
            if after_time:
                nodes_query = nodes_query.where(db.Node.updated_at > after_time)
                links_query = links_query.where(db.Link.created_at > after_time)
            
            # Log query counts before executing
            logger.info(f"Query will retrieve nodes and links for session {session_id}")
            
            # Execute queries
            nodes_list = list(nodes_query)
            links_list = list(links_query)
            
            logger.info(f"Retrieved {len(nodes_list)} nodes and {len(links_list)} links for session {session_id}")
            
            # Format nodes
            nodes = []
            for node in nodes_list:
                node_data = {
                    'id': node.node_id,  # Just show node_id to user, not the composite key
                    'prompt': node.prompt,
                    'depth': node.depth
                }
                
                # Add response if it exists
                if node.response:
                    node_data['response'] = node.response
                
                # Add attributes if they exist
                if node.attributes:
                    for key, value in node.attributes.items():
                        node_data[key] = value
                
                nodes.append(node_data)
            
            # Format links
            links = []
            for link in links_list:
                link_data = {
                    'source': link.source_id,
                    'target': link.target_id
                }
                
                # Add edge and link types if they exist
                if link.edge_type:
                    link_data['edge_type'] = link.edge_type
                if link.link_type:
                    link_data['link_type'] = link.link_type
                
                # Add similarity if it exists
                if link.similarity is not None:
                    link_data['similarity'] = link.similarity
                
                # Add attributes if they exist
                if link.attributes:
                    for key, value in link.attributes.items():
                        link_data[key] = value
                
                links.append(link_data)
            
            return {
                'directed': True,
                'multigraph': True,
                'graph': {},
                'nodes': nodes,
                'links': links
            }
        except Exception as e:
            logger.error(f"Error formatting graph JSON for session {session_id}: {e}", exc_info=True)
            return {
                'directed': True,
                'multigraph': True,
                'graph': {},
                'nodes': [],
                'links': []
            }
    
    def get_session_updates(self, session_id: str, client_id: str, 
                           force_full: bool = False) -> Dict:
        """
        Get updates for a client with a specific session.
        
        Args:
            session_id: ID of the session
            client_id: ID of the client
            force_full: Whether to force a full update
            
        Returns:
            Dictionary with update information
        """
        try:
            # Ensure DB is connected
            if self.db.is_closed():
                db.initialize_db()
                
            # Log detailed information
            logger.info(f"Getting updates for session {session_id}, client {client_id}, force_full={force_full}")
            
            # Get client tracking info
            try:
                client = db.ClientTracking.get(db.ClientTracking.client_id == client_id)
                last_check = client.last_check
                logger.info(f"Found existing client tracking for {client_id}, last check at {last_check}")
                
            except DoesNotExist:
                # Create new client tracking if it doesn't exist
                last_check = datetime.datetime.now() - datetime.timedelta(days=30)  # Get everything
                client = db.ClientTracking.create(
                    client_id=client_id,
                    session_id=session_id,
                    last_check=last_check
                )
                logger.info(f"Created new client tracking for {client_id} with session {session_id}")
            
            # Force full update if requested (ignore last_check)
            if force_full:
                last_check = datetime.datetime.now() - datetime.timedelta(days=30)
                logger.info(f"Forcing full update for client {client_id}")
            
            # Get graph data
            graph_data = self.format_graph_json(session_id, None if force_full else last_check)
            
            # Update client's last check time
            current_time = datetime.datetime.now()
            client.last_check = current_time
            client.save()
            
            # Check if there are updates
            has_updates = len(graph_data['nodes']) > 0 or len(graph_data['links']) > 0
            
            # Get session data for status information
            session_data = self.get_session_data(session_id)
            
            # Check for contract data
            contract_available = self.is_contract_available(session_id)
            
            # Log detailed information about updates
            logger.info(f"Updates for session {session_id}: has_updates={has_updates}, " +
                      f"nodes={len(graph_data['nodes'])}, links={len(graph_data['links'])}")
            
            return {
                'session_id': session_id,
                'client_id': client_id,
                'status': session_data.get('status', 'unknown'),
                'contract_status': session_data.get('contract_status', 'pending'),
                'contract_available': contract_available,
                'force_full': force_full,
                'updates': {
                    'has_updates': has_updates,
                    'is_first_update': force_full,
                    'updates': graph_data
                }
            }
        except Exception as e:
            logger.error(f"Error getting session updates for {session_id}, client {client_id}: {e}", exc_info=True)
            return {
                'session_id': session_id,
                'client_id': client_id,
                'status': 'error',
                'error': str(e),
                'updates': {
                    'has_updates': False,
                    'updates': {
                        'nodes': [],
                        'links': []
                    }
                }
            }
    
    def is_contract_available(self, session_id: str) -> bool:
        """
        Check if contract result is available for a session.
        
        Args:
            session_id: ID of the session to check
            
        Returns:
            True if contract result is available, False otherwise
        """
        try:
            # Ensure DB is connected
            if self.db.is_closed():
                db.initialize_db()
                
            return db.ContractResult.select().where(
                db.ContractResult.session_id == session_id
            ).exists()
        except Exception as e:
            logger.error(f"Error checking contract availability for {session_id}: {e}", exc_info=True)
            return False
    
    def get_contract_data(self, session_id: str) -> Optional[Dict]:
        """
        Get contract data for a session if available.
        
        Args:
            session_id: ID of the session to retrieve contract data for
            
        Returns:
            Contract data dictionary or None if not available
        """
        try:
            # Ensure DB is connected
            if self.db.is_closed():
                db.initialize_db()
                
            contract = db.ContractResult.get(db.ContractResult.session_id == session_id)
            return contract.data
        except DoesNotExist:
            logger.info(f"No contract data found for session {session_id}")
            return None
        except Exception as e:
            logger.error(f"Error getting contract data for {session_id}: {e}", exc_info=True)
            return None
    
    def save_contract_data(self, session_id: str, data: Dict) -> bool:
        """
        Save contract result data for a session.
        
        Args:
            session_id: ID of the session
            data: Contract result data
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Ensure DB is connected
            if self.db.is_closed():
                db.initialize_db()
                
            with db.db_transaction() as transaction:
                # Check if contract result already exists
                contract, created = db.ContractResult.get_or_create(
                    session_id=session_id,
                    defaults={
                        'data': data,
                        'created_at': datetime.datetime.now()
                    }
                )
                
                # Update existing contract result if not created
                if not created:
                    contract.data = data
                    contract.created_at = datetime.datetime.now()
                    contract.save()
                
                # Update session status
                self.update_contract_status(session_id, "completed")
                
                logger.info(f"Saved contract data for session {session_id}")
                return True
        except Exception as e:
            logger.error(f"Error saving contract data for {session_id}: {e}", exc_info=True)
            return False
    
    def get_all_sessions(self) -> List[Dict]:
        """
        Get a list of all sessions.
        
        Returns:
            List of session dictionaries
        """
        try:
            # Ensure DB is connected
            if self.db.is_closed():
                db.initialize_db()
                
            sessions = []
            for session in db.Session.select().order_by(db.Session.created_at.desc()):
                session_data = {
                    'session_id': session.session_id,
                    'client_id': session.client_id,
                    'prompt': session.prompt,
                    'status': session.status,
                    'contract_status': session.contract_status,
                    'created_at': session.created_at.timestamp(),
                    'last_updated': session.last_updated.timestamp(),
                    'error': session.error,
                    'contract_error': session.contract_error,
                    'contract_available': self.is_contract_available(session.session_id)
                }
                
                # Get node and link counts for this session
                node_count = db.Node.select().where(db.Node.session_id == session.session_id).count()
                link_count = db.Link.select().where(db.Link.session_id == session.session_id).count()
                
                session_data['tracking'] = {
                    'node_count': node_count,
                    'link_count': link_count
                }
                
                sessions.append(session_data)
            
            return sessions
        except Exception as e:
            logger.error(f"Error getting all sessions: {e}", exc_info=True)
            return []
    
    def cleanup_old_sessions(self, hours: int = 24) -> int:
        """
        Clean up sessions older than the specified number of hours.
        
        Args:
            hours: Number of hours before a session is considered old
            
        Returns:
            Number of sessions deleted
        """
        try:
            # Ensure DB is connected
            if self.db.is_closed():
                db.initialize_db()
                
            cutoff_time = datetime.datetime.now() - datetime.timedelta(hours=hours)
            old_sessions = db.Session.select().where(db.Session.last_updated < cutoff_time)
            
            count = 0
            for session in old_sessions:
                session_id = session.session_id
                with db.db_transaction() as transaction:
                    # Delete will cascade to related tables due to foreign key constraints
                    db.Session.delete().where(db.Session.session_id == session_id).execute()
                    count += 1
                    logger.info(f"Deleted old session {session_id}")
            
            return count
        except Exception as e:
            logger.error(f"Error cleaning up old sessions: {e}", exc_info=True)
            return 0
    
    def get_database_stats(self) -> Dict:
        """
        Get statistics about the database.
        
        Returns:
            Dictionary with database statistics
        """
        try:
            # Ensure DB is connected
            if self.db.is_closed():
                db.initialize_db()
                
            stats = {
                'session_count': db.Session.select().count(),
                'node_count': db.Node.select().count(),
                'link_count': db.Link.select().count(),
                'client_count': db.ClientTracking.select().count(),
                'contract_count': db.ContractResult.select().count(),
                'timestamp': datetime.datetime.now().timestamp()
            }
            
            return stats
        except Exception as e:
            logger.error(f"Error getting database stats: {e}", exc_info=True)
            return {
                'error': str(e),
                'timestamp': datetime.datetime.now().timestamp()
            }

# Create a global instance of the database manager that can be imported
db_manager = DatabaseManager()
