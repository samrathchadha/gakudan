#!/usr/bin/env python
"""
Database models and connection management for the Expand application.
"""

import os
import datetime
import json
import logging
import threading
import time
from contextlib import contextmanager
from playhouse.pool import PooledPostgresqlExtDatabase
from peewee import (
    Model, TextField, DateTimeField, BooleanField,
    FloatField, ForeignKeyField, IntegerField, 
    SQL, PrimaryKeyField, CompositeKey
)
from playhouse.postgres_ext import JSONField

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("db")

# Database connection settings
DB_HOST = os.environ.get('DB_HOST', '35.192.203.179')
DB_PORT = os.environ.get('DB_PORT', '5432')
DB_NAME = os.environ.get('DB_NAME', 'rev_main')
DB_USER = os.environ.get('DB_USER', 'expand_user')
DB_PASSWORD = os.environ.get('DB_PASSWORD', 'himyNAMEIS123@@')

# Thread-local storage for connection tracking
_connection_local = threading.local()

# Connection pool stats
_pool_stats = {
    'created': 0,
    'closed': 0,
    'active': 0,
    'max_seen': 0
}
_pool_lock = threading.Lock()

# Initialize database with improved connection pooling
database = PooledPostgresqlExtDatabase(
    DB_NAME,
    user=DB_USER,
    password=DB_PASSWORD,
    host=DB_HOST,
    port=DB_PORT,
    max_connections=64,  # Increased from 32
    stale_timeout=60,    # Decreased from 300 to close idle connections sooner
    timeout=30,          # Wait up to 30 seconds for a connection
    autorollback=True,
    register_hstore=False
)

class BaseModel(Model):
    """Base model class that should be inherited by all models."""
    class Meta:
        database = database

class Session(BaseModel):
    """Model representing a user session."""
    session_id = TextField(primary_key=True)
    client_id = TextField()
    prompt = TextField()
    status = TextField()
    contract_status = TextField(default='pending')
    created_at = DateTimeField(default=datetime.datetime.now)
    last_updated = DateTimeField(default=datetime.datetime.now)
    error = TextField(null=True)
    contract_error = TextField(null=True)

    class Meta:
        table_name = 'sessions'

class Node(BaseModel):
    """
    Model representing a node in the graph.
    Uses a composite primary key of session_id and node_id.
    """
    session_id = TextField()
    node_id = TextField()  # Now part of a composite key, not primary key alone
    prompt = TextField(null=True)
    response = TextField(null=True)
    depth = IntegerField(null=True)
    created_at = DateTimeField(default=datetime.datetime.now)
    updated_at = DateTimeField(default=datetime.datetime.now)
    attributes = JSONField(null=True)

    class Meta:
        table_name = 'nodes'
        primary_key = CompositeKey('session_id', 'node_id')  # Create composite primary key
        indexes = (
            (('session_id', 'updated_at'), False),
        )

class Link(BaseModel):
    """Model representing a link between nodes."""
    id = PrimaryKeyField()
    session_id = TextField()
    source_id = TextField()  # Just the node_id without session prefix
    target_id = TextField()  # Just the node_id without session prefix
    edge_type = TextField(null=True)  # 'hierarchy' or 'rag'
    link_type = TextField(null=True)  # For compatibility
    similarity = FloatField(null=True)  # For RAG connections
    created_at = DateTimeField(default=datetime.datetime.now)
    attributes = JSONField(null=True)

    class Meta:
        table_name = 'links'
        indexes = (
            (('session_id', 'created_at'), False),
            (('session_id', 'source_id', 'target_id'), False),
        )

class ClientTracking(BaseModel):
    """Model for tracking client updates."""
    client_id = TextField(primary_key=True)
    session_id = TextField()
    last_check = DateTimeField(default=datetime.datetime.now)

    class Meta:
        table_name = 'client_tracking'

class ContractResult(BaseModel):
    """Model for storing contract results."""
    session_id = TextField(primary_key=True)
    data = JSONField()
    created_at = DateTimeField(default=datetime.datetime.now)

    class Meta:
        table_name = 'contract_results'

# Override the connect and close methods to track connections
_original_connect = database._connect
_original_close = database._close

def _connect_wrapper(*args, **kwargs):
    """Wrapper for the connect method to track connections."""
    conn = _original_connect(*args, **kwargs)
    with _pool_lock:
        _pool_stats['created'] += 1
        _pool_stats['active'] += 1
        _pool_stats['max_seen'] = max(_pool_stats['max_seen'], _pool_stats['active'])
    logger.debug(f"DB connection opened. Active: {_pool_stats['active']}")
    return conn

def _close_wrapper(conn, *args, **kwargs):
    """Wrapper for the close method to track connections."""
    result = _original_close(conn, *args, **kwargs)
    with _pool_lock:
        _pool_stats['closed'] += 1
        _pool_stats['active'] -= 1
    logger.debug(f"DB connection closed. Active: {_pool_stats['active']}")
    return result

# Apply the wrappers
database._connect = _connect_wrapper
database._close = _close_wrapper

@contextmanager
def db_transaction():
    """Context manager for database transactions with improved connection handling."""
    needs_connection = database.is_closed()
    
    try:
        if needs_connection:
            database.connect(reuse_if_open=True)
            logger.debug("Opened new connection for transaction")
        
        with database.atomic() as transaction:
            try:
                yield transaction
                logger.debug("Transaction committed successfully")
            except Exception as e:
                logger.error(f"Database transaction error: {e}", exc_info=True)
                transaction.rollback()
                logger.info("Transaction rolled back due to error")
                raise
                
    except Exception as e:
        # Re-raise any exceptions
        raise
        
    finally:
        # Always close connection we opened in this context
        if needs_connection and not database.is_closed():
            database.close()
            logger.debug("Closed connection after transaction")

def initialize_db():
    """Initialize database connection with connection retry and better error handling."""
    max_retries = 3
    retry_delay = 1  # seconds
    
    for attempt in range(max_retries):
        try:
            if database.is_closed():
                database.connect(reuse_if_open=True)
                logger.info(f"Database connected successfully to {DB_HOST}:{DB_PORT}/{DB_NAME} as {DB_USER}")
                
                # Log pool stats
                logger.info(f"Connection pool stats: {_pool_stats}")
                
                # Force a simple query to verify connection
                database.execute_sql("SELECT 1")
                
                return database
            else:
                logger.info("Database already connected")
                return database
                
        except Exception as e:
            logger.error(f"Database connection error (attempt {attempt+1}/{max_retries}): {e}", exc_info=True)
            if attempt < max_retries - 1:
                logger.info(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                logger.critical("Failed to connect to database after multiple attempts")
                raise

def close_db():
    """Close database connection."""
    if not database.is_closed():
        database.close()
        logger.info("Database connection closed")
        
def get_connection_stats():
    """Get current connection pool statistics."""
    with _pool_lock:
        return dict(_pool_stats)

# Emergency connection cleanup function
def cleanup_connections():
    """Force cleanup of database connections."""
    logger.warning("Performing emergency connection cleanup")
    
    try:
        # Try to close all connections in the pool
        if hasattr(database, '_in_use') and hasattr(database, '_connections'):
            logger.info(f"Before cleanup: {len(database._in_use)} in use, {len(database._connections)} available")
            
            # Close in-use connections
            for conn in list(database._in_use):
                try:
                    database._close(conn)
                except:
                    pass
                    
            # Close available connections
            for conn in list(database._connections):
                try:
                    database._close(conn)
                except:
                    pass
            
            # Reset tracking
            with _pool_lock:
                _pool_stats['active'] = 0
                
            logger.info("Connection pool has been reset")
        else:
            logger.warning("Unable to access connection pool directly")
            
        # Close main connection
        if not database.is_closed():
            database.close()
            
    except Exception as e:
        logger.error(f"Error during connection cleanup: {e}", exc_info=True)

# Initialize tables (should be done on application startup)
def create_tables():
    """Create tables if they don't exist."""
    try:
        # Use a fresh connection
        if not database.is_closed():
            database.close()
            
        with database.connection_context():
            database.create_tables([Session, Node, Link, ClientTracking, ContractResult], safe=True)
            logger.info("Database tables created or verified")
    except Exception as e:
        logger.error(f"Error creating tables: {e}", exc_info=True)
        raise
