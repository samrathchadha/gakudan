#!/usr/bin/env python
"""
Database models and connection management for the Expand application.
"""

import os
import datetime
import json
import logging
from contextlib import contextmanager
from playhouse.pool import PooledPostgresqlExtDatabase
from peewee import (
    Model, TextField, DateTimeField, BooleanField,
    FloatField, ForeignKeyField, IntegerField, 
    SQL, PrimaryKeyField
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

# Initialize database with connection pooling
database = PooledPostgresqlExtDatabase(
    DB_NAME,
    user=DB_USER,
    password=DB_PASSWORD,
    host=DB_HOST,
    port=DB_PORT,
    max_connections=32,
    stale_timeout=300,
    autorollback=True
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
    """Model representing a node in the graph."""
    node_id = TextField(primary_key=True)
    session_id = TextField()
    prompt = TextField(null=True)
    response = TextField(null=True)
    depth = IntegerField(null=True)
    created_at = DateTimeField(default=datetime.datetime.now)
    updated_at = DateTimeField(default=datetime.datetime.now)
    attributes = JSONField(null=True)

    class Meta:
        table_name = 'nodes'
        indexes = (
            (('session_id', 'updated_at'), False),
        )

class Link(BaseModel):
    """Model representing a link between nodes."""
    id = PrimaryKeyField()
    session_id = TextField()
    source_id = TextField()
    target_id = TextField()
    edge_type = TextField(null=True)  # 'hierarchy' or 'rag'
    link_type = TextField(null=True)  # For compatibility
    similarity = FloatField(null=True)  # For RAG connections
    created_at = DateTimeField(default=datetime.datetime.now)
    attributes = JSONField(null=True)

    class Meta:
        table_name = 'links'
        indexes = (
            (('session_id', 'created_at'), False),
            (('source_id', 'target_id'), False),
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

@contextmanager
def db_transaction():
    """Context manager for database transactions."""
    with database.atomic():
        try:
            yield
        except Exception as e:
            logger.error(f"Database transaction error: {e}", exc_info=True)
            raise

def initialize_db():
    """Initialize database connection."""
    database.connect(reuse_if_open=True)
    logger.info("Database connected successfully")
    return database

def close_db():
    """Close database connection."""
    if not database.is_closed():
        database.close()
        logger.info("Database connection closed")

# Initialize tables (should be done on application startup)
def create_tables():
    """Create tables if they don't exist."""
    with database:
        database.create_tables([Session, Node, Link, ClientTracking, ContractResult])
        logger.info("Database tables created or verified")
