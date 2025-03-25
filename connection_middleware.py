#!/usr/bin/env python
"""
Flask middleware for database connection management.
Ensures connections are properly opened and closed for each request.
"""

import functools
import logging
import time
from flask import g, request, current_app

# Configure logging
logger = logging.getLogger("connection_middleware")

def setup_db_middleware(app, db_module):
    """
    Set up middleware for database connection management.
    
    Args:
        app: Flask application object
        db_module: Database module with initialize_db and close_db functions
    """
    # Before request middleware
    @app.before_request
    def before_request():
        """Ensure database is connected before each request."""
        start_time = time.time()
        g.db_connection_time = start_time
        
        # Initialize database connection if needed
        try:
            if getattr(g, 'db_connected', False):
                logger.debug("Database already connected for this request")
            else:
                db_module.initialize_db()
                g.db_connected = True
                logger.debug(f"Database connected for request {request.path}")
        except Exception as e:
            logger.error(f"Error connecting to database: {e}", exc_info=True)
            # Still allow request to proceed - the view function will handle DB errors
    
    # After request middleware
    @app.after_request
    def after_request(response):
        """Log request duration and connection stats."""
        # Calculate request duration
        if hasattr(g, 'db_connection_time'):
            duration = time.time() - g.db_connection_time
            
            # Get connection stats if available
            if hasattr(db_module, 'get_connection_stats'):
                stats = db_module.get_connection_stats()
                logger.info(f"Request to {request.path} completed in {duration:.3f}s. DB connections: {stats['active']}/{stats.get('max_seen', 0)}")
            else:
                logger.info(f"Request to {request.path} completed in {duration:.3f}s")
        
        return response
    
    # Teardown request middleware
    @app.teardown_request
    def teardown_request(exception):
        """Close database connection after request."""
        if exception:
            logger.warning(f"Request error: {exception}")
            
        # Only close connections we opened in this request
        if getattr(g, 'db_connected', False):
            try:
                db_module.close_db()
                logger.debug("Database connection closed after request")
            except Exception as e:
                logger.error(f"Error closing database connection: {e}", exc_info=True)
    
    # Add route for connection management
    @app.route('/api/db/connections', methods=['GET'])
    def get_connection_info():
        """Get information about database connections."""
        from flask import jsonify
        
        try:
            if hasattr(db_module, 'get_connection_stats'):
                stats = db_module.get_connection_stats()
                return jsonify({
                    "status": "success",
                    "connections": stats
                })
            else:
                return jsonify({
                    "status": "error",
                    "message": "Connection statistics not available"
                })
        except Exception as e:
            return jsonify({
                "status": "error",
                "message": str(e)
            })
    
    # Add route for emergency connection cleanup
    @app.route('/api/db/cleanup', methods=['POST'])
    def cleanup_connections():
        """Force cleanup of database connections."""
        from flask import jsonify
        
        try:
            if hasattr(db_module, 'cleanup_connections'):
                db_module.cleanup_connections()
                return jsonify({
                    "status": "success",
                    "message": "Database connections have been cleaned up"
                })
            else:
                return jsonify({
                    "status": "error",
                    "message": "Connection cleanup function not available"
                })
        except Exception as e:
            return jsonify({
                "status": "error",
                "message": str(e)
            })

def with_database(func):
    """
    Decorator to ensure a database connection is available for a function.
    Useful for background tasks that need database access.
    
    Usage:
        @with_database
        def some_task():
            # This function will have a database connection available
            ...
    """
    import db
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        needs_connection = db.database.is_closed()
        
        try:
            if needs_connection:
                db.initialize_db()
                logger.debug(f"Database connected for function {func.__name__}")
            
            return func(*args, **kwargs)
            
        finally:
            if needs_connection and not db.database.is_closed():
                db.close_db()
                logger.debug(f"Database connection closed after function {func.__name__}")
    
    return wrapper
