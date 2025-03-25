#!/usr/bin/env python
"""
Flask API using PostgreSQL database for storage instead of local files.
With improved connection management.
"""

import os
import sys
import time
import uuid
import logging
import threading
import subprocess
from typing import Dict, Any, Optional

from flask import Flask, request, jsonify, g
from flask_cors import CORS

# Import database modules
import db
from db_manager import db_manager
from connection_middleware import setup_db_middleware

# Configure logging with timestamps
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("api_server.log")
    ]
)
logger = logging.getLogger("api_server")

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Setup database connection middleware
setup_db_middleware(app, db)

# Get Python executable
PYTHON_EXECUTABLE = sys.executable

# Thread safety
expand_process_lock = threading.Lock()
contract_process_lock = threading.Lock()

def run_expand(session_id: str, prompt: str, api_key: str):
    """Run the expand/runner.py script as a subprocess with database integration."""
    try:
        logger.info(f"Starting expand process for session {session_id}")
        
        # Update session status
        db_manager.update_session_status(session_id, "processing")
        
        # Prepare environment variables for database connection
        env = os.environ.copy()
        env["SESSION_ID"] = session_id
        env["DB_HOST"] = db.DB_HOST
        env["DB_PORT"] = str(db.DB_PORT)  # Ensure port is a string
        env["DB_NAME"] = db.DB_NAME
        env["DB_USER"] = db.DB_USER
        env["DB_PASSWORD"] = db.DB_PASSWORD
        
        logger.info(f"Running expand with env: HOST={env['DB_HOST']}, PORT={env['DB_PORT']}, DB={env['DB_NAME']}")
        
        with expand_process_lock:
            expand_process = subprocess.Popen(
                [
                    PYTHON_EXECUTABLE,
                    "expand/runner.py",
                    "--prompt", prompt,
                    "--api-key", api_key,
                    "--session-id", session_id
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env
            )
            
            # Capture and log output in real-time
            def log_output(pipe, prefix):
                for line in iter(pipe.readline, ''):
                    if line:
                        logger.info(f"{prefix}: {line.rstrip()}")
            
            # Start threads to log stdout and stderr
            stdout_thread = threading.Thread(
                target=log_output, 
                args=(expand_process.stdout, f"EXPAND-STDOUT[{session_id}]"),
                daemon=True
            )
            stderr_thread = threading.Thread(
                target=log_output, 
                args=(expand_process.stderr, f"EXPAND-STDERR[{session_id}]"),
                daemon=True
            )
            stdout_thread.start()
            stderr_thread.start()
            
            # Wait for process to complete
            logger.info(f"Waiting for expand process to complete for session {session_id}")
            expand_process.wait()
            logger.info(f"Expand process completed with code {expand_process.returncode} for session {session_id}")
            
            # Wait for output logging to finish
            stdout_thread.join(timeout=5)
            stderr_thread.join(timeout=5)
            
            # Update session status
            if expand_process.returncode == 0:
                db_manager.update_session_status(session_id, "expand_completed")
                logger.info(f"Session {session_id} marked as expand_completed")
            else:
                error_msg = f"Expand process failed with code {expand_process.returncode}"
                logger.error(error_msg)
                db_manager.update_session_status(
                    session_id, 
                    "expand_error", 
                    error_msg
                )
            
            logger.info(f"Expand process handler completed for session {session_id}")
            
            # Run contract.py if expand succeeded
            if expand_process.returncode == 0:
                run_contract(session_id, api_key)
    
    except Exception as e:
        logger.error(f"Error running expand for session {session_id}: {e}", exc_info=True)
        db_manager.update_session_status(session_id, "error", str(e))

def run_contract(session_id: str, api_key: str):
    """Run the contract.py script as a subprocess with database integration."""
    try:
        logger.info(f"Starting contract process for session {session_id}")
        
        # Update session contract status
        db_manager.update_contract_status(session_id, "processing")
        
        # Prepare environment variables for database connection
        env = os.environ.copy()
        env["SESSION_ID"] = session_id
        env["DB_HOST"] = db.DB_HOST
        env["DB_PORT"] = str(db.DB_PORT)  # Ensure port is a string
        env["DB_NAME"] = db.DB_NAME
        env["DB_USER"] = db.DB_USER
        env["DB_PASSWORD"] = db.DB_PASSWORD
        
        logger.info(f"Running contract with env: HOST={env['DB_HOST']}, PORT={env['DB_PORT']}, DB={env['DB_NAME']}")
        
        with contract_process_lock:
            contract_process = subprocess.Popen(
                [
                    PYTHON_EXECUTABLE,
                    "contract.py",
                    "--session-id", session_id,
                    "--api-key", api_key
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env
            )
            
            # Capture and log output in real-time
            def log_output(pipe, prefix):
                for line in iter(pipe.readline, ''):
                    if line:
                        logger.info(f"{prefix}: {line.rstrip()}")
            
            # Start threads to log stdout and stderr
            stdout_thread = threading.Thread(
                target=log_output, 
                args=(contract_process.stdout, f"CONTRACT-STDOUT[{session_id}]"),
                daemon=True
            )
            stderr_thread = threading.Thread(
                target=log_output, 
                args=(contract_process.stderr, f"CONTRACT-STDERR[{session_id}]"),
                daemon=True
            )
            stdout_thread.start()
            stderr_thread.start()
            
            # Wait for process to complete
            logger.info(f"Waiting for contract process to complete for session {session_id}")
            contract_process.wait()
            logger.info(f"Contract process completed with code {contract_process.returncode} for session {session_id}")
            
            # Wait for output logging to finish
            stdout_thread.join(timeout=5)
            stderr_thread.join(timeout=5)
            
            # Update session status
            if contract_process.returncode == 0:
                db_manager.update_contract_status(session_id, "completed")
                db_manager.update_session_status(session_id, "completed")
                logger.info(f"Session {session_id} marked as completed")
            else:
                error_msg = f"Contract process failed with code {contract_process.returncode}"
                logger.error(error_msg)
                db_manager.update_contract_status(
                    session_id,
                    "error",
                    error_msg
                )
            
            logger.info(f"Contract process handler completed for session {session_id}")
    
    except Exception as e:
        logger.error(f"Error running contract for session {session_id}: {e}", exc_info=True)
        db_manager.update_contract_status(session_id, "error", str(e))

# Routes
@app.route('/api/status', methods=['GET'])
def status():
    """API status endpoint."""
    return jsonify({
        "status": "online",
        "python_executable": PYTHON_EXECUTABLE,
        "database": db_manager.get_database_stats(),
        "db_connections": db.get_connection_stats() if hasattr(db, 'get_connection_stats') else None
    })

@app.route('/api/sessions', methods=['GET'])
def list_sessions():
    """List all active sessions."""
    sessions = db_manager.get_all_sessions()
    return jsonify({
        "sessions": sessions
    })

@app.route('/api/sessions', methods=['POST'])
def create_session():
    """Create a new session and start processing a prompt."""
    data = request.json
    if not data:
        return jsonify({"error": "Missing request body"}), 400
    
    prompt = data.get('prompt')
    api_key = data.get('api_key')
    
    if not prompt:
        return jsonify({"error": "Missing prompt parameter"}), 400
    
    if not api_key:
        return jsonify({"error": "Missing API key parameter"}), 400
    
    # Create a new session
    session_id, client_id = db_manager.create_session(prompt, api_key)
    
    # Start the expand process in a background thread
    expand_thread = threading.Thread(
        target=run_expand,
        args=(session_id, prompt, api_key),
        daemon=True
    )
    expand_thread.start()
    
    logger.info(f"Created new session {session_id} for client {client_id}")
    
    return jsonify({
        "session_id": session_id,
        "client_id": client_id,
        "status": "processing"
    })

@app.route('/api/sessions/<session_id>', methods=['GET'])
def get_session(session_id):
    """Get details about a specific session."""
    session_data = db_manager.get_session_data(session_id)
    if not session_data:
        return jsonify({"error": "Session not found"}), 404
    
    # Add contract availability information
    session_data["contract_available"] = db_manager.is_contract_available(session_id)
    
    return jsonify(session_data)

@app.route('/api/sessions/<session_id>/updates', methods=['GET'])
def get_updates(session_id):
    """Get updates for a session."""
    # Log request
    logger.info(f"Update request for session {session_id} with args: {dict(request.args)}")
    
    # Get client ID
    client_id = request.args.get('client_id')
    force_full = request.args.get('force_full') == 'true'
    include_contract = request.args.get('include_contract') == 'true'
    
    # Check if session exists
    session_data = db_manager.get_session_data(session_id)
    if not session_data:
        return jsonify({"error": "Session not found"}), 404
    
    # If no client ID provided, create a new one
    if not client_id:
        client_id = str(uuid.uuid4())
        logger.info(f"Created new client ID {client_id} for session {session_id}")
    
    # Get updates for this client
    updates = db_manager.get_session_updates(session_id, client_id, force_full)
    
    # Add contract data if requested and available
    if include_contract and db_manager.is_contract_available(session_id):
        updates["contract_data"] = db_manager.get_contract_data(session_id)
    
    # Log response info
    logger.info(f"Sending update response: session={session_id}, "
                f"status={updates.get('status')}, "
                f"has_updates={updates['updates'].get('has_updates', False)}, "
                f"contract_available={updates.get('contract_available', False)}")
    
    return jsonify(updates)

@app.route('/api/sessions/<session_id>/contract', methods=['GET'])
def get_contract(session_id):
    """Get contract data for a session."""
    # Check if session exists
    session_data = db_manager.get_session_data(session_id)
    if not session_data:
        return jsonify({"error": "Session not found"}), 404
    
    # Get contract data
    contract_data = db_manager.get_contract_data(session_id)
    if not contract_data:
        return jsonify({"error": "Contract data not available"}), 404
    
    return jsonify({
        "session_id": session_id,
        "contract_data": contract_data
    })

@app.route('/api/debug/stats/<session_id>', methods=['GET'])
def debug_stats(session_id):
    """Debug endpoint to check session statistics."""
    try:
        # Check if session exists
        session_data = db_manager.get_session_data(session_id)
        if not session_data:
            return jsonify({"error": "Session not found"}), 404
        
        # Get node and link counts
        node_count = db.Node.select().where(db.Node.session_id == session_id).count()
        link_count = db.Link.select().where(db.Link.session_id == session_id).count()
        
        # Get latest node and link
        latest_node = db.Node.select().where(db.Node.session_id == session_id).order_by(db.Node.updated_at.desc()).first()
        latest_link = db.Link.select().where(db.Link.session_id == session_id).order_by(db.Link.created_at.desc()).first()
        
        stats = {
            "session_id": session_id,
            "node_count": node_count,
            "link_count": link_count,
            "latest_node": {
                "id": latest_node.node_id,
                "updated_at": latest_node.updated_at.timestamp()
            } if latest_node else None,
            "latest_link": {
                "source": latest_link.source_id,
                "target": latest_link.target_id,
                "created_at": latest_link.created_at.timestamp()
            } if latest_link else None,
            "contract_available": db_manager.is_contract_available(session_id)
        }
        
        return jsonify(stats)
    except Exception as e:
        logger.error(f"Error in debug endpoint: {e}", exc_info=True)
        return jsonify({
            "error": str(e)
        })

@app.route('/api/health', methods=['GET'])
def health_check():
    """Simple health check endpoint."""
    try:
        # Test database connection with a simple query
        db.database.execute_sql("SELECT 1").fetchone()
        
        # Get database stats
        db_stats = db_manager.get_database_stats()
        
        # Get connection stats
        conn_stats = db.get_connection_stats() if hasattr(db, 'get_connection_stats') else None
        
        return jsonify({
            "status": "healthy",
            "time": time.time(),
            "database": {
                "connected": True,
                "stats": db_stats
            },
            "connections": conn_stats
        })
    except Exception as e:
        logger.error(f"Health check failed: {e}", exc_info=True)
        return jsonify({
            "status": "unhealthy",
            "time": time.time(),
            "error": str(e)
        }), 500

def cleanup_old_sessions_job():
    """Periodically clean up old sessions."""
    from connection_middleware import with_database
    
    @with_database
    def do_cleanup():
        logger.info("Running cleanup of old sessions")
        deleted_count = db_manager.cleanup_old_sessions(hours=24)
        logger.info(f"Deleted {deleted_count} old sessions")
    
    while True:
        try:
            time.sleep(3600)  # Run once per hour
            do_cleanup()
        except Exception as e:
            logger.error(f"Error in cleanup job: {e}", exc_info=True)
            time.sleep(60)  # Wait a minute before retrying

if __name__ == "__main__":
    # Initialize the database
    db.initialize_db()
    db.create_tables()
    
    # Start cleanup thread
    cleanup_thread = threading.Thread(target=cleanup_old_sessions_job, daemon=True)
    cleanup_thread.start()
    
    # Log startup
    logger.info("Starting API server")
    logger.info(f"Python executable: {PYTHON_EXECUTABLE}")
    logger.info(f"Database host: {db.DB_HOST}")
    logger.info(f"Database port: {db.DB_PORT}")
    logger.info(f"Database name: {db.DB_NAME}")
    logger.info(f"Database user: {db.DB_USER}")
    
    # Start the server
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host="0.0.0.0", port=port)
