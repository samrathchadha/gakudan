#!/usr/bin/env python
"""
Rewritten Flask API to run expand and contract with simpler logic and extensive logging.
"""

import os
import sys
import json
import uuid
import time
import glob
import shutil
import logging
import threading
import subprocess
from typing import Dict, List, Any, Optional

from flask import Flask, request, jsonify
from flask_cors import CORS

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

# Directory where session data will be stored
SESSIONS_DIR = "sessions"
os.makedirs(SESSIONS_DIR, exist_ok=True)

# Get Python executable
PYTHON_EXECUTABLE = sys.executable

# Sessions and tracking
sessions = {}  # Session info
file_tracking = {}  # Track files for each session
client_tracking = {}  # Track last file seen by each client
lock = threading.Lock()  # Thread safety

def run_expand(session_id: str, prompt: str, api_key: str):
    """Run the expand/runner.py script as a subprocess with extensive logging."""
    session_dir = os.path.abspath(os.path.join(SESSIONS_DIR, session_id))
    os.makedirs(session_dir, exist_ok=True)
    try:
        # Create a session directory
        logger.info(f"Starting expand process for session {session_id} in {session_dir}")
        
        expand_process = subprocess.Popen(
            [
                PYTHON_EXECUTABLE,
                "expand/runner.py",
                "--cwd", session_dir,
                "--prompt", prompt,
                "--api-key", api_key
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
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
        with lock:
            if session_id in sessions:
                if expand_process.returncode == 0:
                    sessions[session_id]["status"] = "expand_completed"
                else:
                    sessions[session_id]["status"] = "expand_error"
                    sessions[session_id]["error"] = f"Expand process failed with code {expand_process.returncode}"
            
            # Initialize or update file tracking for this session
            check_for_new_files(session_id)
        
        logger.info(f"Expand process handler completed for session {session_id}")
        
        # Run contract.py if expand succeeded
        if expand_process.returncode == 0:
            run_contract(session_id, api_key)
    
    except Exception as e:
        logger.error(f"Error running expand for session {session_id}: {e}", exc_info=True)
        with lock:
            if session_id in sessions:
                sessions[session_id]["status"] = "error"
                sessions[session_id]["error"] = str(e)

def run_contract(session_id: str, api_key: str):
    """Run the contract-cli.py script as a subprocess with logging."""
    try:
        # Get the session directory
        session_dir = os.path.abspath(os.path.join(SESSIONS_DIR, session_id))
        
        # Check if expand.json exists
        expand_json = os.path.join(session_dir, "expand.json")
        if not os.path.exists(expand_json):
            logger.error(f"expand.json not found for session {session_id}")
            with lock:
                if session_id in sessions:
                    sessions[session_id]["contract_status"] = "error"
                    sessions[session_id]["contract_error"] = "expand.json not found"
            return
        
        logger.info(f"Starting contract process for session {session_id}")
        
        # Run contract-cli.py with arguments
        contract_process = subprocess.Popen(
            [
                PYTHON_EXECUTABLE,
                "contract.py",
                "--cwd", session_dir,
                "--input", "expand.json",
                "--output", "contract.json",
                "--api-key", api_key
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
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
        with lock:
            if session_id in sessions:
                if contract_process.returncode == 0:
                    sessions[session_id]["contract_status"] = "completed"
                    sessions[session_id]["status"] = "completed"
                else:
                    sessions[session_id]["contract_status"] = "error"
                    sessions[session_id]["contract_error"] = f"Contract process failed with code {contract_process.returncode}"
        
        logger.info(f"Contract process handler completed for session {session_id}")
    
    except Exception as e:
        logger.error(f"Error running contract for session {session_id}: {e}", exc_info=True)
        with lock:
            if session_id in sessions:
                sessions[session_id]["contract_status"] = "error"
                sessions[session_id]["contract_error"] = str(e)

def check_for_new_files(session_id: str):
    """Check for new JSON files and update tracking."""
    try:
        # Get the session directory
        json_dir = os.path.join(SESSIONS_DIR, session_id, "json")
        
        if not os.path.exists(json_dir):
            logger.warning(f"JSON directory does not exist for session {session_id}")
            return False
        
        # Find all JSON files
        json_files = glob.glob(os.path.join(json_dir, "expand_*.json"))
        if not json_files:
            logger.warning(f"No JSON files found for session {session_id}")
            return False
        
        # Sort by creation time
        json_files.sort(key=os.path.getctime)
        
        # Initialize tracking if needed
        if session_id not in file_tracking:
            latest_file = json_files[-1]
            file_tracking[session_id] = {
                "latest_stored": latest_file,
                "latest_accessed": None,
                "all_files": json_files
            }
            logger.info(f"Initialized file tracking for session {session_id}: latest_stored={latest_file}")
            return True
        
        # Check if there are new files
        current_latest = file_tracking[session_id]["latest_stored"]
        new_latest = json_files[-1]
        
        if current_latest != new_latest:
            file_tracking[session_id]["latest_stored"] = new_latest
            file_tracking[session_id]["all_files"] = json_files
            logger.info(f"Updated latest file for session {session_id}: {current_latest} -> {new_latest}")
            return True
        
        return False
    
    except Exception as e:
        logger.error(f"Error checking for new files for session {session_id}: {e}", exc_info=True)
        return False

def get_session_updates(session_id: str, client_id: str) -> Dict:
    """Get updates for a client with simplified logic."""
    logger.info(f"Getting updates for session {session_id}, client {client_id}")
    
    # Check for new files first
    with lock:
        check_for_new_files(session_id)
    
    # If no tracking info, return empty
    if session_id not in file_tracking:
        logger.warning(f"No file tracking for session {session_id}")
        return {
            "has_updates": False,
            "updates": {"nodes": [], "links": []}
        }
    
    # Get tracking info
    tracking = file_tracking[session_id]
    latest_stored = tracking["latest_stored"]
    
    # Get client's last seen file
    client_last_file = None
    if client_id in client_tracking:
        client_last_file = client_tracking.get(client_id, {}).get("last_file")
    
    logger.info(f"Client {client_id} last file: {client_last_file}")
    logger.info(f"Latest file for session {session_id}: {latest_stored}")
    
    # If client has never seen a file or we don't have their record, send the full latest file
    if not client_last_file:
        logger.info(f"Client {client_id} has no last file, sending full data")
        try:
            with open(latest_stored, 'r') as f:
                full_data = json.load(f)
            
            # Update client tracking
            client_tracking[client_id] = {
                "session_id": session_id,
                "last_file": latest_stored,
                "last_check": time.time()
            }
            
            # Update session tracking
            file_tracking[session_id]["latest_accessed"] = latest_stored
            
            return {
                "has_updates": True,
                "is_first_update": True,
                "file_path": latest_stored,
                "updates": full_data
            }
        except Exception as e:
            logger.error(f"Error loading JSON data: {e}", exc_info=True)
            return {
                "has_updates": False,
                "error": str(e)
            }
    
    # If client has already seen the latest file, no updates
    if client_last_file == latest_stored:
        logger.info(f"Client {client_id} already has latest file {latest_stored}")
        return {
            "has_updates": False,
            "updates": {"nodes": [], "links": []}
        }
    
    # Client has seen a file, but not the latest one
    logger.info(f"Client has older file, sending diff or new data")
    try:
        # Load both files
        with open(client_last_file, 'r') as f:
            old_data = json.load(f)
        
        with open(latest_stored, 'r') as f:
            new_data = json.load(f)
        
        # For simplicity, count the differences
        old_node_count = len(old_data.get("nodes", []))
        new_node_count = len(new_data.get("nodes", []))
        old_link_count = len(old_data.get("links", []))
        new_link_count = len(new_data.get("links", []))
        
        logger.info(f"Old file: {old_node_count} nodes, {old_link_count} links")
        logger.info(f"New file: {new_node_count} nodes, {new_link_count} links")
        
        # Update client tracking
        client_tracking[client_id] = {
            "session_id": session_id,
            "last_file": latest_stored,
            "last_check": time.time()
        }
        
        # Update session tracking
        file_tracking[session_id]["latest_accessed"] = latest_stored
        
        # Return the full new data - simpler than diffing for now
        return {
            "has_updates": True,
            "file_path": latest_stored,
            "old_file_path": client_last_file,
            "old_counts": {"nodes": old_node_count, "links": old_link_count},
            "new_counts": {"nodes": new_node_count, "links": new_link_count},
            "updates": new_data
        }
    
    except Exception as e:
        logger.error(f"Error processing updates: {e}", exc_info=True)
        return {
            "has_updates": False,
            "error": str(e)
        }

def is_contract_available(session_id: str) -> bool:
    """Check if contract.json is available for a session."""
    contract_path = os.path.join(SESSIONS_DIR, session_id, "contract.json")
    return os.path.exists(contract_path)

def get_contract_data(session_id: str) -> Optional[Dict]:
    """Get contract data for a session if available."""
    contract_path = os.path.join(SESSIONS_DIR, session_id, "contract.json")
    if not os.path.exists(contract_path):
        return None
    
    try:
        with open(contract_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading contract data: {e}", exc_info=True)
        return None

# Routes
@app.route('/api/status', methods=['GET'])
def status():
    """API status endpoint."""
    return jsonify({
        "status": "online",
        "sessions": len(sessions),
        "clients": len(client_tracking),
        "python_executable": PYTHON_EXECUTABLE
    })

@app.route('/api/sessions', methods=['GET'])
def list_sessions():
    """List all active sessions."""
    with lock:
        session_list = []
        for session_id, session_info in sessions.items():
            # Add file tracking info
            session_data = session_info.copy()
            session_data["tracking"] = file_tracking.get(session_id, {})
            session_list.append(session_data)
        
        return jsonify({
            "sessions": session_list
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
    session_id = str(uuid.uuid4())
    client_id = str(uuid.uuid4())
    
    session_info = {
        "session_id": session_id,
        "client_id": client_id,
        "prompt": prompt,
        "status": "processing",
        "contract_status": "pending",
        "created_at": time.time(),
        "last_updated": time.time()
    }
    
    with lock:
        sessions[session_id] = session_info
        client_tracking[client_id] = {
            "session_id": session_id,
            "last_check": time.time()
        }
    
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
    with lock:
        if session_id not in sessions:
            return jsonify({"error": "Session not found"}), 404
        
        # Get the current status
        status_info = sessions[session_id].copy()
        
        # Check for new files
        has_new_files = check_for_new_files(session_id)
        
        # Add file tracking info
        status_info["file_tracking"] = file_tracking.get(session_id, {})
        status_info["has_new_files"] = has_new_files
        
        # Check if contract.json is available
        status_info["contract_available"] = is_contract_available(session_id)
        
        return jsonify(status_info)

@app.route('/api/sessions/<session_id>/files', methods=['GET'])
def list_session_files(session_id):
    """List all files for a session."""
    session_dir = os.path.join(SESSIONS_DIR, session_id)
    json_dir = os.path.join(session_dir, "json")
    
    # Check if directories exist
    session_exists = os.path.exists(session_dir)
    json_dir_exists = os.path.exists(json_dir)
    
    if not session_exists:
        return jsonify({
            "session_exists": False,
            "error": "Session directory does not exist"
        })
    
    # List files
    expand_json_path = os.path.join(session_dir, "expand.json")
    contract_json_path = os.path.join(session_dir, "contract.json")
    
    # List JSON files if directory exists
    json_files = []
    if json_dir_exists:
        json_files = glob.glob(os.path.join(json_dir, "expand_*.json"))
        json_files.sort(key=os.path.getctime)
    
    return jsonify({
        "session_exists": True,
        "json_dir_exists": json_dir_exists,
        "expand_json_exists": os.path.exists(expand_json_path),
        "contract_json_exists": os.path.exists(contract_json_path),
        "json_file_count": len(json_files),
        "json_files": [os.path.basename(f) for f in json_files],
        "tracking": file_tracking.get(session_id, {})
    })

@app.route('/api/sessions/<session_id>/updates', methods=['GET'])
def get_updates(session_id):
    """Get updates for a session."""
    # Log request
    logger.info(f"Update request for session {session_id} with args: {dict(request.args)}")
    
    # Get client ID
    client_id = request.args.get('client_id')
    force_full = request.args.get('force_full') == 'true'
    
    with lock:
        if session_id not in sessions:
            return jsonify({"error": "Session not found"}), 404
        
        if not client_id:
            # Create a new client ID
            client_id = str(uuid.uuid4())
            client_tracking[client_id] = {
                "session_id": session_id,
                "last_check": time.time()
            }
            logger.info(f"Created new client ID {client_id} for session {session_id}")
        elif client_id not in client_tracking:
            # Unknown client ID, register it
            client_tracking[client_id] = {
                "session_id": session_id,
                "last_check": time.time()
            }
            logger.info(f"Registered unknown client ID {client_id} for session {session_id}")
        
        # If forcing full update, clear client's last file
        if force_full and client_id in client_tracking:
            if "last_file" in client_tracking[client_id]:
                del client_tracking[client_id]["last_file"]
                logger.info(f"Forcing full update for client {client_id}")
        
        # Update session activity time
        sessions[session_id]["last_updated"] = time.time()
    
    # Get updates for this client
    updates = get_session_updates(session_id, client_id)
    
    # Add session status information
    session_status = sessions[session_id]["status"]
    
    # Check if contract data is available
    contract_data = None
    contract_available = is_contract_available(session_id)
    if contract_available and request.args.get('include_contract') == 'true':
        contract_data = get_contract_data(session_id)
    
    response = {
        "session_id": session_id,
        "client_id": client_id,
        "status": session_status,
        "contract_status": sessions[session_id].get("contract_status", "pending"),
        "contract_available": contract_available,
        "force_full": force_full,
        "updates": updates
    }
    
    if contract_data:
        response["contract_data"] = contract_data
    
    # Log response info
    logger.info(f"Sending update response: session={session_id}, "
                f"status={session_status}, "
                f"has_updates={updates.get('has_updates', False)}, "
                f"contract_available={contract_available}")
    
    return jsonify(response)

@app.route('/api/sessions/<session_id>/contract', methods=['GET'])
def get_contract(session_id):
    """Get contract data for a session."""
    with lock:
        if session_id not in sessions:
            return jsonify({"error": "Session not found"}), 404
    
    contract_data = get_contract_data(session_id)
    if not contract_data:
        return jsonify({"error": "Contract data not available"}), 404
    
    return jsonify({
        "session_id": session_id,
        "contract_data": contract_data
    })

@app.route('/api/debug/files/<session_id>', methods=['GET'])
def debug_files(session_id):
    """Debug endpoint to check file content."""
    try:
        # Get file paths
        session_dir = os.path.join(SESSIONS_DIR, session_id)
        json_dir = os.path.join(session_dir, "json")
        expand_json = os.path.join(session_dir, "expand.json")
        
        # List all JSON files
        json_files = []
        if os.path.exists(json_dir):
            json_files = glob.glob(os.path.join(json_dir, "expand_*.json"))
            json_files.sort(key=os.path.getctime)
        
        # Get content of latest file
        latest_content = None
        if json_files:
            try:
                with open(json_files[-1], 'r') as f:
                    latest_content = json.load(f)
            except Exception as e:
                latest_content = {"error": str(e)}
        
        # Get content of expand.json
        expand_content = None
        if os.path.exists(expand_json):
            try:
                with open(expand_json, 'r') as f:
                    expand_content = json.load(f)
            except Exception as e:
                expand_content = {"error": str(e)}
        
        return jsonify({
            "session_dir_exists": os.path.exists(session_dir),
            "json_dir_exists": os.path.exists(json_dir),
            "expand_json_exists": os.path.exists(expand_json),
            "json_file_count": len(json_files),
            "latest_json_file": json_files[-1] if json_files else None,
            "latest_json_size": os.path.getsize(json_files[-1]) if json_files else 0,
            "expand_json_size": os.path.getsize(expand_json) if os.path.exists(expand_json) else 0,
            "file_tracking": file_tracking.get(session_id, {}),
            "latest_content_summary": {
                "node_count": len(latest_content.get("nodes", [])) if latest_content else 0,
                "link_count": len(latest_content.get("links", [])) if latest_content else 0
            },
            "expand_content_summary": {
                "node_count": len(expand_content.get("nodes", [])) if expand_content else 0,
                "link_count": len(expand_content.get("links", [])) if expand_content else 0
            }
        })
    except Exception as e:
        logger.error(f"Error in debug endpoint: {e}", exc_info=True)
        return jsonify({
            "error": str(e)
        })

def cleanup_old_sessions():
    """Periodically clean up old sessions."""
    while True:
        time.sleep(3600)  # Run once per hour
        logger.info("Running cleanup of old sessions")
        
        current_time = time.time()
        with lock:
            # Find sessions older than 24 hours
            to_remove = []
            for session_id, session in sessions.items():
                if current_time - session.get("last_updated", 0) > 86400:  # 24 hours
                    to_remove.append(session_id)
            
            # Remove old sessions
            for session_id in to_remove:
                del sessions[session_id]
                
                # Remove from file tracking
                if session_id in file_tracking:
                    del file_tracking[session_id]
                
                # Remove client tracking for this session
                for client_id in list(client_tracking.keys()):
                    if client_tracking[client_id].get("session_id") == session_id:
                        del client_tracking[client_id]
                
                logger.info(f"Removed old session {session_id}")
                
                # Also clean up files
                session_dir = os.path.join(SESSIONS_DIR, session_id)
                if os.path.exists(session_dir):
                    try:
                        shutil.rmtree(session_dir)
                        logger.info(f"Removed session directory {session_dir}")
                    except Exception as e:
                        logger.error(f"Error removing session directory {session_dir}: {e}")

@app.route('/api/health', methods=['GET'])
def health_check():
    """Simple health check endpoint."""
    process_info = {
        "api_uptime": time.time() - app.config.get("start_time", time.time()),
        "session_count": len(sessions),
        "client_count": len(client_tracking),
        "tracking_count": len(file_tracking)
    }
    
    # Check directories
    data_dirs = {
        "sessions_dir_exists": os.path.exists(SESSIONS_DIR),
        "sessions_dir_files": len(os.listdir(SESSIONS_DIR)) if os.path.exists(SESSIONS_DIR) else 0
    }
    
    # Get disk usage
    if os.path.exists(SESSIONS_DIR):
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(SESSIONS_DIR):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                if os.path.exists(fp):
                    total_size += os.path.getsize(fp)
        
        data_dirs["sessions_dir_size"] = total_size
    
    return jsonify({
        "status": "healthy",
        "time": time.time(),
        "process": process_info,
        "directories": data_dirs
    })

if __name__ == "__main__":
    # Record start time
    app.config["start_time"] = time.time()
    
    # Start cleanup thread
    cleanup_thread = threading.Thread(target=cleanup_old_sessions, daemon=True)
    cleanup_thread.start()
    
    # Log startup
    logger.info("Starting API server")
    logger.info(f"Python executable: {PYTHON_EXECUTABLE}")
    logger.info(f"Sessions directory: {os.path.abspath(SESSIONS_DIR)}")
    
    # Start the server
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host="0.0.0.0", port=port)
