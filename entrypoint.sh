#!/bin/bash
set -e

# Display environment info for debugging
echo "=== Environment Information ==="
echo "Python version: $(python --version)"
echo "Database host: ${DB_HOST}"
echo "Database port: ${DB_PORT}"
echo "Database name: ${DB_NAME}"
echo "Database user: ${DB_USER}"

# Create necessary directories
mkdir -p /app/sessions

# Run database diagnostics first
echo "Running database diagnostics..."
python /app/db_diagnostic.py

# Run the application with Gunicorn for better performance
echo "Starting Gunicorn with Flask application..."
exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 "api:app"
