#!/bin/bash
set -e

# Create necessary directories
mkdir -p /app/sessions

# Run the application with Gunicorn for better performance
exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 "api:app"
