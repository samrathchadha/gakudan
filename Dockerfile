FROM python:3.10-slim

WORKDIR /app

# Install dependencies required for sentence-transformers and PostgreSQL
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all the application code
COPY . .

# Make entrypoint script executable
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

# Expose the port the app runs on
EXPOSE 8080

# Environment variables
ENV PORT=8080
ENV PYTHONUNBUFFERED=1
ENV DB_HOST=35.192.203.179
ENV DB_PORT=5432
ENV DB_NAME=rev_main
ENV DB_USER=expand_user
ENV DB_PASSWORD=himyNAMEIS123@@

# Run the application with Gunicorn
ENTRYPOINT ["/app/entrypoint.sh"]
