#!/bin/bash
# Populates a realistic developer workspace for eval

set -e

cd /home/developer

# --- Git config ---
git config --global user.name "Developer"
git config --global user.email "dev@example.com"
git config --global init.defaultBranch main

# --- .bashrc extras ---
cat >> ~/.bashrc << 'BASHRC'
export EDITOR=vim
export PROJECT_ROOT=/home/developer/projects/webapp
alias ll='ls -alF'
alias gs='git status'
BASHRC

# --- Fake project repo: a Python/JS web app ---
mkdir -p projects/webapp/{src,tests,docs,scripts,config}
cd projects/webapp
git init

# Python files
cat > src/main.py << 'EOF'
"""Main application entry point."""
import os
from src.utils import load_config, setup_logging
from src.api import create_app

def main():
    config = load_config()
    setup_logging(config.get("log_level", "INFO"))
    app = create_app(config)
    # TODO: add graceful shutdown handler
    app.run(host="0.0.0.0", port=config.get("port", 8080))

if __name__ == "__main__":
    main()
EOF

cat > src/utils.py << 'EOF'
"""Utility functions."""
import json
import logging
import os

def load_config(path="config/settings.json"):
    """Load configuration from JSON file."""
    if not os.path.exists(path):
        return {"port": 8080, "debug": False, "log_level": "INFO"}
    with open(path) as f:
        return json.load(f)

def setup_logging(level="INFO"):
    """Configure application logging."""
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

def sanitize_input(text):
    """Basic input sanitization."""
    # TODO: implement proper sanitization
    return text.strip()

def format_bytes(size):
    """Format byte size to human readable string."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} PB"
EOF

cat > src/api.py << 'EOF'
"""API route definitions."""

def create_app(config):
    """Create and configure the application."""
    class App:
        def __init__(self, config):
            self.config = config
            self.routes = {}

        def route(self, path):
            def decorator(func):
                self.routes[path] = func
                return func
            return decorator

        def run(self, host="0.0.0.0", port=8080):
            print(f"Server running on {host}:{port}")

    app = App(config)

    @app.route("/health")
    def health():
        return {"status": "ok"}

    @app.route("/api/users")
    def list_users():
        # TODO: implement database query
        return []

    return app
EOF

cat > src/__init__.py << 'EOF'
EOF

cat > tests/test_utils.py << 'EOF'
"""Tests for utility functions."""
from src.utils import format_bytes, sanitize_input

def test_format_bytes():
    assert format_bytes(0) == "0.0 B"
    assert format_bytes(1024) == "1.0 KB"
    assert format_bytes(1048576) == "1.0 MB"

def test_sanitize_input():
    assert sanitize_input("  hello  ") == "hello"
    assert sanitize_input("normal") == "normal"
EOF

cat > tests/test_api.py << 'EOF'
"""Tests for API routes."""
from src.api import create_app

def test_create_app():
    app = create_app({"port": 3000})
    assert "/health" in app.routes
    assert "/api/users" in app.routes
EOF

# JavaScript files
cat > src/frontend.js << 'EOF'
// Frontend entry point
const API_BASE = '/api';

async function fetchUsers() {
    const response = await fetch(`${API_BASE}/users`);
    return response.json();
}

async function checkHealth() {
    const response = await fetch(`${API_BASE}/health`);
    return response.json();
}

// TODO: add error handling for network failures
module.exports = { fetchUsers, checkHealth };
EOF

# Config files
cat > config/settings.json << 'EOF'
{
    "port": 8080,
    "debug": false,
    "log_level": "INFO",
    "database": {
        "host": "localhost",
        "port": 5432,
        "name": "webapp_db"
    },
    "allowed_origins": ["http://localhost:3000"]
}
EOF

cat > requirements.txt << 'EOF'
flask==3.0.0
pytest==7.4.0
requests==2.31.0
gunicorn==21.2.0
EOF

cat > package.json << 'EOF'
{
    "name": "webapp-frontend",
    "version": "1.0.0",
    "scripts": {
        "start": "node src/frontend.js",
        "test": "jest"
    }
}
EOF

cat > .gitignore << 'EOF'
__pycache__/
*.pyc
node_modules/
.env
*.log
dist/
build/
.pytest_cache/
EOF

cat > README.md << 'EOF'
# WebApp

A simple web application with Python backend and JS frontend.

## Setup
```
pip install -r requirements.txt
python -m src.main
```

## Testing
```
pytest tests/
```
EOF

# Scripts
cat > scripts/deploy.sh << 'EOF'
#!/bin/bash
echo "Deploying to production..."
echo "Building frontend..."
echo "Running migrations..."
echo "Restarting services..."
echo "Deploy complete."
EOF
chmod +x scripts/deploy.sh

cat > scripts/backup.sh << 'EOF'
#!/bin/bash
BACKUP_DIR="/home/developer/backups"
DATE=$(date +%Y%m%d_%H%M%S)
mkdir -p "$BACKUP_DIR"
tar -czf "$BACKUP_DIR/webapp_$DATE.tar.gz" -C /home/developer/projects webapp
echo "Backup created: $BACKUP_DIR/webapp_$DATE.tar.gz"
EOF
chmod +x scripts/backup.sh

# Docs
cat > docs/architecture.md << 'EOF'
# Architecture

## Components
- **API Server**: Python-based REST API
- **Frontend**: JavaScript client
- **Database**: PostgreSQL (config in settings.json)

## Directory Structure
- `src/` - Source code
- `tests/` - Test files
- `config/` - Configuration
- `scripts/` - Deployment and maintenance scripts
- `docs/` - Documentation
EOF

# Git commits to create realistic history
git add -A
git commit -m "Initial project setup" --quiet

# Add some log files and misc
cd /home/developer
mkdir -p logs backups .ssh tmp

# Log files of various ages
for i in $(seq 1 5); do
    echo "$(date -d "-${i} days" 2>/dev/null || date) - Application started" > logs/app_day${i}.log
    echo "$(date -d "-${i} days" 2>/dev/null || date) - Request processed" >> logs/app_day${i}.log
    echo "$(date -d "-${i} days" 2>/dev/null || date) - Connection closed" >> logs/app_day${i}.log
done

# Some larger files for disk usage tests
dd if=/dev/zero of=tmp/large_file_1.dat bs=1M count=10 2>/dev/null
dd if=/dev/zero of=tmp/large_file_2.dat bs=1M count=5 2>/dev/null
dd if=/dev/zero of=tmp/small_file.dat bs=1K count=100 2>/dev/null

# Nested directory structure for find tests
mkdir -p projects/old_project/{src,build,dist}
echo "print('old code')" > projects/old_project/src/legacy.py
echo "print('old test')" > projects/old_project/src/test_legacy.py
echo "compiled" > projects/old_project/build/output.o
echo "bundled" > projects/old_project/dist/bundle.js

# Duplicate files for dedup tests
echo "duplicate content here" > tmp/file_a.txt
echo "duplicate content here" > tmp/file_b.txt
echo "unique content" > tmp/file_c.txt

# Files with specific extensions for search tests
echo "# Notes from meeting" > docs_notes.md
echo "Important data" > data.csv
echo '<?xml version="1.0"?><root></root>' > config.xml

# Add a second git commit in the webapp
cd /home/developer/projects/webapp
echo "" >> src/utils.py
echo "def get_version():" >> src/utils.py
echo '    return "1.0.0"' >> src/utils.py
git add -A
git commit -m "Add version helper function" --quiet

# Create a branch
git checkout -b feature/user-auth --quiet
cat > src/auth.py << 'EOF'
"""Authentication module."""

def authenticate(username, password):
    """Authenticate a user (stub)."""
    # TODO: implement real auth
    return username == "admin" and password == "admin"

def generate_token(user_id):
    """Generate an auth token."""
    import hashlib
    return hashlib.sha256(f"token_{user_id}".encode()).hexdigest()
EOF
git add -A
git commit -m "Add authentication module (WIP)" --quiet
git checkout main --quiet

echo "Workspace setup complete."
