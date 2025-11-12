#!/bin/bash

# Script to update the application on AWS EC2
# Usage: ./update_ec2.sh

echo "========================================="
echo "Updating application on AWS EC2"
echo "========================================="

# Navigate to the project directory
cd /path/to/your/project || exit 1

echo "Step 1: Fetching latest changes from GitHub..."
git fetch origin

echo "Step 2: Pulling latest changes..."
git pull origin main

echo "Step 3: Checking for any new dependencies..."
# If you have a requirements.txt for Python
if [ -f "IAAS/LLMAcadamic-Advisor/api/requirements.txt" ]; then
    echo "Installing/updating Python dependencies..."
    pip3 install -r IAAS/LLMAcadamic-Advisor/api/requirements.txt --user
fi

# If you have a package.json for Node.js
if [ -f "IAAS/client/package.json" ]; then
    echo "Installing/updating Node.js dependencies..."
    cd IAAS/client
    npm install
    cd ../..
fi

echo "Step 4: Restarting services..."
# Stop existing Flask server if running
pkill -f "python.*api.py" || echo "No Flask server found running"

# Start Flask server in background (adjust path and port as needed)
echo "Starting Flask server..."
cd IAAS/LLMAcadamic-Advisor/api
nohup python3 api.py > /var/log/flask_app.log 2>&1 &
echo "Flask server started. PID: $!"

# If you're using systemd or PM2 for process management, use those instead:
# sudo systemctl restart your-flask-service
# pm2 restart your-app-name

echo "========================================="
echo "Update completed!"
echo "========================================="
echo "Flask server should be running on port 5002"
echo "Check logs with: tail -f /var/log/flask_app.log"

