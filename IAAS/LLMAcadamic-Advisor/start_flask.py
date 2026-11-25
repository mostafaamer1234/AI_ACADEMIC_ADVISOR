#!/usr/bin/env python3
"""
Flask Server Startup Script
This script properly sets up the Python path and starts the Flask server.
"""

import sys
import os

# Add the current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Now we can import and run the Flask app
if __name__ == '__main__':
    try:
        # Import the Flask app from api module
        from api.api import app
        
        print("🚀 Starting Flask LLM Server...")
        print(f"📁 Working directory: {current_dir}")
        print(f"🐍 Python path: {sys.path[:3]}...")  # Show first 3 paths
        
        # Run the Flask app
        # Debug mode exposes the Werkzeug debugger (remote code execution); never enable it on a public host.
        app.run(debug=os.getenv("FLASK_DEBUG") == "1", port=int(os.getenv("PORT", "5002")), host=os.getenv("HOST", "127.0.0.1"))
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("🔍 Make sure you're running this from the LLMAcadamic-Advisor directory")
        print("📋 Required files:")
        print("   - api/api.py")
        print("   - LLM/advisor_engine.py")
        print("   - LLM/majorsGraph.py")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error starting Flask server: {e}")
        sys.exit(1)
