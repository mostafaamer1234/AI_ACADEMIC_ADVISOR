
import sys
import os
import io
import logging
import hashlib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add the parent directory to Python path BEFORE importing LLM modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Now import LLM modules
from LLM.main_copy import setup_environment
from LLM.main_copy import aggregate_and_generate_response
from LLM.main_copy import generate_conversational_response
from LLM.majorsGraph import MajorGraph

# Import Flask modules
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS, cross_origin
app = Flask(__name__)
CORS(app, origins=['http://localhost:3000', 'http://127.0.0.1:3000', 'http://13.58.145.194', '*'])

# In-memory storage for conversation history (keyed by user_id/session_id)
# In production, use a database or Redis
conversation_storage = {}
user_transcript_hash = {}


def get_cache_key(user_id: str, doc_hash: str, user_question: str, 
                  model_version: str, prompt_version: str) -> str:
    """Generate cache key with all required scoping."""
    import hashlib
    import json
    key_data = {
        "user_id": user_id,
        "doc_hash": doc_hash,
        "question": user_question,
        "model": model_version,
        "prompt": prompt_version
    }
    key_str = json.dumps(key_data, sort_keys=True)
    return hashlib.sha256(key_str.encode()).hexdigest()

def get_user_transcript_hash(user_id: str) -> str:
    """Get transcript hash for user (in-memory)."""
    return user_transcript_hash.get(user_id, "no_transcript")

def compute_file_hash(file_path: str) -> str:
    """Compute SHA256 hash of a file."""
    hasher = hashlib.sha256()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            hasher.update(chunk)
    return hasher.hexdigest()

RESPONSE_CACHE = {}  # In production: Redis
CACHE_TTL_HOURS = 24

@app.route('/llm/generate-response', methods=['POST'])
def generate_graduation_plan():
    try:
        data = request.get_json()
        user_question = data.get('user_question', '')
        conversation_history = data.get('conversation_history', [])
        user_id = data.get('user_id', 'default')  # Use session_id or user_id from frontend
        
        if not user_question:
            return jsonify({"error": "User question is required"}), 400

        # Get doc_hash if transcript exists
        doc_hash = get_user_transcript_hash(user_id)
        model_version = "gpt-4o"  # Or get from config
        prompt_version = "v1"  # Or get from config
        
        # Check cache only when a transcript hash exists; otherwise skip to avoid stale plans
        cache_key = get_cache_key(user_id, doc_hash, user_question, model_version, prompt_version)
        if doc_hash != "no_transcript" and cache_key in RESPONSE_CACHE:
            cached = RESPONSE_CACHE[cache_key]
            from datetime import datetime, timedelta
            if datetime.now() - cached['timestamp'] < timedelta(hours=CACHE_TTL_HOURS):
                # Get existing conversation history
                if user_id not in conversation_storage:
                    conversation_storage[user_id] = []
                if conversation_history and len(conversation_history) > 0:
                    conversation_storage[user_id] = conversation_history
                
                return jsonify({
                    "graduation_plan": cached['response'],
                    "conversation_history": conversation_storage[user_id],
                    "cached": True
                }), 200

        # Get existing conversation history for this user
        # Use the history from frontend if provided, otherwise use stored history
        if user_id not in conversation_storage:
            conversation_storage[user_id] = []
        
        # Update with frontend history if provided (sync frontend and backend state)
        if conversation_history and len(conversation_history) > 0:
            conversation_storage[user_id] = conversation_history
        
        # Generate response with existing conversation history (before adding current message)
        result = generate_conversational_response(
            user_question, 
            conversation_storage[user_id]
        )
        
        # Cache response (only if transcript hash exists)
        if doc_hash != "no_transcript":
            from datetime import datetime
            RESPONSE_CACHE[cache_key] = {
                "response": result,
                "timestamp": datetime.now()
            }
        
        # Add user message and AI response to history after generation
        conversation_storage[user_id].append({
            'role': 'user',
            'content': user_question
        })
        conversation_storage[user_id].append({
            'role': 'assistant',
            'content': result
        })
        
        # Keep only last 50 messages to prevent memory issues
        if len(conversation_storage[user_id]) > 50:
            conversation_storage[user_id] = conversation_storage[user_id][-50:]

        return jsonify({
            "graduation_plan": result,
            "conversation_history": conversation_storage[user_id],
            "cached": False
        }), 200
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500


@app.route('/llm/clear-conversation', methods=['POST'])
def clear_conversation():
    """Clear conversation history for a user"""
    try:
        data = request.get_json()
        user_id = data.get('user_id', 'default')
        
        if user_id in conversation_storage:
            conversation_storage[user_id] = []
        
        return jsonify({"message": "Conversation cleared successfully"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/llm/setup-environment', methods=['POST'])
def handle_setup_environment():
    if 'transcript' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    transcript = request.files['transcript']
    user_id = request.form.get('user_id', 'default')  # Get user_id from form data
    
    # Validate file
    if transcript.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not transcript.filename.lower().endswith('.pdf'):
        return jsonify({'error': 'Only PDF files are allowed'}), 400
    
    try:
        # Save file temporarily to a more accessible location
        import tempfile
        import os
        
        # Create a temporary file with proper extension
        temp_dir = tempfile.gettempdir()
        temp_filename = f"transcript_{os.urandom(8).hex()}.pdf"
        temp_path = os.path.join(temp_dir, temp_filename)
        
        # Save the uploaded file
        transcript.save(temp_path)

        # Track transcript hash per user for cache scoping
        doc_hash = compute_file_hash(temp_path)
        user_transcript_hash[user_id] = doc_hash
        
        # Verify the file was saved and is readable
        if not os.path.exists(temp_path) or os.path.getsize(temp_path) == 0:
            return jsonify({'error': 'Failed to save uploaded file'}), 500
        
        # Create a mock request object with the saved file
        class MockFile:
            def __init__(self, path, original_filename):
                self.path = path
                self.filename = original_filename
            
            def save(self, destination):
                import shutil
                shutil.copy2(self.path, destination)
        
        # Create mock request object
        class MockRequest:
            def __init__(self, file_path, filename):
                self.files = {'transcript': MockFile(file_path, filename)}
        
        mock_request = MockRequest(temp_path, transcript.filename)
        
        # Call setup_environment with user_id
        setup_environment(mock_request, user_id=user_id)
        
        # Clean up temporary file
        try:
            os.unlink(temp_path)
        except:
            pass
            
        return jsonify({
            "message": "Environment setup successfully",
            "doc_hash": doc_hash
        }), 200
        
    except Exception as e:
        # Clean up temporary file on error
        try:
            if 'temp_path' in locals() and os.path.exists(temp_path):
                os.unlink(temp_path)
        except:
            pass
            
        import traceback
        error_details = str(e)
        traceback_str = traceback.format_exc()
        logger.error(f"Setup environment error: {error_details}")
        logger.error(f"Traceback: {traceback_str}")
        # Also print to stderr so systemd can capture it
        import sys
        print(f"Setup environment error: {error_details}", file=sys.stderr)
        print(f"Traceback: {traceback_str}", file=sys.stderr)
        return jsonify({"error": f"Setup failed: {error_details}"}), 500


@app.route('/llm/generate-major-graph', methods=['POST'])
def generate_major_graph():
    try:
        data = request.get_json()
        major_name = data.get('major_name', '')

        if not major_name:
            return jsonify({"error": "Major name is required"}), 400

        # Initialize MajorGraph with the CSV file path
        csv_path = os.path.join(os.path.dirname(__file__), '..', 'LLM', 'all_abington_majors_combined(5).csv')
        
        # Check if CSV file exists
        if not os.path.exists(csv_path):
            logger.error(f"CSV file not found at: {csv_path}")
            return jsonify({"error": f"CSV file not found at: {csv_path}"}), 500
        
        logger.info(f"Generating graph for major: {major_name}")
        mg = MajorGraph(csv_path)

        # Generate the graph and get the image buffer
        image_buffer = mg.draw_major_graph(major_name)

        # Return the image as a response
        return send_file(
            image_buffer,
            mimetype='image/png',
            as_attachment=False
        )

    except ValueError as e:
        # This is raised when major is not found
        logger.error(f"Major not found: {str(e)}")
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        logger.error(f"Error generating major graph: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({"error": f"Failed to generate graph: {str(e)}"}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5002)
