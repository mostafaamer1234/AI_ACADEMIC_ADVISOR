
import sys
import os
import io

# Add the parent directory to Python path BEFORE importing LLM modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Now import LLM modules
from LLM.main_copy import setup_environment
from LLM.main_copy import aggregate_and_generate_response
from LLM.majorsGraph import MajorGraph

# Import Flask modules
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS, cross_origin
app = Flask(__name__)
CORS(app, origins=['http://localhost:3000', 'http://127.0.0.1:3000'])


@app.route('/llm/generate-response', methods=['POST'])
def generate_graduation_plan():
    try:
        data = request.get_json()
        user_question = data.get('user_question', '')

        if not user_question:
            return jsonify({"error": "User question is required"}), 400

        result = aggregate_and_generate_response(user_question)

        return jsonify({"graduation_plan": result}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/llm/setup-environment', methods=['POST'])
def handle_setup_environment():
    if 'transcript' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    transcript = request.files['transcript']
    
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
        
        # Call setup_environment
        setup_environment(mock_request)
        
        # Clean up temporary file
        try:
            os.unlink(temp_path)
        except:
            pass
            
        return jsonify({"message": "Environment setup successfully"}), 200
        
    except Exception as e:
        # Clean up temporary file on error
        try:
            if 'temp_path' in locals() and os.path.exists(temp_path):
                os.unlink(temp_path)
        except:
            pass
            
        import traceback
        error_details = str(e)
        print(f"Setup environment error: {error_details}")
        print(f"Traceback: {traceback.format_exc()}")
        return jsonify({"error": f"Setup failed: {error_details}"}), 500


@app.route('/llm/generate-major-graph', methods=['POST'])
def generate_major_graph():
    try:
        data = request.get_json()
        major_name = data.get('major_name', '')

        if not major_name:
            return jsonify({"error": "Major name is required"}), 400

        # Initialize MajorGraph with the CSV file path
        csv_path = os.path.join(os.path.dirname(
            __file__), '..', 'LLM', '/Users/mostafa/newFianlLLMRepo/IAAS/LLMAcadamic-Advisor/LLM/all_abington_majors_combined(5).csv')
        mg = MajorGraph(csv_path)

        # Generate the graph and get the image buffer
        image_buffer = mg.draw_major_graph(major_name)

        # Return the image as a response
        return send_file(
            image_buffer,
            mimetype='image/png',
            as_attachment=False
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5002)
