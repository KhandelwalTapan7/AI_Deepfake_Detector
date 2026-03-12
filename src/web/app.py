from flask import Flask, render_template, request, jsonify, session
import os
import sys
import json
from datetime import datetime
import hashlib
import uuid

# Fix the path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(src_dir)

# Add paths
sys.path.insert(0, project_root)
sys.path.insert(0, src_dir)

# Now import using absolute paths
from web.utils.helpers import webapp_helper, make_json_serializable

app = Flask(__name__, 
            template_folder=os.path.join(current_dir, 'templates'),
            static_folder=os.path.join(current_dir, 'static'))

# Secret key for session management
app.secret_key = 'deepfake-detector-secret-key-2026'

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB
app.config['UPLOAD_FOLDER'] = os.path.join(project_root, 'uploads')
app.config['HISTORY_FOLDER'] = os.path.join(project_root, 'history')
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

# Create necessary folders
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['HISTORY_FOLDER'], exist_ok=True)

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def save_to_history(filename, result, image_hash):
    """Save analysis to history"""
    history_file = os.path.join(app.config['HISTORY_FOLDER'], 'history.json')
    
    # Load existing history
    history = []
    if os.path.exists(history_file):
        try:
            with open(history_file, 'r') as f:
                history = json.load(f)
        except:
            history = []
    
    # Add new entry
    history.append({
        'id': str(uuid.uuid4()),
        'filename': filename,
        'timestamp': datetime.now().isoformat(),
        'result': result,
        'image_hash': image_hash
    })
    
    # Keep only last 50 entries
    history = history[-50:]
    
    # Save back
    with open(history_file, 'w') as f:
        json.dump(history, f, indent=2)
    
    return history

def get_image_hash(image_data):
    """Generate hash for image to detect duplicates"""
    return hashlib.md5(image_data).hexdigest()

@app.route('/')
def index():
    """Render main page"""
    try:
        return render_template('upload_image.html')
    except Exception as e:
        return f"Template error: {e}. Make sure upload_image.html exists in {app.template_folder}"

@app.route('/analyze', methods=['POST'])
def analyze():
    """Analyze uploaded image"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'File type not allowed'}), 400
    
    try:
        # Read file data for hash
        file_data = file.read()
        image_hash = get_image_hash(file_data)
        
        # Reset file pointer for reading again
        file.seek(0)
        
        # Analyze image
        result = webapp_helper.analyze_image(file)
        
        # Convert any numpy types to Python native types
        serializable_result = make_json_serializable(result)
        
        if 'error' in serializable_result:
            return jsonify(serializable_result), 500
        
        # Save to history
        save_to_history(file.filename, serializable_result, image_hash)
        
        return jsonify(serializable_result)
    
    except Exception as e:
        print(f"Error in analyze endpoint: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/batch_analyze', methods=['POST'])
def batch_analyze():
    """Analyze multiple images"""
    if 'files[]' not in request.files:
        return jsonify({'error': 'No files uploaded'}), 400
    
    files = request.files.getlist('files[]')
    
    if len(files) == 0:
        return jsonify({'error': 'No files selected'}), 400
    
    # Validate files
    for file in files:
        if not allowed_file(file.filename):
            return jsonify({'error': f'File type not allowed: {file.filename}'}), 400
    
    try:
        results = []
        for file in files:
            # Read file data for hash
            file_data = file.read()
            image_hash = get_image_hash(file_data)
            file.seek(0)
            
            # Analyze image
            result = webapp_helper.analyze_image(file)
            serializable_result = make_json_serializable(result)
            
            results.append({
                'filename': file.filename,
                'result': serializable_result,
                'image_hash': image_hash
            })
            
            # Save to history
            save_to_history(file.filename, serializable_result, image_hash)
        
        return jsonify({'results': results})
    
    except Exception as e:
        print(f"Error in batch analyze: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/history', methods=['GET'])
def get_history():
    """Get analysis history"""
    history_file = os.path.join(app.config['HISTORY_FOLDER'], 'history.json')
    
    if os.path.exists(history_file):
        try:
            with open(history_file, 'r') as f:
                history = json.load(f)
            return jsonify({'history': history})
        except:
            return jsonify({'history': []})
    else:
        return jsonify({'history': []})

@app.route('/history/<history_id>', methods=['DELETE'])
def delete_history(history_id):
    """Delete a history entry"""
    history_file = os.path.join(app.config['HISTORY_FOLDER'], 'history.json')
    
    if os.path.exists(history_file):
        try:
            with open(history_file, 'r') as f:
                history = json.load(f)
            
            # Filter out the entry to delete
            history = [h for h in history if h['id'] != history_id]
            
            # Save back
            with open(history_file, 'w') as f:
                json.dump(history, f, indent=2)
            
            return jsonify({'success': True})
        except:
            return jsonify({'error': 'Failed to delete'}), 500
    
    return jsonify({'error': 'History not found'}), 404

@app.route('/stats', methods=['GET'])
def get_stats():
    """Get statistics about analyses"""
    history_file = os.path.join(app.config['HISTORY_FOLDER'], 'history.json')
    
    stats = {
        'total_analyses': 0,
        'real_count': 0,
        'ai_count': 0,
        'deepfake_count': 0,
        'avg_confidence': 0,
        'by_hour': {},
        'by_day': {}
    }
    
    if os.path.exists(history_file):
        try:
            with open(history_file, 'r') as f:
                history = json.load(f)
            
            stats['total_analyses'] = len(history)
            
            if history:
                confidences = []
                for entry in history:
                    result = entry['result']
                    stats['real_count'] += 1 if result['class'] == 'Real Image' else 0
                    stats['ai_count'] += 1 if result['class'] == 'AI Generated' else 0
                    stats['deepfake_count'] += 1 if result['class'] == 'Deepfake' else 0
                    confidences.append(result['confidence'])
                    
                    # Time-based stats
                    timestamp = datetime.fromisoformat(entry['timestamp'])
                    hour = timestamp.strftime('%H:00')
                    day = timestamp.strftime('%Y-%m-%d')
                    
                    stats['by_hour'][hour] = stats['by_hour'].get(hour, 0) + 1
                    stats['by_day'][day] = stats['by_day'].get(day, 0) + 1
                
                stats['avg_confidence'] = sum(confidences) / len(confidences)
        
        except Exception as e:
            print(f"Error loading stats: {e}")
    
    return jsonify(stats)

@app.route('/model/info', methods=['GET'])
def model_info():
    """Get model information"""
    return jsonify({
        'name': 'DeepGuard AI v2.0',
        'classes': ['Real Image', 'AI Generated', 'Deepfake'],
        'input_size': '224x224',
        'framework': 'PyTorch',
        'backend': 'ResNet50 with Attention',
        'accuracy': '94.2% on test set',
        'last_trained': '2026-03-01',
        'model_loaded': webapp_helper.model is not None
    })

@app.route('/compare', methods=['POST'])
def compare_images():
    """Compare multiple images"""
    if 'files[]' not in request.files:
        return jsonify({'error': 'No files uploaded'}), 400
    
    files = request.files.getlist('files[]')
    
    if len(files) < 2:
        return jsonify({'error': 'Need at least 2 images to compare'}), 400
    
    try:
        comparisons = []
        results = []
        
        # Analyze all images
        for file in files:
            file.seek(0)
            result = webapp_helper.analyze_image(file)
            serializable_result = make_json_serializable(result)
            results.append({
                'filename': file.filename,
                'result': serializable_result
            })
        
        # Generate comparisons
        for i in range(len(results)):
            for j in range(i+1, len(results)):
                comparisons.append({
                    'image1': results[i]['filename'],
                    'image2': results[j]['filename'],
                    'same_class': results[i]['result']['class'] == results[j]['result']['class'],
                    'confidence_diff': abs(results[i]['result']['confidence'] - results[j]['result']['confidence'])
                })
        
        return jsonify({
            'results': results,
            'comparisons': comparisons
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/export', methods=['POST'])
def export_results():
    """Export results to CSV/JSON"""
    format_type = request.json.get('format', 'json')
    
    history_file = os.path.join(app.config['HISTORY_FOLDER'], 'history.json')
    
    if not os.path.exists(history_file):
        return jsonify({'error': 'No data to export'}), 404
    
    try:
        with open(history_file, 'r') as f:
            history = json.load(f)
        
        if format_type == 'json':
            return jsonify(history)
        elif format_type == 'csv':
            # Create CSV
            import csv
            from io import StringIO
            
            output = StringIO()
            writer = csv.writer(output)
            
            # Write header
            writer.writerow(['Timestamp', 'Filename', 'Prediction', 'Confidence', 
                           'Real_Prob', 'AI_Prob', 'Deepfake_Prob'])
            
            # Write data
            for entry in history:
                result = entry['result']
                writer.writerow([
                    entry['timestamp'],
                    entry['filename'],
                    result['class'],
                    result['confidence'],
                    result['probabilities'].get('Real Image', 0),
                    result['probabilities'].get('AI Generated', 0),
                    result['probabilities'].get('Deepfake', 0)
                ])
            
            return output.getvalue(), 200, {
                'Content-Type': 'text/csv',
                'Content-Disposition': 'attachment; filename=deepfake_analysis.csv'
            }
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/settings', methods=['POST'])
def update_settings():
    """Update user settings"""
    settings = request.json
    
    # Store settings in session
    session['settings'] = settings
    
    return jsonify({'success': True, 'settings': settings})

@app.route('/settings', methods=['GET'])
def get_settings():
    """Get user settings"""
    settings = session.get('settings', {
        'theme': 'light',
        'auto_save': True,
        'confidence_threshold': 0.6,
        'batch_size': 10
    })
    
    return jsonify(settings)

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': webapp_helper.model is not None,
        'template_folder': app.template_folder,
        'static_folder': app.static_folder,
        'upload_folder': app.config['UPLOAD_FOLDER'],
        'history_folder': app.config['HISTORY_FOLDER']
    })

@app.errorhandler(413)
def too_large(e):
    """Handle file too large error"""
    return jsonify({'error': 'File too large. Maximum size is 16MB.'}), 413

@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors"""
    return jsonify({'error': 'Resource not found'}), 404

@app.errorhandler(500)
def server_error(e):
    """Handle 500 errors"""
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 DEEPGUARD AI - Advanced Deepfake Detector")
    print("="*60)
    print("\n📱 Access the app at:")
    print("   ➜ http://127.0.0.1:5000")
    print("   ➜ http://localhost:5000")
    print("\n📁 Template folder:", app.template_folder)
    print("📁 Static folder:", app.static_folder)
    print("📁 Upload folder:", app.config['UPLOAD_FOLDER'])
    print("📁 History folder:", app.config['HISTORY_FOLDER'])
    print("\n🔧 Features enabled:")
    print("   ✅ Single image analysis")
    print("   ✅ Batch processing")
    print("   ✅ History tracking")
    print("   ✅ Statistics dashboard")
    print("   ✅ Image comparison")
    print("   ✅ Export results (CSV/JSON)")
    print("   ✅ Settings management")
    print("\n⌨️  Press CTRL+C to stop the server")
    print("="*60 + "\n")
    
    # Check if template exists
    template_path = os.path.join(app.template_folder, 'upload_image.html')
    if os.path.exists(template_path):
        print("✅ Template file found")
    else:
        print(f"❌ Template file not found at: {template_path}")
    
    # Run the app
    app.run(debug=True, host='127.0.0.1', port=5000)