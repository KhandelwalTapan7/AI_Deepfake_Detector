"""
AURORA Deepfake Detector - Complete Working Application
Architecture : EfficientNet-B0  (matches Colab training notebook)
Classes      : ai_generated | deepfake | real  (alphabetical, as trained)
"""

from flask import Flask, render_template, request, jsonify
import os
import json
import uuid
import hashlib
import traceback
from datetime import datetime
from PIL import Image
from werkzeug.utils import secure_filename
import cv2
import numpy as np

# PyTorch imports
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torchvision import models, transforms
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️ PyTorch not installed. Install with: pip install torch torchvision")

# ============================================================
# Flask App Initialization
# ============================================================
app = Flask(__name__)
app.secret_key = 'aurora-deepfake-secret-2026'

# ============================================================
# Configuration
# ============================================================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

UPLOAD_FOLDER   = os.path.join(PROJECT_ROOT, 'uploads')
HISTORY_FOLDER  = os.path.join(PROJECT_ROOT, 'history')
MODEL_FOLDER    = os.path.join(PROJECT_ROOT, 'models')
TEMPLATE_FOLDER = os.path.join(PROJECT_ROOT, 'templates')

for folder in [UPLOAD_FOLDER, HISTORY_FOLDER, MODEL_FOLDER, TEMPLATE_FOLDER]:
    os.makedirs(folder, exist_ok=True)

app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024
app.config['UPLOAD_FOLDER']      = UPLOAD_FOLDER
app.config['HISTORY_FOLDER']     = HISTORY_FOLDER
app.config['MODEL_FOLDER']       = MODEL_FOLDER
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}

# ============================================================
# Constants  — must match the notebook exactly
# ============================================================
CLASS_NAMES = ['ai_generated', 'deepfake', 'real']

CLASS_DISPLAY = {
    'ai_generated': 'AI Generated',
    'deepfake':     'Deepfake',
    'real':         'Real Image',
}

MODEL_ACCURACY = 95.27
IMG_SIZE       = 224
RESIZE_TO      = 232
MEAN           = [0.485, 0.456, 0.406]
STD            = [0.229, 0.224, 0.225]
CONFIDENCE_THRESHOLD = 0.50

# ============================================================
# Model Architecture — matches Cell 6 exactly (EfficientNet-B0)
# ============================================================
class DeepfakeDetector(nn.Module):
    def __init__(self, num_classes=3, dropout=0.4):
        super().__init__()
        self.backbone = models.efficientnet_b0(weights=None)
        in_f = self.backbone.classifier[1].in_features  # 1280 for B0
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(in_f, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(p=dropout / 2, inplace=True),
            nn.Linear(256, num_classes),
        )
        self.num_classes = num_classes

    def forward(self, x):
        return self.backbone(x)


# ============================================================
# Global Variables
# ============================================================
_model        = None
_device       = None
_model_loaded = False
_transform    = None


# ============================================================
# Model Loading
# ============================================================
def find_model_file():
    search_paths = [
        os.path.join(MODEL_FOLDER, 'deepfake_detector_export.pth'),
        os.path.join(MODEL_FOLDER, 'best_model.pth'),
        os.path.join(MODEL_FOLDER, 'model.pth'),
        os.path.join(PROJECT_ROOT, 'models', 'deepfake_detector_export.pth'),
        os.path.join(PROJECT_ROOT, 'models', 'best_model.pth'),
        os.path.join(PROJECT_ROOT, 'deepfake_detector_export.pth'),
        os.path.join(PROJECT_ROOT, 'best_model.pth'),
        './models/deepfake_detector_export.pth',
        './models/best_model.pth',
        './deepfake_detector_export.pth',
        './best_model.pth',
    ]
    for path in search_paths:
        if os.path.exists(path):
            return path
    return None


def load_model():
    global _model, _device, _model_loaded, _transform, CLASS_NAMES, CLASS_DISPLAY

    if not TORCH_AVAILABLE:
        print("⚠️ PyTorch not available — using forensic fallback mode")
        return False

    model_path = find_model_file()
    if model_path is None:
        print("⚠️ No model checkpoint found — using forensic fallback mode")
        print(f"   Searched in: {MODEL_FOLDER}")
        return False

    try:
        _device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"📱 Device: {_device}")

        checkpoint = torch.load(model_path, map_location=_device)

        if 'model_state_dict' in checkpoint:
            state_dict   = checkpoint['model_state_dict']
            epoch        = checkpoint.get('epoch', '?')
            accuracy     = checkpoint.get('val_accuracy', 0)
            arch         = checkpoint.get('model_arch', 'efficientnet_b0')
            ckpt_classes = checkpoint.get('class_names', CLASS_NAMES)
            print(f"📊 Checkpoint epoch={epoch} | val_acc={accuracy*100:.2f}% | arch={arch}")
            print(f"   Checkpoint class_names: {ckpt_classes}")
        else:
            state_dict   = checkpoint
            ckpt_classes = CLASS_NAMES
            print("📊 Checkpoint loaded (direct state dict)")

        if ckpt_classes != CLASS_NAMES:
            print(f"⚠️  Class name mismatch — using checkpoint names: {ckpt_classes}")
            CLASS_NAMES   = ckpt_classes
            CLASS_DISPLAY = {c: c.replace('_', ' ').title() for c in ckpt_classes}

        num_classes = len(CLASS_NAMES)
        _model = DeepfakeDetector(num_classes=num_classes).to(_device)
        _model.load_state_dict(state_dict, strict=True)
        _model.eval()

        _transform = transforms.Compose([
            transforms.Resize(RESIZE_TO),
            transforms.CenterCrop(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])

        _model_loaded = True
        print(f"✅ Model loaded successfully from: {model_path}")
        print(f"   Classes : {CLASS_NAMES}")
        print(f"   Accuracy: {MODEL_ACCURACY}%")
        return True

    except Exception as e:
        print(f"❌ Model load error: {e}")
        traceback.print_exc()
        return False


# ============================================================
# STARTUP — runs at import time, works with Gunicorn AND python app.py
# This is the KEY fix: Gunicorn never runs if __name__ == '__main__'
# ============================================================
def _startup():
    print("\n" + "=" * 60)
    print("  🔍 AURORA STARTUP (Gunicorn-compatible)")
    print("=" * 60)
    print(f"  Architecture : EfficientNet-B0")
    print(f"  Project root : {PROJECT_ROOT}")
    print(f"  Model folder : {MODEL_FOLDER}")

    # Step 1: Download model if not present
    try:
        from download_model import download_model
        download_model()
    except ImportError:
        print("  ℹ️  download_model.py not found — skipping download")
    except Exception as e:
        print(f"  ⚠️  Download error: {e}")
        traceback.print_exc()

    # Step 2: Load model into memory
    load_model()

    print("=" * 60)
    print(f"  Mode: {'Neural Network ✅' if _model_loaded else 'Forensic fallback ⚠️'}")
    print("=" * 60 + "\n")

# Called at module import — Gunicorn imports app.py, so this always runs
_startup()


# ============================================================
# Neural Network Prediction
# ============================================================
def predict_with_model(image_path):
    global _model, _device, _transform

    try:
        image        = Image.open(image_path).convert('RGB')
        input_tensor = _transform(image).unsqueeze(0).to(_device)

        with torch.no_grad():
            logits = _model(input_tensor)
            probs  = torch.softmax(logits, dim=1)[0].cpu().numpy()

        pred_idx   = int(np.argmax(probs))
        confidence = float(probs[pred_idx])
        pred_class = CLASS_NAMES[pred_idx]
        display    = CLASS_DISPLAY.get(pred_class, pred_class)

        print(f"\n🔬 Analyzing: {os.path.basename(image_path)}")
        for i, name in enumerate(CLASS_NAMES):
            bar = '█' * int(probs[i] * 40)
            print(f"   {name:>16}: {probs[i]*100:6.2f}%  {bar}")
        print(f"   → Prediction: {display} ({confidence*100:.2f}% confidence)")

        if pred_class == 'real':
            recommendation = (
                f"✅ AUTHENTIC IMAGE — {confidence*100:.1f}% confidence. "
                "This appears to be a genuine photograph."
            )
        elif pred_class == 'ai_generated':
            recommendation = (
                f"🤖 AI GENERATED — {confidence*100:.1f}% confidence "
                "this image was created by artificial intelligence."
            )
        else:
            recommendation = (
                f"🚨 DEEPFAKE DETECTED — {confidence*100:.1f}% confidence "
                "of facial manipulation."
            )

        prob_dict = {CLASS_NAMES[i]: round(float(probs[i]), 3) for i in range(len(CLASS_NAMES))}

        return {
            'class':                display,
            'class_key':            pred_class,
            'confidence':           round(confidence, 3),
            'real_probability':     prob_dict.get('real', 0.0),
            'ai_probability':       prob_dict.get('ai_generated', 0.0),
            'deepfake_probability': prob_dict.get('deepfake', 0.0),
            'fake_probability':     round(1.0 - prob_dict.get('real', 0.0), 3),
            'recommendation':       recommendation,
            'analysis_mode':        'neural_network',
            'model_accuracy':       MODEL_ACCURACY,
        }

    except Exception as e:
        print(f"Prediction error: {e}")
        traceback.print_exc()
        return None


# ============================================================
# Forensic Analysis (Fallback when model is unavailable)
# ============================================================
def forensic_analysis(image_path):
    try:
        img = cv2.imread(image_path)
        if img is None:
            return None

        gray          = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        faces    = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
        has_face = len(faces) > 0

        if laplacian_var < 50:
            pred_class = 'ai_generated'
            confidence = 0.65
        elif laplacian_var > 200:
            pred_class = 'real'
            confidence = 0.70
        elif has_face and laplacian_var < 120:
            pred_class = 'deepfake'
            confidence = 0.60
        else:
            pred_class = 'real'
            confidence = 0.60

        display = CLASS_DISPLAY.get(pred_class, pred_class)

        print(f"\n🔬 Forensic analysis: {os.path.basename(image_path)}")
        print(f"   Sharpness     : {laplacian_var:.1f}")
        print(f"   Face detected : {has_face}")
        print(f"   → Prediction  : {display} ({confidence*100:.1f}%)")

        return {
            'class':                display,
            'class_key':            pred_class,
            'confidence':           round(confidence, 3),
            'real_probability':     0.6 if pred_class == 'real' else 0.2,
            'ai_probability':       0.6 if pred_class == 'ai_generated' else 0.2,
            'deepfake_probability': 0.6 if pred_class == 'deepfake' else 0.2,
            'fake_probability':     0.4 if pred_class == 'real' else 0.7,
            'recommendation':       f"Forensic heuristic suggests this is {display.lower()}.",
            'analysis_mode':        'forensic',
            'model_accuracy':       MODEL_ACCURACY,
        }

    except Exception as e:
        print(f"Forensic error: {e}")
        return None


def analyze_image(image_path):
    if _model_loaded and _model is not None:
        result = predict_with_model(image_path)
        if result:
            return result
    return forensic_analysis(image_path)


# ============================================================
# Helpers
# ============================================================
def allowed_file(filename):
    return (
        '.' in filename
        and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']
    )


def get_image_hash(image_data):
    return hashlib.md5(image_data).hexdigest()


def save_to_history(filename, result, image_hash):
    history_file = os.path.join(HISTORY_FOLDER, 'history.json')
    history = []
    if os.path.exists(history_file):
        try:
            with open(history_file, 'r') as f:
                history = json.load(f)
        except Exception:
            history = []

    history.append({
        'id':         str(uuid.uuid4()),
        'filename':   filename,
        'timestamp':  datetime.now().isoformat(),
        'result':     result,
        'image_hash': image_hash,
    })
    history = history[-100:]
    with open(history_file, 'w') as f:
        json.dump(history, f, indent=2)
    return history


# ============================================================
# Routes
# ============================================================
@app.route('/')
def index():
    try:
        return render_template('upload_image.html')
    except Exception:
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>AURORA Deepfake Detector</title>
            <style>
                * { margin:0; padding:0; box-sizing:border-box; }
                body { font-family:Arial,sans-serif; background:linear-gradient(135deg,#0f0c29,#302b63,#24243e); min-height:100vh; color:#fff; }
                .container { max-width:800px; margin:0 auto; padding:40px 20px; text-align:center; }
                h1 { font-size:2.5rem; margin-bottom:10px; }
                .upload-area { border:2px dashed rgba(255,255,255,.3); border-radius:20px; padding:50px; margin:30px 0; cursor:pointer; background:rgba(255,255,255,.05); }
                .upload-area:hover { background:rgba(255,255,255,.1); }
                .btn { background:linear-gradient(135deg,#8b5cf6,#14b8a6); color:#fff; border:none; padding:12px 30px; border-radius:50px; font-size:16px; cursor:pointer; margin-top:20px; }
                #results { margin-top:30px; background:rgba(0,0,0,.3); border-radius:20px; padding:20px; display:none; }
                .bar-wrap { background:rgba(255,255,255,.2); border-radius:10px; height:8px; margin-top:15px; overflow:hidden; }
                .bar-fill { height:100%; border-radius:10px; transition:width .5s; }
            </style>
        </head>
        <body>
        <div class="container">
            <h1>🛡️ AURORA</h1>
            <p>Deepfake Detection · EfficientNet-B0 · 95.27% Accuracy</p>
            <div class="upload-area" id="uploadZone">
                <p>📁 Click or drag &amp; drop an image</p>
                <input type="file" id="fileInput" accept="image/*" style="display:none;">
            </div>
            <div id="fileName"></div>
            <button class="btn" id="analyzeBtn">🔍 Analyze Image</button>
            <div id="results"></div>
        </div>
        <script>
        const zone=document.getElementById('uploadZone'),inp=document.getElementById('fileInput'),
              btn=document.getElementById('analyzeBtn'),res=document.getElementById('results'),
              fn=document.getElementById('fileName');
        let file=null;
        zone.onclick=()=>inp.click();
        zone.ondragover=e=>{e.preventDefault();zone.style.background='rgba(255,255,255,.15)';};
        zone.ondragleave=()=>{zone.style.background='rgba(255,255,255,.05)';};
        zone.ondrop=e=>{e.preventDefault();zone.style.background='rgba(255,255,255,.05)';file=e.dataTransfer.files[0];fn.innerHTML=`✅ ${file.name}`;};
        inp.onchange=e=>{file=e.target.files[0];fn.innerHTML=`✅ ${file.name}`;};
        btn.onclick=async()=>{
            if(!file){alert('Please select an image first.');return;}
            btn.disabled=true;btn.innerHTML='⏳ Analyzing…';
            res.style.display='block';res.innerHTML='<p>Running EfficientNet-B0 analysis…</p>';
            const fd=new FormData();fd.append('file',file);
            const r=await fetch('/analyze',{method:'POST',body:fd});
            const d=await r.json();
            const pct=(d.confidence*100).toFixed(1);
            let badge,color;
            const k=d.class_key||'';
            if(k==='real'){badge='🟢 REAL IMAGE';color='#4CAF50';}
            else if(k==='ai_generated'){badge='🤖 AI GENERATED';color='#FFC107';}
            else{badge='🔴 DEEPFAKE';color='#f44336';}
            res.innerHTML=`<div style="background:rgba(0,0,0,.3);border-radius:16px;padding:20px;">
              <h2>${badge}</h2><p style="font-size:32px;font-weight:bold;">${pct}%</p>
              <p>${d.recommendation||''}</p>
              <div class="bar-wrap"><div class="bar-fill" style="width:${pct}%;background:${color};"></div></div>
              <p style="margin-top:15px;font-size:12px;opacity:.6;">Real: ${(d.real_probability*100).toFixed(1)}% | AI: ${(d.ai_probability*100).toFixed(1)}% | Deepfake: ${(d.deepfake_probability*100).toFixed(1)}%</p>
              <p style="font-size:11px;opacity:.5;">Model: EfficientNet-B0 · ${d.model_accuracy}% accuracy · mode: ${d.analysis_mode}</p>
            </div>`;
            btn.disabled=false;btn.innerHTML='🔍 Analyze Image';
        };
        </script>
        </body>
        </html>
        """


@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    if not allowed_file(file.filename):
        return jsonify({'error': 'File type not allowed'}), 400

    try:
        file_data  = file.read()
        image_hash = get_image_hash(file_data)
        file.seek(0)

        filename  = secure_filename(
            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
        )
        temp_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(temp_path)

        result = analyze_image(temp_path)

        if os.path.exists(temp_path):
            os.remove(temp_path)

        if result is None:
            return jsonify({'error': 'Analysis failed'}), 500

        save_to_history(file.filename, result, image_hash)
        return jsonify(result)

    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/batch_analyze', methods=['POST'])
def batch_analyze():
    if 'files[]' not in request.files:
        return jsonify({'error': 'No files uploaded'}), 400

    files   = request.files.getlist('files[]')
    results = []

    for file in files:
        if not allowed_file(file.filename):
            continue
        try:
            file_data  = file.read()
            image_hash = get_image_hash(file_data)
            file.seek(0)

            filename  = secure_filename(
                f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
            )
            temp_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(temp_path)

            result = analyze_image(temp_path)

            if os.path.exists(temp_path):
                os.remove(temp_path)

            if result:
                results.append({
                    'filename':   file.filename,
                    'result':     result,
                    'image_hash': image_hash,
                })
                save_to_history(file.filename, result, image_hash)

        except Exception as e:
            results.append({'filename': file.filename, 'error': str(e)})

    return jsonify({'results': results, 'total': len(results)})


@app.route('/history', methods=['GET'])
def get_history():
    history_file = os.path.join(HISTORY_FOLDER, 'history.json')
    if os.path.exists(history_file):
        try:
            with open(history_file, 'r') as f:
                return jsonify({'history': json.load(f)})
        except Exception:
            pass
    return jsonify({'history': []})


@app.route('/history/<history_id>', methods=['DELETE'])
def delete_history(history_id):
    history_file = os.path.join(HISTORY_FOLDER, 'history.json')
    if os.path.exists(history_file):
        try:
            with open(history_file, 'r') as f:
                history = json.load(f)
            history = [h for h in history if h['id'] != history_id]
            with open(history_file, 'w') as f:
                json.dump(history, f, indent=2)
            return jsonify({'success': True})
        except Exception:
            return jsonify({'error': 'Failed to delete'}), 500
    return jsonify({'error': 'History not found'}), 404


@app.route('/stats', methods=['GET'])
def get_stats():
    history_file = os.path.join(HISTORY_FOLDER, 'history.json')
    stats = {
        'total_analyses': 0,
        'real_count':     0,
        'ai_count':       0,
        'deepfake_count': 0,
        'avg_confidence': 0,
        'model_mode':     'neural_network' if _model_loaded else 'forensic',
        'model_accuracy': MODEL_ACCURACY,
    }

    if os.path.exists(history_file):
        try:
            with open(history_file, 'r') as f:
                history = json.load(f)
            stats['total_analyses'] = len(history)
            if history:
                confidences = []
                for entry in history:
                    r = entry['result']
                    k = r.get('class_key', '')
                    stats['real_count']     += 1 if k == 'real' else 0
                    stats['ai_count']       += 1 if k == 'ai_generated' else 0
                    stats['deepfake_count'] += 1 if k == 'deepfake' else 0
                    confidences.append(r.get('confidence', 0))
                stats['avg_confidence'] = sum(confidences) / len(confidences)
        except Exception:
            pass

    return jsonify(stats)


@app.route('/model/status', methods=['GET'])
def model_status():
    return jsonify({
        'loaded':        _model_loaded,
        'mode':          'neural_network' if _model_loaded else 'forensic',
        'device':        str(_device) if _device else 'N/A',
        'architecture':  'EfficientNet-B0',
        'classes':       CLASS_NAMES,
        'class_display': CLASS_DISPLAY,
        'accuracy':      MODEL_ACCURACY,
        'preprocessing': f'Resize({RESIZE_TO}) → CenterCrop({IMG_SIZE}) → Normalize(ImageNet)',
    })


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status':         'healthy',
        'model_loaded':   _model_loaded,
        'analyzer_ready': True,
        'accuracy':       MODEL_ACCURACY,
    })


# ============================================================
# Error Handlers
# ============================================================
@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large. Maximum size is 50 MB.'}), 413

@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Resource not found'}), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify({'error': 'Internal server error'}), 500


# ============================================================
# Entry Point — only for local: python app.py
# Gunicorn uses _startup() above (called at import time)
# ============================================================
if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000, threaded=True)