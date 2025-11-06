import os
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, session, flash
from werkzeug.utils import secure_filename
from datetime import datetime

app = Flask(__name__)
app.secret_key = "cnrd_complete_fixed_2024"

# Configuration
UPLOAD_FOLDER = os.path.join('static', 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
MAX_FILE_SIZE = 16 * 1024 * 1024

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Create directories
for directory in [UPLOAD_FOLDER, 'static', 'models', 'models/saved_models', 
                  'templates', 'blood_analysis', 'image_processing']:
    os.makedirs(directory, exist_ok=True)

# Paths
MODEL_DIR = os.path.join("models", "saved_models")
CSV_FILE = "predictions_log.csv"

# Load models - FIXED VERSION
def load_models():
    """Load only actual ML models, skip scalers and preprocessors"""
    models = {}
    exclude_files = ['scaler.pkl', 'scaler.joblib', 'preprocessor.pkl', 
                    'vectorizer.pkl', 'encoder.pkl']
    
    if os.path.exists(MODEL_DIR):
        print(f"\n📂 Scanning: {MODEL_DIR}")
        for f in os.listdir(MODEL_DIR):
            if f.endswith(".pkl") and f not in exclude_files:
                try:
                    model_name = f.split(".pkl")[0]
                    model_path = os.path.join(MODEL_DIR, f)
                    loaded_model = joblib.load(model_path)
                    
                    # Verify it has predict method (is a model, not a scaler)
                    if hasattr(loaded_model, 'predict'):
                        models[model_name] = loaded_model
                        print(f"  ✅ Loaded: {model_name}")
                    else:
                        print(f"  ⚠️  Skipped: {model_name} (no predict method)")
                        
                except Exception as e:
                    print(f"  ❌ Error loading {f}: {e}")
    
    print(f"📊 Total models loaded: {len(models)}\n")
    return models

models = load_models()

# Initialize analyzers - COMPLETE VERSION
print("🔧 Initializing analyzers...")

# Blood Analyzer
try:
    from blood_analysis import BloodAnalyzer
    blood_analyzer = BloodAnalyzer()
    BLOOD_AVAILABLE = True
    print("  ✅ Blood analyzer ready")
except Exception as e:
    print(f"  ⚠️  Blood analyzer fallback: {e}")
    BLOOD_AVAILABLE = False
    class BloodAnalyzer:
        def analyze_blood_parameters(self, values):
            if not values or all(v == 0 for v in values):
                return 0.0
            risk = 0.0
            if len(values) > 0 and values[0] > 0 and values[0] < 10:
                risk += 0.3
            if len(values) > 9 and values[9] > 0 and values[9] < 3.0:
                risk += 0.3
            return min(risk, 1.0)
        
        def get_parameter_interpretation(self, values):
            params = ['Hemoglobin', 'RBC', 'WBC', 'Platelets', 'Hematocrit',
                     'MCV', 'MCH', 'MCHC', 'Protein', 'Albumin']
            interp = {}
            for i, (name, val) in enumerate(zip(params, values)):
                if val > 0:
                    status = "Normal"
                    if i == 0 and val < 11:
                        status = "Low"
                    elif i == 9 and val < 3.5:
                        status = "Low"
                    interp[name] = {
                        'value': f"{val}",
                        'status': status,
                        'normal_range': 'Varies by age',
                        'clinical_significance': 'Basic assessment'
                    }
            return interp
    
    blood_analyzer = BloodAnalyzer()

# Face Analyzer
try:
    from image_processing import FaceAnalyzer
    face_analyzer = FaceAnalyzer()
    FACE_AVAILABLE = True
    print("  ✅ Face analyzer ready")
except Exception as e:
    print(f"  ⚠️  Face analyzer fallback: {e}")
    FACE_AVAILABLE = False
    class FaceAnalyzer:
        def analyze_face(self, path):
            try:
                import cv2
                if not os.path.exists(path):
                    return 0.0
                img = cv2.imread(path)
                if img is None:
                    return 0.0
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                brightness = gray.mean()
                if brightness < 80:
                    return 0.6
                elif brightness < 120:
                    return 0.3
                return 0.1
            except:
                return 0.0
    
    face_analyzer = FaceAnalyzer()

# Growth Chart Analyzer
try:
    from image_processing import GrowthChartAnalyzer
    growth_analyzer = GrowthChartAnalyzer()
    GROWTH_AVAILABLE = True
    print("  ✅ Growth analyzer ready")
except Exception as e:
    print(f"  ⚠️  Growth analyzer fallback: {e}")
    GROWTH_AVAILABLE = False
    class GrowthChartAnalyzer:
        def analyze_chart(self, path):
            try:
                import cv2
                if not os.path.exists(path):
                    return 0.0
                img = cv2.imread(path)
                if img is None:
                    return 0.0
                h, w = img.shape[:2]
                return 0.3 if h > w else 0.2
            except:
                return 0.0
    
    growth_analyzer = GrowthChartAnalyzer()

print("✅ All analyzers initialized\n")

# User database
users = {'demo': 'demo', 'admin': 'password', 'test': 'test123'}

# Parameter names
PARAMETER_NAMES = [
    "Age", "Gender", "Weight", "Height", "BMI",
    "Mother's Education", "Household Income", "Meals per Day",
    "Vaccination Status", "Access to Clean Water", "Region",
    "Birth Weight", "Family Size", "Father's Education",
    "Food Habits", "Inherited Diseases", "Appetite Level",
    "Place of Birth", "Sanitation Access", "Breastfeeding Duration"
]

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def safe_float(value, default=0.0):
    try:
        return float(value) if value and str(value).strip() else default
    except (ValueError, TypeError):
        return default

# ==================== ROUTES ====================

@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()
        
        if username in users and users[username] == password:
            session["username"] = username
            flash(f"Welcome {username}!", "success")
            return redirect(url_for("index"))
        else:
            flash("Invalid credentials!", "error")
    
    return render_template("login.html")

@app.route("/register", methods=["POST"])
def register():
    username = request.form.get("new_username", "").strip()
    password = request.form.get("new_password", "").strip()
    confirm = request.form.get("confirm_password", "").strip()
    
    if not username or not password:
        flash("All fields required!", "error")
        return redirect(url_for("login"))
    
    if password != confirm:
        flash("Passwords don't match!", "error")
        return redirect(url_for("login"))
    
    if username in users:
        flash("Username exists!", "error")
        return redirect(url_for("login"))
    
    users[username] = password
    session["username"] = username
    flash("Registration successful!", "success")
    return redirect(url_for("index"))

@app.route("/index")
def index():
    if "username" not in session:
        return redirect(url_for("login"))
    return render_template("index.html", username=session.get("username"))

@app.route("/predict", methods=["POST"])
def predict():
    if "username" not in session:
        return redirect(url_for("login"))
    
    try:
        print("\n" + "="*70)
        print(f"🔍 PREDICTION REQUEST: {session['username']}")
        print("="*70)
        
        # ============ DEMOGRAPHIC FEATURES ============
        features = []
        parameters = {}
        
        param_fields = ['age', 'gender', 'weight', 'height', 'bmi', 'mother_edu', 
                       'income', 'meals', 'vaccination', 'clean_water', 'region',
                       'birth_weight', 'family_size', 'father_edu', 'food_habits',
                       'diseases', 'appetite', 'place_birth', 'sanitation', 'breastfeeding']
        
        print("\n📋 DEMOGRAPHIC DATA:")
        for field, label in zip(param_fields, PARAMETER_NAMES):
            value = request.form.get(field, "")
            parameters[label] = value if value else "Not provided"
            features.append(safe_float(value, 0))
            print(f"  {label}: {value}")
        
        # ============ IMAGE ANALYSIS ============
        print("\n📸 IMAGE ANALYSIS:")
        face_score = 0.0
        growth_score = 0.0
        image_paths = {}
        
        # Face image
        if 'face_image' in request.files:
            file = request.files['face_image']
            if file and file.filename and allowed_file(file.filename):
                filename = secure_filename(f"face_{session['username']}_{int(datetime.now().timestamp())}.jpg")
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                try:
                    file.save(filepath)
                    image_paths['face'] = filename
                    face_score = face_analyzer.analyze_face(filepath)
                    print(f"  ✅ Face analyzed: {face_score:.3f}")
                except Exception as e:
                    print(f"  ❌ Face error: {e}")
            else:
                print(f"  ⚠️  No face image uploaded")
        else:
            print(f"  ⚠️  No face image in request")
        
        # Growth chart
        if 'growth_chart' in request.files:
            file = request.files['growth_chart']
            if file and file.filename and allowed_file(file.filename):
                filename = secure_filename(f"growth_{session['username']}_{int(datetime.now().timestamp())}.jpg")
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                try:
                    file.save(filepath)
                    image_paths['growth'] = filename
                    growth_score = growth_analyzer.analyze_chart(filepath)
                    print(f"  ✅ Growth chart analyzed: {growth_score:.3f}")
                except Exception as e:
                    print(f"  ❌ Growth error: {e}")
            else:
                print(f"  ⚠️  No growth chart uploaded")
        else:
            print(f"  ⚠️  No growth chart in request")
        
        # ============ BLOOD PARAMETERS ============
        print("\n🩸 BLOOD PARAMETERS:")
        blood_fields = ['hemoglobin', 'rbc_count', 'wbc_count', 'platelet_count',
                       'hematocrit', 'mcv', 'mch', 'mchc', 'protein', 'albumin']
        
        blood_parameters = {}
        blood_values = []
        
        for field in blood_fields:
            value = request.form.get(field, "")
            display_name = field.replace('_', ' ').title()
            blood_parameters[display_name] = value if value else "Not provided"
            blood_values.append(safe_float(value, 0))
            if value:
                print(f"  {display_name}: {value}")
        
        blood_score = 0.0
        blood_interpretations = {}
        
        has_blood_data = any(v > 0 for v in blood_values)
        
        if has_blood_data:
            try:
                blood_score = blood_analyzer.analyze_blood_parameters(blood_values)
                blood_interpretations = blood_analyzer.get_parameter_interpretation(blood_values)
                print(f"  ✅ Blood analysis complete: {blood_score:.3f}")
                print(f"  📊 Parameters interpreted: {len(blood_interpretations)}")
            except Exception as e:
                print(f"  ❌ Blood analysis error: {e}")
        else:
            print(f"  ⚠️  No blood parameters provided")
        
        # ============ ML MODEL PREDICTIONS ============
        print("\n🤖 ML MODEL PREDICTIONS:")
        demographic_risk = 0.5
        
        if models and len(features) >= 20:
            X = np.array(features).reshape(1, -1)
            predictions = []
            
            for model_name, model in models.items():
                try:
                    pred = int(model.predict(X)[0])
                    predictions.append(pred)
                    result = "At Risk" if pred == 1 else "Healthy"
                    print(f"  {model_name}: {result}")
                except Exception as e:
                    print(f"  {model_name}: ERROR - {e}")
            
            if predictions:
                risk_votes = sum(1 for p in predictions if p == 1)
                healthy_votes = len(predictions) - risk_votes
                demographic_risk = risk_votes / len(predictions)
                print(f"\n  📊 Voting: {risk_votes} At Risk, {healthy_votes} Healthy")
        else:
            print(f"  ⚠️  Using default demographic risk (no models/insufficient data)")
        
        # ============ COMBINED RISK CALCULATION ============
        print("\n📊 RISK SCORE BREAKDOWN:")
        print(f"  Demographic Risk: {demographic_risk:.3f} (40% weight)")
        print(f"  Face Score:       {face_score:.3f} (30% weight)")
        print(f"  Growth Score:     {growth_score:.3f} (20% weight)")
        print(f"  Blood Score:      {blood_score:.3f} (10% weight)")
        
        total_risk = (
            demographic_risk * 0.4 +
            face_score * 0.3 +
            growth_score * 0.2 +
            blood_score * 0.1
        )
        
        print(f"  TOTAL RISK:       {total_risk:.3f}")
        
        # Risk classification
        if total_risk < 0.3:
            final_prediction = "Low Risk"
            risk_level = "healthy"
        elif total_risk < 0.7:
            final_prediction = "Medium Risk"
            risk_level = "moderate"
        else:
            final_prediction = "High Risk"
            risk_level = "severe"
        
        confidence = min(max(total_risk * 100, 10), 95)
        
        print(f"\n🎯 FINAL ASSESSMENT:")
        print(f"  Prediction: {final_prediction}")
        print(f"  Confidence: {confidence:.1f}%")
        print(f"  Risk Level: {risk_level}")
        print("="*70 + "\n")
        
        # ============ SAVE TO CSV ============
        log_data = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "username": session["username"]
        }
        log_data.update(parameters)
        log_data.update(blood_parameters)
        log_data.update({
            "face_image": image_paths.get('face', ''),
            "growth_chart": image_paths.get('growth', ''),
            "face_score": round(face_score, 4),
            "growth_score": round(growth_score, 4),
            "blood_score": round(blood_score, 4),
            "demographic_risk": round(demographic_risk, 4),
            "total_risk_score": round(total_risk, 4),
            "final_prediction": final_prediction,
            "confidence": round(confidence, 2),
            "has_blood_data": has_blood_data,
            "has_face_image": bool(image_paths.get('face')),
            "has_growth_chart": bool(image_paths.get('growth'))
        })
        
        try:
            df = pd.DataFrame([log_data])
            if not os.path.exists(CSV_FILE):
                df.to_csv(CSV_FILE, index=False)
            else:
                df.to_csv(CSV_FILE, mode="a", header=False, index=False)
            print(f"✅ Data logged to {CSV_FILE}")
        except Exception as e:
            print(f"⚠️  CSV save error: {e}")
        
        # ============ RENDER RESULTS ============
        return render_template("results.html",
                             parameters=parameters,
                             blood_parameters=blood_parameters,
                             blood_interpretations=blood_interpretations,
                             final_prediction=final_prediction,
                             risk_level=risk_level,
                             confidence=confidence,
                             face_score=round(face_score, 3),
                             growth_score=round(growth_score, 3),
                             blood_score=round(blood_score, 3),
                             demographic_risk=round(demographic_risk, 3),
                             total_risk_score=round(total_risk, 3),
                             image_paths=image_paths,
                             has_blood_data=has_blood_data)
    
    except Exception as e:
        print(f"\n❌ PREDICTION ERROR: {e}")
        import traceback
        traceback.print_exc()
        flash(f"Error during analysis: {str(e)}", "error")
        return redirect(url_for("index"))

@app.route("/logout")
def logout():
    session.clear()
    flash("Logged out successfully", "info")
    return redirect(url_for("login"))

@app.route("/history")
def history():
    if "username" not in session:
        return redirect(url_for("login"))
    
    records = []
    if os.path.exists(CSV_FILE):
        try:
            df = pd.read_csv(CSV_FILE)
            user_history = df[df['username'] == session['username']].tail(10)
            records = user_history.to_dict('records')
        except Exception as e:
            print(f"History error: {e}")
    
    return render_template("history.html", records=records, username=session["username"])

@app.errorhandler(413)
def file_too_large(error):
    flash("File too large! Max 16MB", "error")
    return redirect(url_for("index"))

@app.errorhandler(404)
def not_found(error):
    return render_template("error.html", error="Page not found"), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template("error.html", error="Internal server error"), 500

if __name__ == "__main__":
    print("\n" + "="*70)
    print("🏥 CHILD NUTRITION RISK DETECTOR")
    print("="*70)
    print(f"📊 ML Models: {len(models)}")
    print(f"🩸 Blood Analysis: {'✅ Active' if BLOOD_AVAILABLE else '⚠️  Fallback'}")
    print(f"👤 Face Analysis: {'✅ Active' if FACE_AVAILABLE else '⚠️  Fallback'}")
    print(f"📈 Growth Analysis: {'✅ Active' if GROWTH_AVAILABLE else '⚠️  Fallback'}")
    print(f"👤 Users: {', '.join(users.keys())}")
    print("="*70)
    print("🌐 Server: http://localhost:5000")
    print("👤 Demo: demo / demo")
    print("="*70 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)