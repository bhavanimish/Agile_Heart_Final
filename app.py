from flask import Flask, render_template, request, redirect, url_for, session, flash
import numpy as np
import pickle
import joblib
import sqlite3
import pennylane as qml
import pennylane.numpy as pnp
from tensorflow.keras.models import load_model

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Change this in production

# Configure Flask to use templates and static folders
app.template_folder = 'templates'
app.static_folder = 'static'

# Ensure the database exists
def init_db():
    con = sqlite3.connect("users.db")
    cur = con.cursor()
    cur.execute("CREATE TABLE IF NOT EXISTS users (username TEXT PRIMARY KEY, password TEXT)")
    con.commit()
    con.close()

# ML/DL feature names (all 13)
ml_dl_features = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
    'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
]

# QML feature names (only 4)
qml_features = ['age', 'trestbps', 'chol', 'thalach']

# ML: Load Random Forest model and scaler
try:
    # Load the models and scalers
    with open('random_forest_model.pkl', 'rb') as f:
        rf_model = pickle.load(f)
    
    # Load the standard scaler for ML/DL models
    scaler = joblib.load('scaler.pkl')
    
    # Load the QML-specific scaler
    qml_scaler = joblib.load('qml_scaler.pkl')
    
    # Load DL model
    dl_model = load_model('dl_model.h5')
    
    # Load QML weights
    qml_weights = np.load('qml_weights.npy')
    print("All models and scalers loaded successfully!")
except FileNotFoundError as e:
    print(f"Warning: Model or scaler file not found - {e}")
    # Create dummy models/data for development
    rf_model = None
    scaler = None
    qml_scaler = None
    dl_model = None
    qml_weights = np.random.rand(3, 4)  # Match the shape in your QML training script
    print("Using dummy models for development")

# Set up quantum device for QML
dev = qml.device("default.qubit", wires=4)

@qml.qnode(dev)
def circuit(weights, x):
    for i in range(4):
        qml.RY(x[i], wires=i)
    qml.templates.BasicEntanglerLayers(weights, wires=range(4))
    return qml.expval(qml.PauliZ(0))

# --- Auth Routes ---

@app.route("/", methods=["GET"])
def index():
    if "user" in session:
        return redirect(url_for("predict"))
    return redirect(url_for("login"))

@app.route("/register", methods=["GET", "POST"])
def register():
    # Initialize the database if it doesn't exist
    init_db()
    
    error = None
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        
        if not username or not password:
            error = "Username and password are required"
        else:
            try:
                con = sqlite3.connect("users.db")
                cur = con.cursor()
                cur.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
                con.commit()
                con.close()
                flash("Registration successful, please login")
                return redirect(url_for("login"))
            except sqlite3.IntegrityError:
                error = "Username already exists"
            finally:
                con.close()
    
    return render_template("register.html", error=error)

@app.route("/login", methods=["GET", "POST"])
def login():
    # Initialize the database if it doesn't exist
    init_db()
    
    error = None
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        
        con = sqlite3.connect("users.db")
        cur = con.cursor()
        cur.execute("SELECT * FROM users WHERE username=? AND password=?", (username, password))
        user = cur.fetchone()
        con.close()
        
        if user:
            session["user"] = username
            return redirect(url_for("predict"))
        else:
            error = "Invalid username or password"
    
    return render_template("login.html", error=error)

@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect(url_for("login"))

# --- Prediction Route ---

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if "user" not in session:
        return redirect(url_for("login"))

    prediction = None
    # Default model type for initial page load
    model_type = request.form.get("model_type", "ml")
    
    if request.method == "POST":
        try:
            model_type = request.form["model_type"]
            print(f"Selected model type: {model_type}")
            
            if model_type == "ml":
                # For ML model, collect all 13 features
                features = []
                for feature in ml_dl_features:
                    value = request.form.get(feature)
                    if value is None or value == '':
                        raise ValueError(f"Missing value for feature: {feature}")
                    features.append(float(value))
                
                features = np.array(features).reshape(1, -1)
                print(f"ML Features: {features}")
                
                if scaler is not None:
                    features_scaled = scaler.transform(features)
                    print(f"Scaled features: {features_scaled}")
                else:
                    features_scaled = features  # Fallback for development
                
                if rf_model is not None:
                    pred = rf_model.predict(features_scaled)
                    prediction = "Positive" if pred[0] == 1 else "Negative"
                    print(f"ML Prediction: {prediction}")
                else:
                    prediction = "Heart Disease Predicted"
                    print("Heart Disease Predicted")

            elif model_type == "dl":
                # For DL model, collect all 13 features
                features = []
                for feature in ml_dl_features:
                    value = request.form.get(feature)
                    if value is None or value == '':
                        raise ValueError(f"Missing value for feature: {feature}")
                    features.append(float(value))
                
                features = np.array(features).reshape(1, -1)
                print(f"DL Features: {features}")
                
                if scaler is not None:
                    features_scaled = scaler.transform(features)
                    print(f"Scaled features: {features_scaled}")
                else:
                    features_scaled = features  # Fallback for development
                
                if dl_model is not None:
                    pred = dl_model.predict(features_scaled)
                    prediction = "Positive" if pred[0][0] > 0.5 else "Negative"
                    print(f"DL Prediction: {prediction}, raw output: {pred[0][0]}")
                else:
                    prediction = "Heart Disease Predicted"
                    print("Heart Disease Predicted")
                
            elif model_type == "qml":
                # For QML, only collect the 4 required features
                qml_input = []
                for feature in qml_features:
                    value = request.form.get(feature)
                    if value is None or value == '':
                        raise ValueError(f"Missing value for feature: {feature}")
                    qml_input.append(float(value))
                
                qml_input = np.array(qml_input).reshape(1, -1)
                print(f"QML Input (original): {qml_input}")
                
                # Scale the QML inputs using the dedicated QML scaler
                if qml_scaler is not None:
                    qml_input_scaled = qml_scaler.transform(qml_input)
                    print(f"QML Input (scaled with dedicated scaler): {qml_input_scaled}")
                else:
                    # Simple fallback normalization if no scaler is available
                    qml_input_scaled = qml_input / 100.0
                    print(f"QML Input (normalized fallback): {qml_input_scaled}")
                
                if qml_weights is not None:
                    weights = pnp.array(qml_weights)
                    # Pass the scaled inputs to the quantum circuit
                    result = circuit(weights, qml_input_scaled[0])
                    prediction = "Positive" if result > 0 else "Negative"
                    print(f"QML Prediction: {prediction}, raw output: {result}")
                else:
                    prediction = "QML Model not available"
                    print("QML Model not available")
            
            else:
                prediction = "Invalid model type"
                print(f"Invalid model type: {model_type}")
                
        except Exception as e:
            prediction = f"Error: {str(e)}"
            print(f"Error during prediction: {str(e)}")

    # Pass different feature sets based on selected model
    return render_template(
        "index.html", 
        prediction=prediction, 
        model_type=model_type,
        ml_dl_features=ml_dl_features,
        qml_features=qml_features,
        username=session.get("user")
    )

if __name__ == "__main__":
    app.run(debug=True)