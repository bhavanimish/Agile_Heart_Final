import pennylane as qml
from pennylane.optimize import NesterovMomentumOptimizer
import pennylane.numpy as pnp
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# Load dataset
try:
    df = pd.read_csv("heart.csv")
    print("✅ Dataset loaded successfully.")
except FileNotFoundError:
    print("❌ Error: heart.csv not found. Please make sure the file exists in the current directory.")
    exit(1)

# Define the features to use for QML
qml_features = ['age', 'trestbps', 'chol', 'thalach']

# Create a dataset with only the QML features
X_qml = df[qml_features]
y = df["target"]

# Scale features specifically for QML
qml_scaler = StandardScaler()
X_qml_scaled = qml_scaler.fit_transform(X_qml)

# Save the QML-specific scaler
joblib.dump(qml_scaler, "qml_scaler.pkl")
print("✅ QML Scaler saved to qml_scaler.pkl.")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_qml_scaled, y, test_size=0.2, random_state=42)

# Convert y_train from pandas Series to numpy array for proper indexing
y_train_np = y_train.to_numpy()
y_test_np = y_test.to_numpy()

# Show debug info
print("✅ Data loaded and scaled.")
print(f"🔹 Sample input (scaled): {X_train[0]}")
print(f"🔹 Sample label: {y_train_np[0]}")

# Set up quantum device
dev = qml.device("default.qubit", wires=4)

@qml.qnode(dev)
def circuit(weights, x):
    # Encode the classical data into quantum states
    for i in range(4):
        qml.RY(x[i], wires=i)
    
    # Apply parameterized quantum circuit
    qml.templates.BasicEntanglerLayers(weights, wires=range(4))
    
    # Return the expectation value
    return qml.expval(qml.PauliZ(0))

# Cost function
def cost(weights, X, Y):
    loss = 0
    for i in range(len(X)):
        x = X[i]
        y = Y[i]
        pred = circuit(weights, x)
        # Convert label 0/1 to -1/+1
        target = 2*y - 1
        loss += (pred - target)**2
    return loss / len(X)

def accuracy(weights, X, Y):
    correct = 0
    for i in range(len(X)):
        x = X[i]
        y = Y[i]
        pred = circuit(weights, x)
        # Classify as 1 if pred > 0, else 0
        pred_class = 1 if pred > 0 else 0
        if pred_class == y:
            correct += 1
    return correct / len(X)

if __name__ == "__main__":
    print("Starting QML training...")
    
    # Initialize weights: 3 layers, 4 wires
    np.random.seed(42)  # For reproducibility
    weights = pnp.array(np.random.uniform(0, np.pi, size=(3, 4)), requires_grad=True)
    print(f"🔹 Initial weights shape: {weights.shape}")
    
    opt = NesterovMomentumOptimizer(stepsize=0.1)
    
    steps = 20  # Increase for better results, reduce for faster debugging
    for i in range(steps):
        weights = opt.step(lambda w: cost(w, X_train, y_train_np), weights)
        
        if (i + 1) % 5 == 0:
            current_loss = cost(weights, X_train, y_train_np)
            train_acc = accuracy(weights, X_train, y_train_np)
            print(f"📉 Step {i+1}/{steps} - Loss: {current_loss:.4f} - Train Accuracy: {train_acc:.4f}")
    
    # Evaluate on test set
    test_acc = accuracy(weights, X_test, y_test_np)
    print(f"✅ Final test accuracy: {test_acc:.4f}")
    
    # Save trained weights
    np.save("qml_weights.npy", weights)
    print("✅ qml_weights.npy saved successfully.")