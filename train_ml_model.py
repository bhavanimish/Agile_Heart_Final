import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pickle
import joblib
import os

print("Starting ML model training...")

# Load dataset - handle both possible filenames
try:
    df = pd.read_csv("heart.csv")
    print("✅ Loaded dataset from heart.csv")
except FileNotFoundError:
    try:
        df = pd.read_csv("heart .csv")
        print("✅ Loaded dataset from 'heart .csv' (with space)")
    except FileNotFoundError:
        print("❌ Error: Could not find heart disease dataset")
        print("Please ensure 'heart.csv' or 'heart .csv' is in the current directory")
        exit(1)

# Split features and target
X = df.drop("target", axis=1)
y = df["target"]
print(f"🔹 Dataset shape: {X.shape} features, {len(y)} samples")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"🔹 Training on {len(X_train)} samples, testing on {len(X_test)} samples")

# Scale features for better model performance
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save the scaler for inference
joblib.dump(scaler, "scaler.pkl")
print("✅ StandardScaler saved to scaler.pkl")

# Train RandomForest model on scaled data
print("🔹 Training Random Forest model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Evaluate model performance
train_accuracy = model.score(X_train_scaled, y_train)
test_accuracy = model.score(X_test_scaled, y_test)
print(f"🔹 Training accuracy: {train_accuracy:.4f}")
print(f"🔹 Testing accuracy: {test_accuracy:.4f}")

# Save model
try:
    with open("random_forest_model.pkl", "wb") as f:
        pickle.dump(model, f)
    print("✅ Random Forest model saved to random_forest_model.pkl")
    
    # Verify the file exists
    if os.path.exists("random_forest_model.pkl"):
        file_size = os.path.getsize("random_forest_model.pkl") / (1024 * 1024)  # Size in MB
        print(f"✅ Verified file exists: {file_size:.2f} MB")
    else:
        print("❌ Warning: File was not created successfully")
except Exception as e:
    print(f"❌ Error saving model: {str(e)}")

# Print feature importances
feature_importance = list(zip(X.columns, model.feature_importances_))
feature_importance.sort(key=lambda x: x[1], reverse=True)
print("\nFeature importance:")
for feature, importance in feature_importance[:5]:  # Show top 5 features
    print(f"  • {feature}: {importance:.4f}")

print("\n✅ ML model training complete!")
print("Make sure both 'random_forest_model.pkl' and 'scaler.pkl' are in the same directory as your Flask app")