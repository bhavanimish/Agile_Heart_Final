import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import joblib
import os

print("Starting DL model training...")

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

# Scale features - important for neural networks
dl_scaler = StandardScaler()
X_train_scaled = dl_scaler.fit_transform(X_train)
X_test_scaled = dl_scaler.transform(X_test)

# Save the DL-specific scaler
joblib.dump(dl_scaler, "dl_scaler.pkl")
print("✅ DL StandardScaler saved to dl_scaler.pkl")

# Build a more robust neural network with regularization
model = Sequential([
    Dense(32, activation='relu', input_shape=(X.shape[1],), kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    Dropout(0.3),  # Add dropout to prevent overfitting
    Dense(16, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

# Compile model with early stopping
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Print model summary
model.summary()

# Add early stopping to prevent overfitting
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

# Train the model with early stopping
print("🔹 Training Deep Learning model...")
history = model.fit(
    X_train_scaled, y_train,
    epochs=50,
    batch_size=16,
    validation_data=(X_test_scaled, y_test),
    callbacks=[early_stopping],
    verbose=1
)

# Evaluate model
train_loss, train_acc = model.evaluate(X_train_scaled, y_train, verbose=0)
test_loss, test_acc = model.evaluate(X_test_scaled, y_test, verbose=0)
print(f"🔹 Training accuracy: {train_acc:.4f}")
print(f"🔹 Testing accuracy: {test_acc:.4f}")

# Save the model
try:
    model.save("dl_model.h5")
    print("✅ Deep Learning model saved to dl_model.h5")
    
    # Verify the file exists
    if os.path.exists("dl_model.h5"):
        file_size = os.path.getsize("dl_model.h5") / (1024 * 1024)  # Size in MB
        print(f"✅ Verified file exists: {file_size:.2f} MB")
    else:
        print("❌ Warning: File was not created successfully")
except Exception as e:
    print(f"❌ Error saving model: {str(e)}")

print("\n✅ DL model training complete!")
print("Make sure both 'dl_model.h5' and 'dl_scaler.pkl' are in the same directory as your Flask app")