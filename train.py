import pandas as pd
import numpy as np
import os

# --- CONFIGURATION ---
DATA_FILE = "cortex_training_v2_fixed.csv"
WEIGHTS_FILE = "snn_weights.npy"

if not os.path.exists(DATA_FILE):
    print("ERROR: CSV file not found. Run repair_v2.py first!")
    exit()

print("--- TRAINING BIOLOGICAL SNN ---")

# 1. Load Data
df = pd.read_csv(DATA_FILE)

# 2. Normalize Data (0 to 1 range)
# We use absolute values because a -30 degree roll is same danger as +30
X = df[["roll", "pitch", "vibration_z"]].copy()
X = X.abs()
max_vals = X.max()
X = X / max_vals # Normalization

inputs = X.to_numpy()
labels = np.where(df["label"] == "FAULT", 1, 0)

# 3. Evolutionary Learning
# We want to find 3 numbers (weights) that multiply with Roll, Pitch, Vib
# such that the result is > 1.0 ONLY when it's a crash.

best_acc = 0.0
best_weights = np.random.uniform(0.0, 1.0, 3)

for i in range(2000):
    # Mutate weights
    test_weights = best_weights + np.random.uniform(-0.5, 0.5, 3)
    test_weights = np.clip(test_weights, 0, 10) # Keep positive
    
    # Calculate Neuron Potential for ALL rows
    # Potential = (Roll*W1) + (Pitch*W2) + (Vib*W3)
    potentials = np.dot(inputs, test_weights)
    
    # If Potential > 1.0, it Spikes (Predicted Crash)
    predictions = np.where(potentials > 1.0, 1, 0)
    
    acc = np.mean(predictions == labels)
    
    if acc > best_acc:
        best_acc = acc
        best_weights = test_weights
        print(f"Gen {i}: Accuracy {best_acc*100:.1f}%")

print("--------------------------------")
print(f"FINAL WEIGHTS: {best_weights}")
print(f"Max Scale Factors used: \n{max_vals}")
print("--------------------------------")

# Save weights AND the normalization factors (Crucial for the drone to match the data)
np.save(WEIGHTS_FILE, best_weights)
# We also need to save the max values to scale live data correctly
np.save("snn_normalization.npy", max_vals.to_numpy()) 

print("Brain Saved.")