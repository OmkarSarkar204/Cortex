import pandas as pd
import numpy as np
import os

# --- CONFIGURATION ---
DATA_FILE = "cortex_training_v2_fixed.csv"
WEIGHTS_FILE = "snn_weights.npy"

if not os.path.exists(DATA_FILE):
    print("ERROR: Data file not found!")
    exit()

print("--- TRAINING BIOLOGICAL SNN (LIF MODEL) ---")

# 1. Load Data
df = pd.read_csv(DATA_FILE)

# 2. Normalize Data (Neurons need inputs between 0 and 1)
# We focus on Pitch and Vibration (Roll is symmetric, so absolute roll works)
X = df[["roll", "pitch", "vibration_z"]].copy()
X['roll'] = X['roll'].abs() 
X['pitch'] = X['pitch'].abs()
X['vibration_z'] = X['vibration_z'].abs()

# Normalize columns to 0-1 range
X = (X - X.min()) / (X.max() - X.min())
inputs = X.to_numpy()

# Labels: 0 for SAFE, 1 for FAULT
labels = np.where(df["label"] == "FAULT", 1, 0)

# --- THE LIF NEURON ---
class LIFNeuron:
    def __init__(self, n_inputs):
        # Biologically plausible parameters
        self.dt = 1.0           # Time step
        self.tau = 10.0         # Membrane time constant (Decay)
        self.threshold = 1.0    # Firing threshold
        self.v_rest = 0.0       # Resting potential
        
        # Synaptic Weights (The "Learning" part)
        # We start with random strong connections
        self.weights = np.random.uniform(0.5, 1.0, n_inputs)
        
    def check_spike(self, input_current):
        # The LIF Equation: V(t) = V(t-1) + (Input - V(t-1))/tau
        # For a classifier, we simplify: Potential = Weighted Sum of Inputs
        potential = np.dot(input_current, self.weights)
        
        # Does it cross the threshold?
        if potential > self.threshold:
            return 1 # SPIKE!
        else:
            return 0 # Silence

# --- TRAINING (Evolutionary Search) ---
# Since SNNs are hard to use gradient descent on, we use a simple
# "Best Fit" search to find the perfect synaptic weights.

best_acc = 0.0
best_weights = None

print("Optimizing Synapses...")
for i in range(1000): # Try 1000 different brain structures
    # Mutate weights slightly
    test_weights = np.random.uniform(-1.0, 2.0, 3) 
    
    # Run Simulation
    # Calculate potential for ALL rows at once (Vectorized LIF)
    potentials = np.dot(inputs, test_weights)
    spikes = np.where(potentials > 1.0, 1, 0)
    
    # Calculate Accuracy
    accuracy = np.mean(spikes == labels)
    
    if accuracy > best_acc:
        best_acc = accuracy
        best_weights = test_weights
        print(f"Generation {i}: New Best Accuracy = {best_acc*100:.2f}%")

print("--------------------------------")
print(f"FINAL SNN ACCURACY: {best_acc*100:.2f}%")
print(f"Learned Weights: {best_weights}")
print("--------------------------------")

# Save the Synaptic Weights
np.save(WEIGHTS_FILE, best_weights)
print(f"Brain saved to {WEIGHTS_FILE}")