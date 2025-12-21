import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# --- FIND THE LOG FILE ---
# Try looking on Desktop first, then current folder
paths_to_check = [
    os.path.join(os.path.expanduser("~"), "Desktop", "brain_activity.csv"),
    "brain_activity.csv"
]

LOG_FILE = None
for p in paths_to_check:
    if os.path.exists(p):
        LOG_FILE = p
        break

if LOG_FILE is None:
    print("ERROR: Could not find 'brain_activity.csv'")
    print("Make sure you ran the Logger Drone and crashed it first!")
    exit()

print(f"Plotting data from: {LOG_FILE}")
df = pd.read_csv(LOG_FILE)

# --- SETUP PLOT ---
plt.figure(figsize=(14, 8))
plt.style.use('ggplot') # Nice style

# 1. EXCITATION (Red) - "Danger Signal"
plt.plot(df['time'], df['excitation'], color='red', alpha=0.5, label='Excitation (Sensor Danger)')

# 2. INHIBITION (Green) - "Pilot Intent"
# We plot this as negative to visually show it "cancelling out" the red
plt.plot(df['time'], -df['inhibition'], color='green', alpha=0.5, label='Inhibition (Your Stick Input)')

# 3. MEMBRANE POTENTIAL (Blue) - "The Result"
# This is the actual voltage inside the neuron
plt.plot(df['time'], df['voltage'], color='blue', linewidth=2, label='Neuron Voltage (Panic Level)')

# 4. THRESHOLD (Orange Dashed)
plt.axhline(y=1.5, color='orange', linestyle='--', label='Firing Threshold')

# 5. MARK SPIKES
spikes = df[df['spike'] == 1]
if not spikes.empty:
    plt.scatter(spikes['time'], spikes['voltage'], color='black', marker='x', s=100, label='NEURON SPIKE', zorder=10)
    # Highlight the crash zone
    crash_start = spikes['time'].iloc[0]
    plt.axvspan(crash_start, df['time'].iloc[-1], color='red', alpha=0.1, label='EMERGENCY LANDING')

# Labels and Titles
plt.title("EVIDENCE OF INTELLIGENCE: Predictive Coding SNN", fontsize=16)
plt.xlabel("Time (seconds)", fontsize=12)
plt.ylabel("Neural Units", fontsize=12)
plt.legend(loc='upper left', frameon=True, fancybox=True, framealpha=1, shadow=True)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

# Show it
print("Displaying Graph...")
plt.tight_layout()
plt.show()