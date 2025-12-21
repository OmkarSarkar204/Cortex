import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

DATA_FILE = "cortex_training_v2_fixed.csv"
MODEL_FILE = "cortex_brain.pkl"

if not os.path.exists(DATA_FILE):
    print("ERROR: Run repair_v2.py first!")
    exit()

df = pd.read_csv(DATA_FILE)
X = df[["roll", "pitch", "vibration_z"]]
y = df["label"]

clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
clf.fit(X, y)

print(f"Brain Trained. Accuracy: {clf.score(X, y)*100:.1f}%")
with open(MODEL_FILE, 'wb') as f:
    pickle.dump(clf, f)
print(f"Saved to {MODEL_FILE}")