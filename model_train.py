import os
import numpy as np
import librosa
import joblib
from tqdm import tqdm
from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
from sklearn.preprocessing import StandardScaler 

# ---------- Feature Extraction ----------
def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=None)
        if len(y) < 2048:
            return None
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        return np.mean(mfcc.T, axis=0)
    except Exception as e:
        return None

# ---------- Dataset Loader ----------
def load_dataset(base_path):
    X, y = [], []
    valid_exts = ('.wav', '.mp3', '.flac', '.wav.noisered', '.opus', '.ogg')

    for label_dir, label in [('real', 0), ('fake', 1)]:
        folder = os.path.join(base_path, label_dir)
        if not os.path.exists(folder):
            print(f"Missing folder: {folder}")
            continue

        files = [fn for fn in os.listdir(folder) if fn.lower().endswith(valid_exts)]
        print(f"\n Loading {label_dir} ({len(files)} files)...")

        for fn in tqdm(files, desc=f"Processing {label_dir}", unit="file"):
            path = os.path.join(folder, fn)
            features = extract_features(path)
            if features is not None:
                X.append(features)
                y.append(label)

    return shuffle(np.array(X), np.array(y), random_state=42)

# ---------- Train + Evaluate + Save ----------
dataset_path = r"D:\final year project6" 
print(" Loading dataset...")
X, y = load_dataset(dataset_path)

print("\n Training model...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ------ FEATURE SCALING: FIT AND TRANSFORM ***
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
print(" Features standardized using StandardScaler.")


clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# ---------- Predictions ----------
y_pred = clf.predict(X_test)

# ---------- Evaluation Metrics ----------
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print("\n ------ Model Evaluation ------")
print(f" Accuracy:           {accuracy*100:.3f}%")
print(f" Precision:          {precision*100:.3f}%")
print(f" Recall:             {recall*100:.3f}%")
print(f" F1-Score:           {f1*100:.3f}%")
print("\n Confusion Matrix:")
print(conf_matrix)
print("\n Classification Report:")
print(class_report)

# ---------- Save Model & SCALER ----------
os.makedirs("models", exist_ok=True)
joblib.dump(clf, "models/deepfake_model.pkl")
# ------- SAVE THE SCALER -----
joblib.dump(scaler, "models/deepfake_scaler.pkl") 
print("\n Model saved to models/deepfake_model.pkl")
print(" Scaler saved to models/deepfake_scaler.pkl")