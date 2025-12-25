from flask import Flask, request, render_template, send_file, url_for
import joblib
import librosa
import numpy as np
import os
from reportlab.pdfgen import canvas
# *** NEW IMPORT ***
from sklearn.preprocessing import StandardScaler 

app = Flask(__name__)

# ---------- Load trained model and SCALER ----------
model = joblib.load("models/deepfake_model.pkl")
# *** NEW LINE: LOAD THE SCALER ***
scaler = joblib.load("models/deepfake_scaler.pkl") 

# ---------- Feature Extraction (RESTORED) ----------
def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=None)
        if y is None or len(y) < 2048:
            return None, None, None

        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        features = np.mean(mfcc.T, axis=0)

        return features, y, sr

    except Exception as e:
        print(f"Error extracting features from {file_path}: {e}")
        return None, None, None


# ---------- Noise / Corruption Detection (RESTORED) ----------
def detect_noise(y):
    if y is None:
        return True
    return np.std(y) < 0.005


# ---------- PDF Generator (RESTORED) ----------
def generate_pdf(result, confidence, duration, noise_flag, filename):
    os.makedirs("static", exist_ok=True)
    pdf_path = os.path.join("static", f"{filename}_report.pdf")

    c = canvas.Canvas(pdf_path)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(100, 780, "AI Deepfake Audio Detection Report")

    c.setFont("Helvetica", 12)
    c.drawString(60, 740, f"Result: {result}")
    c.drawString(60, 720, f"Confidence Score: {confidence:.3f}%")
    c.drawString(60, 700, f"Duration: {duration:.3f} sec")
    c.drawString(60, 680, f"Noise Detected: {noise_flag}")
    c.drawString(60, 650, "Generated using AI Deepfake Audio Detector")
    c.save()

    return pdf_path


# ---------- Routes ----------
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    # --- File Upload and Saving (RESTORED) ---
    file = request.files.get("audio")
    if not file or file.filename == "":
        return render_template("index.html", result="❌ No file uploaded")

    os.makedirs("uploads", exist_ok=True)
    filepath = os.path.join("uploads", file.filename)
    file.save(filepath)
    # --- END RESTORED ---

    # --- File size check (RESTORED) ---
    if os.path.getsize(filepath) > 10_000_000:
        return render_template("index.html", result="⚠️ File too large (>10MB)")
    # --- END RESTORED ---

    # --- Extract Features ---
    # The error "extract_features" and "filepath" are resolved here.
    features, y, sr = extract_features(filepath)
    if features is None:
        return render_template("index.html", result="⚠️ Audio too short or unreadable")

    # --- Duration (RESTORED) ---
    duration = librosa.get_duration(y=y, sr=sr)
    if duration < 1.0: 
        return render_template("index.html", result="⚠️ Audio too short (<1 sec)")
    # --- END RESTORED ---

    # *** NEW STEP: APPLY SCALER ***
    # The scaler expects a 2D array, so we reshape the 1D feature array before transforming.
    features_scaled = scaler.transform(features.reshape(1, -1))[0]
    
    # --- Noise (RESTORED) ---
    noise_flag = "Yes" if detect_noise(y) else "No"
    noise_color = "red" if noise_flag == "Yes" else "lime"
    # --- END RESTORED ---

    # --- Prediction ---
    # Use the SCALED features for prediction
    prob = model.predict_proba([features_scaled])[0] 
    pred = np.argmax(prob)
    confidence = prob[pred] * 100

    # --- Result Formatting (RESTORED) ---
    result_text = "Real Voice" if pred == 0 else "Fake Voice"
    color = "lime" if pred == 0 else "red"
    # --- END RESTORED ---

    # --- PDF Report (RESTORED) ---
    filename = os.path.splitext(file.filename)[0]
    pdf_path = generate_pdf(result_text, confidence, duration, noise_flag, filename)

    return render_template(
        "index.html",
        result=result_text,
        confidence=f"{confidence:.3f}%",
        duration=f"{duration:.3f} sec",
        noise_flag=noise_flag,
        noise_color=noise_color,
        color=color,
        pdf_url=pdf_path
    )


@app.route("/download/<path:filename>")
def download(filename):
    return send_file(filename, as_attachment=True)


if __name__ == "__main__":
    app.run(debug=True)