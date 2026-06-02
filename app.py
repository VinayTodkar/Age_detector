import cv2
import numpy as np
import os
import pandas as pd
import joblib
from flask import Flask, render_template, request, jsonify
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# --- Configuration ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FEEDBACK_FILE = os.path.join(BASE_DIR, "feedback_data.csv")
ML_MODEL_FILE = os.path.join(BASE_DIR, "correction_model.pkl")
SCALER_FILE = os.path.join(BASE_DIR, "scaler.pkl")

# OpenCV Model Paths
face_proto = os.path.join(BASE_DIR, "opencv_face_detector.pbtxt")
face_model = os.path.join(BASE_DIR, "opencv_face_detector_uint8.pb")
age_proto = os.path.join(BASE_DIR, "age_deploy.prototxt")
age_model = os.path.join(BASE_DIR, "age_net.caffemodel")

# Load OpenCV Models
face_net = cv2.dnn.readNetFromTensorflow(face_model, face_proto)
age_net = cv2.dnn.readNetFromCaffe(age_proto, age_model)

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
age_buckets = [1, 5, 10, 17.5, 28.5, 40.5, 50.5, 80] # Numeric midpoints for calculation

# --- Machine Learning Functions ---

def load_correction_model():
    if os.path.exists(ML_MODEL_FILE) and os.path.exists(SCALER_FILE):
        return joblib.load(ML_MODEL_FILE), joblib.load(SCALER_FILE)
    return None, None

def train_correction_model():
    if not os.path.exists(FEEDBACK_FILE):
        return False
    
    df = pd.read_csv(FEEDBACK_FILE)
    if len(df) < 5: # Need at least 5 samples to train
        return False

    # Features: [Predicted_Age_Midpoint, Confidence_Score]
    X = df[['predicted_midpoint', 'confidence']].values
    # Target: Actual_Age
    y = df['actual_age'].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_scaled, y)

    joblib.dump(model, ML_MODEL_FILE)
    joblib.dump(scaler, SCALER_FILE)
    return True

def save_feedback(predicted_midpoint, confidence, actual_age):
    new_data = {
        'predicted_midpoint': [predicted_midpoint],
        'confidence': [confidence],
        'actual_age': [actual_age]
    }
    df_new = pd.DataFrame(new_data)
    
    if os.path.exists(FEEDBACK_FILE):
        df_existing = pd.read_csv(FEEDBACK_FILE)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_combined = df_new
    
    df_combined.to_csv(FEEDBACK_FILE, index=False)
    return True

# --- OpenCV Processing ---

def detect_and_predict_age(frame):
    frame_height = frame.shape[0]
    frame_width = frame.shape[1]
    
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], False, False)
    face_net.setInput(blob)
    detections = face_net.forward()
    
    results = []
    
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.7:
            x1 = int(detections[0, 0, i, 3] * frame_width)
            y1 = int(detections[0, 0, i, 4] * frame_height)
            x2 = int(detections[0, 0, i, 5] * frame_width)
            y2 = int(detections[0, 0, i, 6] * frame_height)
            
            pad = 60
            face = frame[max(0, y1-pad):min(y2+pad, frame_height), 
                         max(0, x1-pad):min(x2+pad, frame_width)]
            
            if face.size > 0:
                blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
                age_net.setInput(blob)
                age_preds = age_net.forward()
                
                age_idx = age_preds[0].argmax()
                base_age_label = age_list[age_idx]
                base_age_midpoint = age_buckets[age_idx]
                base_confidence = float(np.max(age_preds[0]) * 100)
                
                # Apply ML Correction if model exists
                final_age_label = base_age_label
                final_age_value = base_age_midpoint
                
                ml_model, scaler = load_correction_model()
                if ml_model:
                    input_data = np.array([[base_age_midpoint, base_confidence]])
                    input_scaled = scaler.transform(input_data)
                    corrected_age = ml_model.predict(input_scaled)[0]
                    final_age_value = int(corrected_age)
                    final_age_label = str(final_age_value) # Show specific age instead of bucket

                results.append({
                    "base_label": base_age_label,
                    "age": final_age_label,
                    "midpoint": final_age_value,
                    "confidence": f"{base_confidence:.2f}%",
                    "box": [x1, y1, x2, y2]
                })
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"Age: {final_age_label}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    return frame, results

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    npimg = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    
    if frame is None:
        return jsonify({'error': 'Invalid image file'}), 400

    result_frame, predictions = detect_and_predict_age(frame)
    
    _, buffer = cv2.imencode('.jpg', result_frame)
    import base64
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    
    return jsonify({
        'image': f"data:image/jpeg;base64,{img_base64}",
        'predictions': predictions
    })

@app.route('/feedback', methods=['POST'])
def feedback():
    data = request.json
    # data structure: { "predicted_midpoint": 28.5, "confidence": 90.0, "actual_age": 3 }
    
    try:
        save_feedback(data['predicted_midpoint'], data['confidence'], data['actual_age'])
        # Attempt to retrain if we have enough data
        trained = train_correction_model()
        return jsonify({'status': 'success', 'retrained': trained})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == '__main__':
    # Initial training attempt on startup
    train_correction_model()
    app.run(debug=True)   