# 🧠 Smart Age Detection System with Self-Learning Feedback

An intelligent web-based application that detects faces and predicts age using Deep Learning (OpenCV). Unlike standard models, this system features a **human-in-the-loop feedback mechanism** that allows the model to self-correct and improve accuracy over time using Scikit-Learn.

## 🚀 Key Features
- **Face Detection & Age Prediction:** Uses OpenCV DNN with pre-trained Caffe/TensorFlow models.
- **Interactive Dashboard:** Clean, responsive web UI built with Flask, HTML, CSS, and JS.
- **Self-Learning Correction:** Users can provide feedback (Correct, Slightly Greater/Less, Wrong).
- **Adaptive ML Model:** A Random Forest Regressor learns from feedback to correct future predictions automatically.
- **Persistent Learning:** Feedback and trained models are saved locally, so the system gets smarter every time you use it.

## 🛠️ Tech Stack
- **Backend:** Python, Flask
- **Computer Vision:** OpenCV, NumPy
- **Machine Learning:** Scikit-Learn, Pandas, Joblib
- **Frontend:** HTML5, CSS3, JavaScript

## 📋 Prerequisites
Ensure you have Python installed. Then install the required libraries:
```bash
pip install flask opencv-python numpy scikit-learn pandas joblib

⚙️ Installation & Setup
Clone the repository:
git clone https://github.com/VinayTodkar/Age_detector.git
cd Age_detector

Download Model Files: This project requires specific pre-trained model files. Place the following files in the root directory:
opencv_face_detector.pbtxt
opencv_face_detector_uint8.pb
age_deploy.prototxt
age_net.caffemodel
Note: These files are not included in the repo due to size limits. You can download them using the commands below or manually from the links provided.PowerShell Commands to Download:
Invoke-WebRequest -Uri "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face/master/opencv_face_detector.pbtxt" -OutFile "opencv_face_detector.pbtxt"
Invoke-WebRequest -Uri "https://github.com/spmallick/learnopencv/raw/master/AgeGender/opencv_face_detector_uint8.pb" -OutFile "opencv_face_detector_uint8.pb"
Invoke-WebRequest -Uri "https://raw.githubusercontent.com/GilLevi/AgeGenderDeepLearning/master/models/age_deploy.prototxt" -OutFile "age_deploy.prototxt"
Invoke-WebRequest -Uri "https://github.com/GilLevi/AgeGenderDeepLearning/raw/master/models/age_net.caffemodel" -OutFile "age_net.caffemodel"

Run the Application:
python app.py

Open in Browser: Navigate to http://127.0.0.1:5000 in your web browser.

🎯 How to Use
Upload an Image: Drag and drop a .jpg or .png file into the dashboard.
View Prediction: The system will display the detected face(s) and predicted age.
Provide Feedback:
  Click Correct if the age is right.
  Click Slightly Greater/Less if it's close.
  Click Completely Wrong and enter the actual age.
Automatic Improvement: After 5+ feedback entries, the internal ML model retrains automatically. New predictions will reflect your corrections!

## Project Structure

Age_detector/
│
├── app.py                # Flask backend & ML logic
├── templates/
│   └── index.html        # Frontend dashboard
├── opencv_face_detector.pbtxt
├── opencv_face_detector_uint8.pb
├── age_deploy.prototxt
├── age_net.caffemodel
├── feedback_data.csv     # Stores user feedback (auto-generated)
├── correction_model.pkl  # Trained ML correction model (auto-generated)
├── scaler.pkl            # Scaler for ML model (auto-generated)
└── README.md

🧠 How the Learning Works
Base Prediction: OpenCV predicts an age range (e.g., "25-32").
User Feedback: You indicate if this is correct or wrong.
Data Logging: The system saves the predicted_age, confidence, and actual_age to feedback_data.csv.
Retraining: Once enough data is collected, a Random Forest Regressor is trained to learn the error pattern.
Correction: Future predictions are adjusted by this ML model, effectively "correcting" the base OpenCV model for your specific use case.
🤝 Contributing
Feel free to fork this project and submit pull requests. For major changes, please open an issue first to discuss what you would like to change.

📄 License
 - This project is open source and available under the MIT License.

👤 Author
 - Vinay Sunil Todkar

GitHub: Vinay Todkar
LinkedIn: Vinay Todkar
