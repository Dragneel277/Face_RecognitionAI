from flask import Flask, render_template, Response
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

app = Flask(__name__)

# Load your trained model
model = load_model('your_model.h5')

# Function to predict face using the loaded model
def predict_face(frame):
    # Preprocess the frame for prediction
    face = cv2.resize(frame, (224, 224))  # Resize to match the input shape of the model
    face = img_to_array(face)
    face = np.expand_dims(face, axis=0)
    face = face / 255.0  # Normalize

    prediction = model.predict(face)[0][0]

    return prediction

# Route for the home page
@app.route('/')
def index():
    return render_template('index.html')

# Route for the student profile page
@app.route('/profile')
def profile():
    return render_template('profile.html')

# Route for real-time face detection
@app.route('/real_time_detection')
def real_time_detection():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Function to generate frames for real-time detection
def generate_frames():
    camera = cv2.VideoCapture(0)
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # Use the prediction function
            prediction = predict_face(frame)
            label = 'Student' if prediction > 0.5 else 'Others'

            # Display the label on the frame
            cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    camera.release()

if __name__ == '__main__':
    app.run(debug=True)
