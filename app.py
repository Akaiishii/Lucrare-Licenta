from flask import Flask, request, render_template, redirect, url_for, flash
import cv2
import numpy as np
import pickle
from utils import get_face_landmarks

app = Flask(__name__)
app.secret_key = 'supersecretkey'

# Load the pre-trained model
with open('model', 'rb') as f:
    model = pickle.load(f)

emotions = ['HAPPY', 'SAD', 'SURPRISED']


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash("No file part")
        return redirect(request.url)

    file = request.files['file']
    if file.filename == '':
        flash("No selected file")
        return redirect(request.url)

    if file:
        # Save the original file
        original_filepath = './static/original_image.jpg'
        file.save(original_filepath)

        # Load and resize the image with aspect ratio preservation
        image = cv2.imread(original_filepath)
        h, w = image.shape[:2]

        if h < 100 or w < 100:
            flash("The size of the image is too small, please choose another.")
            return redirect(request.url)

        if h > 500 or w > 500:
            scale = min(500 / h, 500 / w)
            image = cv2.resize(image, (int(w * scale), int(h * scale)))

        resized_filepath = './static/uploaded_image.jpg'
        cv2.imwrite(resized_filepath, image)

        # Process the image for emotion detection
        face_landmarks = get_face_landmarks(image)

        if len(face_landmarks) == 1404:
            prediction = model.predict_proba([face_landmarks])[0]
            emotion_index = np.argmax(prediction)
            emotion = emotions[emotion_index]
            confidence = prediction[emotion_index]
            return render_template('index.html', emotion=emotion, confidence=confidence)

    return redirect(url_for('index'))


if __name__ == '__main__':
    app.run(debug=True)
