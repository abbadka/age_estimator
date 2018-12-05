# https://blog.keras.io/building-a-simple-keras-deep-learning-rest-api.html used for parts of this code
# face detector code inspired from http://dlib.net/face_alignment.py.html

import os
from flask import Flask, render_template, request, Response, send_from_directory
from PIL import Image
import io
import pandas as pd
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
from keras.applications.resnet50 import ResNet50, preprocess_input
import keras.backend as K
import random
import json
import dlib
import cv2

# Constants used in the different functions

image_location = 'static/challenge_images/'
face_predictor_model = 'shape_predictor_5_face_landmarks.dat'
target_size = (224, 224)


# The metrics used by our model

def soft_accuracy_5(y_true, y_pred):
    return K.mean(abs(y_pred - y_true) < 5)


def soft_accuracy_10(y_true, y_pred):
    return K.mean(abs(y_pred - y_true) < 10)


# Loads our feature extractor, our primary model
# Additionally loads the face detector, and thecd
# predictor, which are used to identify the position of the face in the image
def load_models():
    global feature_extractor
    global model
    global face_detector
    global face_predictor

    feature_extractor = ResNet50(weights='imagenet', include_top=False)
    model = load_model('v19_augmentation_weights.best.hdf5', custom_objects={
                       'soft_accuracy_5': soft_accuracy_5, 'soft_accuracy_10': soft_accuracy_10})
    face_detector = dlib.get_frontal_face_detector()
    face_predictor = dlib.shape_predictor(face_predictor_model)


# Uses our feature extractor to extract bottleneck features before passing them to the model
def extract_features(img):
    img = np.expand_dims(img, axis=0)
    return feature_extractor.predict(preprocess_input(img))

# Detects a face in the image.
# If there is more than one face, it will take the first.


def detect_face(img):
    detected_faces = face_detector(img, 1)

    faces = dlib.full_object_detections()
    # for our purposes, we will return the first face we see
    for i, face_rect in enumerate(detected_faces):
        faces.append(face_predictor(img, face_rect))
        identified_face = dlib.get_face_chip(img, faces[0], size=224)
        return identified_face

# Uses a combination of the other methods to make a prediction based on a
# given image, used by both the challenge and the estimator portions.
# Returns NO_FACE if no face is found


def model_predict(img):
    face = detect_face(img)
    if face is not None:
        features = extract_features(face)
    else:
        return 'NO_FACE'
    pred = model.predict(features)
    return '{0:.2f}'.format(pred[0][0])

# creates the flask app and required end points


def create_app():
    app = Flask(__name__)

    challenge_files = [name for name in os.listdir(
        image_location) if name.endswith('jpg')]

    @app.route('/')
    def index():
        return render_template('index.html')

    @app.route('/estimator')
    def estimator():
        return render_template('estimator.html')

    @app.route('/challenge')
    def challenge():
        return render_template('challenge.html')

    @app.route("/predict", methods=["POST"])
    def predict():
        img = request.files["file"].read()
        img = np.fromstring(img, np.uint8)
        img = cv2.imdecode(img, cv2.IMREAD_UNCHANGED)
        return model_predict(img)

    @app.route("/challenge_image", methods=["GET"])
    def get_challenge_image():
        challenge_image = challenge_files[random.randrange(
            len(challenge_files))]
        source_age = challenge_image.split('_')[0]
        img = cv2.imread(image_location + challenge_image)
        predicted_age = model_predict(img)
        image_data = {
            'file_name': challenge_image,
            'source_age': source_age,
            'predicted_age': predicted_age
        }
        return Response(json.dumps(image_data), mimetype='application/json')

    @app.route('/image/<filename>', methods=["GET"])
    def send_image(filename):
        return send_from_directory(image_location, filename)

    # https://stackoverflow.com/questions/34066804/disabling-caching-in-flask
    @app.after_request
    def add_header(r):
        """
        Add headers to both force latest IE rendering engine or Chrome Frame,
        and also to cache the rendered page for 10 minutes.
        """
        r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        r.headers["Pragma"] = "no-cache"
        r.headers["Expires"] = "0"
        r.headers['Cache-Control'] = 'public, max-age=0'
        return r

    return app


# Initialize the flask app
if __name__ == "__main__":
    app = create_app()
    load_models()

    challenge_files = [name for name in os.listdir(
        image_location) if name.endswith('jpg')]

    # Without this, keras does not work properly
    # https://github.com/jrosebr1/simple-keras-rest-api/issues/5
    app.run(debug=False, threaded=False)
