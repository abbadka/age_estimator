import dlib
#import cv2

# inspired from http://dlib.net/face_alignment.py.html

face_detector = dlib.get_frontal_face_detector()
predictor_model = 'shape_predictor_5_face_landmarks.dat'

# Load all the models we need: a detector to find the faces, a shape predictor
# to find face landmarks so we can precisely localize the face
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_model)


def detect_face(filename):
    # Load the image using Dlib
    img = dlib.load_rgb_image(filename)
    detected_faces = face_detector(img, 1)

    faces = dlib.full_object_detections()
    # for our purposes, we will return the first face we see
    for i, face_rect in enumerate(detected_faces):
        faces.append(predictor(img, face_rect))
        identified_face = dlib.get_face_chip(img, faces[0], size=224)
        return identified_face
