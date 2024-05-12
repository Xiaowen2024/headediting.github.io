from flask import Flask, request, send_file
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


base_options = python.BaseOptions(model_asset_path='face_landmarker_v2_with_blendshapes.task')
options = vision.FaceLandmarkerOptions(base_options=base_options,
                                       output_face_blendshapes=True,
                                       output_facial_transformation_matrixes=True,
                                       num_faces=2)
detector = vision.FaceLandmarker.create_from_options(options)


app = Flask(__name__)


source_image = None
target_image = None
source_landmarks = None
target_landmarks = None

@app.route('/upload', methods=['POST'])
def get_images():
    download_file("https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task", "face_landmarker_v2_with_blendshapes.task")
    global source_image, target_image
    
    if 'file' not in request.files:
        return 'No image uploaded'
    image = request.files['file']
    
    if not source_image: 
        source_image = image
        return 'Source image uploaded'
    elif not target_image:
        target_image = image
        return 'Target image uploaded'
    else:
        return 'Both images uploaded'

def get_landmarks(source_image, target_image):
    global source_landmarks, target_landmarks
    source_image = cv2.imread(source_image)
    target_image = cv2.imread(target_image)
    
    source_image = cv2.cvtColor(source_image, cv2.COLOR_BGR2RGB)
    target_image = cv2.cvtColor(target_image, cv2.COLOR_BGR2RGB)

    source_image_loader = mp.Image(image_format=mp.ImageFormat.SRGB, data = source_image)
    target_image_loader = mp.Image(image_format=mp.ImageFormat.SRGB, data = target_image)
    source_detection_result =  detector.detect(source_image_loader)
    target_detection_result = detector.detect(target_image_loader)
    source_landmarks = source_detection_result.face_landmarks
    target_landmarks = target_detection_result.face_landmarks
    return source_landmarks, target_landmarks

def warp_image(source_image, target_image, source_landmarks, target_landmarks):
    source_left_eye = (source_landmarks[0][468].x * source_image.shape[1], source_landmarks[0][468].y * source_image.shape[0])
    source_right_eye = (source_landmarks[0][473].x * source_image.shape[1], source_landmarks[0][473].y * source_image.shape[0])
    target_left_eye = (target_landmarks[0][468].x * target_image.shape[1], target_landmarks[0][468].y * target_image.shape[0])
    target_right_eye = (target_landmarks[0][473].x * target_image.shape[1], target_landmarks[0][473].y * target_image.shape[0])
    source_eyes_center = ((source_left_eye[0] + source_right_eye[0]) // 2, (source_left_eye[1] + source_right_eye[1]) // 2)
    target_eyes_center = ((target_left_eye[0] + target_right_eye[0]) // 2, (target_left_eye[1] + target_right_eye[1]) // 2)
    source_eyes_dx = source_eyes_center[0] - target_eyes_center[0]
    source_eyes_dy = source_eyes_center[1] - target_eyes_center[1]
    height, width, _ = source_image.shape
    transformation_matrix = np.float32([[1, 0, source_eyes_dx], [0, 1, source_eyes_dy]])
    translated_image = cv2.warpAffine(target_image, transformation_matrix, (width, height))
    return translated_image

def get_face_mask(convex_hull, image):
    hull = get_face_contour(source_landmarks, source_image) 
    mask = np.zeros(source_image.shape[:2], dtype=np.uint8)
    mask = cv2.drawContours(mask, [hull], -1, (255, 255, 255), -1, cv2.LINE_AA)
    return mask

def send_mask(mask):
    # Convert the mask to a PIL Image
    mask_image = Image.fromarray(mask)

    # Create a BytesIO object to hold the image data
    image_buffer = BytesIO()

    # Save the mask image to the buffer in PNG format
    mask_image.save(image_buffer, format='PNG')

    # Set the buffer position to the start of the buffer
    image_buffer.seek(0)

    # Send the mask image as a file attachment
    return send_file(image_buffer, mimetype='image/png', as_attachment=True, attachment_filename='mask.png')

def get_face_contour(face_landmarks, image):
    indices = [10, 297, 67, 21, 127, 94, 58, 136, 149, 148, 377, 378, 365, 367, 435, 366, 447, 389, 284]
    face_contour = []
    for i in indices:
        face_contour.append((face_landmarks[i].x * image.shape[1], face_landmarks[i].y * image.shape[0]))
    convex_hull = cv2.convexHull(np.array(face_contour, dtype=np.float32)) 
    return convex_hull 

def download_file(url, filename):
    response = request.get(url)
    if response.status_code == 200:
        with open(filename, 'wb') as f:
            f.write(response.content)
        print(f"File downloaded successfully as {filename}")
    else:
        print("Failed to download file")



if __name__ == '__main__':
    app.run(debug=True)
