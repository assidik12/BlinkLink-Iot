import cv2
import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras
import os
import time

# --- Configuration ---
# File containing pre-computed face embeddings.
# It's crucial that 'encode_face.py' is run first to generate this file.
embeddings_file = "./face_embeddings_tf.pkl"

# Path to the Haar Cascade XML file for frontal face detection.
# This path is typically found within the OpenCV installation.
face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'

# Input size for the face embedding model.
# This must match the size used during the encoding process.
face_embedding_input_size = (160, 160)

# Tolerance level for face recognition.
# A lower value means stricter matching (closer faces).
# This is a critical parameter for balancing false positives and false negatives.
RECOGNITION_TOLERANCE = 0.2

# --- Model and Data Loading ---
def load_face_recognition_assets():
    """
    Loads the face embedding model, pre-computed face embeddings,
    and the Haar Cascade classifier.
    """
    try:
        # Define a dummy model structure for inference.
        # In a real-world scenario, you would load a pre-trained model
        # like FaceNet or ArcFace. This dummy model serves as a placeholder
        # to ensure the input/output shapes are compatible with the embedding process.
        def create_dummy_face_embedding_model_for_inference():
            model = keras.Sequential([
                keras.layers.Input(shape=(face_embedding_input_size[0], face_embedding_input_size[1], 3)),
                keras.layers.Conv2D(32, (3, 3), activation='relu'),
                keras.layers.MaxPooling2D((2, 2)),
                keras.layers.Conv2D(64, (3, 3), activation='relu'),
                keras.layers.MaxPooling2D((2, 2)),
                keras.layers.Flatten(),
                keras.layers.Dense(128, activation=None) # No activation for embeddings
            ])
            return model
        
        face_embedding_model = create_dummy_face_embedding_model_for_inference()
        print("Face embedding model loaded (dummy for inference).")

        # Load pre-computed face embeddings and corresponding names.
        # The 'known_face_expressions' are loaded but not used in this version,
        # as per the original code's modification.
        with open(embeddings_file, 'rb') as f:
            known_face_encodings, known_face_names = pickle.load(f)
        print(f"Total {len(known_face_encodings)} face embeddings loaded.")

        # Initialize Haar Cascade Classifier for face detection.
        face_cascade = cv2.CascadeClassifier(face_cascade_path)
        if face_cascade.empty():
            raise IOError(f"Could not load Haar Cascade Classifier from: {face_cascade_path}")
        print("Haar Cascade Classifier loaded.")

        return face_embedding_model, known_face_encodings, known_face_names, face_cascade

    except FileNotFoundError:
        print(f"Error: Embeddings file '{embeddings_file}' not found.")
        print("Please ensure you have run 'encode_face.py' to generate the embeddings.")
        exit()
    except Exception as e:
        print(f"Error loading model or data: {e}")
        exit()

# --- Face Recognition Function ---
def find_closest_face(face_encoding, known_encodings, known_names, tolerance):
    """
    Compares a detected face encoding against a database of known face encodings
    to find the closest match. Uses Euclidean distance for similarity.

    Args:
        face_encoding (np.array): The embedding of the detected face.
        known_encodings (list of np.array): List of embeddings of known faces.
        known_names (list of str): List of names corresponding to known_encodings.
        tolerance (float): The maximum distance for a face to be considered a match.

    Returns:
        tuple: (name_of_closest_face, distance_to_closest_face).
               Returns ("Unknown", distance) if no face is within tolerance.
    """
    if not known_encodings:
        return "Unknown", 1.0  # Return a high distance if no known encodings

    # Calculate Euclidean distances between the detected face and all known faces.
    # np.linalg.norm computes the Euclidean norm (L2 norm).
    distances = np.linalg.norm(known_encodings - face_encoding, axis=1)
    
    # Find the index of the closest known face.
    min_distance_index = np.argmin(distances)
    min_distance = distances[min_distance_index]

    return known_names[min_distance_index], min_distance
    # If the minimum distance is below the tolerance, the face is recognized.
    # if min_distance < tolerance - 3:
    #     return known_names[min_distance_index], min_distance
    # else:
    #     return "Unknown", min_distance

# --- Main Execution Block ---
if __name__ == "__main__":
    face_embedding_model, known_face_encodings, known_face_names, face_cascade = load_face_recognition_assets()

    # Initialize webcam. '0' typically refers to the default webcam.
    video_capture = cv2.VideoCapture(0)

    if not video_capture.isOpened():
        print("Error: Could not open webcam. Exiting.")
        exit()

    print("Webcam opened. Press 'q' to quit.")

    frame_count = 0
    start_time = time.time()
    fps = 0 # Initialize fps

    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("Failed to grab frame. Exiting.")
            break

        # Flip the frame horizontally for a mirrored view (common for webcams).
        frame = cv2.flip(frame, 1)

        # Downsize the frame for faster processing.
        # This is a performance optimization; a smaller frame means faster detection,
        # but potentially less accurate for very small faces.
        processing_scale_factor = 0.5
        small_frame = cv2.resize(frame, (0, 0), fx=processing_scale_factor, fy=processing_scale_factor)
        
        # Convert to grayscale for Haar Cascade, which typically operates on grayscale images.
        gray_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)

        # Detect faces using Haar Cascade.
        # scaleFactor: How much the image size is reduced at each image scale.
        # minNeighbors: How many neighbors (detections) each candidate rectangle
        #               should have to retain a detection. Higher values reduce false positives.
        # minSize: Minimum possible object size. Objects smaller than this are ignored.
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Process each detected face.
        for (x, y, w, h) in faces:
            # Scale coordinates back to the original frame size.
            x_orig, y_orig, w_orig, h_orig = (int(coord / processing_scale_factor) for coord in (x, y, w, h))

            # Draw a rectangle around the detected face on the original frame.
            cv2.rectangle(frame, (x_orig, y_orig), (x_orig + w_orig, y_orig + h_orig), (0, 255, 0), 2)

            # Extract the Region of Interest (ROI) for the face.
            face_roi_rgb = frame[y_orig:y_orig + h_orig, x_orig:x_orig + w_orig]

            try:
                # Pre-process the face ROI for the embedding model:
                # 1. Resize to the model's expected input size.
                # 2. Convert BGR (OpenCV default) to RGB.
                # 3. Normalize pixel values to [0, 1].
                # 4. Add a batch dimension (expected by TensorFlow models).
                face_for_embedding = cv2.resize(cv2.cvtColor(face_roi_rgb, cv2.COLOR_BGR2RGB), face_embedding_input_size)
                face_for_embedding = face_for_embedding.astype('float32') / 255.0
                face_for_embedding = np.expand_dims(face_for_embedding, axis=0)

                # Get face embedding using the model.
                # `verbose=0` suppresses prediction output.
                face_embedding = face_embedding_model.predict(face_for_embedding, verbose=0)[0]

                # Perform face recognition.
                name, distance = find_closest_face(face_embedding, known_face_encodings, known_face_names, RECOGNITION_TOLERANCE)
                
                # Display the recognized name and distance.
                text = f"{name} (Dist: {distance:.2f})"
                cv2.putText(frame, text, (x_orig, y_orig - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

            except Exception as e:
                # Handle potential errors during face processing (e.g., if ROI is invalid).
                cv2.putText(frame, "Error", (x_orig, y_orig - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
                print(f"Error processing face at ({x_orig},{y_orig}): {e}")

        # Calculate and display FPS.
        frame_count += 1
        current_time = time.time()
        if current_time - start_time >= 1: # Update FPS every second
            fps = frame_count / (current_time - start_time)
            frame_count = 0
            start_time = current_time
        
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv2.LINE_AA)

        # Display the output frame.
        cv2.imshow('Real-time Face Recognition', frame)

        # Exit loop if 'q' key is pressed.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release webcam and destroy all OpenCV windows.
    video_capture.release()
    cv2.destroyAllWindows()